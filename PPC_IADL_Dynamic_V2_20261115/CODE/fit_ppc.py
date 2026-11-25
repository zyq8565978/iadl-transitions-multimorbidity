"""Frozen V2 history contrasts; sample-conditional PPC models, no search."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
from pathlib import Path
import hashlib,json,warnings
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy.special import expit,logit
from scipy.stats import t
from statsmodels.stats.sandwich_covariance import cov_cluster

ROOT=Path(__file__).resolve().parents[1]
PRIVATE=ROOT/'PRIVATE'; TABLES=ROOT/'TABLES'
COMPARE={'H11_vs_H01':('H11','H01'),'H10_vs_H00':('H10','H00')}
SCENARIOS=['primary','age50_index','earliest','old_cognition','common_fixed','common_old','strict_medication']
FIELDS=['age1','sex1','education1','chronic_count1','cesd1','cog_z1','wave','gap01','gap12']

class DiagnosticFailure(ValueError):
    def __init__(self,message,diagnostics):
        super().__init__(message);self.diagnostics=diagnostics

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def save(rows,name):
    pd.DataFrame(rows).to_csv(TABLES/(name+'.csv'),index=False,encoding='utf-8-sig')

def count(d):
    return dict(windows=len(d),people=int(d.person.nunique()),families=int(d.cluster.nunique()))

def eligible(q,tag='fixed',strict=False):
    h='history_old' if strict else 'history'
    return q.ordered_entry&q[h].ne('unknown')&q[f'd_{tag}1'].le(1)&q[['chronic_count1','cesd1','cog_z1']].notna().all(axis=1)

def stages(q,comparison,spec):
    a,b=COMPARE[comparison]
    tag='old' if spec in ['old_cognition','common_old'] else 'fixed'
    h='history_old' if spec=='strict_medication' else 'history'
    kval='K_old1' if spec=='strict_medication' else 'K1'
    select=eligible(q,tag,spec=='strict_medication')
    if spec=='earliest':select&=q.first_eligible
    if spec=='age50_index':select&=q.age1.ge(50)
    initial=q.loc[select&q[h].isin([a,b])].copy()
    initial['arm']=initial[h]; initial['hx']=initial.arm.eq(a).astype(int)
    initial['y']=initial[f'y_{tag}']; initial['p0']=initial[f'code_{tag}1']
    initial['kgrp']=initial[kval].clip(upper=3)
    observed=initial.loc[initial.y.notna()].copy()
    complete=observed.dropna(subset=FIELDS+['p0','kgrp']).copy()
    context=['wave','p0','kgrp'] if comparison=='H11_vs_H01' else ['wave','p0']
    shared=restrict(complete,context)
    return [initial,observed,complete,shared],context

def restrict(d,context):
    if d.empty:return d.copy()
    supported=d.groupby(context,observed=True).arm.nunique()
    return d.loc[pd.MultiIndex.from_frame(d[context]).isin(supported.index[supported.eq(2)])].copy()

def common_sets(q,comparison):
    aa,ca=stages(q,comparison,'common_fixed');bb,cb=stages(q,comparison,'common_old')
    shared=aa[-1].index.intersection(bb[-1].index)
    while True:
        a=restrict(aa[2].loc[shared],ca);b=restrict(bb[2].loc[shared],cb)
        nxt=a.index.intersection(b.index)
        if nxt.equals(shared):break
        shared=nxt
    aa[-1]=a.loc[shared].copy();bb[-1]=b.loc[shared].copy()
    assert aa[-1].index.equals(bb[-1].index)
    return {'common_fixed':(aa,ca),'common_old':(bb,cb)}

def support_rows(source,comparison,spec,stageset,context):
    rows=[];missing=[];cells=[]
    for stage,d in zip(['eligible_baseline','observed_endpoint','complete_covariates','shared_background'],stageset):
        for arm in COMPARE[comparison]:
            z=d.loc[d.arm.eq(arm)]
            rows.append(dict(source=source,comparison=comparison,specification=spec,stage=stage,arm=arm,**count(z),
                             events=int(z.y.eq(1).sum()),nonevents=int(z.y.eq(0).sum()),unknown_endpoint=int(z.y.isna().sum())))
    obs=stageset[1]
    for field in FIELDS+['p0','kgrp']:
        missing.append(dict(source=source,comparison=comparison,specification=spec,field=field,observed_endpoint_n=len(obs),missing_n=int(obs[field].isna().sum()),
                            missing_people=int(obs.loc[obs[field].isna(),'person'].nunique())))
    for keys,z in stageset[2].groupby(context+['arm'],observed=True):
        cells.append(dict(source=source,comparison=comparison,specification=spec,**dict(zip(context+['arm'],keys)),**count(z),
                          events=int(z.y.eq(1).sum()),nonevents=int(z.y.eq(0).sum()),in_shared_background=bool(z.index.isin(stageset[3].index).all())))
    return rows,missing,cells

def matrix(d,source,comparison,layer):
    d=d.copy();d['a10']=(d.age1-65)/10; d['g01']=d.gap01/12;d['g12']=d.gap12/12
    d['extra_disease']=(d.chronic_count1-1).clip(lower=0)
    formula='hx + a10 + I(a10 ** 2) + C(sex1) + '+('education1' if source=='HRS' else 'C(education1)')
    formula+=' + C(p0) + extra_disease + C(wave) + g01 + g12'
    if comparison=='H11_vs_H01':formula+=' + C(kgrp)'
    if layer=='M_severity':formula+=' + cesd1 + cog_z1'
    ds=patsy.dmatrix(formula,d,return_type='dataframe',NA_action='raise')
    keep=[c for c in ds if c=='Intercept' or ds[c].nunique()>1]
    return ds[keep],formula,[c for c in ds if c not in keep]

def sandwich(x,y,p,cluster):
    h=x.T@(x*(p*(1-p))[:,None]);hi=np.linalg.inv(h)
    fam,index=np.unique(cluster,return_inverse=True)
    scores=np.zeros((len(fam),x.shape[1]));np.add.at(scores,index,x*(y-p)[:,None])
    n,k=x.shape;g=len(fam)
    vc=hi@(scores.T@scores)@hi*(g/(g-1))*(n-1)/(n-k)
    return (vc+vc.T)/2,h

def functional(value,grad,cov,crit,risk=False):
    variance=float(grad@cov@grad)
    if not np.isfinite(variance) or variance < -1e-10:raise ValueError('invalid functional variance')
    se=np.sqrt(max(0,variance))
    lo,hi=(expit(logit(value)+np.array([-1,1])*crit*se/(value*(1-value))) if risk else value+np.array([-1,1])*crit*se)
    return [float(value),float(se),float(lo),float(hi)]

def standardized(x,b,cov,j,crit,mask):
    z=x[mask].copy(); u=z.copy();v=z.copy();u[:,j]=1;v[:,j]=0
    p1=expit(u@b);p0=expit(v@b)
    g1=(u*(p1*(1-p1))[:,None]).mean(axis=0);g0=(v*(p0*(1-p0))[:,None]).mean(axis=0)
    estimates=[functional(p1.mean(),g1,cov,crit,True),functional(p0.mean(),g0,cov,crit,True),functional(p1.mean()-p0.mean(),g1-g0,cov,crit)]
    row={}
    for name,values in zip(['risk_index','risk_reference','risk_difference'],estimates):
        row.update({name+suffix:value for suffix,value in zip(['','_se','_lo','_hi'],values)})
    return row,u,v,g1,g0

def fit_one(d,source,comparison,spec,layer):
    ds,formula,constants=matrix(d,source,comparison,layer)
    x=ds.to_numpy(float);y=d.y.to_numpy(float);clusters=d.cluster.to_numpy(str)
    n,k=x.shape;g=len(set(clusters));j=ds.columns.get_loc('hx')
    if np.linalg.matrix_rank(x)!=k or g<=1 or n<=k:raise ValueError('design rank or degrees of freedom insufficient')
    if len(set(y))!=2 or d.hx.nunique()!=2:raise ValueError('only one outcome or history group')
    with warnings.catch_warnings(record=True) as warn:
        warnings.simplefilter('always');fit=sm.GLM(y,x,family=sm.families.Binomial()).fit(maxiter=150,tol=1e-11)
    b=np.asarray(fit.params);p=np.asarray(fit.fittedvalues)
    if not fit.converged or not np.isfinite(b).all() or any('separation' in str(w.message).lower() for w in warn):raise ValueError('nonconvergence or separation')
    cov,h=sandwich(x,y,p,clusters);eig=np.linalg.eigvalsh(h);condition=np.linalg.cond(h)
    score=float(abs(x.T@(y-p)).max()); error=float(abs(cov-np.asarray(cov_cluster(fit,clusters,use_correction=True))).max())
    if not np.isfinite(cov).all() or eig.min()<=1e-10 or condition>1e12 or score>1e-5 or error>1e-6 or np.linalg.eigvalsh(cov).min()<-1e-8:
        stem=f'ppc_{source}_{comparison}_{spec}_{layer}_FAILED'
        diag=dict(information_min=float(eig.min()),information_condition=float(condition),score_max=score,sandwich_error=error,
                  covariance_min_eigenvalue=float(np.linalg.eigvalsh(cov).min()),max_abs_coefficient=float(abs(b).max()),formula=formula,
                  fitted_probability_min=float(p.min()),fitted_probability_max=float(p.max()),warnings=';'.join(str(w.message) for w in warn))
        np.savez_compressed(PRIVATE/(stem+'.npz'),X=x,y=y,cluster=clusters,beta=b,cov=cov,information=h,terms=np.asarray(ds.columns,dtype=str),row_index=d.index.to_numpy(int))
        d.to_parquet(PRIVATE/(stem+'_rows.parquet'),index=True)
        (PRIVATE/(stem+'_diagnostics.json')).write_text(json.dumps(diag,indent=2),encoding='utf-8')
        raise DiagnosticFailure('nonfinite covariance, weak information, score, or sandwich check',diag)
    crit=t.ppf(.975,g-1);m=np.ones(n,bool)
    row,u,v,g1,g0=standardized(x,b,cov,j,crit,m)
    e=np.zeros(k);e[j]=1; beta,se,lo,hi=functional(b[j],e,cov,crit)
    row.update(log_or=beta,log_or_se=se,log_or_lo=lo,log_or_hi=hi,odds_ratio=float(np.exp(beta)),odds_ratio_lo=float(np.exp(lo)),odds_ratio_hi=float(np.exp(hi)),
               nominal_p=float(2*t.sf(abs(beta/se),g-1)),rank=k,information_min=float(eig.min()),information_condition=float(condition),
               score_max=score,sandwich_error=error,prediction_min=float(p.min()),prediction_max=float(p.max()),
               max_leverage=float(np.max(((x@np.linalg.inv(h))*x).sum(axis=1)*p*(1-p))),
               formula=formula,constant_columns_not_estimated=';'.join(constants),warnings=';'.join(str(w.message) for w in warn),
               standardized_target='shared-background observed-endpoint complete-baseline person-window empirical distribution',
               repeated_window_inference='family-connected HC1 sandwich; t(G-1)',reference_distribution_uncertainty='conditional on empirical reference and fixed measurement parameters')
    stem=f'ppc_{source}_{comparison}_{spec}_{layer}'
    np.savez_compressed(PRIVATE/(stem+'.npz'),X=x,y=y,cluster=clusters,beta=b,cov=cov,information=h,terms=np.asarray(ds.columns,dtype=str),
         X_index=u,X_reference=v,gradient_index=g1,gradient_reference=g0,row_index=d.index.to_numpy(int),p0=d.p0.to_numpy(float),person=d.person.to_numpy(str),window=d.window.to_numpy(str))
    d.to_parquet(PRIVATE/(stem+'_rows.parquet'),index=True)
    strata=[]
    for name,mask in [('zero_domain',d.p0.eq(0).to_numpy()),('one_domain',d.p0.ne(0).to_numpy()),('P_only',d.p0.eq(1).to_numpy()),('N',d.p0.eq(0).to_numpy())]:
        if not mask.any():continue
        zz=d.loc[mask]; out,*_=standardized(x,b,cov,j,crit,mask)
        strata.append(dict(source=source,comparison=comparison,specification=spec,model_layer=layer,stratum=name,**count(zz),**out,
              status='estimated_shared_coefficients',interaction_test_performed=False))
    coeff=[]
    for name,value,se0 in zip(ds.columns,b,np.sqrt(np.diag(cov))):
        coeff.append(dict(source=source,comparison=comparison,specification=spec,model_layer=layer,term=name,estimate=value,se=se0))
    return row,strata,coeff

def run(source):
    lock=json.loads((ROOT/'PROTOCOL_LOCK.json').read_text(encoding='utf-8'))
    assert digest(ROOT/lock['protocol'])==lock['sha256']
    path=PRIVATE/f'{source}_windows.parquet';q=pd.read_parquet(path)
    assert not q.duplicated(['person','window']).any()
    results=[];support=[];missing=[];cells=[];strata=[];coef=[]
    specs=[s for s in SCENARIOS if s!='strict_medication' or source=='HRS']
    for comparison,(a,b) in COMPARE.items():
        common=common_sets(q,comparison) if source!='CHARLS' else {}
        for spec in specs:
            if source=='CHARLS':
                for layer in ['M_basic','M_severity']:
                    results.append(dict(source=source,comparison=comparison,specification=spec,model_layer=layer,index_arm=a,reference_arm=b,
                        status='NOT_ESTIMABLE_MEASUREMENT_PROTOCOL_BREAK',detail='2018/2020 delayed-memory exposure follows three learning trials; no fixed2013-comparable four-test PPC endpoint'))
                continue
            parts,context=common[spec] if spec in common else stages(q,comparison,spec)
            rr,mm,cc=support_rows(source,comparison,spec,parts,context);support+=rr;missing+=mm;cells+=cc
            d=parts[-1]
            for layer in ['M_basic','M_severity']:
                row=dict(source=source,comparison=comparison,specification=spec,model_layer=layer,index_arm=a,reference_arm=b,role='primary_contrast' if comparison=='H11_vs_H01' else 'key_secondary_contrast',**count(d))
                for arm,key in [(a,'index'),(b,'reference')]:
                    z=d.loc[d.arm.eq(arm)]
                    row.update({key+'_windows':len(z),key+'_people':z.person.nunique(),key+'_families':z.cluster.nunique(),key+'_events':int(z.y.eq(1).sum()),key+'_nonevents':int(z.y.eq(0).sum()),key+'_raw_risk':z.y.mean()})
                try:
                    if d.empty or d.arm.nunique()!=2:raise ValueError('no shared background containing both history groups')
                    out,ss,cf=fit_one(d,source,comparison,spec,layer)
                    row.update(status='estimated',**out);strata+=ss;coef+=cf
                except (ValueError,np.linalg.LinAlgError,sm.tools.sm_exceptions.PerfectSeparationError) as err:
                    row.update(status='not_estimable',detail=str(err))
                    if isinstance(err,DiagnosticFailure):row.update(err.diagnostics)
                results.append(row)
                save(results,source+'_ppc_results')
                print(source,comparison,spec,layer,row['status'],flush=True)
    save(results,source+'_ppc_results')
    save(support,source+'_ppc_support');save(missing,source+'_ppc_covariate_missing');save(strata,source+'_ppc_standardized_strata');save(coef,source+'_ppc_coefficients')
    # Exact cells are for disclosure-reviewed aggregation, not automatic public release.
    pd.DataFrame(cells).to_csv(PRIVATE/(source+'_ppc_exact_support_cells.csv'),index=False)
    (TABLES/(source+'_ppc_run.json')).write_text(json.dumps(dict(source=source,protocol_sha256=lock['sha256'],input_sha256=digest(path),finished_at=datetime.now(timezone.utc).isoformat(),models_estimated=sum(r['status']=='estimated' for r in results),models_total=len(results)),indent=2),encoding='utf-8')

def refresh_failed(source):
    """Same specification, only enrich diagnostics for already non-estimable fits."""
    path=TABLES/(source+'_ppc_results.csv');results=pd.read_csv(path)
    q=pd.read_parquet(PRIVATE/f'{source}_windows.parquet')
    for i,row in results.loc[results.status.eq('not_estimable')].iterrows():
        spec=row.specification;comparison=row.comparison
        parts,_=common_sets(q,comparison)[spec] if spec in ['common_fixed','common_old'] else stages(q,comparison,spec)
        try:
            fit_one(parts[-1],source,comparison,spec,row.model_layer)
            raise RuntimeError('A failed fit unexpectedly became estimable without model/data change; inspect before updating')
        except DiagnosticFailure as err:
            for key,value in err.diagnostics.items():
                if isinstance(value,str) and key in results:results[key]=results[key].astype(object)
                results.loc[i,key]=value
            print(source,comparison,spec,row.model_layer,err.diagnostics,flush=True)
    results.to_csv(path,index=False,encoding='utf-8-sig')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('source',choices=['HRS','ELSA','CHARLS']);p.add_argument('--refresh-failed-diagnostics',action='store_true');a=p.parse_args()
    refresh_failed(a.source) if a.refresh_failed_diagnostics else run(a.source)

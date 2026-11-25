"""V4 B: authorized local execution; no writes to previous versions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import tempfile

for var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[var]='1'

import numpy as np
import pandas as pd
import patsy
from scipy.special import expit
from scipy.stats import f, t
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster

ROOT=Path(__file__).resolve().parents[1]
V3=ROOT.parent/'PPC_IADL_Dynamic_V3_20261116'
NEXT={'CHARLS':{2015:2018,2018:2020},'HRS':{2014:2016,2016:2018,2018:2020},
      'ELSA':{2:3,3:4,4:5,7:8,8:9}}
BITS={'physical':1,'psychological':2,'cognitive':4}
INITIAL={'physical':'chronic_count1','psychological':'cesd_z1','cognitive':'cog_z1'}
FIXED_TAIL='age10 + I(age10 ** 2) + C(sex1) + {education} + {clinical} + C(index_wave) + gap01_years'


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()


def fit_formula(source, comp, domain, model):
    clinical=['C(code_fixed1)','extra_disease','cesd_z1','cog_z1']
    if model=='index_included':
        clinical[0]='C(other_domain_code)'
        clinical.remove({'physical':'extra_disease','psychological':'cesd_z1','cognitive':'cog_z1'}[domain])
    history={'ri_linear':'hx * time2','ri_quadratic':'hx * (time2 + I(time2 ** 2))',
             'index_included':'hx * (post + post_time2)'}[model]
    return history+' + '+FIXED_TAIL.format(education='education1' if source=='HRS' else 'C(education1)',
            clinical=' + '.join(clinical))+(' + C(Kgroup)' if comp=='H11_minus_H01' else '')


def working_inverse(x, person, residual_var, random_var):
    """Apply (sigma_e^2 I + sigma_b^2 J)^{-1} by person, without dense blocks."""
    sums=np.zeros((person.max()+1,x.shape[1]))
    np.add.at(sums,person,x)
    sizes=np.bincount(person)
    factor=random_var/(residual_var+random_var*sizes)
    return (x-factor[person,None]*sums[person])/residual_var


def robust(x, score_rows, bread, family):
    _,codes=np.unique(family,return_inverse=True)
    groups=int(codes.max())+1
    scores=np.zeros((groups,x.shape[1]))
    np.add.at(scores,codes,score_rows)
    if groups<=x.shape[1] or len(x)<=x.shape[1]:
        raise ValueError('Insufficient family / row degrees of freedom')
    cov=bread@(scores.T@scores)@bread*(groups/(groups-1))*((len(x)-1)/(len(x)-x.shape[1]))
    cov=(cov+cov.T)/2
    if not np.isfinite(cov).all() or np.linalg.eigvalsh(cov).min()<-1e-9:
        raise ValueError('Invalid sandwich')
    return cov,groups


def fit_ri(x,y,d,rscript):
    people=pd.factorize(d.person)[0]
    work=ROOT/'PRIVATE'
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='b_fit_',dir=work) as scratch:
        directory=Path(scratch)
        a=pd.DataFrame(x,columns=[f'x{i}' for i in range(x.shape[1])])
        a.insert(0,'person',people)
        a.insert(0,'y',y)
        a.to_csv(directory/'input.csv',index=False)
        fit=subprocess.run([rscript,'--vanilla','../../CODE/b_nlme.R','.'],cwd=scratch,
                           capture_output=True,text=True,timeout=180,check=False)
        if fit.returncode:
            if 'convergence' in fit.stderr.lower():
                raise ValueError('nlme optimizer did not converge under frozen controls')
            raise ValueError(f'nlme failed, exit {fit.returncode}')
        beta=pd.read_csv(directory/'coef.csv').beta.to_numpy()
        model_cov=pd.read_csv(directory/'vcov.csv').to_numpy()
        diag=pd.read_csv(directory/'diagnostics.csv').fillna('').iloc[0].to_dict()
    e,b=diag['residual_var'],diag['random_intercept_var']
    if not np.isfinite(beta).all() or e<=0 or b<0 or not np.isfinite(e+b):
        raise ValueError('nlme returned inadmissible coefficients or variances')
    vix=working_inverse(x,people,e,b)
    vi_y=working_inverse(y[:,None],people,e,b).ravel()
    bread=np.linalg.inv(x.T@vix)
    residual=y-x@beta
    v_res=working_inverse(residual[:,None],people,e,b).ravel()
    cov,groups=robust(x,x*v_res[:,None],bread,d.cluster.to_numpy())
    cov_err=float(np.max(np.abs(model_cov-bread)))
    beta_err=float(np.max(np.abs(beta-bread@x.T@vi_y)))
    score_max=float(np.max(np.abs(x.T@v_res)))
    if cov_err>1e-6 or beta_err>1e-6 or score_max>1e-4:
        raise ValueError('Independent GLS and REML numeric QA disagree')
    diag.update(model_covariance_check_error=cov_err,gls_coefficient_check_error=beta_err,
                score_max=score_max,icc=b/(e+b),variance_boundary=b/(e+b)<1e-6,
                bread_condition=float(np.linalg.cond(bread)))
    return beta,cov,groups,diag


def fit_index(x,y,d,bounded):
    fit=(sm.GLM(y,x,family=sm.families.Binomial()).fit(maxiter=150,tol=1e-10)
         if bounded else sm.OLS(y,x).fit(method='qr'))
    beta=np.asarray(fit.params)
    mu=np.asarray(fit.fittedvalues)
    resid=y-mu
    w=mu*(1-mu) if bounded else np.ones(len(y))
    bread=np.linalg.inv(x.T@(w[:,None]*x))
    cov,groups=robust(x,x*resid[:,None],bread,d.cluster.to_numpy())
    err=float(np.max(np.abs(cov-cov_cluster(fit,d.cluster,use_correction=True))))
    score=float(np.max(np.abs(x.T@resid)))
    if not getattr(fit,'converged',True) or err>1e-6 or score>1e-5 or not np.isfinite(beta).all():
        raise ValueError('B2 convergence or independent sandwich check failed')
    return beta,cov,groups,dict(sandwich_check_error=err,score_max=score,bread_condition=float(np.linalg.cond(bread)))


def arm_mean(di,beta,target,arm,year,bounded):
    post=(np.asarray(year)>0).astype(int)
    x=np.asarray(patsy.build_design_matrices([di],target.assign(hx=arm,time2=year-2,
                                post=post,post_time2=post*(year-2)))[0])
    eta=x@beta
    if not bounded:
        return float(eta.mean()), x.mean(axis=0)
    p=expit(eta)
    if not ((p>0)&(p<1)).all():
        raise ValueError('Degenerate disease-count prediction')
    return float(7*p.mean()), (7*x*(p*(1-p))[:,None]).mean(axis=0)


def subtract(a,b):
    return a[0]-b[0],a[1]-b[1]


def output_mean_rows(di,beta,cov,groups,ref,year,bounded,infer,base,start=None):
    at={h:arm_mean(di,beta,ref,h,year,bounded) for h in (1,0)}
    summary=[('standardized_mean',h,v,bounded) for h,v in at.items()]
    summary.append(('standardized_mean_difference',None,subtract(at[1],at[0]),False))
    if start is not None:
        prior={h:arm_mean(di,beta,ref,h,start,bounded) for h in (1,0)}
        change={h:subtract(at[h],prior[h]) for h in prior}
        summary.extend(('start_mean',h,v,bounded) for h,v in prior.items())
        summary.append(('start_mean_difference',None,subtract(prior[1],prior[0]),False))
        summary.extend(('within_history_change',h,v,False) for h,v in change.items())
        summary.append(('between_history_change_difference',None,subtract(change[1],change[0]),False))
    critical=float(t.ppf(.975,groups-1))
    return [dict(**base,functional=k,arm=h,**infer(v[0],v[1],cov,critical,b)) for k,h,v,b in summary]


def next_targets(d,target,source,dates):
    frames=[]
    for iw,nw in NEXT[source].items():
        year=float(dates[nw]-dates[iw])/12
        observed=d.loc[d.index_wave.eq(iw)&d.wave.eq(nw)]
        s=observed.groupby('hx').years.agg(['min','max','size']).reindex([0,1])
        if s['min'].le(year).all() and s['max'].ge(year).all() and s['size'].ge(30).all():
            frames.append(target.loc[target.index_wave.eq(iw)].assign(next_year=year))
    return pd.concat(frames) if frames else target.iloc[:0].copy()


def run(rscript):
    old=runpy.run_path(str(V3/'CODE/long_v3.py'))
    source_lock=json.loads((V3/'TABLES/LONG_RUN_V3.json').read_text(encoding='utf-8'))
    frozen_results=pd.read_csv(V3/'TABLES/LONG_RESULTS_V3.csv')
    out=[]; qa=[]; curves=[]; support=[]
    for source in old['LAST']:
        folder=old['DIAG'] if source=='CHARLS' else old['OLD']
        for stem,sha in source_lock['sources'][source].items():
            if digest(folder/'PRIVATE'/f'{source}_{stem}.parquet')!=sha:
                raise RuntimeError(f'{source} frozen V3 {stem} hash mismatch')
        panel,entry,obs=old['prepare'](source)
        dates=panel.groupby('wave').month.median()
        for comp in old['COMPARE']:
            for domain, field, last in zip(old['DOMAINS'],old['DOMAINS'].values(),old['LAST'][source]):
                bounded=domain=='physical'
                d,_=old['sample'](entry,obs,comp,domain,6.)
                d=old['derived'](d)
                base={'source':source,'comparison':comp,'domain':domain,
                      'n_people':d.person.nunique(),'n_families':d.cluster.nunique(),'n_post_records':len(d)}
                old_formula=fit_formula(source,comp,domain,'ri_linear')
                x0=patsy.dmatrix(old_formula,d,return_type='dataframe',NA_action='raise')
                with np.load(V3/'PRIVATE'/f'LONG_{source}_{comp}_{domain}_06.npz',allow_pickle=False) as z:
                    exactly={'X':np.asarray(x0),'y':d[field].to_numpy(float)/(7 if bounded else 1),
                             'person':d.person.to_numpy(str),'cluster':d.cluster.to_numpy(str),
                             'years':d.years.to_numpy(float),'terms':x0.columns.to_numpy(str)}
                    if not all(np.array_equal(z[k],v) for k,v in exactly.items()):
                        raise RuntimeError('V3 person/outcome/design identity failed')
                prior=frozen_results.loc[frozen_results.source.eq(source)&frozen_results.comparison.eq(comp)&
                       frozen_results.domain.eq(domain)&frozen_results.functional.eq('standardized_mean_difference')]
                wave_gap={w:float(dates[last]-dates[w]) for w in entry.index_wave.unique()}
                opportunity={w:(int(np.floor(g/12+.5)),g) for w,g in wave_gap.items()}
                wave_sets={yr:old['support_for_year'](d,entry,opportunity,yr,base,[]) for yr in (2.,4.,6.)}
                target=d.drop_duplicates('person')
                for row in prior.itertuples(index=False):
                    sites=';'.join(map(str,wave_sets[row.time_years]))
                    assert sites==(row.included_index_waves if pd.notna(row.included_index_waves) else '')
                    assert target.index_wave.isin(wave_sets[row.time_years]).sum()==row.target_people
                common=base|dict(exact_V3_identity=True,exact_V3_target=True,scale=old['UNITS'][domain])
                eligible3=d.groupby(['index_wave','wave','hx']).person.nunique().unstack('hx').ge(30).all(axis=1)
                # Both history columns are present overall; an absent arm within a wave is not support.
                waves3=eligible3[eligible3].groupby(level='index_wave').size()
                quad=source in ('HRS','ELSA') and bool(waves3.ge(3).any()) and not bounded
                support.append(dict(**common,eligible_quadratic=quad,
                                    supported_three_post_wave_indexes=';'.join(map(str,waves3[waves3.ge(3)].index)),
                                    **{f'V3_{int(y)}y_waves':';'.join(map(str,s)) for y,s in wave_sets.items()}))
                bspec=([] if bounded else ['ri_linear'])+(['ri_quadratic'] if quad else [])+['index_included']
                for model in bspec:
                    metrics=common|dict(model=model)
                    post_index=model!='index_included'
                    if post_index:
                        data=d
                    else:
                        initial=obs.loc[obs.person.isin(target.person)&obs.wave.eq(obs.index_wave)].copy()
                        assert len(initial)==len(target) and initial.person.is_unique and initial.years.eq(0).all()
                        np.testing.assert_array_equal(initial[field].to_numpy(),initial[INITIAL[domain]].to_numpy())
                        initial['hx']=initial.history.eq(old['COMPARE'][comp][0]).astype(int)
                        data=old['derived'](pd.concat([initial,d],ignore_index=True))
                        data=data.assign(post=data.years.gt(0).astype(int),
                                         other_domain_code=data.code_fixed1.astype(int)&(7-BITS[domain]))
                        data['post_time2']=data.post*data.time2
                        metrics['n_index_records']=len(initial)
                    metrics['n_total_records']=len(data)
                    try:
                        design=(x0 if model=='ri_linear' else
                                patsy.dmatrix(fit_formula(source,comp,domain,model),data,
                                              return_type='dataframe',NA_action='raise'))
                        x=np.asarray(design)
                        y=data[field].to_numpy(float)/(7 if bounded else 1)
                        if np.linalg.matrix_rank(x)!=x.shape[1] or not np.isfinite(y).all():
                            raise ValueError('Rank or response check failed')
                        if bounded and not np.all((y>=0)&(y<=1)):
                            raise ValueError('Illegal disease count')
                        beta,cov,groups,diag=(fit_ri(x,y,data,rscript) if post_index else fit_index(x,y,data,bounded))
                        if diag['bread_condition']>1e12:
                            raise ValueError('Ill-conditioned information matrix')
                        qa.append(dict(**metrics,status='estimated',parameters=x.shape[1],**diag))
                        state=dict(**metrics,status='estimated')
                        di=design.design_info
                        if post_index:
                            terms=[i for i,c in enumerate(di.column_names) if 'hx:' in c]
                            b=beta[terms]; v=cov[np.ix_(terms,terms)]
                            fr=float(b@np.linalg.solve(v,b)/len(b))
                            out.append(dict(**state,functional='joint_history_time_test',F=fr,df1=len(b),
                                            df2=groups-1,nominal_p=f.sf(fr,len(b),groups-1)))
                            if model=='ri_linear':
                                i=di.column_names.index('hx:time2')
                                out.append(dict(**state,functional='history_time_interaction',nominal_p=2*t.sf(abs(beta[i]/np.sqrt(cov[i,i])),groups-1),
                                    **old['infer'](beta[i],np.eye(len(beta))[i],cov,t.ppf(.975,groups-1))))
                        base_target=data.drop_duplicates('person')
                        for yr,waves in wave_sets.items():
                            ref=base_target.loc[base_target.index_wave.isin(waves)]
                            s=state|dict(target_kind='fixed_landmark',time_years=yr,target_people=len(ref),
                                         target_index_waves=';'.join(map(str,waves)))
                            if not len(ref):
                                out.append(s|dict(status='time_not_supported',functional='standardized_mean_difference'))
                                continue
                            out.extend(output_mean_rows(di,beta,cov,groups,ref,yr,bounded,old['infer'],s,None if post_index else 0.))
                        if post_index:
                            yr=4. if wave_sets[4.] else 2.
                            ref=target.loc[target.index_wave.isin(wave_sets[yr])]
                            if len(ref):
                                bounds=d.loc[d.index_wave.isin(wave_sets[yr])].groupby(['index_wave','hx']).years.agg(['min','max'])
                                lo,hi=np.ceil(bounds['min'].max()*4)/4,np.floor(bounds['max'].min()*4)/4
                                for thisyear in np.arange(lo,hi+.01,.25):
                                    s=dict(**state,target_kind='fixed_curve_target',time_years=thisyear,reference_landmark=yr,
                                           target_people=len(ref))
                                    curves.extend(output_mean_rows(di,beta,cov,groups,ref,thisyear,bounded,old['infer'],s,lo))
                        else:
                            refs=next_targets(d,base_target,source,dates)
                            s=dict(**state,target_kind='next_planned_wave',time_years=np.nan,target_people=len(refs),
                                   target_index_waves=';'.join(map(str,sorted(refs.index_wave.unique()))))
                            if len(refs):
                                s['reference_mean_next_years']=refs.next_year.mean()
                                out.extend(output_mean_rows(di,beta,cov,groups,refs,refs.next_year.to_numpy(),
                                                           bounded,old['infer'],s,0.))
                            else:
                                out.append(s|dict(status='time_not_supported',functional='between_history_change_difference'))
                    except (ValueError,ArithmeticError,np.linalg.LinAlgError,subprocess.TimeoutExpired) as error:
                        qa.append(dict(**metrics,status='not_estimable',detail=str(error)))
                    print(source,comp,domain,model,qa[-1]['status'],flush=True)
    for rows,name in ((out,'B_RESULTS'),(qa,'B_NUMERIC_QA'),(curves,'B_CURVES'),(support,'B_TIME_SUPPORT')):
        pd.DataFrame(rows).to_csv(ROOT/'TABLES'/f'{name}.csv',index=False,encoding='utf-8-sig')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--rscript',required=True)
    parser.add_argument('--lock',type=Path,required=True)
    parser.add_argument('--lock-sha256',required=True)
    parser.add_argument('--branch-sha256',required=True)
    args=parser.parse_args()
    if args.lock.resolve().parent!=ROOT or digest(args.lock)!=args.lock_sha256:
        raise RuntimeError('Current V4 execution lock is not verified')
    if (digest(ROOT/'B_SPEC_CN.md')!=args.branch_sha256 or
            json.loads(args.lock.read_text(encoding='utf-8-sig'))['empirical_execution']!='AUTHORIZED_BY_V4'):
        raise RuntimeError('Frozen B specification or V4 authorization not verified')
    run(args.rscript)

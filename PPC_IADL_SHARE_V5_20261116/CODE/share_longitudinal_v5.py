"""SHARE V3/V4 same-record longitudinal bridge, 0 through 6 years."""
import argparse
import json
import os
import runpy
import subprocess
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import pandas as pd
import patsy
from scipy.stats import t
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from share_association_v5 import ROOT, TABLE, load, verify, digest, coarse

V3 = ROOT.parent/'PPC_IADL_Dynamic_V3_20261116'
V4 = ROOT.parent/'PPC_IADL_Dynamic_V4_20261116'
OLD = runpy.run_path(str(V3/'CODE/long_v3.py'))
B = runpy.run_path(str(V4/'CODE/b_longitudinal.py'))
B['fit_ri'].__globals__['ROOT']=ROOT
DOMAINS,COMPARE,BASE=OLD['DOMAINS'],OLD['COMPARE'],OLD['BASE']
UNITS = {'physical':'condition_count','psychological':'fixed_W5_EURO_D_SD','cognitive':'mean_subtest_Z'}
COUNTS = {'n_people','n_records','n_families','n_post_records','target_people','available_people',
          'nearby_people','records','people','families','n_index_records','n_total_records'}
EXACT=ROOT/'PRIVATE'/'SHARE_LONG'
FAILURE=(ValueError,ArithmeticError,np.linalg.LinAlgError,PerfectSeparationError,subprocess.TimeoutExpired)


def save(name,rows):
    d=pd.DataFrame(rows)
    d.to_csv(EXACT/(name+'_exact.csv'),index=False,encoding='utf-8-sig')
    for c in COUNTS.intersection(d.columns):
        d[c]=d[c].map(lambda n:coarse(n) if pd.notna(n) else np.nan)
    d=d.drop(columns=['warnings'],errors='ignore')
    d.to_csv(TABLE/(name+'.csv'),index=False,encoding='utf-8-sig')


def prepare():
    p=pd.read_parquet(ROOT/'PRIVATE'/'SHARE_panel.parquet')
    q=load()
    assert not p.duplicated(['person','wave']).any()
    eligible=q.loc[q.entry_eligible].copy()
    e=eligible.loc[eligible.first_eligible].copy()
    e['index_wave'],e['Kgroup']=e.wave,e.K1.clip(upper=3)
    want=eligible.sort_values(['month1','wave']).drop_duplicates('person')['person'].sort_values().tolist()
    assert e.person.sort_values().tolist()==want and e.person.is_unique
    if 'cesd_z1' not in e:
        e=e.merge(p[['person','wave','cesd_z']].rename(columns={'cesd_z':'cesd_z1'}),
                  validate='many_to_one',on=['person','wave'])
    e=e[['person','cluster','month1','index_wave','country','history','Kgroup',*BASE]]
    p=p[['person','wave','month','country',*DOMAINS.values()]].copy()
    z=e.merge(p,on=['person','country'],how='left',validate='one_to_many')
    z=z.loc[z.wave.ge(z.index_wave)].copy()
    z['years']=(z.month-z.month1)/12
    assert z.loc[z.wave.eq(z.index_wave),'years'].eq(0).all()
    return p,e,z


def contexts(d,comp):
    kk=['country','index_wave','code_fixed1']+(['Kgroup'] if comp=='H11_minus_H01' else [])
    g=d.groupby(kk,observed=True).hx.nunique()
    return d.loc[pd.MultiIndex.from_frame(d[kk]).isin(g.index[g.eq(2)])].copy()


def sample(e,z,comp,field):
    a,b=COMPARE[comp]
    target=e.loc[e.history.isin([a,b])].copy()
    target['hx']=target.history.eq(a).astype(int)
    target=contexts(target.dropna(subset=BASE+['country','cluster']),comp)
    d=z.loc[z.person.isin(target.person)&z.years.gt(0)&z.years.le(6)&z[field].notna()].copy()
    d['hx']=d.history.eq(a).astype(int)
    return OLD['derived'](contexts(d,comp))


def targets(p,d,field,meta):
    dates=p.groupby(['country','wave'],observed=True).month.median()
    valid_dates=p.loc[p[field].notna()].groupby(['country','wave'],observed=True).month.median()
    last=valid_dates.groupby(level='country').max()
    qualified={y:[] for y in (2.,4.,6.)}
    next_refs=[]; support=[]
    persons=d.drop_duplicates('person')
    for (co,w),x in d.groupby(['country','index_wave'],observed=True):
        scope=x.groupby('hx').years.agg(['min','max']).reindex([0,1])
        cap=float(last.get(co,np.nan)-dates.get((co,w),np.nan))/12
        # Same nominal-wave interpretation as V3, plus exact observed-time bracketing.
        for yr in qualified:
            both=bool(scope['min'].le(yr).all() and scope['max'].ge(yr).all())
            eligible=bool(np.isfinite(cap) and np.floor(cap+.5)>=yr and both)
            if eligible:
                qualified[yr].append((co,w))
            g=x.loc[x.years.between(yr-.5,yr+.5)]
            for arm in (0,1):
                side=x.loc[x.hx.eq(arm)]
                q=dict(**meta,target_kind='fixed_landmark',country=co,index_wave=w,arm=arm,
                        time_years=yr,administrative_horizon_years=cap,
                        both_arm_bracket=both,qualified=eligible,available_people=side.person.nunique(),
                        nearby_people=g.loc[g.hx.eq(arm),'person'].nunique())
                if q['available_people']>=10:
                    q.update(observed_min_years=side.years.min(),observed_max_years=side.years.max())
                support.append(q)
        interval=(dates.get((co,w+1),np.nan)-dates.get((co,w),np.nan))/12
        g=x.loc[x.wave.eq(w+1)]
        gg=g.groupby('hx').years.agg(['size','min','max']).reindex([0,1])
        if gg['size'].ge(30).all() and gg['min'].le(interval).all() and gg['max'].ge(interval).all():
            next_refs.append(persons.loc[persons.country.eq(co)&persons.index_wave.eq(w)].assign(next_year=interval))
    result={yr:persons.loc[pd.MultiIndex.from_frame(persons[['country','index_wave']]).isin(s)] for yr,s in qualified.items()}
    return result,pd.concat(next_refs) if next_refs else None,support


def add_initial(d,z,domain,comp):
    a,b=COMPARE[comp]
    field=DOMAINS[domain]
    init=z.loc[z.person.isin(d.person)&z.wave.eq(z.index_wave)].copy()
    assert init.person.is_unique and init.years.eq(0).all() and len(init)==d.person.nunique()
    np.testing.assert_allclose(init[field],init[B['INITIAL'][domain]])
    init['hx']=init.history.eq(a).astype(int)
    data=OLD['derived'](pd.concat([init,d],ignore_index=True))
    return data.assign(post=data.years.gt(0).astype(int),post_time2=data.years.gt(0)*data.time2,
                       other_domain_code=data.code_fixed1.astype(int)&(7-B['BITS'][domain]))


def other_domain(ref,domain):
    return ref.assign(other_domain_code=ref.code_fixed1.astype(int)&(7-B['BITS'][domain]))


def fit_rows(p,e,z,comp,domain,rscript):
    bounded=domain=='physical'; field=DOMAINS[domain]
    meta=dict(source='SHARE',comparison=comp,domain=domain,scale=UNITS[domain])
    d=sample(e,z,comp,field)
    result=[]; diagnostic=[]
    frames,nxt,support=targets(p,d,field,meta)
    for name in ('marginal_post','ri_linear','index_included'):
        if bounded and name=='ri_linear':
            continue
        b2=name=='index_included'
        meta_fit=dict(**meta,model=name,n_people=d.person.nunique(),n_post_records=len(d),n_families=d.cluster.nunique())
        try:
            if d.hx.nunique()!=2 or d.groupby('hx').wave.nunique().min()<2:
                raise ValueError('Both histories need two or more actual post-index assessment waves')
            df=add_initial(d,z,domain,comp) if b2 else d
            if b2:
                meta_fit['n_index_records']=df.years.eq(0).sum()
            meta_fit['n_total_records']=len(df)
            formula=B['fit_formula']('SHARE',comp,domain,'index_included' if b2 else 'ri_linear')+' + C(country)'
            design=patsy.dmatrix(formula,df,return_type='dataframe',NA_action='raise')
            x=np.asarray(design); y=df[field].to_numpy(float)/(7 if bounded else 1)
            if np.linalg.matrix_rank(x)!=x.shape[1] or not np.isfinite(y).all() or (bounded and not np.all((y>=0)&(y<=1))):
                raise ValueError('Rank, response or disease-count range check failed')
            if name=='ri_linear':
                beta,cov,G,checks=B['fit_ri'](x,y,df,rscript)
            else:
                beta,cov,G,checks=B['fit_index'](x,y,df,bounded)
            if checks['bread_condition']>1e12:
                raise ValueError('Ill-conditioned model information')
            diagnostic.append(dict(**meta_fit,status='estimated',parameters=x.shape[1],**checks,formula=formula))
            ok=dict(**meta_fit,status='estimated')
            di=design.design_info
            if not b2:
                ind=di.column_names.index('hx:time2')
                slope=OLD['infer'](beta[ind],np.eye(len(beta))[ind],cov,t.ppf(.975,G-1))
                result.append(dict(**ok,functional='history_time_interaction',
                                   unit='logit_disease_fraction_per_year' if bounded else UNITS[domain]+'_per_year',
                                   nominal_p=2*t.sf(abs(slope['estimate']/slope['se']),G-1),**slope))
            for yr,ref in frames.items():
                q=dict(**ok,target_kind='fixed_landmark',time_years=yr,target_people=len(ref),
                       n_country_index_blocks=len(ref[['country','index_wave']].drop_duplicates()))
                if ref.empty or ref.groupby('hx').person.nunique().min()<10:
                    result.append({**q,'status':'time_not_supported','functional':'standardized_mean_difference'})
                else:
                    target=other_domain(ref,domain) if b2 else ref
                    result.extend(B['output_mean_rows'](di,beta,cov,G,target,yr,bounded,OLD['infer'],q,0. if b2 else None))
            if b2:
                b=dict(**ok,target_kind='next_planned_wave',time_years=np.nan)
                if nxt is not None:
                    b.update(target_people=len(nxt),reference_mean_next_years=nxt.next_year.mean())
                    target=other_domain(nxt,domain)
                    result.extend(B['output_mean_rows'](di,beta,cov,G,target,nxt.next_year.to_numpy(),bounded,OLD['infer'],b,0.))
                else:
                    result.append({**b,'status':'time_not_supported','functional':'between_history_change_difference'})
        except FAILURE as err:
            detail=str(err) if isinstance(err,ValueError) else type(err).__name__
            if '\n' in detail or len(detail)>220:
                detail=type(err).__name__
            diagnostic.append(dict(**meta_fit,status='not_estimable',detail=detail))
        print('SHARE',comp,domain,name,diagnostic[-1]['status'],flush=True)
    return result,diagnostic,support


def main(rscript,lock_sha):
    verify(lock_sha)
    gate=json.loads((ROOT/'QA'/'SHARE_PRIMARY_GATE.json').read_text(encoding='utf-8'))
    if gate['share_lock_sha256']!=lock_sha or gate['rows_sha256']!=digest(TABLE/'SHARE_PPC_PRIMARY.csv'):
        raise RuntimeError('A frozen primary fit checkpoint is required before the SHARE B models')
    EXACT.mkdir(parents=True,exist_ok=True)
    p,e,z=prepare()
    outputs=[]; qa=[]; supports=[]
    for c in COMPARE:
        for dom in DOMAINS:
            r,d,s=fit_rows(p,e,z,c,dom,rscript)
            outputs.extend(r); qa.extend(d); supports.extend(s)
    save('SHARE_LONG_RESULTS',outputs)
    save('SHARE_LONG_DIAGNOSTICS',qa)
    save('SHARE_LONG_TIME_SUPPORT',supports)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--lock-sha256',required=True)
    ap.add_argument('--rscript',required=True)
    args=ap.parse_args()
    main(args.rscript,args.lock_sha256)

"""V6 next-assessment association gates and only the frozen original-model repair."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import runpy
from time import perf_counter

from firth_engine_v6 import FitFail, family_plan, fit, functionals
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
V2=ROOT.parent/'PPC_IADL_Dynamic_V2_20261115'
V3=ROOT.parent/'PPC_IADL_Dynamic_V3_20261116'
V4=ROOT.parent/'PPC_IADL_Dynamic_V4_20261116'
V5=ROOT.parent/'PPC_IADL_SHARE_V5_20261116'
P=ROOT/'PRIVATE'/'FIRTH'; T=ROOT/'TABLES'; QA=ROOT/'QA'
SOURCES=('CHARLS','HRS','ELSA','SHARE')
COMPS=('H11_vs_H01','H10_vs_H00')
LAYERS=('M_basic','M_severity')
B=499; MIN_OK=450
KEYS=['source','comparison','model_layer','specification']
NUMCOLS=['log_or','log_or_se','log_or_lo','log_or_hi','odds_ratio','odds_ratio_lo','odds_ratio_hi',
         'risk_index','risk_index_lo','risk_index_hi','risk_difference','risk_difference_lo','risk_difference_hi',
         'risk_reference','risk_reference_lo','risk_reference_hi']


def init(locksha):
    p=ROOT/'V6_EXECUTION_LOCK.json'
    if not p.is_file() or sha256(p.read_bytes()).hexdigest()!=locksha:
        raise ValueError('V6 analysis lock is missing or has wrong SHA-256')
    for q in (P,T,QA): q.mkdir(parents=True,exist_ok=True)


def src_seed(src,comp):
    return 6111606+10*SOURCES.index(src)+COMPS.index(comp)


def open_snapshot(directory,src,comp,spec,layer):
    stem=directory/f'ppc_{src}_{comp}_{spec}_{layer}'
    if not stem.with_suffix('.npz').is_file():
        stem=stem.with_name(stem.name+'_FAILED')
    d=pd.read_parquet(stem.with_name(stem.name+'_rows.parquet'))
    with np.load(stem.with_suffix('.npz'),allow_pickle=False) as f:
        a={k:f[k] for k in ('X','y','cluster','terms','beta','cov')}
        assert np.array_equal(f['row_index'],d.index.to_numpy())
    assert np.array_equal(a['y'],d.y) and np.array_equal(a['cluster'],d.cluster.astype(str))
    assert len(a['terms'])==a['X'].shape[1] and a['terms'][0]=='Intercept' and 'hx' in a['terms']
    assert np.array_equal(a['X'][:,list(a['terms']).index('hx')],d.hx)
    assert not d.duplicated(['person','window']).any() and d.hx.nunique()==2
    return a,d,stem


def original(src,comp,layer):
    where=(V5/'PRIVATE/SHARE_ASSOC' if src=='SHARE' else V3/'PRIVATE/PPC')
    a,d,stem=open_snapshot(where,src,comp,'primary',layer)
    if src=='SHARE' and comp==COMPS[0]:
        assert len(d)==504 and d.person.nunique()==480 and d.country.nunique()==13
        assert sum(d.groupby('country').y.nunique().eq(1))==1
        assert (a['X'].shape[1]==35)==(layer=='M_severity')
    return a,d,stem


def field_counts(d):
    return dict(windows=len(d), people=d.person.nunique(), families=d.cluster.nunique())


def safe_count(n):
    return '0' if n==0 else ('<10' if n<10 else str(int((n+5)//10)*10))


def bootpack(src,comp,d,label):
    groups=d.country.to_numpy(dtype=str) if src=='SHARE' else np.repeat('all',len(d))
    seed=src_seed(src,comp)
    p=P/f'FIRTH_PLAN_{src}_{comp}_{label}.npz'
    pid=d.person.to_numpy(dtype=str); rr=d.index.to_numpy(dtype=int)
    fam=d.cluster.to_numpy(dtype=str)
    if p.exists():
        with np.load(p,allow_pickle=False) as z:
            assert np.array_equal(pid,z['person']) and np.array_equal(rr,z['row_index']) and np.array_equal(fam,z['cluster'])
            assert np.array_equal(groups,z['strata']) and int(z['seed'])==seed
            plan,inv=z['family_multiplicities'],z['row_to_family']
    else:
        plan,inv=family_plan(fam,groups,seed,B)
        np.savez_compressed(p,person=pid,row_index=rr,cluster=fam,strata=groups,seed=seed,
                            family_multiplicities=plan,row_to_family=inv)
    assert len(plan)==B
    return plan,inv,seed


def modelspec(src,comp,spec,layer):
    return dict(source=src,comparison=comp,model_layer=layer,specification=spec)


def estimates(a,d,src,comp,spec,layer,pair):
    row=dict(**modelspec(src,comp,spec,layer), **field_counts(d), estimator='Firth slopes + FLIC intercept',
             infer='household percentile CI, draw SD; SHARE resampled within country')
    for h,pref in ((1,'index'),(0,'reference')):
        z=d.loc[d.hx.eq(h)]
        row.update({f'{pref}_windows_protected':safe_count(len(z)),f'{pref}_events_protected':safe_count(z.y.eq(1).sum()),
                    f'{pref}_nonevents_protected':safe_count(z.y.eq(0).sum())})
    x=a['X']; y=a['y']; j=list(a['terms']).index('hx'); k=int(x.shape[1]); started=perf_counter()
    row.update(design_terms=k,design_rank=int(np.linalg.matrix_rank(x)))
    if src=='SHARE':
        row.update(countries=int(d.country.nunique()),country_levels_no_outcome_variation=int(d.groupby('country').y.nunique().eq(1).sum()))
    try:
        if row['design_rank']!=k or d.hx.nunique()!=2:
            raise FitFail('original_design_rank_or_contrast')
        b,diag=fit(x,y)
        stat=functionals(x,y,b,j,np.ones(len(y)))
        row.update(**diag,**dict(zip(('log_or','risk_index','risk_reference','risk_difference','flic_intercept_shift'),stat)))
        row['odds_ratio']=float(np.exp(row['log_or']));row['status']='point_only'
    except FitFail as e:
        return dict(**row,status='not_estimable',failure=str(e),bootstrap_attempts=0,bootstrap_success=0)
    plan,inv,seed=bootpack(src,comp,d,pair)
    failures=Counter(); rescues=0; constant_redraw=0
    beta_draws=np.full((B,5),np.nan)
    for i,draw in enumerate(plan):
        w=draw[inv].astype(float)
        try:
            b2,q=fit(x,y,w,warm=b)
            beta_draws[i]=functionals(x,y,b2,j,w)
            rescues+=int(q['algorithm']=='bfgs_rescue')
            constant_redraw+=int(q['dropped_redraw_constant_columns']>0)
        except (FitFail,np.linalg.LinAlgError) as e:
            failures[str(e)]+=1
    np.savez_compressed(P/f'FIRTH_DRAW_{src}_{comp}_{spec}_{layer}.npz',draws=beta_draws)
    ok=np.isfinite(beta_draws).all(axis=1)
    row.update(bootstrap_attempts=B,bootstrap_success=int(ok.sum()),bootstrap_failed=int((~ok).sum()),
               bootstrap_rescue=rescues,bootstrap_constant_level=constant_redraw,
               failure=';'.join(f'{e}:{n}' for e,n in sorted(failures.items())), seed=seed,elapsed_seconds=perf_counter()-started)
    if ok.sum()>=MIN_OK:
        row['status']='estimated'
        for k,name in enumerate(('log_or','risk_index','risk_reference','risk_difference')):
            z=beta_draws[ok,k]
            row[name+'_se']=float(z.std(ddof=1));row[name+'_lo'],row[name+'_hi']=np.quantile(z,[.025,.975])
        row['odds_ratio_lo']=float(np.exp(row['log_or_lo']));row['odds_ratio_hi']=float(np.exp(row['log_or_hi']))
    # Mean outcome calibration is implemented before shifting HX to each counterfactual.
    return row


def frozen_row(src,comp,layer,spec,path,alias,saved,route,matched=None):
    old=saved.loc[saved.comparison.eq(comp)&saved.specification.eq(alias)&saved.model_layer.eq(layer)]
    assert len(old)==1
    o=old.iloc[0]
    a,d,stem=open_snapshot(path,src,comp,alias,layer)
    if matched is not None:
        assert np.array_equal(d.index.to_numpy(),matched.index.to_numpy())
    if o.status!='estimated':
        return estimates(a,d,src,comp,spec,layer,route)
    c={k:float(o[k]) for k in NUMCOLS}
    j=list(a['terms']).index('hx')
    beta=float(a['beta'][j]);se=float(np.sqrt(a['cov'][j,j]))
    np.testing.assert_allclose([c['log_or'],c['log_or_se']],[beta,se], atol=1e-12)
    crit=t.ppf(.975,d.cluster.nunique()-1); np.testing.assert_allclose([c['log_or_lo'],c['log_or_hi']],beta+np.array([-1,1])*crit*se, atol=1e-10)
    return dict(**modelspec(src,comp,spec,layer),**field_counts(d),**c, status='estimated',
                estimator='Frozen logistic MLE',infer='Family HC1 sandwich t(G-1)',evidence_file=stem.with_suffix('.npz').name)


def record(data,name):
    q=T/name
    tmp=q.with_suffix('.tmp')
    pd.DataFrame(data).to_csv(tmp,index=False,encoding='utf-8-sig')
    tmp.replace(q)


def primary(src):
    output=[]
    for comp in COMPS:
        a0,d0,_=original(src,comp,LAYERS[0]);a1,d1,_=original(src,comp,LAYERS[1])
        assert d0.index.equals(d1.index) and np.array_equal(a0['y'],a1['y'])
        for a,d,l in ((a0,d0,LAYERS[0]),(a1,d1,LAYERS[1])):
            row=estimates(a,d,src,comp,'primary',l,'original')
            output.append(row)
            record(output,f'FIRTH_MODEL_RESULTS_{src}.csv')
            print(src,comp,l,row['status'],row.get('bootstrap_success'),round(row.get('elapsed_seconds',0),2),flush=True)


def join():
    fit=pd.concat([pd.read_csv(T/f'FIRTH_MODEL_RESULTS_{s}.csv') for s in SOURCES],ignore_index=True)
    main=[]
    for src in SOURCES:
        path=V5/'PRIVATE/SHARE_ASSOC' if src=='SHARE' else V3/'PRIVATE/PPC'
        release=V5/'TABLES/SHARE_PPC_PRIMARY.csv' if src=='SHARE' else V3/f'TABLES/PPC_RESULTS_{src}.csv'
        archived=pd.read_csv(release)
        for comp in COMPS:
            for layer in LAYERS:
                a,d,stem=original(src,comp,layer)
                old=archived.loc[archived.comparison.eq(comp)&archived.model_layer.eq(layer)&archived.specification.eq('primary')].iloc[0]
                row=dict(**modelspec(src,comp,'primary',layer),**field_counts(d),
                         v5_status=old.status,v5_information_min=old.get('information_min'),
                         v5_information_condition=old.get('information_condition'), v5_score_max=old.get('score_max'),
                         v5_max_abs_beta=float(max(abs(a['beta']))),v5_rank=int(np.linalg.matrix_rank(a['X'])),
                         v5_width=int(a['X'].shape[1]),firth_full_country_design=src=='SHARE')
                row.update({'v5_'+c:old.get(c) for c in NUMCOLS})
                rr=fit.loc[(fit.source==src)&(fit.comparison==comp)&(fit.model_layer==layer)].iloc[0]
                row.update({'firth_'+c:rr[c] for c in NUMCOLS if c in rr})
                row.update(firth_status=rr.status,bootstrap_attempts=rr.bootstrap_attempts,bootstrap_success=rr.bootstrap_success,
                           v5_failure=('finite rank but quasi-separated fixed-effect nuisance; no original coefficient/CI released'
                                       if old.status!='estimated' else ''),
                           same_ordered_rows=True)
                main.append(row)
    record(fit,'FIRTH_MODEL_RESULTS.csv'); record(main,'FIRTH_MLE_BRIDGE.csv')
    record([x for x in main if x['source']=='SHARE'],'SHARE_MLE_FIRTH_FLIC_BRIDGE.csv')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--lock-sha256',required=True)
    p.add_argument('mode',choices=['primary','join']); p.add_argument('--source',choices=SOURCES)
    args=p.parse_args(); init(args.lock_sha256)
    primary(args.source) if args.mode=='primary' else join()

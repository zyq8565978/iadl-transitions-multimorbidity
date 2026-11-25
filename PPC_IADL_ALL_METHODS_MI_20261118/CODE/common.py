"""Read-only inherited estimators; new MI bindings, checkpoints and pooling."""
from inventory import R,W,S,load
from pathlib import Path
from functools import lru_cache
from collections import Counter
import sys,json,hashlib,time
import numpy as np
import pandas as pd
from scipy.stats import t as td
from scipy.special import expit
sys.path.insert(0,str(W/'PPC_IADL_ZHOU_COVARIATE_QC_20261117/CODE'))
import run_qc as qc
old=qc.old;SOURCES=old.SOURCES;COMPS=old.COMPS
pm=load('allmi_previous_pool',W/'PPC_IADL_LONGITUDINAL_MI_20261117/CODE/fit_completed.py')
pool_values=pm.pool_values

def write(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

@lru_cache(maxsize=1)
def completed(mi):
    frames=[]
    for s in SOURCES:
        n=node_binding(s)
        a=pd.read_csv(R/f'PRIVATE/{s}_mi_{mi:03d}.csv')
        assert len(a)==len(n)
        for v in ('smoking','alcohol','bmi'):
            known=n[v].notna();np.testing.assert_allclose(a.loc[known,v],n.loc[known,v],atol=1e-10,rtol=1e-12)
        assert a.notna().all().all() and a.smoking.isin([0,1]).all() and a.alcohol.isin([0,1]).all() and a.bmi.between(10,80).all()
        a['key']=n.key;a['bmi_cat']=pd.cut(a.bmi,[-np.inf,18.5,25,30,np.inf],right=False,labels=False)
        a['bmi_whole_wave_missing']=n.bmi.isna()&n.bmi_reason.eq('structurally_unavailable')
        frames.append(a)
    return pd.concat(frames,ignore_index=True).set_index('key',verify_integrity=True)

@lru_cache(maxsize=4)
def node_binding(source):
    return pd.read_parquet(R/f'PRIVATE/{source}_node_binding.parquet',columns=['key','smoking','alcohol','bmi','bmi_reason'])

def keys(d,wave='wave',source='source',person='person'):
    src=d[source].astype(str);p=d[person].astype(str)
    for s in SOURCES:
        take=src.eq(s);p.loc[take]=p.loc[take].str.removeprefix(s+'|').str.removeprefix(s+':')
    return src+'|'+p+'|'+pd.to_numeric(d[wave],errors='raise').astype(int).astype(str)

def extras(d,mi,wave='wave',columns=('smoking','alcohol','bmi'),stress=False):
    a=completed(mi).loc[keys(d,wave)].reset_index(drop=True)
    if stress:
        a=a.copy();mask=a.bmi_whole_wave_missing
        a.loc[mask,'bmi']=(a.loc[mask,'bmi']-2).clip(lower=10)
        a['bmi_cat']=pd.cut(a.bmi,[-np.inf,18.5,25,30,np.inf],right=False,labels=False)
    data=[];names=[]
    for s in SOURCES:
        mask=d.source.eq(s).to_numpy()
        if not mask.any():continue
        for v in columns:
            for level in ((0,2,3) if v=='bmi' else (None,)):
                q=a.bmi_cat.eq(level).to_numpy(float) if v=='bmi' else a[v].to_numpy(float)
                data.append(np.where(mask,q,0));names.append(s+':MI_'+v+(str(level) if level is not None else ''))
    return np.array(data).T,names

def effect_gate():
    gate=json.loads((R/'QA/MI_DIAGNOSTIC_GATE.json').read_text())
    assert gate['allow_effect_analysis'] and gate['M']==50

def binary(d,y,x,src,stem,mi,engine='MLE',B=199):
    stamp=f'{stem}_mi{mi:03d}';cache=R/f'PRIVATE/{stamp}.npz';status=R/f'QA/{stamp}.json'
    if cache.exists() and status.exists():
        z=np.load(cache);return z['q'],z['U'],json.loads(status.read_text())
    tick=time.time();x=x[:,qc.independent(x)];y=np.asarray(y,float)
    info=dict(stem=stem,mi=mi,people=int(d.person.nunique()),windows=len(d),families=int(d.family.nunique()),events=int(y.sum()),engine=engine)
    if engine=='MLE':
        q,inf,qa,b=qc.mle_influence(d,y,x);U=inf.T@inf
    else:
        b,qa=qc.fit(x,y);q=old.flic_stat(x,y,src,b,np.ones(len(y)))
        write(R/f'QA/{stamp}_point.json',dict(**info,point=q.tolist(),diagnostics=qa))
        seed=71118000+int.from_bytes(hashlib.sha256(stem.encode()).digest()[:3],'big')
        plan,inv=qc.family_plan(d.family.to_numpy(str),d.stratum.to_numpy(str),seed,B)
        boot=R/f'PRIVATE/{stamp}_bootstrap.npz';draw=np.full((B,4),np.nan);done=0;fail=Counter()
        if boot.exists():
            z=np.load(boot);draw=z['draws'];done=int(z['done'])
        for i in range(done,B):
            try:draw[i]=qc.boot_fit(x,y,src,plan[i,inv].astype(float),b)
            except (ValueError,AssertionError,np.linalg.LinAlgError) as exc:fail[str(exc)]+=1
            if (i+1)%50==0 or i+1==B:
                np.savez_compressed(boot,draws=draw,done=i+1)
                write(R/f'QA/{stamp}_progress.json',dict(done=i+1,failures=dict(fail),seconds=time.time()-tick))
        good=draw[np.isfinite(draw).all(1)];minok=450 if B==499 else 180
        U=np.cov(good,rowvar=False) if len(good)>=minok else np.full((4,4),np.nan)
        info.update(bootstrap_valid=len(good),bootstrap_attempted=B,failures=dict(fail))
    info.update(status='ESTIMATED' if np.isfinite(U).all() else 'POINT_ONLY_INSUFFICIENT_VALID_BOOTSTRAP',diagnostics=qa,seconds=time.time()-tick)
    np.savez_compressed(cache,q=q,U=U,beta=b)
    write(status,info)
    return q,U,info

def pooled(stem,labels,scale,metadata=None):
    qs=[];us=[];met=[]
    for mi in range(1,51):
        p=R/f'PRIVATE/{stem}_mi{mi:03d}.npz'
        if not p.exists():continue
        z=np.load(p);qs.append(z['q']);us.append(z['U']);met.append(json.loads((R/f'QA/{stem}_mi{mi:03d}.json').read_text()))
    base=dict(stem=stem,MI_completed=len(qs),**(metadata or {}))
    if len(qs)==50 and np.isfinite(qs).all() and not np.isfinite(us).all():
        q=np.mean(qs,axis=0);rows=[]
        for j,label in enumerate(labels):
            rows.append(dict(**base,functional=label,scale='ratio' if scale[j]=='log' else scale[j],estimate=float(np.exp(q[j]) if scale[j]=='log' else q[j]),
                 status='MI_POINT_ONLY_NO_VALID_POOLED_VARIANCE',valid_covariance_count=int(sum(np.isfinite(u).all() for u in us)),
                 people=met[0].get('people'),windows=met[0].get('windows'),families=met[0]['families']))
        pd.DataFrame(rows).to_csv(R/f'TABLES/{stem}.csv',index=False,encoding='utf-8-sig');return rows
    if len(qs)!=50 or not np.isfinite(us).all():
        row=dict(**base,status='NOT_POOLED_INCOMPLETE_VALID_MI',valid_covariance_count=int(sum(np.isfinite(u).all() for u in us)))
        pd.DataFrame([row]).to_csv(R/f'TABLES/{stem}.csv',index=False);return [row]
    dfcom=min(x['families'] for x in met)-1
    q,w,b,total,df,se,lo,hi,fmi,mc=pool_values(np.asarray(qs),np.asarray(us),dfcom)
    rows=[]
    for j,label in enumerate(labels):
        transform=np.exp if scale[j]=='log' else lambda v:v
        testable=scale[j]!='probability' and not label.startswith('mean_')
        rows.append(dict(**base,functional=label,scale='ratio' if scale[j]=='log' else scale[j],estimate=transform(q[j]),lower=transform(lo[j]),upper=transform(hi[j]),
             pooled_analysis_scale=q[j],SE_analysis_scale=se[j],df=df[j],FMI=fmi[j],MCSE=mc[j],MCSE_over_SE=mc[j]/se[j] if se[j]>0 else 0.,
             p_two_sided=2*td.sf(abs(q[j]/se[j]),df[j]) if se[j]>0 and testable else np.nan,
             test_null_analysis_scale=0 if testable else None,status='ESTIMATED',people=met[0].get('people'),windows=met[0].get('windows'),families=met[0]['families'],events=met[0].get('events')))
    np.savez_compressed(R/f'QA/{stem}_pooled_covariance.npz',mean=q,within=w,between=b,total=total)
    pd.DataFrame(rows).to_csv(R/f'TABLES/{stem}.csv',index=False,encoding='utf-8-sig')
    return rows

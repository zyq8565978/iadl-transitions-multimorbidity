"""Bounded, paired covariate QC; all old inputs are read-only."""
import os
for v in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'): os.environ[v]='1'
from pathlib import Path
import argparse,hashlib,json,sys,time
from collections import Counter
import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.special import expit
from scipy.stats import t as tdist
R=Path(__file__).resolve().parents[1]; W=R.parent
sys.path.insert(0,str(W/'PPC_IADL_ZHOU_ADAPTATION_20261117'))
import run_ipd as old
from firth_engine_v6 import fit,family_plan
P=R/'PRIVATE'; T=R/'TABLES'; Q=R/'QA'
B=499

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def writejson(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def independent(x):
    u=np.linalg.qr(x[:,:2],mode='reduced')[0]
    z=x[:,2:]-u@(u.T@x[:,2:]); _,r,piv=qr(z,mode='economic',pivoting=True)
    tol=max(z.shape)*np.finfo(float).eps*max(1,abs(r).max())
    k=int((abs(r.diagonal())>tol).sum()); cols=np.r_[0,1,2+np.sort(piv[:k])]
    a=x[:,cols]; mapping=np.linalg.lstsq(a,x,rcond=None)[0]
    assert np.linalg.matrix_rank(a)==a.shape[1]
    for h in (None,0,1):
        v=x.copy()
        if h is not None:v[:,1]=h
        assert np.max(abs(v[:,cols]@mapping-v))<1e-7,'contrast_column_space_lost'
    return cols

def overlay(comp,d):
    frames=[]
    for s in old.SOURCES:
        a=pd.read_parquet(P/f'{s}_{comp}_bound.parquet')
        a['person']=s+'|'+a.person.astype(str);a['source']=s
        frames.append(a)
    a=pd.concat(frames,ignore_index=True)
    assert np.array_equal(a.person,d.person) and np.array_equal(a.window,d.window)
    assert np.array_equal(a.y,d.y) and np.array_equal(a.hx,d.hx)
    return a

def extras(a,src,timing,health=False,behav=True):
    c=[]; names=[]
    for j,s in enumerate(old.SOURCES):
        cols=[]; ns=[]
        if health:cols.extend([a.cesd0.to_numpy(float),a.cog_z0.to_numpy(float)]);ns+=['prior_cesd','prior_cognition']
        if behav:
            cols.extend([a['smoking_'+timing].to_numpy(float),a['alcohol_'+timing].to_numpy(float)])
            ns+=['native_current_smoking','weekly_or_recent_alcohol']
            for k in (0,2,3):cols.append(a['bmi_cat_'+timing].eq(k).to_numpy(float));ns.append('BMI_category_'+str(k))
        for v,n in zip(cols,ns):c.append(np.where(src==j,v,0));names.append(s+':'+n)
    return np.asarray(c).T,names

def prepare(comp,module,timing):
    d,y,x,src=old.load(comp,'M_severity');a=overlay(comp,d)
    mask=a[['smoking_'+timing,'alcohol_'+timing,'bmi_'+timing]].notna().all(axis=1).to_numpy()
    if module!='primary':
        spec={'prior_health':'prior_health_adjusted','psychological':'physical_only_psychological','cognitive':'physical_only_cognitive'}[module]
        allowed=pd.read_parquet(W/f'PPC_IADL_ONE_STAGE_SUPPLEMENTS_20261117/ANCILLARY/PRIVATE/{comp}_{spec}_rows.parquet')
        keys=pd.MultiIndex.from_frame(allowed[['person','window']]); mask &= pd.MultiIndex.from_frame(d[['person','window']]).isin(keys)
        if module=='prior_health':mask &= a[['cesd0','cog_z0']].notna().all(axis=1).to_numpy()
        else:
            bit=2 if module=='psychological' else 4
            y=(d.code_fixed2.astype(int).to_numpy()&bit!=0).astype(float)
    d=d.loc[mask].reset_index(drop=True);a=a.loc[mask].reset_index(drop=True);y=y[mask];src=src[mask];base=x[mask]
    d['y']=y
    added,names=extras(a,src,timing)
    matrices=[base,np.c_[base,added]]
    labels=['M0_CC','M1_Zhou_t' if timing=='t' else 'M2_Pre_tminus1']
    if module=='prior_health':
        health,hn=extras(a,src,timing,health=True,behav=False)
        matrices=[base,np.c_[base,health],np.c_[base,health,added]]
        labels=['M0_CC_prior_health_intersection','M_PH_CC','M_PH_plus_three_t'];names=hn+names
    maps=[independent(v) for v in matrices];xs=[v[:,c] for v,c in zip(matrices,maps)]
    assert all(np.isfinite(v).all() for v in xs) and len(np.unique(d.hx))==2
    audit=dict(module=module,timing=timing,comparison=comp,rank=[v.shape[1] for v in xs],
        original_columns=[v.shape[1] for v in matrices],retained_column_indices=[c.tolist() for c in maps],
        no_country_or_covariate_deleted=True,equivalent_column_space_and_both_counterfactuals_checked=True,
        new_columns=names,sources=d.source.value_counts().to_dict(),
        member_sha256=hashlib.sha256(d[['person','window']].to_csv(index=False).encode()).hexdigest(),
        source_wave_counts=d.groupby(['source','wave']).size().to_dict())
    audit['source_wave_counts']={str(k):v for k,v in audit['source_wave_counts'].items()}
    return d,y,xs,src,labels,audit

def mle_influence(d,y,x):
    beta,V,G,qa=old.logistic(x,y,d.family)
    p=expit(x@beta);H=(x.T*(p*(1-p)))@x
    gid,fam=pd.factorize(d.family,sort=True);S=np.zeros((G,len(beta)));np.add.at(S,gid,x*(y-p)[:,None])
    influence=np.linalg.solve(H,S.T).T*np.sqrt((G/(G-1))*((len(d)-1)/(len(d)-len(beta))))
    risks=[];grads=[]
    for h in (1,0):
        z=x.copy();z[:,1]=h;ph=expit(z@beta)
        risks.append(ph.mean());grads.append(z.T@(ph*(1-ph))/len(d))
    grads=np.stack([np.eye(len(beta))[1],*grads,grads[0]-grads[1]])
    ans=np.r_[beta[1],risks,risks[0]-risks[1]]; fi=influence@grads.T
    np.testing.assert_allclose(fi.T@fi,grads@V@grads.T,rtol=2e-6,atol=1e-10)
    return ans,fi,qa,beta

def boot_fit(x,y,src,w,warm):
    try:b,qa=fit(x,y,w,warm=warm)
    except ValueError as e:
        if str(e)!='design_rank_deficient':raise
        cols=independent(x[w>0]);bb,qa=fit(x[:,cols],y,w,warm=warm[cols]);b=np.zeros(x.shape[1]);b[cols]=bb
    return old.flic_stat(x,y,src,b,w)

def run(comp,module,timing):
    gate=json.loads((R/'REPLICATION_GATE.json').read_text());assert gate['status']=='PASS'
    assert sha(R/'VARIABLE_RULES_LOCK.md')==(R/'VARIABLE_RULES_LOCK.sha256').read_text().split()[0]
    stem=comp+'_'+module+'_'+timing; tick=time.monotonic()
    if (T/f'{stem}_results.csv').exists():print('COMPLETED CACHE',stem,flush=True);return
    d,y,xs,src,labels,audit=prepare(comp,module,timing);writejson(Q/f'{stem}_design.json',audit)
    d[['person','family','source','window','hx','y']].to_parquet(P/f'{stem}_rows.parquet',index=False)
    if min(int(((d.hx==h)&(y==v)).sum()) for h in (0,1) for v in (0,1))<10:
        pd.DataFrame([dict(comparison=comp,module=module,timing=timing,status='NON_ESTIMABLE',reason='pooled_arm_outcome_support_below10')]).to_csv(T/f'{stem}_results.csv',index=False);return
    base=dict(comparison=comp,module=module,timing=timing,windows=len(d),people=d.person.nunique(),families=d.family.nunique(),events=int(y.sum()),
        sources=';'.join(d.source.unique()),target='same observed-outcome person-window empirical reference',
        covariate_interpretation='source-specific approximate additional adjustment; not uniformly Zhou-equivalent')
    method='Firth_source_FLIC' if comp=='H11_vs_H01' else 'MLE_family_HC1'
    centers=[];influences=[];qas=[];betas=[];mle_error=None
    if method=='MLE_family_HC1':
        try:
            for x in xs:
                v,inf,qa,b=mle_influence(d,y,x);centers.append(v);influences.append(inf);qas.append(qa);betas.append(b)
        except (ValueError,AssertionError,np.linalg.LinAlgError) as e:
            method='Firth_source_FLIC';mle_error=str(e);centers=[];qas=[];betas=[]
    if method=='Firth_source_FLIC':
        for x in xs:
            b,qa=fit(x,y);centers.append(old.flic_stat(x,y,src,b,np.ones(len(y))));betas.append(b);qas.append(qa)
    centers=np.asarray(centers)
    writejson(Q/f'{stem}_point_checkpoint.json',dict(**base,estimator=method,labels=labels,centers=centers.tolist(),MLE_failure=mle_error,diagnostics=qas))
    print('POINT SAVED',stem,method,'n=',len(d),'seconds=',round(time.monotonic()-tick,2),flush=True)
    rows=[];deltas=[];boot_valid=None;fail=Counter()
    if method=='Firth_source_FLIC':
        seed=61117100+int.from_bytes(hashlib.sha256(stem.encode()).digest()[:3],'big')
        plan,inv=family_plan(d.family.to_numpy(str),d.stratum.to_numpy(str),seed,B)
        cache=P/f'{stem}_bootstrap.npz';draws=np.full((B,len(xs),4),np.nan);done=0
        if cache.exists():
            cc=np.load(cache);assert int(cc['seed'])==seed;draws=cc['draws'];done=int(cc['done'])
        for i in range(done,B):
            w=plan[i,inv].astype(float)
            for j,x in enumerate(xs):
                try:draws[i,j]=boot_fit(x,y,src,w,betas[j])
                except (ValueError,AssertionError,np.linalg.LinAlgError) as e:fail[f'{labels[j]}:{e}']+=1
            if (i+1)%50==0 or i==B-1:
                np.savez_compressed(cache,draws=draws,done=i+1,seed=seed)
                writejson(Q/f'{stem}_bootstrap_progress.json',dict(completed=i+1,total=B,failures=dict(fail),elapsed_seconds=time.monotonic()-tick))
                print('BOOT',stem,i+1,round(time.monotonic()-tick,1),flush=True)
        for j,label in enumerate(labels):
            good=draws[np.isfinite(draws[:,j]).all(axis=1),j];ok=len(good)>=450
            lo,hi=np.quantile(good,[.025,.975],axis=0) if ok else (np.full(4,np.nan),np.full(4,np.nan))
            rows.append(dict(**base,model=label,estimator=method,status='estimated' if ok else 'point_only',rank=xs[j].shape[1],
                bootstrap_attempted=B,bootstrap_valid=len(good),inference='source/country-stratified family percentile bootstrap',
                **old.columns(centers[j],lo,hi,good.std(0,ddof=1)),**qas[j]))
        for j in range(1,len(labels)):
            ref=j-1 if module=='prior_health' else 0
            mask=np.isfinite(draws[:,[ref,j]]).all(axis=(1,2));diff=draws[mask,j]-draws[mask,ref];ok=len(diff)>=450
            lo,hi=np.quantile(diff,[.025,.975],axis=0) if ok else (np.full(4,np.nan),np.full(4,np.nan))
            deltas.append(dict(**base,reference=labels[ref],adjusted=labels[j],status='estimated' if ok else 'point_only',paired_valid=len(diff),
                inference='same family bootstrap paired functional differences',**old.columns(centers[j]-centers[ref],lo,hi,diff.std(0,ddof=1))))
    else:
        crit=tdist.ppf(.975,base['families']-1)
        for j,label in enumerate(labels):
            se=np.sqrt((influences[j]**2).sum(0))
            rows.append(dict(**base,model=label,estimator=method,status='estimated',rank=xs[j].shape[1],inference='family HC1 t(G-1); fixed empirical reference',
                **old.columns(centers[j],centers[j]-crit*se,centers[j]+crit*se,se),**qas[j]))
        for j in range(1,len(labels)):
            ref=j-1 if module=='prior_health' else 0;delta=centers[j]-centers[ref]
            se=np.sqrt(((influences[j]-influences[ref])**2).sum(0))
            deltas.append(dict(**base,reference=labels[ref],adjusted=labels[j],status='estimated',inference='joint family influence; paired HC1 t(G-1)',
                **old.columns(delta,delta-crit*se,delta+crit*se,se)))
    for row in rows:row['elapsed_seconds']=time.monotonic()-tick;row['MLE_failure']=mle_error
    pd.DataFrame(rows).to_csv(T/f'{stem}_results.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(deltas).drop(columns=['OR','OR_lo','OR_hi'],errors='ignore').to_csv(T/f'{stem}_paired_differences.csv',index=False,encoding='utf-8-sig')
    writejson(Q/f'{stem}_fit_qa.json',dict(status='COMPLETE',elapsed_seconds=time.monotonic()-tick,bootstrap_failures=dict(fail),diagnostics=qas))
    np.savez_compressed(P/f'{stem}_coefficients.npz',**{f'beta_{i}':b for i,b in enumerate(betas)})
    print('COMPLETE',stem,round(time.monotonic()-tick,1),flush=True)

def gate():
    assert (R/'VARIABLE_RULES_LOCK.sha256').exists()
    expected=json.loads((W/'PPC_IADL_ZHOU_ADAPTATION_20261117/ONE_STAGE_LOCK.json').read_text())
    for p,h in expected.items():assert sha(W/p)==h,('frozen_input_hash_changed',p)
    table=pd.read_csv(W/'PPC_IADL_ZHOU_ADAPTATION_20261117/ONE_STAGE_IPD_RESULTS.csv');checks=[]
    for comp in old.COMPS:
        d,y,x,src=old.load(comp,'M_severity')
        checkpoint=Q/f'replication_{comp}.json'
        if checkpoint.exists():ans=json.loads(checkpoint.read_text())
        else:
            ans=old.corrected(d,y,x,src) if comp=='H11_vs_H01' else old.ordinary(d,y,x,src,False)
            writejson(checkpoint,ans)
        est='one_stage_Firth_FLIC_full_sample' if comp=='H11_vs_H01' else 'one_stage_MLE'
        frozen=table.loc[table.comparison.eq(comp)&table.layer.eq('M_severity')&table.estimator.eq(est)].iloc[0]
        keys=[k+s for k in old.KEYS for s in ('','_lo','_hi')]+['OR','OR_lo','OR_hi']
        delta=max(abs(float(ans[k])-float(frozen[k])) for k in keys)
        assert delta<2e-7,(comp,delta)
        assert (len(d),d.person.nunique(),d.family.nunique(),int(y.sum()))==tuple(int(frozen[k]) for k in ('windows','people','families','events'))
        checks.append({**ans,'comparison':comp,'status':'PASS','max_absolute_numeric_difference':delta,
            'windows':len(d),'people':int(d.person.nunique()),'families':int(d.family.nunique()),'events':int(y.sum()),
            'original_confidence_intervals':'actually rerun; original estimator/seed/499 attempts for H11; no CI substituted'})
        print('REPLICATION PASS',comp,delta,flush=True)
    writejson(R/'REPLICATION_GATE.json',dict(status='PASS',frozen_hashes_checked=len(expected),models=checks))

def collect():
    frames=[pd.read_csv(p) for p in T.glob('*_results.csv')]
    if not frames:return
    a=pd.concat(frames,ignore_index=True)
    a.loc[a.module.isin(['primary','prior_health'])].to_csv(R/'LIFESTYLE_BMI_ADJUSTED_RESULTS.csv',index=False,encoding='utf-8-sig')
    a.loc[a.module.isin(['psychological','cognitive'])].to_csv(R/'DOMAIN_RESULTS_UPDATE.csv',index=False,encoding='utf-8-sig')
    p=[pd.read_csv(f) for f in T.glob('*_paired_differences.csv')]
    if p:pd.concat(p,ignore_index=True).to_csv(R/'PAIRED_ADJUSTMENT_DIFFERENCES.csv',index=False,encoding='utf-8-sig')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--gate',action='store_true');ap.add_argument('--collect',action='store_true')
    ap.add_argument('--comparison',choices=old.COMPS);ap.add_argument('--module',choices=['primary','prior_health','psychological','cognitive']);ap.add_argument('--timing',choices=['t','pre'],default='t');a=ap.parse_args()
    if a.gate:gate()
    elif a.collect:collect()
    else:run(a.comparison,a.module,a.timing);collect()

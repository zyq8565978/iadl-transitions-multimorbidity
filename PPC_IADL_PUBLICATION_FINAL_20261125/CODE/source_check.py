"""Only the eight protocol-authorised source-specific binary checks; raw files stay local."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import hashlib,json,sys,zipfile,time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[1]; W=ROOT.parent
CL=W/'PPC_IADL_FINAL_COMBINATION_CLOSURE_20261125';MI=W/'PPC_IADL_ALL_METHODS_MI_20261118';V6=W/'PPC_IADL_V6_Repair_Synthesis_20261116'
for d in ['QA','LOCAL','AGGREGATES','SUBMISSION','REPORT','FIGURES']:(ROOT/d).mkdir(exist_ok=True)
L=ROOT/'LOCAL';Q=ROOT/'QA';A=ROOT/'AGGREGATES'
sys.path.insert(0,str(CL/'CODE'))
from fit_combinations import base,precise_fit,relocated
from firth_engine_v6 import functionals
import numpy as np,pandas as pd
from scipy.stats import t,chi2
from scipy.optimize import minimize_scalar
def js(p,a):p.write_text(json.dumps(a,ensure_ascii=False,indent=2,default=str),encoding='utf8')
def bind():
    base.effect_gate()
    checked={}
    for pr,n in [(CL,'SAFE_DELIVERY_FINAL_COMBINATION_20261125.zip'),(MI,'SAFE_DELIVERY_ALL_METHODS_MI_20261118.zip')]:
        f=pr/n;rec=json.loads((pr/('QA/FINAL_RELEASE_RECEIPT.json' if pr==CL else 'QA/RELEASE_RECEIPT.json')).read_text())
        assert base.digest(f)==rec['sha256']
        with zipfile.ZipFile(f) as z:
            v=json.loads(z.read('RELEASE_MANIFEST.json')); entries=v['files'] if isinstance(v,dict) else {o['path']:o for o in v}
            assert set(z.namelist())==set(entries)|{'RELEASE_MANIFEST.json'}
            for name,o in entries.items():assert hashlib.sha256(z.read(name)).hexdigest()==o['sha256']
            checked[n]=dict(sha256=rec['sha256'],members=len(z.namelist()))
    assert checked['SAFE_DELIVERY_FINAL_COMBINATION_20261125.zip']['sha256']=='448566336f63defb59ed220e0684150fa48ac2a94ef9570a8d9632886b02129c'
    totals={};sets={}
    historic=pd.read_csv(V6/'TABLES/META_INPUTS.csv');historic=historic[(historic.set=='A_original')&(historic.model_layer=='M_severity')]
    for comp in base.COMPS:
        d,y,x,src=base.old.load(comp,'M_severity')
        assert d.code_fixed1.isin([0,1,2,4]).all() and np.array_equal(y,d.code_fixed2.isin([3,5,6,7]))
        for i,s in enumerate(base.SOURCES):
            k=f'{s}_{comp}'; m=src==i;g=d.loc[m].reset_index(drop=True);yy=y[m]
            xx=np.ascontiguousarray(x[m][:,np.r_[True,True,np.ptp(x[m,2:],axis=0)>0]])
            old=historic.loc[historic.source.eq(s)&historic.comparison.eq(comp)].iloc[0];assert (len(g),g.person.nunique(),g.family.nunique())==(old.windows,old.people,old.families)
            assert min(((g.hx==h)&(yy==e)).sum() for h in (0,1) for e in (0,1))>=10
            seed=6111606+10*i+list(base.COMPS).index(comp)
            plan,inv=base.qc.family_plan(g.cluster.to_numpy(str),g.country.to_numpy(str) if s=='SHARE' else np.repeat('all',len(g)),seed,499)
            pre=V6/f'PRIVATE/FIRTH/FIRTH_PLAN_{s}_{comp}_original.npz'
            if pre.exists():
                z=np.load(pre,allow_pickle=False);assert np.array_equal(z['family_multiplicities'],plan) and np.array_equal(z['row_to_family'],inv)
            # Family codes in numeric local files; no participant keys leave input environment.
            np.savez_compressed(L/(k+'_base.npz'),x=xx,y=yy,cluster=pd.factorize(g.family,sort=True)[0],plan=plan,inv=inv)
            totals[k]=dict(source=s,comparison=comp,people=int(g.person.nunique()),families=int(g.family.nunique()),windows=len(g),events=int(yy.sum()),v6_method=old.estimator,seed=seed)
            sets[k]=g
    for m in range(1,51):
        for k,g in sets.items():
            extra,_=base.extras(g,m);assert extra.shape[1]==5 and np.array_equal(extra,extra.astype('uint8'))
            np.save(L/f'{k}_extra{m:02d}.npy',extra.astype('uint8'))
    js(Q/'authority.json',checked); js(Q/'source_inputs.json',totals)
def one(k,m,firth):
    dest=L/f'{k}_{m:02d}_{"F" if firth else "M"}.npz'
    if dest.exists():return
    z=np.load(L/(k+'_base.npz'));xx=np.c_[z['x'],np.load(L/f'{k}_extra{m:02d}.npy')];xx=np.ascontiguousarray(xx[:,base.qc.independent(xx)]);y=z['y'];diag={}
    if not firth:
        v,inf,diag,b=base.qc.mle_influence(pd.DataFrame(dict(family=z['cluster'])),y,xx);u=inf.T@inf;good=-1
    else:
        b,diag=precise_fit(xx,y);v=functionals(xx,y,b,1,np.ones(len(y)))[:4]
        draws=[];fail=Counter()
        for w in z['plan'][:,z['inv']].astype(float):
            try:
                try:bb,_=base.qc.fit(xx,y,w,warm=b)
                except ValueError as e:
                    if str(e)!='design_rank_deficient':raise
                    sel=base.qc.independent(xx[w>0]);bnew,_=base.qc.fit(xx[:,sel],y,w,warm=b[sel]);bb=np.zeros_like(b);bb[sel]=bnew
                draws.append(functionals(xx,y,bb,1,w)[:4])
            except (ValueError,AssertionError,np.linalg.LinAlgError) as e:fail[str(e)]+=1
        good=len(draws);u=np.cov(draws,rowvar=False) if good>=450 else np.full((4,4),np.nan);diag['failed_draws']=dict(fail)
    assert len(v)==4 and np.isclose(v[1]-v[2],v[3]) and np.isfinite(v).all()
    np.savez_compressed(dest,q=v,U=u,valid=good,rank=xx.shape[1]);js(L/(dest.stem+'.json'),diag)
def worker(a):
    k,m,f=a
    try:one(k,m,f);return k,m,''
    except (ValueError,AssertionError,np.linalg.LinAlgError) as e:return k,m,type(e).__name__+': '+str(e)
def synth(results):
    outs=[]
    for c in base.COMPS:
        g=results.loc[results.comparison.eq(c)&results.functional.eq('OR')&results.status.eq('Estimated')];k=len(g)
        r=dict(comparison=c,sources='; '.join(g.source),k=k)
        if k<3:outs.append(r|dict(status='Not estimable'));continue
        y=np.log(g.estimate.to_numpy());v=g.se.to_numpy()**2
        def reml(a):
            w=1/(v+a);mu=w@y/w.sum();return .5*(np.log(v+a).sum()+np.log(w.sum())+w@((y-mu)**2))
        o=minimize_scalar(reml,bounds=(0,max(np.ptp(y)**2,max(v),.01)*10),method='bounded',options={'xatol':1e-13})
        a=0 if reml(0)<=o.fun+1e-13 else float(o.x);w=1/(v+a);mu=w@y/w.sum();wf=1/v;q=wf@((y-wf@y/wf.sum())**2);vt=(k-1)*wf.sum()/(wf.sum()**2-(wf**2).sum())
        raw=w@((y-mu)**2)/(k-1);se=np.sqrt(max(1,raw)/w.sum());cr=t.ppf(.975,k-1);vpi=t.ppf(.975,k-2)*np.sqrt(a+se**2)
        outs.append(r|dict(status='Estimated',estimate=np.exp(mu),lower=np.exp(mu-cr*se),upper=np.exp(mu+cr*se),p=2*t.sf(abs(mu/se),k-1),tau2=a,I2=100*a/(a+vt),Q=q,Q_p=chi2.sf(q,k-1),q_raw=raw,pi_low=np.exp(mu-vpi),pi_high=np.exp(mu+vpi)))
    pd.DataFrame(outs).to_csv(A/'source_synthesis.csv',index=False)
def run():
    if not (Q/'source_inputs.json').exists():bind()
    specs=json.loads((Q/'source_inputs.json').read_text());attempts={k:[] for k in specs};forced={k for k,z in specs.items() if z['source']=='SHARE' and z['comparison']==base.COMPS[0]}
    with ProcessPoolExecutor(max_workers=min(6,os.cpu_count() or 2)) as ex:
        for k,m,err in ex.map(worker,[(k,m,False) for k in specs if k not in forced for m in range(1,51)],chunksize=3):
            if err:attempts[k].append((m,err))
        fset=forced|{k for k in specs if attempts[k]};js(Q/'source_methods.json',dict(Firth=sorted(fset),MLE_preflight_failures=attempts))
        print('MLE checked; full-spec Firth:',sorted(fset),flush=True)
        for k,m,err in ex.map(worker,[(k,m,True) for k in sorted(fset) for m in range(1,51)],chunksize=1):
            if err:js(L/f'{k}_{m:02d}_F_error.json',{'error':err})
            if m%10==0:print(k,m,flush=True)
    rows=[]
    for k,rec in specs.items():
        f=k in fset;files=[L/f'{k}_{m:02d}_{"F" if f else "M"}.npz' for m in range(1,51)]
        zs=[np.load(p) for p in files if p.exists()];meta=rec|dict(estimator='Firth/FLIC' if f else 'MLE with family HC1',M=len(zs),M_cov=sum(np.isfinite(z['U']).all() for z in zs))
        if len(zs)!=50:
            for name in ['OR','risk_history','risk_reference','risk_difference']:rows.append(meta|dict(functional=name,status='Not estimable'))
            continue
        qs=np.array([z['q'] for z in zs]);us=np.array([z['U'] for z in zs]);full=np.isfinite(us).all();mean=qs.mean(0)
        if full:_,_,_,_,df,se,lo,hi,fmi,mc=base.pool_values(qs,us,rec['families']-1)
        for j,name in enumerate(['OR','risk_history','risk_reference','risk_difference']):
            tr=np.exp if j==0 else lambda a:a
            row=meta|dict(functional=name,estimate=tr(mean[j]),status='Estimated' if full else 'Point only',min_valid_draws=min(int(z['valid']) for z in zs) if f else None)
            if full:row.update(lower=tr(lo[j]),upper=tr(hi[j]),df=df[j],se=se[j],p=2*t.sf(abs(mean[j]/se[j]),df[j]) if j in (0,3) else None,FMI=fmi[j],MCSE=mc[j],MCSE_over_SE=mc[j]/se[j])
            rows.append(row)
    out=pd.DataFrame(rows);out.to_csv(A/'source_MI_functionals.csv',index=False);synth(out);print(out.query("functional=='OR'")[['source','comparison','M','M_cov','estimate','lower','upper']],flush=True)
if __name__=='__main__':run()

"""Bounded parallel execution of disjoint MI jobs with the original draw plans and thresholds."""
from binding import *
from fit_combinations import precise_fit,pool
from block_draw import BlockDraw
from functools import lru_cache
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import time,os,ctypes,shutil
cache=ROOT/'PRIVATE/dispatch';cache.mkdir(exist_ok=True)
def atomic(p,obj):
    tmp=p.with_suffix('.writing');dump(tmp,obj);os.replace(tmp,p)
@lru_cache(1)
def arrays(comp):
    d,_,x,src=load_risk(comp);names=['Intercept','history','source:HRS','source:ELSA','source:SHARE']
    for s in base.SOURCES:
        a,_,_=base.old.original(s,comp,'M_severity')
        names.extend(s+':'+str(t) for t in a['terms'] if t not in ('Intercept','hx'))
    return d,x,src,names
def matrix(comp,mi,stem):
    d,x,src,names=arrays(comp);a,more=base.extras(d,mi);xx=np.column_stack([x,a]);idx=base.qc.independent(xx)
    assert list(idx[:2])==[0,1] and len(names+more)==xx.shape[1]
    pd.DataFrame(dict(term=names+more,retained=np.isin(np.arange(xx.shape[1]),idx))).to_csv(cache/f'{stem}_mi{mi:03}_design.csv',index=False)
    return d,np.ascontiguousarray(xx[:,idx]),src
def decide(comp,k):
    stem=f'COMB_{comp}_state{k}';out=cache/(stem+'_engine.json')
    if out.exists():return json.loads(out.read_text())
    su=pd.read_csv(ROOT/'PRIVATE/support.csv').query('comparison==@comp and code==@k').iloc[0]
    engine='FIRTH';fail=None
    if not su.arm_support:engine='NO_ARM_SUPPORT'
    elif comp=='H10_vs_H00':
        orig=ROOT/'QA'/(stem+'_estimator.json')
        if orig.exists() and json.loads(orig.read_text())['engine']=='FIRTH':
            fail=json.loads(orig.read_text())['MLE_failure']
        else:
            engine='MLE'
            for mi in range(1,51):
                d,x,src=matrix(comp,mi,stem);y=d.code_fixed2.eq(k).to_numpy(float)
                try:q,inf,qa,b=base.qc.mle_influence(d,y,x)
                except (ValueError,AssertionError,np.linalg.LinAlgError) as exc:
                    engine='FIRTH';fail=dict(mi=mi,reason=str(exc));break
                nm=f'{stem}_mi{mi:03}';np.savez_compressed(cache/(nm+'_tryMLE.npz'),q=q,U=inf.T@inf,beta=b)
                atomic(cache/(nm+'_tryMLE.json'),qa)
    if engine=='FIRTH' and not su.source_calibration_possible:engine='NO_FLIC_ROOT'
    obj=dict(stem=stem,comparison=comp,code=k,engine=engine,MLE_failure=fail,events=int(su.events),windows=int(su.windows),families=int(su.families),people=int(su.people))
    atomic(out,obj);return obj
def full(job):
    pl,mi=job;stem=pl['stem'];nm=f'{stem}_mi{mi:03}';comp=pl['comparison'];k=pl['code']
    result=cache/(nm+'.npz');receipt=cache/(nm+'.json');oldp=ROOT/'PRIVATE'/(nm+'.npz');oldj=ROOT/'QA'/(nm+'.json')
    if receipt.exists():return nm
    if oldj.exists() and oldp.exists():
        shutil.copyfile(oldp,result);atomic(receipt,json.loads(oldj.read_text())|{'implementation':'native checkpoint'});return nm
    info=pl|dict(mi=mi);start=time.monotonic()
    if pl['engine']=='MLE':
        shutil.copyfile(cache/(nm+'_tryMLE.npz'),result)
        atomic(receipt,info|dict(status='ESTIMATED',diagnostics=json.loads((cache/(nm+'_tryMLE.json')).read_text()),implementation='original family HC1'));return nm
    d,x,src=matrix(comp,mi,stem);y=d.code_fixed2.eq(k).to_numpy(float)
    if not (cache/(nm+'_point.npz')).exists() or not (cache/(nm+'_point.json')).exists():
        try:b,qa=precise_fit(x,y);q=base.old.flic_stat(x,y,src,b,np.ones(len(y)))
        except (ValueError,AssertionError,np.linalg.LinAlgError) as ex:
            atomic(receipt,info|dict(status='NON_ESTIMABLE',reason=str(ex)));return nm
        atomic(cache/(nm+'_point.json'),info|dict(point=q.tolist(),diagnostics=qa))
        tmp=cache/(nm+'_worker_point.npz');np.savez_compressed(tmp,q=q,b=b);os.replace(tmp,cache/(nm+'_point.npz'))
    z=np.load(cache/(nm+'_point.npz'));q=z['q'];b=z['b'];qa=json.loads((cache/(nm+'_point.json')).read_text())['diagnostics']
    rg=ROOT/'QA/FIXED_PLAN_FLIC_ROOT_GATE.csv'
    if rg.exists():
        proof=pd.read_csv(rg).set_index('stem').loc[stem]
        assert isinstance(proof.bootstrap_variance_mathematically_impossible,(bool,np.bool_))
        if proof.bootstrap_variance_mathematically_impossible:
            assert int(proof.planned)==199 and int(proof.max_possible_valid)<180 and int(proof.max_possible_valid)+int(proof.no_root_draws)==199
            rlog=cache/(nm+'_redraws.json');actual=json.loads(rlog.read_text()) if rlog.exists() else {}
            np.savez_compressed(result,q=q,U=np.full((4,4),np.nan),beta=b)
            atomic(receipt,info|dict(status='POINT_ONLY_STRUCTURAL_BOOTSTRAP_ROOT_GATE',diagnostics=qa,bootstrap_attempted=actual.get('completed',0),
                actual_successes_in_committed_draws=actual.get('completed',0)-actual.get('failures',0),root_plans_all_checked=199,
                max_possible_valid=int(proof.max_possible_valid),required=180,
                implementation='199 calibration feasibility checks; omitted redundant fits only after exact impossibility proof'));return nm
    seed=71118000+int.from_bytes(hashlib.sha256(stem.encode()).digest()[:3],'big')
    plan,inv=base.qc.family_plan(d.family.to_numpy(str),d.stratum.to_numpy(str),seed,199);F=BlockDraw(x,y,src)
    boot=cache/(nm+'_redraws.npz');rlog=cache/(nm+'_redraws.json');draw=np.full((199,4),np.nan);done=0;err={}
    if boot.exists():
        saved=np.load(boot);draw=saved['draws'];done=int(saved['done']);err=json.loads(str(saved['errors']))
    use_native=False;delta=None
    for i in range(done,199):
        w=plan[i,inv].astype(float)
        try:
            if i==0:
                native=base.qc.boot_fit(x,y,src,w,b);fast=F(w,b);delta=float(max(abs(native-fast)))
                use_native=not np.allclose(fast,native,atol=2e-8,rtol=1e-7);draw[i]=native
                atomic(cache/(nm+'_equivalence.json'),dict(draw=i+1,max_diff=delta,use_native=use_native))
            else:
                if i==done and done and (cache/(nm+'_equivalence.json')).exists():use_native=json.loads((cache/(nm+'_equivalence.json')).read_text())['use_native']
                draw[i]=base.qc.boot_fit(x,y,src,w,b) if use_native else F(w,b)
        except (ValueError,AssertionError,np.linalg.LinAlgError) as ex:err[str(i+1)]=str(ex)
        if (i+1)%25==0 or i==198:
            tmp=cache/(nm+'_pending.npz');np.savez_compressed(tmp,draws=draw,done=i+1,errors=json.dumps(err));os.replace(tmp,boot)
            atomic(rlog,dict(completed=i+1,failures=len(err),elapsed_s=time.monotonic()-start))
    good=draw[np.isfinite(draw).all(1)];u=np.cov(good,rowvar=False) if len(good)>=180 else np.full((4,4),np.nan)
    np.savez_compressed(result,q=q,U=u,beta=b);atomic(receipt,info|dict(status='ESTIMATED' if np.isfinite(u).all() else 'POINT_ONLY_INSUFFICIENT_VALID_BOOTSTRAP',bootstrap_valid=len(good),bootstrap_attempted=199,failures=dict(Counter(err.values())),diagnostics=qa,implementation='source-block score with native verification and rescue',seconds=time.monotonic()-start));return nm
def main():
    import argparse
    opt=argparse.ArgumentParser();opt.add_argument('--workers',type=int,default=6);args=opt.parse_args()
    assert 1<=args.workers<=6 and json.loads((ROOT/'QA/binding.json').read_text())['pass_all']
    mem=(ctypes.c_ulonglong*8)();mem[0]=64;ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
    nw=min(args.workers,max(1,int((mem[2]/2**30-5)/1.4)))
    plans=[decide(c,k) for k in TARGETS for c in base.COMPS]
    atomic(ROOT/'QA/EXECUTION_PLAN.json',dict(workers=nw,blas_threads=1,plan_by_spec=plans))
    if not (ROOT/'QA/FIXED_PLAN_FLIC_ROOT_GATE.csv').exists():
        from flic_root_gate import run as root_proof
        root_proof()
    active=[z for z in plans if z['engine'] in ('FIRTH','MLE')]
    jobs=[(p,m) for m in range(1,51) for p in active]
    # Native reference runs advance from MI 1; handle their completed MIs and later MIs first.
    jobs.sort(key=lambda pm: (int(pm[0]['code']==3 and pm[1]<8),pm[1],pm[0]['code']))
    with ProcessPoolExecutor(max_workers=nw) as ex:
        futs={ex.submit(full,j):j for j in jobs}
        for i,f in enumerate(as_completed(futs)):
            f.result()
            if (i+1)%10==0:print(i+1,'/',len(jobs),'MI fits or archived checkpoints',flush=True)
    from closure_results import combine
    combine(plans)
if __name__=='__main__':main()

"""Eight locked one-versus-rest specifications, no recalculation of MI or old effects."""
from binding import *
import argparse,types,time,inspect
from scipy.optimize import minimize
from scipy.linalg import cho_solve

def relocated(fun,**changes):
    return types.FunctionType(fun.__code__,dict(fun.__globals__,**changes),fun.__name__,fun.__defaults__,fun.__closure__)

def precise_fit(x,y):
    try:return base.qc.fit(x,y)
    except ValueError as e:
        if str(e)!='adjusted_score_unresolved':raise
    steps=[]
    def same_objective(fun,b,**kw):
        evaluate=inspect.currentframe().f_back.f_locals['evaluate']
        res=minimize(fun,b,**kw)
        for i in range(3):
            f,g,cf=evaluate(res.x);step=float(np.max(abs(cho_solve(cf,g,check_finite=False))))
            steps.append((float(f),step,float(np.max(abs(g)))/len(y)))
            if step<=2e-7 and steps[-1][2]<=1e-8 or i==2:break
            alt=minimize(fun,res.x,**(kw|dict(options=dict(maxiter=500,gtol=1e-10))))
            if not np.isfinite(alt.fun) or alt.fun>-f+1e-10:break
            res=alt
        return res
    opt=relocated(base.qc.fit.__globals__['optimize'],minimize=same_objective)
    b,qa=relocated(base.qc.fit,optimize=opt)(x,y)
    return b,qa|dict(full_only_precision_continuation=steps)

qcopy=types.SimpleNamespace(**{k:v for k,v in vars(base.qc).items() if not k.startswith('__')})
qcopy.fit=precise_fit
fit=relocated(base.binary,R=ROOT,qc=qcopy)
pool=relocated(base.pooled,R=ROOT)
def run(comp,k):
    assert json.loads((ROOT/'QA/binding.json').read_text())['pass_all']
    d,old_y,x,src=load_risk(comp); y=(d.code_fixed2.to_numpy(int)==k).astype(float)
    stem=f'COMB_{comp}_state{k}';dest=ROOT/'TABLES'/f'{stem}.csv'
    if dest.exists():return
    rec=pd.read_csv(ROOT/'PRIVATE/support.csv').query('comparison==@comp and code==@k').iloc[0]
    meta=dict(comparison=comp,code=k,label=STATES[k],role='post_result_secondary')
    if not rec.arm_support:
        pd.DataFrame([dict(**meta,status='NON_ESTIMABLE_POOLED_ARM_OUTCOME_SUPPORT_BELOW_10',people=rec.people,windows=rec.windows,families=rec.families,events=rec.events,MI_completed=0)]).to_csv(dest,index=False);return
    def design(mi):
        extra,n=base.extras(d,mi);xx=np.column_stack((x,extra))
        names=['Intercept','history','source:HRS','source:ELSA','source:SHARE']
        for s in base.SOURCES:
            a,_,_=base.old.original(s,comp,'M_severity')
            names.extend(s+':'+str(t) for t in a['terms'] if t not in ('Intercept','hx'))
        names.extend(n);assert len(names)==xx.shape[1]
        cols=base.qc.independent(xx);assert list(cols[:2])==[0,1]
        pd.DataFrame(dict(term=names,retained=np.isin(np.arange(len(names)),cols))).to_csv(ROOT/'QA'/f'{stem}_mi{mi:03d}_design.csv',index=False)
        return xx
    engine='FIRTH' if comp==base.COMPS[0] else 'MLE'; ledger=ROOT/'QA'/f'{stem}_estimator.json'
    if ledger.exists():engine=json.loads(ledger.read_text())['engine']
    if engine=='MLE':
        # Tentative MLEs are isolated. A single failure switches the entire specification.
        provisional=ROOT/'PRIVATE'/('MLE_PREFLIGHT_'+stem);provisional.mkdir(exist_ok=True)
        fail=None
        for mi in range(1,51):
            file=provisional/f'{mi:03d}.npz'
            if file.exists():continue
            xx=design(mi);xx=xx[:,base.qc.independent(xx)]
            try:
                q,inf,qa,beta=base.qc.mle_influence(d,y,xx)
            except (ValueError,AssertionError,np.linalg.LinAlgError) as e:
                fail=dict(mi=mi,error=type(e).__name__,reason=str(e));engine='FIRTH';break
            np.savez_compressed(file,q=q,U=inf.T@inf,beta=beta)
            dump(provisional/f'{mi:03d}.json',qa)
        dump(ledger,dict(engine=engine,MLE_failure=fail,all_50_uniform=True))
        if engine=='MLE':
            import shutil
            for mi in range(1,51):
                shutil.copyfile(provisional/f'{mi:03d}.npz',ROOT/f'PRIVATE/{stem}_mi{mi:03d}.npz')
                dump(ROOT/f'QA/{stem}_mi{mi:03d}.json',dict(stem=stem,mi=mi,engine=engine,people=int(rec.people),windows=int(rec.windows),families=int(rec.families),events=int(rec.events),status='ESTIMATED',diagnostics=json.loads((provisional/f'{mi:03d}.json').read_text())))
    if engine=='FIRTH':
        if not ledger.exists():dump(ledger,dict(engine=engine,reason='preassigned H11',all_50_uniform=True))
        if not rec.source_calibration_possible:
            pd.DataFrame([dict(**meta,engine=engine,status='NON_ESTIMABLE_SOURCE_FLIC_CALIBRATION',people=rec.people,windows=rec.windows,families=rec.families,events=rec.events,MI_completed=0)]).to_csv(dest,index=False);return
        for mi in range(1,51):
            status=ROOT/f'QA/{stem}_mi{mi:03d}.json'
            if status.exists():continue
            try:fit(d,y,design(mi),src,stem,mi,'FIRTH',B=199)
            except (ValueError,AssertionError,np.linalg.LinAlgError) as e:
                dump(status,dict(stem=stem,mi=mi,engine=engine,status='NON_ESTIMABLE',reason=str(e),error=type(e).__name__))
            if mi in (1,5) or mi%10==0:print(stem,mi,'of 50',flush=True)
    pool(stem,['OR','risk_history','risk_reference','risk_difference'],['log','probability','probability','probability_difference'],meta|dict(engine=engine))
    print('Closed',stem,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('comparison',choices=base.COMPS);a.add_argument('code',nargs='*',type=int);v=a.parse_args()
    for k in (v.code or TARGETS):
        assert k in TARGETS;run(v.comparison,k)

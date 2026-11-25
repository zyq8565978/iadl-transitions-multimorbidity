"""Corrected three-state diagnostic fits and conservative, narrowly labeled results."""
import argparse, hashlib, json, subprocess, time
from datetime import datetime
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize
from panel_v6_engine import R, RT, likelihood, generic, rcheck, qmat
from panel_v6_data import (v5, SOURCES, ARMS, input_data, module_inputs, frozen_inputs, death_and_sequences,
                           support_masks, fit_info, calibration_survival, compatible_interval_ci, output, safe_n)

QA=R/'QA'
NBOOT=300
BOOT_MIN=255
PRELOCK=QA/'PANEL_V6_PRE_EFFECT_LOCK.json'
MAIN_GATE=QA/'PANEL_V6_EXECUTE.json'
v5.output=output  # V5 fallback can only write to this run's output function.

def filehash(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def gate(s):
    top=json.loads(MAIN_GATE.read_text(encoding='utf-8'))
    assert top['primary_synthesis_completed'] is True and top['authorized'] is True
    lock=json.loads(PRELOCK.read_text(encoding='utf-8'))
    assert lock['stage']=='PANEL_FROZEN_BEFORE_REAL_EFFECTS'
    print('V6 PANEL lock',filehash(PRELOCK),flush=True)
    for rel,sha in lock['locked_components'].items():
        assert filehash(R/rel)==sha,rel
    for p in frozen_inputs(s)+module_inputs(s):
        assert filehash(p)==lock['locked_inputs'][p.name],p.name
    assert filehash(Path(v5.__file__))==lock['frozen_v5_panel_dependency']
    p=R/'PRIVATE'/f'panel_admin_V6_{s}.csv'
    assert (filehash(p) if p.exists() else None)==lock['documented_administrative_end_inputs'][s]

def panel_lock():
    """Freeze dates, implementation and the outcome-independent criteria before any real panel fit."""
    assert not PRELOCK.exists()
    ns=['PANEL_V6_SPEC.md','CODE/panel_v6_data.py','CODE/panel_v6_engine.py','CODE/panel_v6_likelihood.cpp',
        'CODE/panel_v6_models.py','CODE/panel_v6_verify.R','CODE/panel_v6_sequence_test.py','QA/PANEL_V6_SYNTHETIC_QA.json',
        'QA/PANEL_V6_SEQUENCE_QA.json']
    document=[R/'PRIVATE'/f'panel_admin_V6_{s}.csv' for s in SOURCES]
    raw=[p for s in SOURCES for p in frozen_inputs(s)+module_inputs(s)]
    prev={'stage':'PANEL_FROZEN_BEFORE_REAL_EFFECTS','locked_at':datetime.now().astimezone().isoformat(),
          'locked_components':{p:filehash(R/p) for p in ns},
          'frozen_v5_panel_dependency':filehash(Path(v5.__file__)),
          'locked_inputs':{p.name:filehash(p) for p in raw},
          'documented_administrative_end_inputs':{s:filehash(p) if p.exists() else None for s,p in zip(SOURCES,document)}}
    PRELOCK.write_text(json.dumps(prev,indent=2,ensure_ascii=False),encoding='utf-8')
    print('V6 panel freeze SHA-256',filehash(PRELOCK),flush=True)

def covars(a,s,c,label):
    x, cols, _=v5.design(a,s,c,compact=label=='compact')
    if label=='direction_specific':
        z=np.zeros((len(a),4,3)); z[:,1,:]=cols[['count','symptom','cognition']].to_numpy()
        x=np.concatenate([x,z],axis=2) # BA-specific deviation from AB effect.
    return x,cols

def dated_living_reference(a,raw):
    r,_,_=v5.sequences(a,raw.loc[~raw.status.eq('D')].copy())
    return r

def fit_status(f):
    if f is None or not np.isfinite(f.fun) or not np.isfinite(f.x).all(): return 'nonfinite_likelihood'
    if max(abs(f.jac))>=.015: return 'gradient_unconverged'
    if max(abs(f.x[:4]))>=13 or (len(f.x)>4 and max(abs(f.x[4:]))>=12): return 'real_boundary_or_scaling_limit'
    return 'stationary'

def minimize_ladder(f,b,w=None,inv=None,secondary=True):
    opts={'gtol':1e-5,'maxiter':2000}
    options=[('BFGS',opts if inv is None else dict(opts,hess_inv0=inv)),
             ('L-BFGS-B',dict(opts,ftol=1e-13,maxls=50))]
    attempts=[];current=np.array(b,copy=True)
    for method,opt in options[:2 if secondary else 1]:
        try:
            z=minimize(lambda a:f(a,w),current,method=method,jac=True,options=opt)
        except (np.linalg.LinAlgError,ValueError,FloatingPointError):
            z=None
        attempts.append((method,fit_status(z),z))
        if attempts[-1][1]=='stationary':return z,attempts
        if z is not None and np.isfinite(z.fun) and np.isfinite(z.x).all(): current=z.x
    return None,attempts

def model_fit(x,r,offset,start):
    fn=likelihood(x,r,offset)
    active=np.diff(offset)>0
    d=np.asarray(x[active].reshape(-1,x.shape[-1]))
    rank=np.linalg.matrix_rank(d) if len(d) else 0
    ans=dict(p=x.shape[-1],design_rank=int(rank),status='rank_or_identification_failure')
    if rank!=x.shape[-1]:return None,ans
    alternate=start.copy();alternate[:4]+=np.log([.65,1.50,1.35,.75])
    attempt=[minimize_ladder(fn,p) for p in (start,alternate)]
    ans['numerical_attempts']='|'.join(';'.join(f'{m}:{s}' for m,s,_ in o) for _,o in attempt)
    candidates=[z for z,_ in attempt if z is not None]
    if len(candidates)!=2:
        ans['status']='insufficient_stationary_likelihood_solutions'
        return None,ans
    f=min(candidates,key=lambda z:z.fun)
    gb=abs(candidates[0].fun-candidates[1].fun)
    bs=max(abs(candidates[0].x-candidates[1].x))
    ans.update(nll=float(f.fun),max_abs_gradient=float(max(abs(f.jac))),start_nll_range=float(gb),
               start_coeff_range=float(bs),iterations=int(f.nit))
    if gb>.002 or bs>.12:
        ans['status']='non_unique_local_solution'
        return None,ans
    h=v5.hessian(fn,f.x)
    l=np.linalg.eigvalsh(h)
    if np.any(h.diagonal()<=0) or l[0]<=0:
        ans['status']='nonpositive_observed_information'
        return None,ans
    s=np.sqrt(h.diagonal())
    sc=np.linalg.eigvalsh(h/s[:,None]/s[None,:])
    ans.update(hessian_min=float(l[0]),hessian_condition=float(l[-1]/l[0]),scaled_information_min=float(sc[0]),
               scaled_information_condition=float(sc[-1]/max(sc[0],1.e-20)))
    if l[0]<.002 or sc[0]<=0 or sc[-1]/sc[0]>2.e7:
        ans['status']='weak_identification_after_scaling'
        return None,ans
    inv=np.linalg.solve(h,np.eye(len(f.x)))
    ans['status']='stationary_reproduced'
    return (f.x,(inv+inv.T)*.5,fn),ans

def independent_fit_check(eta,r,s,c,label):
    q=RT/f'check_{s}_{c}_{label}'
    first=list(dict.fromkeys(r.pid))
    typelist=sorted(r.state.unique().tolist())
    include=[int(r.loc[r.state.eq(st),'pid'].iloc[0]) for st in typelist]
    rng=np.random.default_rng(441061)
    pids=np.unique(np.r_[include,rng.choice(first,size=min(120,len(first)),replace=False)])
    rsub=r.loc[r.pid.isin(pids),['pid','dt','state']].copy()
    tab={p:j for j,p in enumerate(pids)}
    rsub['rid']=rsub.pid.map(tab)
    nll=-sum(np.log(generic(eta[i],list(z[['dt','state']].itertuples(index=False,name=None)))) for i,z in rsub.groupby('pid'))
    # The fast engine allows subject-specific rates as saturated features, so
    # compare each independently with the generic matrix result at the fit.
    e=0.; official=[]
    for i,z in rsub.groupby('pid'):
        rows=list(z[['dt','state']].itertuples(index=False,name=None))
        v=-np.log(generic(eta[i],rows))
        impl=likelihood(np.eye(4)[None,:,:],z,np.array([0,len(z)],np.int32))
        e+=impl(eta[i])[0]
        if i in include:
            try: chk,_=rcheck(eta[i],rows,q/f'type_{include.index(i)}')
            except (ValueError, subprocess.SubprocessError, OSError):
                official.append(np.inf)
            else: official.append(abs(chk-v))
    return dict(generic_nll_absdiff=float(abs(e-nll)),official_fixture_max_absdiff=float(max(official,default=np.inf)),
                independent_checked_sequences=len(pids),independent_msm='1.9')

def calibration_alive(a,x,eta):
    rows=[]
    # Index-origin prediction at actual scheduled regular A/B visit.
    g=(a.country+' / '+a.cluster).to_numpy()
    for (slot,h),z in x.loc[x.occasion.isin([1,2])&x.state.isin([1,2])].assign(
            h=lambda y:a.h.to_numpy()[y.pid]).groupby(['occasion','h']):
        elapsed=z.time_years.to_numpy(float)
        state=v5.probabilities(np.exp(eta[z.pid]),elapsed)
        expected=state[:,1]/state[:,:2].sum(axis=1)
        seen=z.state.eq(2).to_numpy()
        m,b=compatible_interval_ci(seen-expected,g[z.pid])
        disclose=len(seen)>=25 and min(seen.sum(),len(seen)-seen.sum())>=10
        with_nan=lambda y:round(float(y),4) if disclose else np.nan
        rows.append(dict(planned_followup=int(slot),history=('index' if h else 'reference'),
                         observed_measured_living=safe_n(len(z)),actual_B=safe_n(int(seen.sum())),
                         observed_B_conditional=with_nan(seen.mean()),expected_B_conditional=with_nan(expected.mean()),
                         average_calibration_residual=with_nan(m),descriptive_family_margin=with_nan(b),
                         excludes_death_and_unmeasured_in_both_parts=True,
                         conditional_measurement_check_not_population_validation=True))
    return pd.DataFrame(rows)

def prediction(x,cols,coef,masks,w,months=False):
    base=x@coef
    hx=cols.hx.to_numpy()
    hh=np.array([coef[4],coef[5],coef[6],coef[6]])
    q=[np.exp(base+(h-hx[:,None])*hh) for h in (0,1)]
    out={}
    for T,mask in masks.items():
        if not mask.any() or w[mask].sum()==0:continue
        for m,f in [('probability',v5.probabilities),('months',v5.occupancy)]:
            if m=='months' and not (months and T==4):continue
            p=np.vstack([np.average(f(z[mask],T),axis=0,weights=w[mask]) for z in q])
            out[T,m]=np.vstack([p,p[1]-p[0]])
    return out

def bootstrap(a,x,cols,b,inv,fn,masks,month,s,c):
    """Use all locked draw indices; never replace failures with a new seed."""
    f=(a.country+' / '+a.cluster).factorize(sort=True)[0]
    grp=pd.DataFrame(dict(f=f,c=a.country)).drop_duplicates().groupby('c').f.apply(np.array).tolist()
    seed=12620+10*SOURCES.index(s)+list(ARMS).index(c)
    rng=np.random.default_rng(seed); draws=[]; failures=Counter(); coefs=[]; reasons=[]
    job=RT/f'{s}_{c}_bootstrap';job.mkdir(parents=True,exist_ok=True)
    for i in range(NBOOT):
        j=np.concatenate([rng.choice(k,size=len(k),replace=True) for k in grp])
        w=np.bincount(j,minlength=int(f.max())+1)[f].astype(float)
        fit=None; status='household_arm_disappeared'
        if all(w[a.h.eq(h)].sum()>0 for h in (0,1)):
            fit,at=minimize_ladder(fn,b,w,inv)
            if fit is None: status=at[-1][1]
            else:
                pred=prediction(x,cols,fit.x,masks,w,month)
                if pred: draws.append(pred); status='estimated'
                else: status='target_reference_draw_disappeared'
        if status!='estimated': failures[status]+=1
        coefs.append(fit.x if fit is not None else np.full(len(b),np.nan))
        reasons.append(status)
        if (i+1)%25==0 or i+1==NBOOT:
            np.savez_compressed(job/'v6_bootstrap_draws.npz',beta=np.asarray(coefs),reason=np.asarray(reasons),
                                seed=seed,panel_lock=filehash(PRELOCK))
            print(f'{s} {c} bootstrap {i+1}/{NBOOT}, successful={len(draws)}',flush=True)
    return draws,failures

def infer(s,c):
    began=time.monotonic()
    a,o,d=input_data(s,c)
    t,r,offset,actual=death_and_sequences(a,o,d)
    info,ready=fit_info(a,t,r); output('INFORMATION',s,c,info)
    masks,support=support_masks(a,o,actual,s); output('TIME_SUPPORT',s,c,support)
    initial_cal,_=calibration_survival(a,t,masks,occupancy=True)
    output('MORTALITY_CALIBRATION',s,c,initial_cal)
    x0,cols=covars(a,s,c,'shared')
    usable=dated_living_reference(a,o)
    z=np.log([.1,.23,.018,.035])
    designs=[]; diag=[]
    if ready:
        u=likelihood(np.tile(np.eye(4)[None,:,:],(len(a),1,1)),r,offset)
        base,_=minimize_ladder(u,z)
        if base is not None:z=base.x
        for label in ['shared','compact']:
            x,cols=covars(a,s,c,label)
            fit,log=model_fit(x,r,offset,np.r_[z,np.zeros(x.shape[-1]-4)])
            diag.append(dict(spec=label,**log))
            if fit is not None:
                designs.append((label,x,cols,fit))
                break
        if designs and designs[0][0]=='shared':
            x,cols=covars(a,s,c,'direction_specific')
            start=np.r_[designs[0][3][0],np.zeros(3)]
            fit,log=model_fit(x,r,offset,start)
            diag.append(dict(spec='direction_specific',**log))
            if fit is not None:designs.append(('direction_specific',x,cols,fit))
        elif designs:
            diag.append(dict(spec='direction_specific',status='not_supported_after_common_nuisance_clock_recovery'))
    disposition=dict(source=s,comparison=c,absolute_2yr='unavailable',absolute_4yr='unavailable',occupancy_4yr='unavailable',
                     legacy_v5_death_dependent_release='superseded_do_not_cite',reason='')
    if not ready or not designs:
        reason='insufficient_structural_reversal_or_timed_death' if not ready else 'identification_or_numerical_gate'
        v5.fallback(a,usable,cols,s,c,reason)
        disposition['reason']=reason
    else:
        for i,(label,x,cols,(beta,h_inv,fn)) in enumerate(designs):
            eta=x@beta
            fixed=independent_fit_check(eta,r,s,c,label)
            diag.append(dict(spec=label+'_independent_check',**fixed))
            if max(fixed['generic_nll_absdiff'],fixed['official_fixture_max_absdiff'])>1e-6:
                if i==0:
                    v5.fallback(a,usable,cols,s,c,'likelihood_independent_check_failed')
                    disposition['reason']='likelihood_independent_check_failed'
                continue
            rates=np.exp(eta)
            fixed_times={T:v5.probabilities(rates,T) for T in (1,2,3,4)}
            cal,support_ok=calibration_survival(a,t,masks,fixed_times,occupancy=True)
            if i==0:
                output('MORTALITY_CALIBRATION',s,c,cal)
                output('LIVING_CALIBRATION',s,c,calibration_alive(a,actual,eta))
            else:
                output('DIRECTION_SPECIFIC_MORTALITY_DIAGNOSTIC',s,c,cal)
                output('DIRECTION_SPECIFIC_LIVING_DIAGNOSTIC',s,c,calibration_alive(a,actual,eta))
            assert cols.columns[0]=='hx'
            allow={T:m.copy() if support_ok[T,T] else np.zeros_like(m) for T,m in masks.items()}
            monthly=bool(allow[2].any() and allow[4].any() and all(support_ok[T,4] for T in (1,2,3,4)))
            values=prediction(x,cols,beta,allow,np.ones(len(a)),monthly)
            pr=RT/f'{s}_{c}_{label}';pr.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(pr/'point.npz',beta=beta,hessian_inverse=h_inv)
            if not values and i==0:
                disposition['reason']='no_supported_fixed_absolute_mortality_target'
                v5.fallback(a,usable,cols,s,c,disposition['reason'])
                continue
            # The one finite direction-specification sensitivity is a diagnostic
            # unless the primary shared-clock fit already authorizes this target.
            if i!=0:continue
            boot,failed=bootstrap(a,x,cols,beta,h_inv,fn,allow,monthly,s,c)
            diag.append(dict(spec=label+'_family_bootstrap',attempted=NBOOT,successful=len(boot),
                             failed_by_reason=json.dumps(failed)))
            points=[]
            for (T,metric),p in values.items():
                sims=[z[T,metric] for z in boot if (T,metric) in z]
                if len(sims)<BOOT_MIN:continue
                key='occupancy_4yr' if metric=='months' else f'absolute_{T}yr'
                disposition[key]='released_as_corrected_working_model'
                lo,hi=np.quantile(sims,[.025,.975],axis=0)
                for arm,n in enumerate(['reference','index','index_minus_reference']):
                    for j,state in enumerate(['A','B','D']):
                        points.append(dict(year=T,measure=metric,role=n,state=state,estimate=p[arm,j],lo=lo[arm,j],hi=hi[arm,j],
                                           family_draws=len(sims),target=safe_n(int(allow[T].sum()))))
                assert np.allclose(p.sum(axis=1),[48.,48.,0] if metric=='months' else [1.,1.,0])
            if points:
                output('STATE_ESTIMATES',s,c,pd.DataFrame(points))
            else:
                disposition['reason']='insufficient_stable_cluster_resampling'
                v5.fallback(a,usable,cols,s,c,disposition['reason'])
    pd.DataFrame(diag or [{'spec':'main','status':disposition['reason']}]).pipe(lambda a:output('NUMERICAL_DIAGNOSTICS',s,c,a))
    disposition['elapsed_seconds']=round(time.monotonic()-began,2)
    output('DISPOSITION',s,c,pd.DataFrame([disposition]))
    print(s,c,disposition,flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',choices=SOURCES)
    p.add_argument('--comparison',choices=ARMS)
    p.add_argument('--audit',action='store_true')
    p.add_argument('--lock',action='store_true')
    z=p.parse_args()
    if z.lock:
        panel_lock()
        return
    if not z.source:p.error('--source is required for audit / fit')
    if not z.audit:gate(z.source)
    for c in [z.comparison] if z.comparison else ARMS:
        if z.audit:
            a,o,d=input_data(z.source,c)
            t,r,_,x=death_and_sequences(a,o,d)
            info,ready=fit_info(a,t,r);output('INFORMATION',z.source,c,info)
            masks,support=support_masks(a,o,x,z.source);output('TIME_SUPPORT',z.source,c,support)
            cal,met=calibration_survival(a,t,masks,occupancy=True)
            output('MORTALITY_SUPPORT_PREFIT',z.source,c,cal)
            print(z.source,c,'structure',ready,'fixed cutoffs with adequate mortality resolution',[k for k,v in met.items() if v],flush=True)
        else:
            infer(z.source,c)

if __name__=='__main__':
    main()

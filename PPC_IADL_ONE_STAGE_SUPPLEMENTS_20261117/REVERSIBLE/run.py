"""One-stage CTMC and distinct two-observed-living endpoint contrasts."""
from pathlib import Path
import os
import sys
import hashlib

sys.dont_write_bytecode = True
for item in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[item] = '1'

import argparse
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parents[1] / 'PPC_IADL_V6_Repair_Synthesis_20261116'
sys.path.insert(0, str(OLD / 'CODE'))
import panel_v6_data as data
import panel_v6_engine as engine
import panel_v6_models as models

TABLES = ROOT / 'TABLES'
PRIVATE = ROOT / 'PRIVATE'
engine.RT = PRIVATE / 'compiled'


def save(name, rows):
    TABLES.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLES / f'{name}.csv', index=False, encoding='utf-8-sig')


def identity():
    for source in data.SOURCES:
        models.gate(source)
    return hashlib.sha256((ROOT / 'SPEC.md').read_bytes()).hexdigest()


def source_data(comp, emit_support):
    items, info, horizons = [], [], []
    for source in data.SOURCES:
        a, observed, deaths = data.input_data(source, comp)
        times, rows, offsets, actual = data.death_and_sequences(a, observed, deaths)
        report, ready = data.fit_info(a, times, rows)
        info.extend(report.assign(source=source, contrast=comp, structural_support=ready).to_dict('records'))
        masks, support = data.support_masks(a, observed, actual, source)
        margins, qualified = data.calibration_survival(a, times, masks, occupancy=True)
        margins = margins.drop(columns=['fitted_expected_death', 'margin_low', 'margin_high', 'compatible_bounds'])
        horizons.extend(margins.assign(source=source, contrast=comp, release='not_released').to_dict('records'))
        group = (source + '|' + a.country + '|' + a.cluster).to_numpy()
        items.append(dict(source=source,a=a,rows=rows,offsets=offsets,group=group,ready=ready,qualified=qualified,
                          observed=models.dated_living_reference(a, observed)))
    if emit_support:
        save(f'CTMC_Structural_{comp}', info)
        save(f'Fixed_Time_Death_Support_{comp}', horizons)
    return items


def hc1(H, scores, families, n, p):
    inv = np.linalg.solve(H, np.eye(len(H)))
    code, unique = pd.factorize(families,sort=False)
    g = len(unique)
    U = np.zeros((g,p))
    np.add.at(U, code, scores)
    assert 1 < p < n and 1 < g <= n
    V = inv @ (U.T @ U) @ inv * (g / (g-1)) * ((n-1)/(n-p))
    assert np.isfinite(V).all() and np.all(V.diagonal()>0)
    return (V+V.T)/2, g


def local_scores(block, b, check_fn):
    X=np.ascontiguousarray(block['x']); offsets=block['offsets']; rows=block['rows']
    dt=np.ascontiguousarray(rows.dt.to_numpy(dtype=np.float64)); st=np.ascontiguousarray(rows.state.to_numpy(dtype=np.int32))
    model=engine.native().ppc_panel_v6_lik
    b=np.ascontiguousarray(b); scores=np.zeros((len(X),len(b))); active=np.flatnonzero(np.diff(offsets)>0)
    one=np.ones(1); total=np.zeros(1); nll=0.
    for i in active:
        lo,hi=offsets[i:i+2]
        offset=np.array([0,hi-lo],np.int32)
        if model(1,len(b),X[i],offset,dt[lo:hi],st[lo:hi],one,b,scores[i],total):
            raise ValueError('invalid participant likelihood during joint score check')
        nll+=total[0]
    nll0,grad=check_fn(b)
    err=max(abs(nll-nll0), np.max(np.abs(scores.sum(axis=0)-grad)))
    if err>1.e-6:
        raise ValueError('participant scores do not recover the fitted likelihood')
    eta=X@b
    generic_error=0.
    sample=np.linspace(0,len(active)-1,min(len(active),30),dtype=int)
    # Ensure every available state/observation type is checked at least once in each source.
    forced=[int(rows.loc[rows.state.eq(s),'pid'].iloc[0]) for s in rows.state.unique()]
    for i in np.unique(np.r_[active[sample],forced]):
        lo,hi=offsets[i:i+2]
        prob=engine.generic(eta[i],list(zip(dt[lo:hi],st[lo:hi])))
        offset=np.array([0,hi-lo],np.int32); gradient=np.zeros(len(b))
        status=model(1,len(b),X[i],offset,dt[lo:hi],st[lo:hi],one,b,gradient,total)
        if status or not np.isfinite(prob) or prob<=0:
            raise ValueError('invalid independent likelihood check')
        generic_error=max(generic_error,abs(total[0]+np.log(prob)))
    if generic_error>1.e-6:
        raise ValueError('independent generic matrix exp likelihood mismatch')
    return scores[active],block['group'][active],err,generic_error


def ctmc_fit(items, comp, spec):
    p=3; intercept=[]; seq=[]; blocks=[]
    for src in items:
        x,cols = models.covars(src['a'],src['source'],comp,spec)
        # V6 puts hx first and expands its masks as AB, BA, AD/BD.
        hx=src['a'].h.to_numpy()
        assert cols.columns[0]=='hx'
        assert np.array_equal(x[:,:,4:7],hx[:,None,None]*np.eye(3)[[0,1,2,2]][None,:,:])
        local=np.r_[np.arange(p,p+4),[0,1,2],np.arange(p+4,p+x.shape[-1]-3)]
        assert len(local)==x.shape[-1]
        intercept.extend(local[:4].tolist())
        p+=x.shape[-1]-3
        seq.append((local, engine.likelihood(x,src['rows'],src['offsets'])))
        blocks.append(dict(**src,x=x))

    def fn(b):
        val=0.; gradient=np.zeros(p)
        for idx, f in seq:
            v,g=f(b[idx]); val+=v; gradient[idx]+=g
        return val,gradient

    g=np.zeros((p,p)); start=np.zeros(p)
    for (idx,_),src in zip(seq,blocks):
        active=np.diff(src['offsets'])>0
        block=src['x'][active].reshape(-1,len(idx)); g[np.ix_(idx,idx)]+=block.T@block
        start[idx[:4]]=np.log([.1,.23,.018,.035])
    scales=np.sqrt(g.diagonal()); denom=np.where(scales>0,scales,1.)
    rank=int(np.linalg.matrix_rank(g/denom[:,None]/denom[None,:],tol=1e-10))
    detail=dict(contrast=comp,spec=spec,p=p,rank=rank,status='rank_failure')
    if rank!=p:
        return [],detail
    inc=np.setdiff1d(np.arange(p),intercept)

    def classify(m):
        if m is None or not np.isfinite(m.fun) or not np.isfinite(m.x).all():
            return 'nonfinite'
        if np.max(np.abs(m.x[intercept]))>=13 or np.max(np.abs(m.x[inc]))>=12:
            return 'coefficient_boundary'
        return 'stationary' if np.max(np.abs(m.jac))<.015 else 'gradient_unconverged'

    def attempt(current):
        ledger=[]
        for method,opt in [('BFGS',dict(gtol=1e-5,maxiter=2000)),
                           ('L-BFGS-B',dict(gtol=1e-5,maxiter=2000,ftol=1e-13,maxls=50))]:
            result=minimize(fn,current,method=method,jac=True,options=opt)
            state=classify(result); ledger.append(dict(method=method,status=state,nll=float(result.fun),
                 gradient=float(np.max(np.abs(result.jac))),iterations=int(result.nit)))
            if state=='stationary':
                return result,ledger
            if np.isfinite(result.fun) and np.isfinite(result.x).all():
                current=result.x
        return None,ledger

    alternative=start.copy(); alternative[intercept]+=np.tile(np.log([.65,1.50,1.35,.75]),len(items))
    f1, ledger1=attempt(start); print(comp,spec,'start 1',ledger1,flush=True)
    f2, ledger2=attempt(alternative); print(comp,spec,'start 2',ledger2,flush=True)
    detail['numerical_attempts']=json.dumps([ledger1,ledger2]); detail['status']='insufficient_stationary_solutions'
    if f1 is None or f2 is None:
        return [],detail
    likelihood_gap=abs(f1.fun-f2.fun); beta_gap=np.max(np.abs(f1.x-f2.x))
    detail.update(nll_gap=likelihood_gap,coefficient_gap=beta_gap,status='local_solutions_disagree')
    if likelihood_gap>.002 or beta_gap>.12:
        return [],detail
    best=min([f1,f2],key=lambda a:a.fun)
    H=models.v5.hessian(fn,best.x); eig=np.linalg.eigvalsh(H)
    detail.update(nll=best.fun,max_abs_gradient=np.max(np.abs(best.jac)),hessian_min=eig[0],status='weak_information')
    if eig[0]<.002 or np.any(H.diagonal()<=0):
        return [],detail
    d=np.sqrt(H.diagonal()); es=np.linalg.eigvalsh(H/d[:,None]/d[None,:]); c=es[-1]/max(es[0],1.e-20)
    detail['scaled_information_condition']=c
    if es[0]<=0 or c>2.e7:
        return [],detail
    scores=[]; families=[]; check=generic=0.
    try:
        for (idx,f),block in zip(seq,blocks):
            sc,fam,e,q=local_scores(block,best.x[idx],f)
            d=np.zeros((len(sc),p)); d[:,idx]=sc; scores.append(d); families.append(fam)
            check=max(check,e); generic=max(generic,q)
        U=np.concatenate(scores); all_family=np.concatenate(families)
        V,G=hc1(H,U,all_family,len(U),p)
    except (ValueError, AssertionError, np.linalg.LinAlgError) as exc:
        detail.update(status='joint_family_inference_failed', inference_error=str(exc))
        return [],detail
    detail.update(status='estimated',max_native_score_difference=check,max_independent_expm_difference=generic,
                  people_informative=data.safe_n(len(U)),families=data.safe_n(G))
    crit=t.ppf(.975,G-1); rows=[]
    for j,edge in enumerate(['A_to_B','B_to_A','A_or_B_to_D']):
        coef=best.x[j]; se=np.sqrt(V[j,j]); lo,hi=coef+crit*se*np.array([-1,1])
        rows.append(dict(contrast=comp,spec=spec,edge=edge,log_intensity_ratio=coef,se=se,
                         intensity_ratio=np.exp(coef),lo=np.exp(lo),hi=np.exp(hi),family_df=G-1))
    return rows,detail


def endpoint_design(items, comp, origin):
    base=[]; cnt=[]; eligible=True
    for d in items:
        x=d['observed']; a=d['a']; src=d['source']
        rows=x.loc[x.origin_exact.eq(origin)&x.state.isin([1,2])&(x.occasion-x.occ_prev).eq(1)].copy()
        _,cols,_=data.v5.design(a,src,comp)
        X=cols.iloc[rows.pid].reset_index(drop=True)
        X['gap_years']=rows.dt.to_numpy()-rows.dt.mean()
        h=X.pop('hx').to_numpy() if 'hx' in X else np.empty(0)
        y=rows.state.eq(3-origin).to_numpy(float)
        groups=d['group'][rows.pid.to_numpy()]
        enough=(len(rows)>0 and len(np.unique(h))==2 and len(np.unique(y))==2)
        eligible &= enough
        for group in (0,1):
            match=h==group; total=int(match.sum()); events=int(y[match].sum())
            cnt.append(dict(contrast=comp,source=src,origin='A' if origin==1 else 'B',history=group,
                            windows=data.safe_n(total),people=data.safe_n(rows.pid.iloc[match].nunique()),
                            families=data.safe_n(len(np.unique(groups[match]))),events=data.safe_n(events)
                           if min(events,total-events)>=10 else 'suppressed_small_or_complement',
                            actual_gap_median_years=float(rows.dt.iloc[match].median()) if total>=10 else np.nan))
        if not enough:
            continue
        dropped=X.columns[X.nunique()<=1].tolist(); X=X.drop(columns=dropped)
        base.append((X.to_numpy(),h,y,groups,src,dropped))
    if not eligible:
        return None, cnt, None
    n=sum(len(y) for _,_,y,_,_,_ in base); p=1+sum(x.shape[1]+1 for x,*_ in base)
    X=np.zeros((n,p)); start=0; j=1; y=[]; group=[]; inactive={}
    for x,h,d,g,src,drop in base:
        stop=start+len(x); X[start:stop,0]=h; X[start:stop,j]=1.
        X[start:stop,j+1:j+1+x.shape[1]]=x
        y.extend(d); group.extend(g); start=stop; j+=x.shape[1]+1; inactive[src]=drop
    return (X,np.array(y),np.array(group)), cnt, inactive


def endpoint(items,comp,origin):
    z,count,inactive=endpoint_design(items,comp,origin)
    edge='A_to_B' if origin==1 else 'B_to_A'
    detail=dict(contrast=comp,edge=edge,status='insufficient_source_level_support')
    if z is None:
        return None,count,detail
    X,y,group=z; N,p=X.shape; scale=np.sqrt(np.mean(X*X,axis=0)); W=X/scale
    rank=np.linalg.matrix_rank(W.T@W,tol=1e-8)
    detail.update(p=p,rank=rank,invariant_within_source=json.dumps(inactive),status='rank_failure')
    if rank!=p:
        return None,count,detail
    history=X[:,0].astype(bool)
    for lam in (0.,.5):
        pen=np.r_[0.,np.repeat(lam,p-1)]/(scale*scale)

        def objective(b):
            eta=W@b; mu=expit(eta)
            return np.sum(np.logaddexp(0,eta)-y*eta)+.5*np.dot(pen,b*b), W.T@(mu-y)+pen*b

        fit=minimize(objective,np.zeros(p),method='BFGS',jac=True,options=dict(gtol=1.e-6,maxiter=800))
        B=fit.x; mu=expit(W@B); v=mu*(1-mu)
        H=(W.T*v)@W+np.diag(pen); eig=np.linalg.eigvalsh(H)
        detail.update(penalty=lam,iterations=fit.nit,max_abs_gradient=np.max(np.abs(fit.jac)),hessian_min=eig[0],
                      max_original_abs_coefficient=np.max(np.abs(B/scale)),status='failed_logistic_gate')
        if not np.isfinite(fit.fun) or np.max(np.abs(fit.jac))>1.e-2 or eig[0]<=1.e-6 or np.max(np.abs(B/scale))>=15:
            continue
        # A finite penalized solution does not substitute for unpenalized information on the protected history effect.
        info=H[0,0]-H[0,1:]@np.linalg.solve(H[1:,1:],H[1:,0]); detail['profile_history_information']=float(info)
        if info<=1.e-4:
            continue
        try:
            V,G=hc1(H,W*(y-mu)[:,None],group,N,p)
        except (AssertionError,np.linalg.LinAlgError):
            continue
        B=B/scale; V=V/scale[:,None]/scale[None,:]
        X0=X.copy(); X0[:,0]=0.; X1=X.copy(); X1[:,0]=1.
        p0=expit(X0@B); p1=expit(X1@B); crit=t.ppf(.975,G-1)
        grad=p1*(1-p1)@X1/N-p0*(1-p0)@X0/N
        rd=float((p1-p0).mean()); se_rd=float(np.sqrt(grad@V@grad)); se_or=np.sqrt(V[0,0])
        detail.update(status='estimated', windows=data.safe_n(N),families=data.safe_n(G))
        return dict(contrast=comp,edge=edge,method=('unpenalized_MLE' if lam==0 else 'nuisance_L2_nominal'),
                    odds_ratio=np.exp(B[0]),or_lo=np.exp(B[0]-crit*se_or),or_hi=np.exp(B[0]+crit*se_or),
                    observed_index_probability=y[history].mean(),observed_reference_probability=y[~history].mean(),
                    standardized_index_probability=p1.mean(),standardized_reference_probability=p0.mean(),
                    risk_difference=rd,rd_lo=rd-crit*se_rd,rd_hi=rd+crit*se_rd,family_df=G-1), count,detail
    return None,count,detail


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('ctmc','endpoint')); p.add_argument('contrast',choices=data.ARMS)
    args=p.parse_args(); sha=identity(); comp=args.contrast
    print('Frozen pre-fit spec SHA256',sha,flush=True)
    items=source_data(comp,args.mode=='ctmc')
    if args.mode=='ctmc':
        result=[]; detail=[]
        if all(d['ready'] for d in items):
            for spec in ('shared','compact'):
                result,diagnosis=ctmc_fit(items,comp,spec); detail.append(diagnosis)
                save(f'CTMC_Diagnostic_{comp}',detail)
                if result:
                    break
        else:
            detail.append(dict(contrast=comp,status='source_structural_support_failed'))
            save(f'CTMC_Diagnostic_{comp}',detail)
        if not result:
            result=[dict(contrast=comp,status='no_publishable_four_source_intensity_ratio')]
        else:
            result=[dict(**r,status='estimated') for r in result]
        save(f'CTMC_Estimates_{comp}',result)
        print(comp,'joint CTMC disposition',detail[-1],flush=True)
    else:
        estimates=[]; counts=[]; diagnostics=[]
        for origin in (1,2):
            value,count,diag=endpoint(items,comp,origin); counts+=count; diagnostics.append(diag)
            if value is not None:
                estimates.append(dict(**value,status='estimated'))
            else:
                estimates.append(dict(contrast=comp,edge=diag['edge'],status=diag['status']))
        save(f'Observed_Endpoint_{comp}',estimates); save(f'Observed_Endpoint_Support_{comp}',counts)
        save(f'Observed_Endpoint_Diagnostic_{comp}',diagnostics)
        print(comp,'observed endpoints',estimates,flush=True)


if __name__=='__main__':
    main()

"""Mean bias-reduced logistic regression and intercept-only FLIC.

Jeffreys penalty is 0.5 log det(X' W X). Frequency multiplicities are
equivalent to expanding repeated rows; leverage is aggregated per unique row.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit


class FitFail(ValueError):
    pass


def design_map(x, w=None):
    """Affine change of basis preserving the frozen support/reference columns."""
    n, k = x.shape
    center = np.zeros(k) if w is None else np.average(x, axis=0, weights=w)
    if w is None:
        center = x.mean(0)
    center[0] = 0
    z = x-center
    z[:,0] = 1
    scale = np.sqrt(np.mean(z*z, axis=0) if w is None else np.average(z*z, axis=0, weights=w))
    if (scale<1e-12).any():
        raise FitFail('design_constant_column')
    z = np.ascontiguousarray(z/scale)
    gram = z.T@z if w is None else z.T@(z*w[:,None])
    eig, vec = eigh(gram)
    if eig[0] <= eig[-1]*1e-11:
        raise FitFail('design_rank_deficient')
    mix = (vec * np.sqrt(n/eig)) @ vec.T
    return scale, center, mix


def optimize(x, y, w, start=None):
    eye = np.eye(x.shape[1])
    def evaluate(b):
        eta=x@b
        p=expit(eta)
        pq=expit(-eta)*p
        ww=w*pq
        info=x.T@(x*ww[:,None])
        try:
            cf=cho_factor(info,check_finite=False)
        except np.linalg.LinAlgError as e:
            raise FitFail('nonpositive_information') from e
        h=np.einsum('ij,ij->i',x@cho_solve(cf,eye,check_finite=False),x)*ww
        f=-np.sum(w*np.where(y==1,np.logaddexp(0,-eta),np.logaddexp(0,eta)))+np.log(np.diag(cf[0])).sum()
        g=x.T@(w*(y-p)+h*(.5-p))
        return f,g,cf
    b = np.zeros(x.shape[1]) if start is None else start.copy()
    try:
        f,g,cf=evaluate(b)
    except FitFail:
        b=np.zeros(x.shape[1]); f,g,cf=evaluate(b)
    for it in range(60):
        step=cho_solve(cf,g,check_finite=False)
        if np.max(np.abs(step))<=1e-9 and np.max(np.abs(g))/max(1,w.sum())<=1e-9:
            return b,'fisher_backtracking',it,float(f),float(np.max(np.abs(g)))
        shrink=min(1,4/max(4.,np.max(np.abs(step))))
        for _ in range(18):
            trial=b+shrink*step
            try:
                fo,go,co=evaluate(trial)
            except FitFail:
                shrink*=.5
                continue
            if fo>=f-1e-10:
                break
            shrink*=.5
        else:
            break
        b,f,g,cf=trial,fo,go,co
    def target(b0):
        try:
            f0,g0,_=evaluate(b0)
            return -f0,-g0
        except FitFail:
            return np.inf,np.zeros_like(b0)
    res=minimize(target,b,jac=True,method='BFGS',options=dict(maxiter=500,gtol=1e-9))
    f,g,cf=evaluate(res.x)
    if max(abs(cho_solve(cf,g,check_finite=False)))>2e-7 or max(abs(g))/max(1,w.sum())>1e-8:
        raise FitFail('adjusted_score_unresolved')
    return res.x,'bfgs_rescue',60+res.nit,float(f),float(max(abs(g)))


def fit(x,y,weights=None,warm=None):
    w=np.ones(len(y)) if weights is None else np.asarray(weights,dtype=float)
    active=w>0
    x=np.asarray(x)[active]; y=np.asarray(y)[active]; w=w[active]
    if not len(y) or not np.any(y==1) or not np.any(y==0):
        raise FitFail('resample_outcome_group_absent')
    if not np.all(x[:,0]==1):
        raise FitFail('design_intercept_missing')
    keep=np.ptp(x,axis=0)>0
    keep[0]=True
    xk=x[:,keep]
    scale,center,mix=design_map(xk,w)
    z=np.ascontiguousarray((xk-center)/scale@mix)
    a0=None
    if warm is not None:
        prior=np.asarray(warm)[keep].copy()
        prior[0]+=center@prior
        a0=np.linalg.solve(mix,prior*scale)
    az,method,it,objective,score=optimize(z,y,w,a0)
    beta=(mix@az)/scale
    beta[0]-=center@beta
    if not np.isfinite(beta).all():
        raise FitFail('nonfinite_coefficients')
    out=np.zeros_like(keep,dtype=float)
    out[keep]=beta
    return out,dict(algorithm=method, iterations=it, objective_scaled=objective,
                    adjusted_score_max_scaled=score, dropped_redraw_constant_columns=int((~keep).sum()))


def flic_shift(eta,y,weight):
    target=np.average(y,weights=weight)
    if not 0<target<1:
        raise FitFail('calibration_outcome_group_absent')
    # Monotone population equation; the outer range spans every eta.
    root=brentq(lambda a:np.average(expit(eta+a),weights=weight)-target,
                float(logit(target)-eta.max()-1),float(logit(target)-eta.min()+1),xtol=1e-13)
    if abs(np.average(expit(eta+root),weights=weight)-target)>1e-11:
        raise FitFail('calibration_error')
    return root


def functionals(x,y,b,j,w):
    take=w>0
    x,y,w=x[take],y[take],w[take]
    if len(np.unique(x[:,j]))!=2:
        raise FitFail('resample_contrast_group_absent')
    eta=x@b
    delta=flic_shift(eta,y,w)
    ref=eta-x[:,j]*b[j]+delta
    p1,p0=(np.average(expit(ref+s*b[j]),weights=w) for s in (1,0))
    return np.array([b[j],p1,p0,p1-p0,delta])


def family_plan(cluster,strata,seed,n=499):
    code,index,inv=np.unique(cluster.astype(str),return_index=True,return_inverse=True)
    groups=strata[index]
    if not np.array_equal(groups[inv],strata):
        raise FitFail('family_crosses_sampling_strata')
    rng=np.random.default_rng(seed)
    plan=np.zeros((n,len(code)),dtype=np.int16)
    for s in np.unique(groups):
        loc=np.flatnonzero(groups==s)
        for i in range(n):
            plan[i,loc]=np.bincount(rng.integers(len(loc),size=len(loc)),minlength=len(loc))
    return plan,inv

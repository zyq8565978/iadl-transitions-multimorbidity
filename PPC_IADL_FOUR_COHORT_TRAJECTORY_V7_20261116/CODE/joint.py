"""True one-stage four-cohort mixture; only outcome records and family IDs are read."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
import pandas as pd
from scipy.special import betaln, digamma, expit, gammaln, logsumexp, logit
from scipy.optimize import minimize, minimize_scalar, linear_sum_assignment
from scipy.stats import betabinom, norm, rankdata

R=Path(__file__).resolve().parents[1]
P,Q,T=(R/n for n in ('PRIVATE','QA','TABLES'))
SOURCE=('CHARLS','HRS','ELSA','SHARE')
DOM=('chronic','symptoms','cognition')
SEED=20261116


def jsonout(x,p):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def evidence_paths():
    return [R/'V7_SPEC.md',R/'CODE/joint.py',P/'long.parquet',P/'index.parquet',T/'PREFIT_SUPPORT.csv',
            R.parent/'PPC_IADL_V6_Repair_Synthesis_20261116/SAFE_DELIVERY_V6.zip']


def prelock():
    files={os.path.relpath(p,R).replace('\\','/'):sha(p) for p in evidence_paths()}
    f=Q/'TRAJECTORY_SPEC_LOCK.json'
    if f.exists():
        assert files==json.loads(f.read_text(encoding='utf-8'))['files'], 'Frozen input/code changed; require versioned amendment'
    else:
        jsonout({'role':'locked before new trajectory fit; IADL data not loaded into trajectory fitter','files':files},f)


class Panel:
    def __init__(self):
        f=pd.read_parquet(P/'long.parquet',columns=['person','source','time','chronic','symptoms','symptoms_max','cognition'])
        self.f=f
        self.person,self.ip=np.unique(f.person.to_numpy(dtype=str),return_inverse=True)
        self.N=len(self.person)
        self.nv=np.bincount(self.ip)
        self.V=len(f)
        assert self.nv.min()>=3 and not f.duplicated(['person','time']).any()
        yr=f.time.to_numpy(dtype=float)
        self.mo=np.rint(12*yr).astype(int)
        assert np.allclose(yr,self.mo/12) and (self.mo.min(),self.mo.max())==(0,72)
        self.s=f.source.map({k:i for i,k in enumerate(SOURCE)}).to_numpy(dtype=int)
        self.si=np.empty(self.N,dtype=int); self.si[self.ip]=self.s
        assert (self.si[self.ip]==self.s).all()
        self.X=np.column_stack([np.eye(4)[self.s], np.eye(4)[self.s]*(self.mo[:,None]/48)])
        self.y=[f[c].to_numpy(dtype=float) for c in DOM]
        assert all(np.isfinite(c).all() for c in self.y)
        self.max=[np.full(self.V,7), f.symptoms_max.to_numpy(dtype=int)]
        assert (self.max[1]==np.array([30,8,8,12])[self.s]).all()
        assert all(((y>=0)&(y<=n)&(y==np.round(y))).all() for y,n in zip(self.y,self.max))
        keys=np.column_stack([self.s,self.mo])
        nodes, self.rnode = np.unique(keys,axis=0,return_inverse=True)
        self.snode=nodes[:,0]
        self.tnode=nodes[:,1]/48
        self.nnode=len(nodes)
        self.Xnode=np.column_stack([np.eye(4)[self.snode],np.eye(4)[self.snode]*self.tnode[:,None]])
        self.discrete=[]
        for y,nn in zip(self.y,self.max):
            a,ri=np.unique(np.column_stack([self.rnode,y,nn]),axis=0,return_inverse=True)
            j,yy,n=a.T
            j=j.astype(int)
            self.discrete.append((j,yy,n,ri,gammaln(n+1)-gammaln(yy+1)-gammaln(n-yy+1)))
        gi=pd.read_parquet(P/'index.parquet',columns=['person','cluster']).set_index('person',verify_integrity=True)
        self.family=gi.loc[self.person,'cluster'].to_numpy(dtype=str)
        assert (np.bincount(self.ip[self.mo==0],minlength=self.N)==1).all()
        self.nuisance()

    def nuisance(self):
        """Estimate K=1 working dispersion separately by source, without exposure."""
        self.c=np.empty((2,4)); self.b0=np.zeros((3,8)); self.sig=np.empty(4)
        for s in range(4):
            use=self.s==s
            t=self.mo[use]/48
            X=np.c_[np.ones(len(t)),t]
            for d in range(2):
                y=self.y[d][use]; n=self.max[d][use]
                def fun(th):
                    p=expit(X@th[:2]); c=np.exp(th[2]); a,b=p*c,(1-p)*c
                    l=betaln(y+a,n-y+b)-betaln(a,b)
                    A=digamma(y+a)-digamma(n+c)-digamma(a)+digamma(c)
                    B=digamma(n-y+b)-digamma(n+c)-digamma(b)+digamma(c)
                    gradient=np.r_[X.T@((A-B)*c*p*(1-p)), ((A*p+B*(1-p))*c).sum()]
                    return -l.mean(),-gradient/len(t)
                b=minimize(fun,[logit(np.clip(y.sum()/n.sum(),.01,.99)),0.,np.log(20.)],jac=True,
                          bounds=[(-10,10),(-6,6),(np.log(.5),np.log(1000.))],method='L-BFGS-B',
                          options={'ftol':1e-13,'gtol':1e-8})
                assert b.success and np.isfinite(b.x).all()
                self.c[d,s]=np.exp(b.x[2]); self.b0[d,[s,4+s]]=b.x[:2]
            y=self.y[2][use]
            b=np.linalg.lstsq(X,y,rcond=None)[0]
            self.b0[2,[s,4+s]]=b
            self.sig[s]=np.sqrt(np.mean((y-X@b)**2))
        assert (self.sig>.1).all()
        features=[]
        for d,y in enumerate(self.y):
            mu=self.X@self.b0[d]; pred=expit(mu)*self.max[d] if d<2 else mu
            rr=y-pred
            if d==1:
                rr=rr/self.max[d]
            m=np.bincount(self.ip,weights=rr)/self.nv
            features.append(m/m.std())
        self.feat=np.column_stack(features)


def unpack(x,k):
    p=8+2*(k-1)
    return x[:3*p].reshape(3,p), x[3*p:].reshape(4,k-1)


def infer(data,x,k,w=None,grad=True):
    """Marginal likelihood over one latent class shared by all visits/domains."""
    B,pp=unpack(x,k)
    w=np.ones(data.N) if w is None else w
    if grad:
        gd=[]
    prior=np.column_stack([np.zeros(4),pp])
    log_prior=prior-logsumexp(prior,axis=1,keepdims=True)
    L=log_prior[data.si].T.copy()
    mats=[]
    for d in range(3):
        base=data.Xnode@B[d,:8]
        ext=np.row_stack([np.zeros(2),B[d,8:].reshape(-1,2)])
        eta=base[None,:]+ext[:,0,None]+ext[:,1,None]*data.tnode[None,:]
        if d<2:
            j,y,n,ri,choose=data.discrete[d]
            mu=expit(eta[:,j]); c=data.c[d,data.snode[j]]
            aa=mu*c; bb=(1-mu)*c
            ln=choose+betaln(y+aa,n-y+bb)-betaln(aa,bb)
            if grad:
                dL=(digamma(y+aa)-digamma(aa)-digamma(n-y+bb)+digamma(bb))*c*mu*(1-mu)
                mats.append(dL)
            val=ln[:,ri]
        else:
            rr=data.y[d][None,:]-eta[:,data.rnode]
            sigma=data.sig[data.s]
            z=rr/sigma
            val=-.5*(z*z+np.log(2*np.pi*sigma**2))
            if grad:
                mats.append(rr/sigma**2)
        for g in range(k):
            L[g]+=np.bincount(data.ip,weights=val[g],minlength=data.N)
    lm=logsumexp(L,axis=0)
    ll=float(w@lm)
    post=np.exp(L-lm)
    if not grad:
        return ll,post
    q=post*w
    for d,r in enumerate(mats):
        if d<2:
            j,_,_,ri,_=data.discrete[d]
            g=np.array([np.bincount(j,weights=r[i]*np.bincount(ri,weights=q[i,data.ip],minlength=len(j)),
                                     minlength=data.nnode) for i in range(k)])
        else:
            g=np.array([np.bincount(data.rnode,weights=r[i]*q[i,data.ip],minlength=data.nnode) for i in range(k)])
        gd.append(np.r_[g.sum(axis=0)@data.Xnode,np.column_stack([g[1:].sum(axis=1),g[1:]@data.tnode]).ravel()])
    pr=np.exp(log_prior)
    obs=np.array([np.bincount(data.si,weights=v,minlength=4) for v in q]).T
    gp=obs[:,1:]-pr[:,1:]*np.bincount(data.si,weights=w,minlength=4)[:,None]
    return ll,np.r_[np.ravel(gd),gp.ravel()]


def start_init(d,k,num):
    B=np.column_stack([d.b0,np.zeros((3,2*(k-1)))])
    if k==1:
        return B.ravel()
    rng=np.random.default_rng(SEED+200*k+num)
    f=d.feat*np.exp(rng.normal(0,.35,size=(1,3)))
    c=f[rng.choice(len(f),size=k,replace=False)]
    for _ in range(20):
        z=((f[:,None]-c[None,:])**2).sum(axis=2).argmin(axis=1)
        c=np.vstack([f[z==i].mean(axis=0) if (z==i).any() else f[rng.integers(len(f))] for i in range(k)])
    fit=d.feat/d.feat.std(axis=0)
    b=np.stack([fit[z==i].mean(0) if (z==i).any() else np.zeros(3) for i in range(k)])
    b=b*np.array([.6,.75,.60])[None,:]
    B[:,:4]+=b[0,:,None]
    for g in range(1,k):
        B[:,8+2*(g-1)]=(b[g]-b[0])
        B[:,9+2*(g-1)]=rng.normal(0,.12,3)
    freq=(np.stack([np.bincount(z[d.si==s],minlength=k) for s in range(4)])+5)
    pp=np.log(freq[:,1:]/freq[:,[0]])
    return np.r_[B.ravel(),pp.ravel()]


def fit(d,k,num,w=None,warm=None):
    tick=time.perf_counter()
    x0=start_init(d,k,num) if warm is None else warm
    n=d.N if w is None else w.sum()
    def fun(x):
        l,g=infer(d,x,k,w)
        return -l/n,-g/n
    bounds=[(-8.,8.)]*(len(x0)-4*(k-1))+[(-9.,9.)]*(4*(k-1))
    opt=minimize(fun,np.clip(x0,-7.999,7.999),method='L-BFGS-B',jac=True,bounds=bounds,
                 options={'ftol':2e-13,'gtol':1e-7,'maxiter':370,'maxcor':13})
    gx=opt.jac
    bound=np.isclose(np.abs(opt.x),np.asarray(bounds)[:,1],atol=1e-5,rtol=0)
    converged=bool(np.isfinite(opt.fun) and not bound.any() and np.max(np.abs(gx))<4e-6)
    return dict(K=k,start=num,loglik=-n*opt.fun,converged=converged,optimizer_success=bool(opt.success),
                max_score_per_person=float(np.max(np.abs(gx))),active_bounds=int(bound.sum()),
                iterations=int(opt.nit),seconds=time.perf_counter()-tick,message=str(opt.message),x=opt.x)


def shape(x,k,tt):
    B,pp=unpack(x,k)
    Z=np.zeros((3,k,4,len(tt)))
    for d in range(3):
        base=B[d,:4,None]+B[d,4:8,None]*tt/4
        u=np.row_stack([np.zeros(2),B[d,8:].reshape(-1,2)])
        eta=base[None,:,:]+u[:,0,None,None]+u[:,1,None,None]*tt[None,None,:]/4
        Z[d]=expit(eta)*(7 if d==0 else 100) if d<2 else eta
    return Z


def profile_order(x,k):
    z=shape(x,k,np.array([0.]))[:,:,:,0].mean(axis=2)
    rank=rankdata(z[0])+rankdata(z[1])+rankdata(-z[2])
    return np.lexsort((z[0],rank))


def ari(ref,z):
    k=max(ref.max(),z.max())+1
    tab=np.bincount(ref*k+z,minlength=k*k).reshape(k,k)
    c=lambda n:(n*(n-1)/2).sum()
    a,b=c(tab.sum(axis=0)),c(tab.sum(axis=1)); expected=a*b/c(np.array([len(ref)],dtype=float))
    denom=(a+b)*.5-expected
    return (c(tab)-expected)/denom if denom>0 else 0.


def compare_curves(d,base,alt,k):
    dims=np.array([1.,10.,1.])[:,None,None]
    u=shape(base,k,np.linspace(0,5,21)).mean(axis=2)/dims
    v=shape(alt,k,np.linspace(0,5,21)).mean(axis=2)/dims
    sq=np.mean((u[:,:,None,:]-v[:,None,:,:])**2,axis=3)
    _,col=linear_sum_assignment(sq.mean(axis=0))
    p=infer(d,base,k,grad=False)[1]; q=infer(d,alt,k,grad=False)[1][col]
    return dict(adjusted_rand=float(ari(p.argmax(axis=0),q.argmax(axis=0))),
                maximum_profile_domain_RMS=float(np.sqrt(sq[:,np.arange(k),col]).max()),class_alignment=','.join(map(str,col+1)))


def qa_math(d):
    """Exact independent observed-density and analytic gradient checks using real outcomes only."""
    k=2; x=start_init(d,k,1)
    ll,post=infer(d,x,k,grad=False)
    B,pp=unpack(x,k)
    check=0.; err=0.
    for idx in np.linspace(0,d.N-1,37).astype(int):
        ix=d.ip==idx; s=d.si[idx]; t=d.mo[ix]/48
        lr=np.r_[0,pp[s]]; lr-=logsumexp(lr)
        for g in range(k):
            for z in range(3):
                a,b=B[z,s],B[z,4+s]
                if g:
                    a+=B[z,8]; b+=B[z,9]
                eta=a+b*t
                if z<2:
                    p=expit(eta); c=d.c[z,s]
                    lr[g]+=betabinom.logpmf(d.y[z][ix],d.max[z][ix],p*c,(1-p)*c).sum()
                else:
                    lr[g]+=norm.logpdf(d.y[z][ix],loc=eta,scale=d.sig[s]).sum()
        err=max(err,np.max(np.abs(np.exp(lr-logsumexp(lr))-post[:,idx])))
    _,grad=infer(d,x,k)
    r=np.random.default_rng(SEED).normal(size=(4,len(x)));r/=np.linalg.norm(r,axis=1,keepdims=True)
    numeric=[]
    for z in r:
        numeric.append((infer(d,x+1e-5*z,k,grad=False)[0]-infer(d,x-1e-5*z,k,grad=False)[0])/2e-5/d.N)
    dgrad=np.max(np.abs(np.asarray(numeric)-(r@grad)/d.N))
    assert err<1e-10 and dgrad<1e-7,(err,dgrad)
    jsonout({'real_outcome_independent_scipy_probability_cases':37,'max_posterior_discrepancy':float(err),
             'max_normalized_gradient_discrepancy':float(dgrad),'loglik_at_numeric_check':ll},Q/'TRAJECTORY_NUMERIC_QA.json')


def csvout(rows,p):
    pd.DataFrame(rows).to_csv(p,index=False,encoding='utf-8-sig')


def adequacy(d, rr):
    k,x=rr['K'],rr['x']
    l,post=infer(d,x,k,grad=False)
    p=post[profile_order(x,k)]; m=p.argmax(axis=0)
    n=np.bincount(m,minlength=k)
    ap=np.array([p[i,m==i].mean() if n[i] else 0 for i in range(k)])
    source_counts=np.column_stack([np.bincount(m[d.si==s],minlength=k) for s in range(4)])
    expct=np.column_stack([p[:,d.si==s].sum(axis=1) for s in range(4)])
    share=source_counts.max(axis=1)/np.maximum(1,n)
    return dict(K=k,loglik=l,BIC=-2*l+len(x)*np.log(d.N),AIC=-2*l+2*len(x),n_common_parameters=len(x),
                min_MAP_share=float(n.min()/d.N),min_MAP_n=int(n.min()),min_AvePP=float(ap.min()),
                max_source_fraction=float(share.max()),min_source_MAP_n=int(source_counts.min()),
                min_expected_source_class=float(expct.min()),
                support_pass=bool(k==1 or (n.min()>=max(50,.05*d.N) and ap.min()>=.70 and share.max()<=.75
                                  and source_counts.min()>=10 and expct.min()>=25)))


def halfweights(d):
    rng=np.random.default_rng(SEED+700)
    wt=np.zeros(d.N)
    for i in range(4):
        keep=d.si==i; fc,inv=np.unique(d.family[keep],return_inverse=True)
        permutation=rng.permutation(len(fc)); side=np.arange(len(fc))
        side[permutation]=side.copy()
        wt[keep]=side[inv]%2
    n=np.bincount(d.si)
    return {'family_half_A':wt,'family_half_B':1-wt,'equal_cohort':(d.N/(4*n))[d.si]}


WORK=None


def init_worker():
    global WORK
    WORK=Panel()


def job(task):
    fit_k,s,role,x=task
    w=None if role=='all' else halfweights(WORK)[role]
    ans=fit(WORK,fit_k,s,w=w,warm=x)
    ans['role']=role
    return ans


def trials(pool,tasks):
    out=[]
    for m in pool.map(job,tasks):
        out.append(m)
        print(f"K={m['K']} {m['role']} start={m['start']} ll={m['loglik']:.5f} score={m['max_score_per_person']:.2g} ok={m['converged']} {m['seconds']:.2f}s",flush=True)
    return out


def publish(d, chosen, admitted):
    k,x=chosen['K'],chosen['x']
    order=profile_order(x,k)
    ll,praw=infer(d,x,k,grad=False)
    p=praw[order]; m=p.argmax(axis=0)
    assert np.max(np.abs(p.sum(axis=0)-1))<1e-10
    clock=np.linspace(0,6,73)
    raw=shape(x,k,clock)[:,order]
    curves=[]; profiles=[]; diagnostics=[]
    denom=np.stack(d.max)[1]
    people_n=np.bincount(d.si)
    B,pp=unpack(x,k)
    psource=np.column_stack([np.zeros(4),pp]); psource=np.exp(psource-logsumexp(psource,axis=1,keepdims=True))[:,order]
    for g in range(k):
        for s,src in enumerate((*SOURCE,'pooled')):
            c=(m==g) if s==4 else (m==g)&(d.si==s)
            n= int(c.sum())
            assert n>=10 or not admitted
            profiles.append(dict(scope=src,profile=g+1,people=n if n>=10 else '<10',
                           share=n/(d.N if s==4 else people_n[s]),mean_MAP_posterior=p[g,c].mean() if n>=10 else np.nan,
                           estimated_membership_share=psource[:,g].mean() if s==4 else psource[s,g]))
            for j,domain in enumerate(DOM):
                pred=raw[j,g].mean(axis=0) if s==4 else raw[j,g,s]
                use=clock<=5.000001 if s==4 else clock<=d.mo[d.s==s].max()/12+1e-8
                for t,v in zip(clock[use],pred[use]):
                    curves.append(dict(domain=domain,profile=g+1,time=t,estimate=v,lo=np.nan,hi=np.nan,scope=src))
    # Residual/check controls are apparent (same training data), not validation.
    for s,src in enumerate(SOURCE):
        idx=(d.s==s)
        for j,domain in enumerate(DOM):
            y=d.y[j][idx].copy(); max_=d.max[j][idx] if j<2 else np.ones(len(y))
            traj=raw[j,:,s][:,d.mo[idx]]
            y_std=y*(100/max_ if j==1 else 1)
            pred=(traj*psource[s,:,None]).sum(axis=0)
            post_pred=(traj*p[:,d.ip[idx]]).sum(axis=0)
            z=pd.DataFrame(dict(ip=d.ip[idx],month=d.mo[idx],resid=y_std-post_pred)).sort_values(['ip','month'])
            period=z.ip.eq(z.ip.shift())
            lag=z.resid.shift().loc[period]
            pair=z.resid.loc[period]
            py0=np.nan
            if j<2:
                fraction=traj/(7 if j==0 else 100)
                c=d.c[j,s]
                py0=float((psource[s,:,None]*betabinom.pmf(0,max_,c*fraction,c*(1-fraction))).sum(axis=0).mean())
            diagnostics.append(dict(source=src,domain=domain,mean_observed=y_std.mean(),mean_marginal_fitted=pred.mean(),
                                    observed_zero_rate=float((y==0).mean()) if j<2 else np.nan,fitted_zero_rate=py0,
                                    adjacent_posterior_mean_residual_correlation=np.corrcoef(pair,lag)[0,1]))
    csvout(curves,T/'TRAJECTORY_CURVES.csv');csvout(profiles,T/'PROFILE_COUNTS.csv');csvout(diagnostics,T/'TRAJECTORY_FIT_DIAGNOSTICS.csv')
    csvout([dict(source=src,chronic_beta_binomial_concentration=d.c[0,i],symptom_beta_binomial_concentration=d.c[1,i],
                 cognition_gaussian_sd=d.sig[i]) for i,src in enumerate(SOURCE)],T/'TRAJECTORY_SOURCE_SCALES.csv')
    if admitted:
        csvout(pd.DataFrame(p.T,columns=[f'p{i+1}' for i in range(k)]).assign(person=d.person),P/'posterior.csv')
    np.savez_compressed(P/'final_working_model.npz',theta=x,nuisance_c=d.c,nuisance_sig=d.sig,K=k,order=order)
    jsonout(dict(admitted=bool(admitted),K=k,loglik=ll,source_rule='source-specific background and prevalence, shared trajectory terms',
             draw='equal-cohort conditional means, no CI',ref_profile=1,status=('COMMON_PROFILES_ADMITTED' if admitted else 'K1_ONLY_NO_STABLE_MULTICLASS'),
             trajectories_sha256=sha(T/'TRAJECTORY_CURVES.csv'), posterior_sha256=sha(P/'posterior.csv') if admitted else None,
             support_sha256=sha(T/'PREFIT_SUPPORT.csv'),authority_sha256=sha(Q/'TRAJECTORY_SPEC_LOCK.json')),
             Q/'CLASS_LOCK.json')


def main():
    arg=argparse.ArgumentParser();arg.add_argument('--workers',type=int,default=3);args=arg.parse_args()
    prelock()
    d=Panel(); qa_math(d)
    grid=[(k,s,'all',None) for k in range(1,5) for s in (range(1) if k==1 else range(1,7))]
    all_runs=[]; stability=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker) as pool:
        all_runs+=trials(pool,grid)
        csvout([{kk:vv for kk,vv in z.items() if kk!='x'} for z in all_runs],T/'TRAJECTORY_CANDIDATE_STARTS.csv')
        winners={}; bestdiag={}
        for k in range(1,5):
            outcomes=[z for z in all_runs if z['K']==k and z['converged']]
            if not outcomes:
                bestdiag[k]=dict(K=k,support_pass=False,fit_disposition='NO_NUMERICALLY_FINITE_STABLE_SOLUTION')
                continue
            top=max(outcomes,key=lambda z:z['loglik'])
            record=adequacy(d,top)
            nclose=sum(top['loglik']-z['loglik']<=.20 for z in outcomes)
            record['converged_starts']=len(outcomes);record['replicating_starts']=nclose
            record['fit_disposition']='ELIGIBLE_OTHER_GATES_PENDING' if k==1 or (nclose>=2 and record['support_pass']) else 'NUMERIC_REPLICATION_OR_SUPPORT_FAIL'
            bestdiag[k]=record
            if k==1 or (nclose>=2 and record['support_pass']):
                winners[k]=top
        assert 1 in winners
        csvout(bestdiag.values(),T/'TRAJECTORY_CANDIDATE_SUMMARY.csv')
        cand=[k for k in winners if k>1 and bestdiag[1]['BIC']-bestdiag[k]['BIC']>10]
        ordered=[]
        while cand:
            m=min(bestdiag[k]['BIC'] for k in cand)
            k=min(k for k in cand if bestdiag[k]['BIC']<=m+2)
            ordered.append(k);cand.remove(k)
        chosen=winners[1];admitted=False
        for k in ordered:
            center=winners[k]
            tasks=[(k,n,key,v) for key in halfweights(d) for n,v in ((91,center['x']),(92,None))]
            t=trials(pool,tasks);all_runs+=t
            flag=[]
            for role in halfweights(d):
                z=[a for a in t if a['role']==role and a['converged']]
                if not z:
                    comparison=dict(adjusted_rand=np.nan,maximum_profile_domain_RMS=np.nan,check_pass=False)
                else:
                    f=max(z,key=lambda a:a['loglik'])
                    comparison=compare_curves(d,center['x'],f['x'],k)
                    comparison['check_pass']=comparison['adjusted_rand']>=.60 and comparison['maximum_profile_domain_RMS']<=.35
                stability.append(dict(K=k,check=role,converged_starts=len(z),**comparison));flag.append(comparison['check_pass'])
            print(f'stability K={k}: {flag}',flush=True)
            if all(flag):
                chosen=center;admitted=True;break
            bestdiag[k]['fit_disposition']='SPLITHALF_OR_SOURCE_REBALANCE_UNSTABLE'
    bestdiag[chosen['K']]['fit_disposition']='SELECTED_STABLE_COMMON_PROFILES' if admitted else 'DESCRIPTIVE_FALLBACK_K1'
    csvout(bestdiag.values(),T/'TRAJECTORY_CANDIDATE_SUMMARY.csv')
    csvout([{kk:vv for kk,vv in z.items() if kk!='x'} for z in all_runs],T/'TRAJECTORY_CANDIDATE_STARTS.csv')
    csvout(stability or [dict(K=1,check='no_multiclass_candidate_reached_gate',check_pass=False)],T/'TRAJECTORY_STABILITY.csv')
    publish(d,chosen,admitted)


if __name__=='__main__':
    main()

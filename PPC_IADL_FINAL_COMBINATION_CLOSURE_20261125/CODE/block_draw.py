"""Exact source-block evaluation of the inherited Jeffreys score, not separate cohort fits.

Only matrix multiplication is factored. Convergence is checked in the same
weighted centred/whitened coordinates. Exceptional redraws go to the original implementation.
"""
from binding import base,np
from scipy.linalg import eigh,cho_factor,cho_solve
from scipy.special import expit
from scipy.optimize import minimize

class BlockDraw:
    def __init__(self,x,y,src):
        self.x,self.y,self.src=x,y,src
        self.blocks=[]
        for s in np.unique(src):
            rows=np.flatnonzero(src==s); block=x[rows];col=np.flatnonzero((block!=0).any(axis=0))
            self.blocks.append((rows,col,np.ascontiguousarray(block[:,col])))
    def __call__(self,w,warm):
        try:return self.fast(w,warm)
        except (ValueError,AssertionError,np.linalg.LinAlgError,FloatingPointError):
            return base.qc.boot_fit(self.x,self.y,self.src,w,warm)
    def fast(self,w,warm):
        p=self.x.shape[1];G=np.zeros((p,p));parts=[];act=int(np.count_nonzero(w))
        for rows,j,design in self.blocks:
            keep=w[rows]>0;weight=w[rows][keep];xx=design[keep];ons=np.ix_(j,j);ys=self.y[rows][keep]
            G[ons]+=xx.T@(weight[:,None]*xx);parts.append((j,ons,xx,ys,weight))
        n=w.sum();avg=G[0]/n;cent=avg.copy();cent[0]=0
        gram=G-n*(np.outer(avg,cent)+np.outer(cent,avg)-np.outer(cent,cent))
        var=np.diag(gram)/n
        if np.any(var<1e-20):raise ValueError('constant: use original')
        sc=np.sqrt(var);eig,V=eigh(gram/sc[:,None]/sc[None,:],check_finite=False)
        if eig[0]<=eig[-1]*1e-11:raise ValueError('rank: use original')
        mix=(V*np.sqrt(act/eig))@V.T;T=mix/sc[:,None];T[0]-=cent@T
        az=np.linalg.solve(T,warm)
        def ev(bz):
            coef=T@bz;Hraw=np.zeros_like(G);tmp=[];ll=0.
            for j,ons,X,y,ww in parts:
                eta=X@coef[j];pr=expit(eta);wt=pr*expit(-eta)*ww
                Hraw[ons]+=X.T@(X*wt[:,None])
                ll-=ww@np.logaddexp(0,(1-2*y)*eta);tmp.append((j,ons,X,y,ww,pr,wt))
            H=T.T@Hraw@T;cf=cho_factor(H,check_finite=False);ll+=np.log(np.diag(cf[0])).sum()
            A=T@cho_solve(cf,T.T,check_finite=False);graw=np.zeros(p)
            for j,ons,X,y,ww,pr,wt in tmp:
                h=wt*np.einsum('ij,ij->i',X@A[ons],X)
                graw[j]+=X.T@(ww*(y-pr)+h*(.5-pr))
            return ll,T.T@graw,cf
        f,g,cf=ev(az)
        for _ in range(60):
            step=cho_solve(cf,g,check_finite=False);mx=np.max(abs(step))
            if mx<=1e-9 and np.max(abs(g))/n<=1e-9:break
            shrink=min(1.,4/max(4.,mx))
            for __ in range(18):
                trial=az+shrink*step;fo,go,co=ev(trial)
                if fo>=f-1e-10:break
                shrink*=.5
            else:break
            az,f,g,cf=trial,fo,go,co
        if np.max(abs(cho_solve(cf,g,check_finite=False)))>1e-9 or np.max(abs(g))/n>1e-9:
            def target(a):
                try:f0,g0,_=ev(a);return -f0,-g0
                except np.linalg.LinAlgError:return np.inf,np.zeros_like(a)
            sol=minimize(target,az,jac=True,method='BFGS',options=dict(maxiter=500,gtol=1e-9))
            az=sol.x;_,gg,cc=ev(az)
            if max(abs(cho_solve(cc,gg,check_finite=False)))>2e-7 or max(abs(gg))/n>1e-8:raise ValueError('rescue: use original')
        b=T@az
        if not np.isfinite(b).all():raise ValueError('unresolved')
        return base.old.flic_stat(self.x,self.y,self.src,b,w)

if __name__=='__main__':
    from binding import load_risk,ROOT,dump
    from fit_combinations import precise_fit
    import time,hashlib,sys,pandas as pd
    sr=sys.argv[1];comp=base.COMPS[int(sr)];k=3;mi=int(sys.argv[2]);d,_,x,src=load_risk(comp)
    extra,_=base.extras(d,mi);x=np.column_stack([x,extra]);x=np.ascontiguousarray(x[:,base.qc.independent(x)]);y=(d.code_fixed2.to_numpy()==k).astype(float)
    stem=f'COMB_{comp}_state{k}';start,_=precise_fit(x,y);F=BlockDraw(x,y,src)
    plan,inv=base.qc.family_plan(d.family.to_numpy(str),d.stratum.to_numpy(str),71118000+int.from_bytes(hashlib.sha256(stem.encode()).digest()[:3],'big'),199)
    output=[]
    for i in (0,12,87):
        w=plan[i,inv].astype(float);t=time.perf_counter()
        try:new=F.fast(w,start);used=True
        except (ValueError,AssertionError,np.linalg.LinAlgError,FloatingPointError):new=F(w,start);used=False
        t1=time.perf_counter();ref=base.qc.boot_fit(x,y,src,w,start);t2=time.perf_counter()
        np.testing.assert_allclose(new,ref,atol=2e-8,rtol=1e-7)
        output.append(dict(draw=i+1,block_used=used,max_absolute_error=float(max(abs(new-ref))),block_seconds=t1-t,native_seconds=t2-t1))
    dump(ROOT/f'QA/BLOCK_EQUIVALENCE_{comp}_{mi:03}.json',output);print(output,flush=True)

"""Fixed same-design, nuisance-only stabilization of the separated multinomial."""
import os
for var in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[var]='1'
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import f,t
from association import read_locked, cohort_overlap, design, support, digest, SOURCES


def fit(x,y,cluster,k,lam):
    n,p=x.shape
    G=int(cluster.max()+1)
    scale=np.maximum(x.std(axis=0),1.)
    a=x/scale
    if np.linalg.matrix_rank(a)!=p or G<=p*(k-1):
        raise ValueError('UNSUPPORTED_RANK_OR_CLUSTERS')
    reg=np.r_[0.,0.,np.repeat(lam,p-2)]
    scores=np.eye(k)[y][:,1:]
    def pred(m,b):
        eta=np.column_stack([np.zeros(len(m)),m@b]); lg=logsumexp(eta,axis=1,keepdims=True)
        return eta,lg,np.exp(eta-lg)
    def obj(v):
        b=v.reshape(p,k-1)
        z,zz,m=pred(a,b)
        loss=(zz[:,0]-z[np.arange(n),y]).sum()+.5*(reg[:,None]*b*b).sum()
        grad = a.T@(m[:,1:]-scores)+reg[:,None]*b
        return loss/n,grad.ravel()/n
    z=minimize(obj,np.zeros(p*(k-1)),jac=True,method='L-BFGS-B',
               options={'ftol':1e-13,'gtol':1e-8,'maxiter':1700,'maxcor':20})
    if not (np.isfinite(z.x).all() and np.max(np.abs(z.jac))<2e-7):
        raise ValueError('OPTIMIZER_OR_GRADIENT_GATE')
    b=z.x.reshape(p,k-1); m=pred(a,b)[2]; q=k-1
    exact_gradient=obj(z.x)[1]
    h=1e-5*np.sin(np.arange(len(z.x))+1)/np.sqrt(len(z.x))
    delta=(obj(z.x+h)[0]-obj(z.x-h)[0])/2
    assert abs(delta-h@exact_gradient)<1e-10
    H=np.zeros((p*q,p*q))
    for i in range(q):
        for j in range(q):
            H[i::q,j::q]=a.T@(a*(m[:,i+1]*(float(i==j)-m[:,j+1]))[:,None])
    H+=np.diag(np.repeat(reg,q))
    ev=np.linalg.eigvalsh(H)
    if ev.min()<=1e-9:
        raise ValueError('PENALIZED_INFORMATION_NOT_POSITIVE')
    U=(a[:,:,None]*(scores-m[:,1:])[:,None,:]).reshape(n,p*q)
    ug=np.zeros((G,p*q)); np.add.at(ug,cluster,U)
    hi=np.linalg.solve(H,np.eye(p*q))
    V=(G/(G-1))*((n-1)/(n-p*q))*hi@(ug.T@ug)@hi
    jj=np.arange(q,2*q); v=V[np.ix_(jj,jj)]; b_exp=b[1]
    if np.linalg.eigvalsh(v).min()<=1e-10:
        raise ValueError('PENALIZED_ROBUST_EXPOSURE_VARIANCE_FAIL')
    se=np.sqrt(np.diag(v)); crit=t.ppf(.975,G-1)
    x0=a.copy(); x0[:,1]=0; x1=a.copy(); x1[:,1]=1
    p0,p1=pred(x0,b)[2].mean(axis=0),pred(x1,b)[2].mean(axis=0)
    W=b_exp@np.linalg.solve(v,b_exp)
    return p0,p1,np.exp(b_exp),np.exp(b_exp-crit*se),np.exp(b_exp+crit*se), f.sf(W/q,q,G-1),ev.min(),np.max(np.abs(z.jac))


def main():
    root=Path(__file__).resolve().parents[1]
    before=digest(root/'QA/CLASS_LOCK.json')
    rows=[]
    d,k=read_locked(root)
    if d is None or k<2:
        raise RuntimeError('No admitted class model')
    for c in [('H11','H01'),('H10','H00')]:
        z,_=cohort_overlap(d,c)
        for s in ('POOLED',*SOURCES):
            b=z if s=='POOLED' else z.loc[z.source.eq(s)]
            n,ng=support(b,list(range(1,k+1)))
            base=dict(source=s,comparison=''.join([c[0],'_vs_',c[1]]),layer='severity',reference=1,people=int(len(b)) if n>=10 and ng>=10 else 10*round(len(b)/10))
            if min(n,ng)<10:
                rows.append(dict(**base,status='NOT_ESTIMATED_BELOW_PROFILE_BY_HISTORY_SUPPORT'))
                continue
            x,_=design(b,c,'severity')
            for lam in (1.,.25,4.):
                try:
                    a=fit(x,b.profile.to_numpy(dtype=int)-1,pd.factorize(b.cluster)[0],k,lam)
                    p0,p1,o,lo,hi,p,emin,g=a
                    for j in range(k):
                        zrow=dict(**base,penalty=lam,status='ESTIMATED_NUISANCE_L2_SENSITIVITY',profile=j+1,
                               standardized_probability_history=p1[j],standardized_probability_reference=p0[j],
                               global_p_desc=p,information_min_eigen=emin,max_normalized_score=g)
                        if j:
                            zrow.update(OR=o[j-1],OR_lo=lo[j-1],OR_hi=hi[j-1])
                        rows.append(zrow)
                except ValueError as e:
                    rows.append(dict(**base,penalty=lam,status=str(e)))
    assert before==digest(root/'QA/CLASS_LOCK.json')
    pd.DataFrame(rows).to_csv(root/'TABLES/IADL_HISTORY_MULTINOMIAL_STABILIZED.csv',index=False,encoding='utf-8-sig')


if __name__=='__main__':
    main()

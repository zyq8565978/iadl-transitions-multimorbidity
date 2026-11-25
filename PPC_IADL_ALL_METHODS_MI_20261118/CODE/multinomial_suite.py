"""Frozen class and observed-state models; retain joint functional uncertainty."""
from common import *
import inspect,argparse
from scipy.special import softmax
sys.path.insert(0,str(W/'PPC_IADL_ZHOU_ADAPTATION_20261117/CODE'))
import association,ridge,observed_state_models as obs

# Reuse the identical inherited fit and family sandwich. Only expose its
# already-computed full parameter/covariance objects instead of discarding them.
txt=inspect.getsource(ridge.fit)
needle='return p0,p1,np.exp(b_exp),np.exp(b_exp-crit*se),np.exp(b_exp+crit*se), f.sf(W/q,q,G-1),ev.min(),np.max(np.abs(z.jac))'
assert needle in txt
txt=txt.replace(needle,'return b,V,a,ev.min(),np.max(np.abs(z.jac))')
ns=dict(ridge.__dict__);exec(compile(txt,'inherited_ridge_fit_return_adapter','exec'),ns)
full_fit=ns['fit']

def functional(x,y,family,k,lam):
    clusters,levels=pd.factorize(family,sort=True)
    b,V,a,emin,score=full_fit(x,y,clusters,k,lam);n,p=a.shape;q=k-1
    vals=[];grads=[];labels=[];scales=[]
    for j in range(q):
        g=np.zeros((p,q));g[1,j]=1
        vals.append(b[1,j]);grads.append(g.ravel());labels.append('OR_class'+str(j+2));scales.append('log')
    means={};gg={}
    for h in (1,0):
        z=a.copy();z[:,1]=h;pr=softmax(np.c_[np.zeros(n),z@b],axis=1)
        means[h]=pr.mean(0);gg[h]=[]
        for j in range(k):
            deriv=pr[:,j,None]*(np.eye(k)[j,1:]-pr[:,1:])
            gg[h].append(np.einsum('np,nq->pq',z,deriv).ravel()/n)
        for j in range(k):
            vals.append(means[h][j]);grads.append(gg[h][j]);labels.append(f'risk_h{h}_class{j+1}');scales.append('probability')
    for j in range(k):
        vals.append(means[1][j]-means[0][j]);grads.append(gg[1][j]-gg[0][j]);labels.append(f'RD_class{j+1}');scales.append('probability_difference')
    J=np.array(grads);U=J@V@J.T
    assert np.isfinite(U).all() and np.linalg.eigvalsh(U).min()>-1e-8
    # Independent directional derivative of every standardized probability.
    v=np.sin(np.arange(b.size)+1).reshape(b.shape);eps=1e-6
    for h in (0,1):
        z=a.copy();z[:,1]=h
        numeric=(softmax(np.c_[np.zeros(n),z@(b+eps*v)],axis=1).mean(0)-softmax(np.c_[np.zeros(n),z@(b-eps*v)],axis=1).mean(0))/(2*eps)
        assert np.max(abs(numeric-np.array(gg[h])@v.ravel()))<1e-6
    return np.array(vals),U,labels,scales,dict(information_min=emin,score=score,families=len(levels))

def run_one(d,y,x,k,stem,meta,lam):
    cells=np.array([np.bincount(y[d.hx.to_numpy()==h],minlength=k) for h in (0,1)])
    if cells.min()<10:
        row=dict(**meta,stem=stem,status='NOT_ESTIMABLE_FROZEN_CELL_SUPPORT',people=d.person.nunique(),windows=len(d))
        pd.DataFrame([row]).to_csv(R/f'TABLES/{stem}.csv',index=False);return
    choice=R/f'QA/{stem}_estimator.json'
    if choice.exists():lam=json.loads(choice.read_text())['penalty']
    else:
        failed=[]
        if lam==0:
            for mi in range(1,51):
                xx=np.c_[x,extras(d,mi)[0]];xx=xx[:,qc.independent(xx)]
                try:functional(xx,y,d.family,k,0.)
                except (ValueError,np.linalg.LinAlgError,AssertionError) as e:
                    failed.append(dict(mi=mi,reason=str(e)));lam=1.;break
        write(choice,dict(penalty=lam,numerical_fallback=failed))
    for mi in range(1,51):
        p=R/f'PRIVATE/{stem}_mi{mi:03d}.npz';jp=R/f'QA/{stem}_mi{mi:03d}.json'
        if p.exists():continue
        extra,_=extras(d,mi);xx=np.c_[x,extra];xx=xx[:,qc.independent(xx)]
        try:
            q,U,labels,scale,qa=functional(xx,y,d.family,k,lam)
            np.savez_compressed(p,q=q,U=U)
            write(jp,dict(**meta,**qa,status='ESTIMATED',people=d.person.nunique(),windows=len(d),labels=labels,scale=scale,penalty=lam))
        except (ValueError,np.linalg.LinAlgError,AssertionError) as e:
            write(jp,dict(**meta,status='NOT_ESTIMABLE',reason=str(e),mi=mi));continue
        if mi%5==0:print(stem,mi,flush=True)
    records=[json.loads(j.read_text()) for j in (R/'QA').glob(stem+'_mi*.json')]
    m=next((v for v in records if 'labels' in v),None)
    if m is None:
        pd.DataFrame([dict(**meta,stem=stem,status='NOT_ESTIMABLE',reason='; '.join(sorted({v.get('reason','') for v in records})))]).to_csv(R/f'TABLES/{stem}.csv',index=False)
    else:pooled(stem,m['labels'],m['scale'],meta)

def trajectory(comp):
    d,k=association.read_locked(W/'PPC_IADL_ZHOU_ADAPTATION_20261117')
    arms=tuple(comp.split('_vs_'));z,_=association.cohort_overlap(d,arms)
    x,names=association.design(z,arms,'severity')
    z['wave']=z.index_wave;z['family']=z.cluster
    run_one(z,z.profile.to_numpy(int)-1,x,k,'TRAJECTORY_'+comp,dict(module='fixed_trajectory_class',comparison=comp,role='exploratory_fixed_labels'),1.)

def forward(comp):
    d,y,x,src=old.load(comp,'M_severity');a=d.code_fixed1.to_numpy(int);b=d.code_fixed2.to_numpy(int)
    for fine in (False,True):
        origins={0:'zero',1:'physical',2:'psychological',4:'cognitive'} if fine else {0:'zero',1:'one'}
        for j,name in origins.items():
            use=a==j if fine or j==0 else a!=0
            end=b[use];z=d.loc[use].reset_index(drop=True);xx=x[use]
            if j==0 and fine:to=np.select([end==0,end==1,end==2,end==4],[0,1,2,3],default=4);k=5
            elif j==0:to=obs.group(end);k=3
            else:to=(obs.group(end)==2).astype(int);k=2
            stem=f'FORWARD_{comp}_{"expanded" if fine else "main"}_{name}'
            # The original observed-state suite used lambda=1 when pure source
            # strata preclude ordinary finite multinomial estimation.
            lam=1. if z.assign(category=to).groupby('stratum').category.nunique().eq(1).any() else 0.
            run_one(z,to,xx,k,stem,dict(module='forward_observed_state',comparison=comp,origin=name,expanded=fine),lam)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('module',choices=['trajectory','forward']);ap.add_argument('comparison',choices=COMPS);a=ap.parse_args()
    effect_gate();globals()[a.module](a.comparison)

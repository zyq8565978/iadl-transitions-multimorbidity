"""Read frozen V6 participant-window matrices. Write only aggregate IPD results here."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import sys
import hashlib
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.special import expit
from scipy.stats import f as fdist, t as tdist

R=Path(__file__).resolve().parent
V6=R.parent/'PPC_IADL_V6_Repair_Synthesis_20261116'
V7=R.parent/'PPC_IADL_FOUR_COHORT_TRAJECTORY_V7_20261116'
sys.path.insert(0,str(V6/'CODE'))
from association_v6 import original, SOURCES, COMPS, LAYERS
from firth_engine_v6 import fit as firth, family_plan, flic_shift

B=499
KEYS=('log_or','risk_index','risk_reference','risk_difference')


def sha(p):
    with p.open('rb') as s:
        return hashlib.file_digest(s,'sha256').hexdigest()


def save(name,rows):
    pd.DataFrame(rows).to_csv(R/name,index=False,encoding='utf-8-sig')


def load(comp,layer,inputs=None):
    xs=[]; ys=[]; ds=[]
    frozen=pd.read_csv(V6/'TABLES/META_INPUTS.csv').query('set == "A_original"')
    for s in SOURCES:
        a,d,stem=original(s,comp,layer)
        old=frozen.loc[frozen.source.eq(s)&frozen.comparison.eq(comp)&frozen.model_layer.eq(layer)].iloc[0]
        terms=list(a['terms']); j=terms.index('hx')
        if inputs is not None and old.estimator=='frozen_MLE_family_HC1':
            np.testing.assert_allclose([a['beta'][j],np.sqrt(a['cov'][j,j])],[old.log_or,old.log_or_se],atol=2e-10)
        if inputs is not None:
            for file in (stem.with_suffix('.npz'),stem.with_name(stem.name+'_rows.parquet')):
                inputs[str(file.relative_to(R.parent))]=sha(file)
        obs=(len(d),d.person.nunique(),d.cluster.nunique())
        assert obs==(int(old.windows),int(old.people),int(old.families))
        d=d.assign(source=s, person=s+'|'+d.person.astype(str), family=s+'|'+d.cluster.astype(str),
                   stratum=(s+'|'+d.country.astype(str) if s=='SHARE' else s))
        assert np.isfinite(a['X']).all() and len(np.unique(a['y']))==2
        ds.append(d);ys.append(a['y']);xs.append(a['X'][:,[i for i in range(a['X'].shape[1]) if i not in (0,j)]])
    d=pd.concat(ds,ignore_index=True); y=np.concatenate(ys)
    cov=np.zeros((len(d),sum(x.shape[1] for x in xs)))
    start=end=0
    for x in xs:
        n,p=x.shape;cov[start:start+n,end:end+p]=x;start+=n;end+=p
    src=pd.Categorical(d.source,categories=SOURCES).codes
    base=np.c_[np.ones(len(d)),d.hx.to_numpy(dtype=float),np.eye(len(SOURCES))[src,1:],cov]
    assert not d.duplicated(['person','window']).any() and np.array_equal(d.y,y)
    assert np.linalg.matrix_rank(base)==base.shape[1]
    assert (d.groupby('family').source.nunique()==1).all()
    return d,y,base,src


def logistic(x,y,family):
    """QR conditioned Newton MLE plus family sandwich, independently evaluated on original X."""
    q, rt=np.linalg.qr(x,mode='reduced'); n,p=x.shape
    z=q*np.sqrt(n)
    inv=np.linalg.solve(rt,np.eye(p))*np.sqrt(n)
    c=np.zeros(p)
    def objective(b):
        eta=z@b
        return np.sum(np.logaddexp(0,eta)-y*eta)
    for it in range(65):
        pr=expit(z@c);v=pr*(1-pr)
        H=(z.T*v)@z
        score=z.T@(y-pr)
        step=np.linalg.solve(H,score)
        if np.max(np.abs(step))<1e-10:
            break
        old=objective(c)
        for k in range(25):
            if objective(c+step*2.**-k)<=old+1e-9:
                c+=step*2.**-k;break
        else:
            raise ValueError('line search did not converge')
    else:
        raise ValueError('convergence gate missed')
    assert np.max(np.abs(score)/n)<1e-9 and np.linalg.eigvalsh(H).min()>1e-7
    gid,G=pd.factorize(family,sort=True);G=len(G)
    S=np.zeros((G,p));np.add.at(S,gid,z*(y-pr)[:,None]); W=np.linalg.solve(H,S.T)
    cov=(G/(G-1))*((n-1)/(n-p))*inv@(W@W.T)@inv.T
    b=inv@c
    xscore=x.T@(y-expit(x@b))
    np.testing.assert_allclose(expit(x@b),pr,rtol=1e-9,atol=1e-10)
    assert G>p and np.isfinite(cov).all()
    return b,cov,G,dict(iterations=it,normalized_score_max=float(np.max(np.abs(score)/n)),
                       original_score_max=float(np.max(abs(xscore))),information_min_qr=float(np.linalg.eigvalsh(H).min()))


def ordinary(d,y,x,src,profile):
    keep=np.ones(len(y),dtype=bool)
    if profile:
        keep=d.groupby('stratum').y.transform('nunique').gt(1).to_numpy()
    a=x[keep]
    cols=np.arange(x.shape[1])
    if not keep.all():
        unit=np.linalg.qr(a[:,:2],mode='reduced')[0]
        rest=a[:,2:]-unit@(unit.T@a[:,2:])
        _,r,piv=qr(rest,mode='economic',pivoting=True)
        rank=(np.abs(np.diag(r))>1e-7).sum()
        cols=np.r_[0,1,np.sort(2+piv[:rank])]
        assert len(cols)==np.linalg.matrix_rank(a)
    x=x[:,cols]; b,V,G,qa=logistic(x[keep],y[keep],d.family[keep])
    pred=[]; gr=[]
    for h in (1,0):
        xx=x.copy();xx[:,1]=h
        p=expit(xx@b);p[~keep]=y[~keep]
        pred.append(p.mean()); gr.append((xx[keep].T@(p[keep]*(1-p[keep])))/len(x))
    j=np.eye(len(b))[1]; grad=np.stack([j,gr[0],gr[1],gr[0]-gr[1]])
    ans=np.r_[b[1],pred,pred[0]-pred[1]]
    se=np.sqrt(np.diag(grad@V@grad.T));crit=tdist.ppf(.975,G-1)
    r=columns(ans,ans-crit*se,ans+crit*se,se)
    return dict(**r,inference='family HC1 t(G-1)',p_log_or_approx=2*tdist.sf(abs(b[1]/se[0]),G-1),
                estimable_families=G if keep.all() else None,
                informative_windows=int(keep.sum()) if keep.all() else None,
                boundary_windows_release='0' if keep.all() else '<10' if (~keep).sum()<10 else '10+',
                retained_full_risk_reference=True,**qa)


def columns(center,lo,hi,se):
    z={}
    for j,k in enumerate(KEYS):
        z.update({k:center[j],k+'_lo':lo[j],k+'_hi':hi[j],k+'_se':se[j]})
    z.update(OR=np.exp(center[0]),OR_lo=np.exp(lo[0]),OR_hi=np.exp(hi[0]))
    return z


def flic_stat(x,y,src,b,w):
    neutral=x@b-x[:,1]*b[1]; eta=neutral+x[:,1]*b[1]
    shifts=np.zeros(len(SOURCES))
    for j in range(len(SOURCES)):
        k=(src==j)&(w>0)
        shifts[j]=flic_shift(eta[k],y[k],w[k])
    pred=np.array([np.average(expit(neutral+h*b[1]+shifts[src]),weights=w) for h in (1,0)])
    assert abs(np.average(expit(eta+shifts[src])-y,weights=w))<1e-10
    return np.r_[b[1],pred,pred[0]-pred[1]]


def corrected(d,y,x,src):
    b,qa=firth(x,y)
    center=flic_stat(x,y,src,b,np.ones(len(y)))
    plan,inv=family_plan(d.family.to_numpy(dtype=str),d.stratum.to_numpy(dtype=str),61116081,n=B)
    good=[];fail=Counter(); constant=0
    for z in plan:
        w=z[inv].astype(float)
        try:
            b2,diag=firth(x,y,w,warm=b)
            good.append(flic_stat(x,y,src,b2,w));constant+=diag['dropped_redraw_constant_columns']>0
        except (ValueError,np.linalg.LinAlgError) as e:
            fail[str(e)]+=1
    ok=len(good)>=450
    q=np.asarray(good)
    missing=np.repeat(np.nan,4)
    lo,hi=np.quantile(q,[.025,.975],axis=0) if ok else (missing,missing)
    return dict(**columns(center,lo,hi,q.std(axis=0,ddof=1) if ok else missing),inference='499 source/country-stratified family bootstrap, percentile; source FLIC',
                status='estimated' if ok else 'point_only',bootstrap_attempted=B,bootstrap_valid=len(good),
                redraw_constant_level=constant,bootstrap_failures=';'.join(f'{k}:{n}' for k,n in fail.items()),
                informative_windows=len(d),retained_full_risk_reference=True,**qa)


def run(arg):
    c,l=arg; tick=perf_counter()
    d,y,x,src=load(c,l)
    base=dict(comparison=c,layer=l,windows=len(d),people=d.person.nunique(),families=d.family.nunique(),events=int(y.sum()),full_design_rank=x.shape[1],
         known_single_outcome_strata=int(d.groupby('stratum').y.nunique().eq(1).sum()))
    if c==COMPS[0]:
        assert base['known_single_outcome_strata']==1
        ans=[dict(**base,estimator='one_stage_Firth_FLIC_full_sample',**corrected(d,y,x,src))]
        diag=ordinary(d,y,x,src,True)
        assert diag['boundary_windows_release']!='0'
        ans.append(dict(**base,estimator='one_stage_profile_boundary_MLE_diagnostic',status='estimated',**diag))
    else:
        assert base['known_single_outcome_strata']==0
        ans=[dict(**base,estimator='one_stage_MLE',status='estimated',**ordinary(d,y,x,src,False))]
    inter=[]
    if c==COMPS[1] and l=='M_severity':
        q=np.c_[x,x[:,1,None]*np.eye(len(SOURCES))[src,1:]]
        b,V,G,_=logistic(q,y,d.family)
        ww=b[-3:]@np.linalg.solve(V[-3:,-3:],b[-3:]); inter.append(dict(comparison=c,layer=l,wald_F=ww/3,df1=3,df2=G-1,p=fdist.sf(ww/3,3,G-1)))
    print(c,l,[(a['estimator'],a['status'],a.get('bootstrap_valid')) for a in ans],round(perf_counter()-tick,2),flush=True)
    return ans,inter


def main():
    inputs={str(p.relative_to(R.parent)):sha(p) for p in (V6/'SAFE_DELIVERY_V6.zip',V7/'SAFE_DELIVERY_V7.zip',V6/'TABLES/META_INPUTS.csv',V6/'TABLES/META_MAIN_AND_SENSITIVITY.csv',V6/'CODE/firth_engine_v6.py',V6/'CODE/association_v6.py',R/'ANALYSIS_SPEC.md',Path(__file__).resolve())}
    assert inputs[str((V6/'SAFE_DELIVERY_V6.zip').relative_to(R.parent))]=='28b5fe3bfd6c9afc1cc87c95cc53843b279e00497d75c1b95e508920b3009f91'
    assert inputs[str((V7/'SAFE_DELIVERY_V7.zip').relative_to(R.parent))]=='8b862b9734ca4f28e8b996eceaf83d05ceb142a57f4f7c12a829cb1e8269d239'
    flow=[];models=[(c,l) for c in COMPS for l in LAYERS]
    membership={}
    for c,l in models:
        d,y,x,s=load(c,l,inputs)
        members=np.array(list(zip(d.source,d.person,d.window,d.y,d.hx)),dtype=str)
        if c in membership:
            assert np.array_equal(membership[c],members)
        else:
            membership[c]=members
            for key,q in d.groupby('source'):
                flow.append(dict(comparison=c,source=key,windows=len(q),people=q.person.nunique(),family_clusters=q.family.nunique(),events=int(q.y.sum())))
    (R/'ONE_STAGE_LOCK.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf-8')
    save('ONE_STAGE_SAMPLE.csv',flow)
    out=[]; inter=[]
    with ProcessPoolExecutor(max_workers=2) as e:
        for a,b in e.map(run,sorted(models,key=lambda z:(z[1]!='M_severity',z[0]))):
            out+=a; inter+=b
            save('ONE_STAGE_IPD_RESULTS.csv',out)
            if inter:save('ONE_STAGE_INTERACTION_QA.csv',inter)
    v6=pd.read_csv(V6/'TABLES/META_MAIN_AND_SENSITIVITY.csv').query('set=="A_original" and excluded=="none"')
    bridge=pd.DataFrame(a for a in out if a['estimator']!='one_stage_profile_boundary_MLE_diagnostic').merge(
        v6[['comparison','model_layer','or','lo_kh','hi_kh','p_kh']],left_on=['comparison','layer'],right_on=['comparison','model_layer'],validate='one_to_one')
    bridge[['comparison','layer','estimator','status','windows','people','events','OR','OR_lo','OR_hi','or','lo_kh','hi_kh','p_kh']].rename(columns={'or':'v6_OR','lo_kh':'v6_lo','hi_kh':'v6_hi','p_kh':'v6_p'}).to_csv(R/'ONE_STAGE_VS_V6.csv',index=False,encoding='utf-8-sig')
    assert all(sha(R.parent/p)==h for p,h in inputs.items()), 'Frozen artifact changed during run'


if __name__=='__main__':
    main()

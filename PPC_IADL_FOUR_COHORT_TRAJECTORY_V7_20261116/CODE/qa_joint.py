"""Independent read-only V7 QA: no imports from the fitter, no parameter estimates."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from scipy.stats import betabinom, norm, rankdata

R = Path(__file__).resolve().parents[1]
P, T, Q = (R/d for d in ('PRIVATE', 'TABLES', 'QA'))
SOURCES = ('CHARLS', 'HRS', 'ELSA', 'SHARE')
DOMAINS = ('chronic', 'symptoms', 'cognition')


def sha(f):
    with f.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def read(f):
    return json.loads(f.read_text(encoding='utf-8-sig'))


def density(d, b, prior, c, sigma):
    scores=d[['person']].copy()
    idx=d.source.map({s:i for i,s in enumerate(SOURCES)}).to_numpy()
    ts=d.time.to_numpy()/4
    for cls in range(prior.shape[1]):
        lnp=0
        for j,domain in enumerate(DOMAINS):
            coeff=b[j]
            eta=coeff[idx]+coeff[idx+4]*ts
            if cls:
                eta=eta+coeff[6+2*cls]+coeff[7+2*cls]*ts
            x=d[domain].to_numpy()
            if j<2:
                p=expit(eta); q=c[j,idx]; n=7 if j==0 else d.symptoms_max.to_numpy()
                lnp+=betabinom.logpmf(x,n,p*q,(1-p)*q)
            else:
                lnp+=norm.logpdf(x,loc=eta,scale=sigma[idx])
        scores[f'l{cls+1}']=lnp
    score=scores.groupby('person',sort=True).sum()
    source=d[['person','source']].drop_duplicates().set_index('person').source.reindex(score.index)
    ii=source.map({s:i for i,s in enumerate(SOURCES)}).to_numpy()
    prior=prior-logsumexp(prior,axis=1,keepdims=True)
    a=score.to_numpy()+prior[ii]
    li=logsumexp(a,axis=1)
    return score.index, ii, li.sum(), np.exp(a-li[:,None]), np.exp(prior)


def means(th, g, src, domain, time):
    j=DOMAINS.index(domain)
    u=th[j]
    eta=u[src]+u[src+4]*(time/4)
    if g:
        eta=eta+u[6+2*g]+u[7+2*g]*(time/4)
    return eta if domain=='cognition' else (7 if j==0 else 100)*expit(eta)


def check_archive():
    z=R/'SAFE_DELIVERY_V7.zip'
    if not z.exists():
        return 'not assembled at this QA step'
    with zipfile.ZipFile(z) as zipf:
        assert zipf.testzip() is None
        names=zipf.namelist()
        assert len(names)==len(set(names))
        manifest=zipf.read('MANIFEST_SHA256.txt').decode('utf-8').splitlines()
        assert len(manifest)==len(names)-1
        entries={}
        for m in manifest:
            h,s=m.split('  ',1)
            assert s not in entries
            entries[s]=h
        assert set(entries)==set(names)-{'MANIFEST_SHA256.txt'}
        for item,h in entries.items():
            f=PurePosixPath(item)
            assert all(s.lower() not in ('private','raw','cache','..','.') for s in f.parts)
            assert not f.is_absolute() and f.suffix.lower() not in ('.parquet','.npz','.dta','.rds')
            assert hashlib.sha256(zipf.read(item)).hexdigest()==h
            assert sha(R/item)==h, f'stale zip member: {item}'
    return f'{len(names)} members'


def main():
    spec=read(Q/'TRAJECTORY_SPEC_LOCK.json')
    for name,h in spec['files'].items():
        assert sha(R/name)==h,name
    for f in pd.read_csv(Q/'PREFIT_INPUT_HASHES.csv').itertuples(index=False):
        assert sha(R.parent/f.path)==f.sha256,f.path
    lock=read(Q/'CLASS_LOCK.json')
    assert sha(Q/'TRAJECTORY_SPEC_LOCK.json')==lock['authority_sha256']
    assert sha(T/'PREFIT_SUPPORT.csv')==lock['support_sha256']
    assert sha(T/'TRAJECTORY_CURVES.csv')==lock['trajectories_sha256']
    ckpt=np.load(P/'final_working_model.npz',allow_pickle=False)
    k=int(ckpt['K']); npar=8+2*(k-1)
    assert k==lock['K']
    b=ckpt['theta'][:3*npar].reshape(3,npar)
    prior=np.c_[np.zeros(4),ckpt['theta'][3*npar:].reshape(4,k-1)]
    c=ckpt['nuisance_c']; sigma=ckpt['nuisance_sig']
    assert (c > .5).all() and (c < 1000).all() and (sigma > .1).all()
    assert np.max(abs(b))<8-1e-5 and np.max(abs(prior))<9-1e-5
    d=pd.read_parquet(P/'long.parquet')
    people,s,like,prob,pr=density(d,b,prior,c,sigma)
    ll_err=abs(like-lock['loglik'])
    assert ll_err<1e-7
    order=ckpt['order']
    z=np.array([[means(b,g,np.arange(4),domain,0).mean() for g in range(k)] for domain in DOMAINS])
    expected=np.lexsort((z[0],rankdata(z[0])+rankdata(z[1])+rankdata(-z[2])))
    assert np.array_equal(order,expected)
    prob=prob[:,order]; pr=pr[:,order]
    assert np.max(abs(prob.sum(axis=1)-1))<1e-11 and np.isfinite(prob).all()
    pos_err=0.
    if lock['admitted']:
        assert sha(P/'posterior.csv')==lock['posterior_sha256']
        p=pd.read_csv(P/'posterior.csv',dtype={'person':'string'}).set_index('person',verify_integrity=True)
        assert set(p.index)==set(people)
        pos_err=np.max(np.abs(prob-p.loc[people,[f'p{j+1}' for j in range(k)]].to_numpy()))
        assert pos_err<1e-10
    mop=prob.argmax(axis=1)
    gdeg=pd.read_csv(T/'PROFILE_COUNTS.csv')
    for rec in gdeg.itertuples(index=False):
        full=rec.scope.lower()=='pooled'
        take=np.ones(len(s),dtype=bool) if full else s==SOURCES.index(rec.scope)
        j=rec.profile-1
        chosen=take&(mop==j)
        assert int(rec.people)==chosen.sum()
        assert abs(float(rec.share)-chosen.sum()/take.sum())<1e-12
        assert abs(rec.mean_MAP_posterior-prob[chosen,j].mean())<1e-12
        model_prior=pr[:,j].mean() if full else pr[SOURCES.index(rec.scope),j]
        assert abs(rec.estimated_membership_share-model_prior)<1e-12
    candidates=pd.read_csv(T/'TRAJECTORY_CANDIDATE_SUMMARY.csv')
    winner=candidates.loc[candidates.K.eq(k)].iloc[0]
    assert abs(winner.BIC-(-2*like+len(ckpt['theta'])*np.log(len(people))))<1e-7
    assert abs(winner.loglik-like)<1e-7
    st=pd.read_csv(T/'TRAJECTORY_STABILITY.csv')
    if lock['admitted']:
        q=st[st.K.eq(k)]
        assert set(q.check)=={'family_half_A','family_half_B','equal_cohort'}
        assert (q.check_pass.eq(True)&q.adjusted_rand.ge(.60)&q.maximum_profile_domain_RMS.le(.35)).all()
        assert all(winner[j]>=limit for j,limit in [('min_MAP_share',.05),('min_MAP_n',50),('min_AvePP',.70),
                     ('min_source_MAP_n',10),('min_expected_source_class',25),('replicating_starts',2)])
        assert winner.max_source_fraction<=.75
    # Every equal-cohort curve coordinate and full source-specific range from the checkpoint.
    curves=pd.read_csv(T/'TRAJECTORY_CURVES.csv')
    assert not curves.duplicated(['scope','domain','profile','time']).any()
    err=0.
    for (scope,domain,profile),u in curves.groupby(['scope','domain','profile']):
        if scope=='pooled':
            xs=np.stack([means(b,order[profile-1],j,domain,u.time.to_numpy()) for j in range(4)])
            exact=xs.mean(axis=0)
            assert len(u)==61 and np.isclose(u.time.max(),5)
        else:
            exact=means(b,order[profile-1],SOURCES.index(scope),domain,u.time.to_numpy())
        err=max(err,np.max(abs(exact-u.estimate.to_numpy())))
    assert err<1e-10 and curves[['lo','hi']].isna().all().all()
    if (Q/'DISPLAY_QA.json').exists():
        f=read(Q/'DISPLAY_QA.json')
        for suffix,h in f['figure_hashes'].items():
            assert sha(R/f'FIGURES/Figure_V7_Four_cohort_joint_trajectories.{suffix}')==h
        assert f['curve_source_sha256']==lock['trajectories_sha256']
        assert f['class_lock_sha256']==sha(Q/'CLASS_LOCK.json')
    print(f'selected K={k}; full independent absolute likelihood delta={ll_err:.3g}; '
          f'max posterior delta={pos_err:.3g}; max curve delta={err:.3g}; archive={check_archive()}')


if __name__=='__main__':
    main()

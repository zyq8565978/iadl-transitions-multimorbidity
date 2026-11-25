"""One-stage local observation models; no participant-level files are written."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k] = '1'

from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path
import runpy
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.linalg import block_diag, cho_factor, cho_solve
from scipy.special import expit

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
V3=WORK/'PPC_IADL_Dynamic_V3_20261116'
V5=WORK/'PPC_IADL_SHARE_V5_20261116'
V6=WORK/'PPC_IADL_V6_Repair_Synthesis_20261116'
sys.path.insert(0,str(V6/'CODE'))
from firth_engine_v6 import FitFail, family_plan, fit as firth, flic_shift
O=runpy.run_path(str(V3/'CODE/ppc_v3.py'))
S=runpy.run_path(str(V5/'CODE/share_association_v5.py'))
SOURCES=('CHARLS','HRS','ELSA','SHARE')
COMP=tuple(O['COMPARE'])
DELTAS=O['DELTA']
IPW=('IPW_uncapped','IPW_p99_cap')
B, MIN_OK=199, 180
HASHES={}


def sha(p):
    with p.open('rb') as f:
        value=file_digest(f,'sha256').hexdigest()
    HASHES[p.relative_to(WORK).as_posix()]=value
    return value


def verify():
    a=json.loads((V3/'QA/PPC_BRANCH_LOCK.json').read_text(encoding='utf-8-sig'))
    b=json.loads((V5/'SHARE_ANALYSIS_LOCK.json').read_text(encoding='utf-8-sig'))
    for s in SOURCES[:3]:
        for p in O['sources'](s):
            if sha(p)!=a['inputs'][s][p.name]:
                raise RuntimeError(f'{s} frozen input hash mismatch')
    for k in ('PRIVATE/SHARE_windows.parquet','PRIVATE/SHARE_panel.parquet',
              'CODE/share_association_v5.py'):
        if sha(V5/k)!=b['inputs'][k]:
            raise RuntimeError('SHARE frozen input hash mismatch')
    if DELTAS!={k:tuple(x) for k,x in a['delta'].items()}:
        raise RuntimeError('delta authority mismatch')


def low(n):
    return '<10' if n<10 else str(int((n+5)//10*10))


class Newton:
    """Warm, preconditioned binomial likelihood with replicated frequency/analytic weights."""
    def __init__(self,x,y):
        self.y=np.asarray(y)
        q,r=np.linalg.qr(x,mode='reduced')
        if np.min(np.abs(np.diag(r))) < 1e-9:
            raise FitFail('constant_or_aliased_point_design')
        n=len(x)
        self.x=q*np.sqrt(n)
        self.r=r/np.sqrt(n)
        self.i=np.linalg.solve(self.r,np.eye(r.shape[0]))

    def fit(self,w,warm=None):
        keep=w>0
        if np.unique(self.y[keep]).size<2:
            raise FitFail('resample_response_group_absent')
        x,y = self.x, self.y
        b=np.zeros(x.shape[1]) if warm is None else self.r@warm
        def likelihood(c):
            eta=x@c
            return np.dot(w,y*eta-np.logaddexp(0,eta))
        for step in range(35):
            eta=x@b
            p=expit(eta)
            h=x.T@(x*(w*p*expit(-eta))[:,None])
            try:
                cf=cho_factor(h)
                grad=x.T@(w*(y-p))
                dx=cho_solve(cf,grad)
            except np.linalg.LinAlgError as e:
                raise FitFail('nonpositive_information') from e
            if np.max(np.abs(dx))<1e-8 and np.max(np.abs(grad))/w.sum()<1e-8:
                spectrum=np.linalg.eigvalsh(h)
                if spectrum[0]<=1e-8 or spectrum[-1]/spectrum[0]>1e11:
                    raise FitFail('weak_information_or_separation')
                return self.i@b
            prev=likelihood(b)
            for shrink in range(18):
                z=b+(2.**-shrink)*dx
                if likelihood(z)>=prev-1e-9:
                    b=z
                    break
            else:
                raise FitFail('likelihood_line_search')
        raise FitFail('likelihood_did_not_converge')


class Target:
    def __init__(self,s,q,comp):
        self.s=s
        d=(S['select_target'](q,comp) if s=='SHARE' else O['obs_prepare'](q,s,comp))
        assert len(d)>0 and d.cluster.notna().all()
        assert (d.legal2 & d.proxy2.eq(0)).all()
        assert not d.status2.eq('confirmed_death').any()
        if s=='HRS':
            assert d.age1.ge(65).all() and not d.c3_route.eq('structural_skip').any()
        design=(S['matrix']() if s=='SHARE' else O['BASE_MATRIX'])(d,s,comp,'M_severity')[0]
        y=d.y.to_numpy(float)
        obs=np.isfinite(y)
        keep=(design.nunique().gt(1)|design.columns.isin(['Intercept']))
        design=design.loc[:,keep]
        assert design.columns[0]=='Intercept' and 'hx' in design
        self.d,self.obs,self.y=d,obs,y
        self.x=design.to_numpy(float)
        self.xa=design.drop(columns='g12',errors='ignore').to_numpy(float)
        self.hx=d.hx.to_numpy(int)
        assert (self.hx==self.x[:,design.columns.get_loc('hx')]).all()
        self.background=design.drop(columns=['Intercept','hx']).to_numpy(float)
        self.family=(s+'|'+d.cluster.astype(str)).to_numpy(str)
        self.strata=(s+'|'+d.country.astype(str)).to_numpy(str) if s=='SHARE' else np.repeat(s,len(d))
        self.a=None
        self.alpha=None
        self.method='Firth / FLIC' if s=='SHARE' else 'ordinary MLE'
        self.status='estimated'
        self.stats=[]
        for h in (0,1):
            b=self.hx==h
            n1,n0=int((self.obs&b).sum()),int((~self.obs&b).sum())
            self.stats.append(dict(source=s,comparison=comp,history='H1' if h else 'H0',
                                   target_windows=low(b.sum()),observed_windows=low(n1),
                                   incomplete_windows=low(n0),response_model=self.method))
            if n1<20 or n0<10:
                self.status='minimum_local_support'
        if self.status=='estimated':
            try:
                w=np.ones(len(d))
                if s!='SHARE':
                    # The historical point-eligibility numeric gate is checked first.
                    self.alpha=O['validated_binomial'](self.xa,obs.astype(float))
                    self.a=Newton(self.xa,obs.astype(float))
                self.response(w,collect=True)
            except (FitFail,ValueError,np.linalg.LinAlgError) as exc:
                self.status=str(exc).split('\n')[0][:140]
        for o in self.stats:
            o['ipw_point_gate']=self.status

    def response(self,w,collect=False):
        obs,hx,s=self.obs,self.hx,self.s
        active=w>0
        r=active&obs
        if any(not (r[hx==a].any() and (active&~obs)[hx==a].any()) for a in (0,1)):
            raise FitFail('local_redraw_observation_group_missing')
        if s=='SHARE':
            a,_=firth(self.xa,obs.astype(float),w,warm=self.alpha)
            eta=self.xa@a
            shift=flic_shift(eta[active],obs[active],w[active])
        else:
            a=self.a.fit(w,self.alpha)
            if collect:
                np.testing.assert_allclose(self.xa@a,self.xa@self.alpha,atol=3e-6)
            eta=self.xa@a
            shift=0
        if collect:
            self.alpha=a
        p=expit(eta+shift)
        W=np.zeros(len(p))
        W[r]=1/p[r]
        bounds=[]
        for h in (0,1):
            take=hx==h
            ps=p[active&take]
            z=r&take
            ww=W[z]
            ess=np.dot(w[z],ww)**2 / np.dot(w[z],ww**2)
            frac=ess/w[z].sum()
            bounds.append((ps.min(),ps.max(),frac))
            if collect:
                self.stats[h].update(min_probability=ps.min(), max_probability=ps.max(),
                                     ESS_fraction=frac,p99_weight=np.quantile(ww,.99),
                                     effective_observed_windows=low(ess))
        if min(k[0] for k in bounds)<.05 or min(k[2] for k in bounds)<.5 or max(k[0] for k in bounds)>min(k[1] for k in bounds):
            raise FitFail('local_probability_overlap_or_ess_gate')
        cap=np.quantile(np.repeat(W[r],w[r].astype(int)),.99)
        return W,np.minimum(W,cap)


class Pool:
    def __init__(self, targets, comp):
        self.targets=targets
        self.people=sum(t.d.person.nunique() for t in targets)
        self.obs=np.concatenate([t.obs for t in targets])
        self.y=np.concatenate([t.y for t in targets])
        self.hx=np.concatenate([t.hx for t in targets])
        self.s=np.concatenate([np.repeat(i,len(t.d)) for i,t in enumerate(targets)])
        self.family=np.concatenate([t.family for t in targets])
        self.strata=np.concatenate([t.strata for t in targets])
        self.use_firth=comp=='H11_vs_H01'
        self.comp=comp
        x=np.c_[np.ones(len(self.y)),self.hx,np.eye(len(targets))[self.s,1:],
                block_diag(*(t.background for t in targets))]
        self.x=x
        self.warm={}
        xobs=x[self.obs]
        if np.linalg.matrix_rank(xobs)<xobs.shape[1]:
            raise FitFail('pooled_compatible_target_not_covered_by_observed_design')
        self.fixed=None if self.use_firth else Newton(xobs,self.y[self.obs])

    def fit(self,w,analytic,key,collect=False):
        obs=self.obs
        wo=w[obs].copy()
        if analytic is not None:
            wo*=analytic[obs]
            wo*=w[obs].sum()/wo.sum()
        active=w>0
        mask=obs[active]
        px=self.x[active]
        xv=self.x[obs]
        # A level present solely in newly incomplete/zero-response target rows
        # cannot receive predictions by assigning an arbitrary coefficient.
        for i in range(len(self.targets)):
            target=active&(self.s==i)
            observed=target&obs
            if not np.any(observed):
                raise FitFail('source_observed_redraw_absent')
            if np.any((np.ptp(self.x[observed],axis=0)==0)&(np.ptp(self.x[target],axis=0)>0)):
                raise FitFail('active_target_level_not_observed')
        if self.use_firth:
            b,_=firth(xv,self.y[obs],wo,warm=self.warm.get(key))
        else:
            b=self.fixed.fit(wo,warm=self.warm.get(key))
        if collect:
            self.warm[key]=b
        eta=px@b
        if self.use_firth:
            ss=self.s[active]
            for i in range(len(self.targets)):
                ta=ss==i
                keep=ta&mask
                eta[ta]+=flic_shift(eta[keep],self.y[active][keep],wo[w[obs]>0][ss[mask]==i])
        base=eta-px[:,1]*b[1]
        return b[1],base,w[active],~mask

    def evaluate(self,fitted,delta):
        hx,base,w,unobs=fitted
        h1=np.average(expit(base+hx+delta[0]*unobs),weights=w)
        h0=np.average(expit(base+delta[1]*unobs),weights=w)
        return np.array([h1,h0,h1-h0])

    def info(self):
        return dict(comparison=self.comp,sources=';'.join(t.s for t in self.targets),
                    target_windows=low(len(self.y)),observed_windows=low(self.obs.sum()),
                    incomplete_windows=low((~self.obs).sum()),target_people=low(self.people),
                    families=low(len(np.unique(self.family))),
                    outcome_model='common-H Firth with source FLIC' if self.use_firth else 'common-H ordinary MLE')


def pool_run(ts,comp,role):
    names=list(DELTAS) if role=='delta_four_sources' else ['matched_unweighted',*IPW]
    source_spec=';'.join(t.s for t in ts)
    if not ts:
        return [dict(comparison=comp,analysis=role,method=k,status='not_estimable_no_eligible_sources')
                for k in names]
    stats=dict(comparison=comp,analysis=role,sources=source_spec)
    try:
        pool=Pool(ts,comp)
    except (FitFail,ValueError,np.linalg.LinAlgError) as e:
        return [dict(**stats,method=k,status='not_estimable',detail=str(e)) for k in names]
    head=pool.info()|dict(analysis=role)
    seed=2026111705+10*COMP.index(comp)+(role=='delta_four_sources')
    plan,inv=family_plan(pool.family,pool.strata,seed,B)
    pos=np.cumsum([0]+[len(t.d) for t in ts])
    error={k:Counter() for k in names}
    draws={k:[] for k in names}

    def fit_all(w,point):
        out={}
        try:
            u=pool.fit(w,None,'plain',point)
            if role=='delta_four_sources':
                out={k:pool.evaluate(u,DELTA) for k,DELTA in DELTAS.items()}
            else:
                out['matched_unweighted']=pool.evaluate(u,(0,0))
        except (FitFail,ValueError,np.linalg.LinAlgError) as e:
            bad=names if role=='delta_four_sources' else ['matched_unweighted']
            for k in bad:
                error[k][str(e)]+=1
        if role=='delta_four_sources':
            return out
        try:
            weighted=[t.response(w[a:z]) for t,a,z in zip(ts,pos[:-1],pos[1:])]
        except (FitFail,ValueError,np.linalg.LinAlgError) as e:
            for k in IPW:
                error[k][str(e)]+=1
            return out
        for j,k in enumerate(IPW):
            try:
                a=np.concatenate([zz[j] for zz in weighted])
                out[k]=pool.evaluate(pool.fit(w,a,k,point),(0,0))
            except (FitFail,ValueError,np.linalg.LinAlgError) as e:
                error[k][str(e)]+=1
        return out

    initial=fit_all(np.ones(len(inv)),True)
    for draw in plan:
        for k,v in fit_all(draw[inv].astype(float),False).items():
            draws[k].append(v)
    out=[]
    for k in names:
        row=dict(**head,method=k,bootstrap_attempts=B,bootstrap_valid=len(draws[k]),seed=seed,
                 detail=';'.join(f'{a}:{b}' for a,b in sorted(error[k].items())))
        if k not in initial:
            out.append(dict(**row,status='not_estimable_point'))
            continue
        vals=np.asarray(draws[k])
        row['status']='estimated' if len(vals)>=MIN_OK else 'point_only'
        for j,label in enumerate(('risk_index','risk_reference','risk_difference')):
            row[label]=initial[k][j]
            if len(vals)>=MIN_OK:
                row[label+'_lo'],row[label+'_hi']=np.quantile(vals[:,j],[.025,.975])
        out.append(row)
    return out


def run():
    verify()
    q={s:S['load']() if s=='SHARE' else O['linked'](s) for s in SOURCES}
    answers=[]; support=[]
    for comp in COMP:
        start=perf_counter()
        targets=[Target(s,q[s],comp) for s in SOURCES]
        support.extend([x for t in targets for x in t.stats])
        pd.DataFrame(support).to_csv(HERE/'SOURCE_SUPPORT.csv',index=False,encoding='utf-8-sig')
        known=[t for t in targets if t.status=='estimated']
        print(comp,'IPW sources',[t.s for t in known],'support failures',
              [(t.s,t.status) for t in targets if t not in known],flush=True)
        for tt,role in ((targets,'delta_four_sources'),(known,'ipw_supported_sources')):
            rr=pool_run(tt,comp,role)
            answers.extend(rr)
            pd.DataFrame(answers).to_csv(HERE/'RESULTS.csv',index=False,encoding='utf-8-sig')
            print(comp,role,[(r['method'],r['status'],r.get('bootstrap_valid')) for r in rr],
                  round(perf_counter()-start,1),flush=True)
    pd.DataFrame(answers).to_csv(HERE/'RESULTS.csv',index=False,encoding='utf-8-sig')
    qa=dict(spec_sha256=sha(HERE/'SPEC_CN.md'),script_sha256=sha(Path(__file__).resolve()),
            hashes=HASHES,bootstrap_attempts=B,bootstrap_minimum=MIN_OK,
            privacy='Only coarsened support and model functionals; no participant output.')
    (HERE/'INPUT_HASHES.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')


if __name__=='__main__':
    run()

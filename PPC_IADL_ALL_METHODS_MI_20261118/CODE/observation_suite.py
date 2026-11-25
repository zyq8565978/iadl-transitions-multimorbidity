"""MI-adjusted local observation assumptions; never fill absent visits/deaths."""
from common import *
import argparse,copy
from scipy.linalg import qr
ob=load('allmi_observation_models',S/'OBSERVATION/run_observation.py')

def targets(raw,comp,mi):
    out=[]
    for s in SOURCES:
        t=ob.Target(s,raw[s],comp)
        z=t.d.assign(source=s);ex,_=extras(z,mi)
        ex=ex[:,np.ptp(ex,axis=0)>1e-12]
        t.background=np.c_[t.background,ex]
        aa=np.c_[t.xa,ex]
        _,rr,pivot=qr(aa,mode='economic',pivoting=True)
        tol=max(aa.shape)*np.finfo(float).eps*max(1,abs(rr).max())
        use=np.sort(pivot[:(abs(rr.diagonal())>tol).sum()]);t.xa=aa[:,use]
        assert np.max(abs(t.xa@np.linalg.lstsq(t.xa,aa,rcond=None)[0]-aa))<1e-7
        t.a=None;t.alpha=None
        enough=all((t.obs[t.hx==h].sum()>=20 and (~t.obs)[t.hx==h].sum()>=10) for h in (0,1))
        t.status='minimum_local_support'
        if enough:
            try:
                if s!='SHARE':
                    t.alpha=ob.O['validated_binomial'](t.xa,t.obs.astype(float));t.a=ob.Newton(t.xa,t.obs.astype(float))
                t.response(np.ones(len(t.d)),collect=True);t.status='estimated'
            except (ValueError,np.linalg.LinAlgError) as e:t.status=str(e)
        out.append(t)
    return out

def run(comp):
    effect_gate();ob.verify()
    raw={s:ob.S['load']() if s=='SHARE' else ob.O['linked'](s) for s in SOURCES}
    pre=R/f'QA/OBSERVATION_{comp}_source_gate.json'
    if pre.exists():compatible=json.loads(pre.read_text())['all_MI_supported_sources']
    else:
        good=set(SOURCES);logs=[]
        for mi in range(1,51):
            ts=targets(raw,comp,mi)
            for t in ts:
                logs.append(dict(mi=mi,source=t.s,status=t.status))
                if t.status!='estimated':good.discard(t.s)
        compatible=[s for s in SOURCES if s in good]
        write(pre,dict(all_MI_supported_sources=compatible,gate_log=logs))
    for mi in range(1,51):
        ts=targets(raw,comp,mi)
        for role,tt in [('delta_four_sources',ts),('ipw_supported_sources',[t for t in ts if t.s in compatible])]:
            stem=f'OBSERVATION_{comp}_{role}';jp=R/f'QA/{stem}_mi{mi:03d}.json'
            if jp.exists():continue
            if len(tt)<2:
                write(jp,dict(status='NOT_ESTIMABLE_FEWER_THAN_TWO_SUPPORTED_SOURCES',mi=mi));continue
            try:p=ob.Pool(tt,comp)
            except (ValueError,np.linalg.LinAlgError) as e:write(jp,dict(status='NOT_ESTIMABLE',reason=str(e),mi=mi));continue
            names=list(ob.DELTAS) if role=='delta_four_sources' else ['matched_unweighted',*ob.IPW]
            pos=np.cumsum([0]+[len(t.d) for t in tt]);fail=Counter()
            def fit(w,point):
                out={}
                try:
                    plain=p.fit(w,None,'plain',point)
                    if role=='delta_four_sources':
                        for name in names:out[name]=p.evaluate(plain,ob.DELTAS[name])
                    else:out['matched_unweighted']=p.evaluate(plain,(0,0))
                except (ValueError,np.linalg.LinAlgError) as e:fail['unweighted:'+str(e)]+=1
                if role=='ipw_supported_sources':
                    try:weighted=[t.response(w[a:b]) for t,a,b in zip(tt,pos[:-1],pos[1:])]
                    except (ValueError,np.linalg.LinAlgError) as e:fail['response:'+str(e)]+=1;return out
                    for j,name in enumerate(ob.IPW):
                        try:
                            ww=np.concatenate([z[j] for z in weighted]);out[name]=p.evaluate(p.fit(w,ww,name,point),(0,0))
                        except (ValueError,np.linalg.LinAlgError) as e:fail[name+':'+str(e)]+=1
                return out
            point=fit(np.ones(len(p.y)),True)
            seed=71118150+COMPS.index(comp)*10+(role=='delta_four_sources')
            plan,inv=ob.family_plan(p.family,p.strata,seed,199);draw=np.full((199,len(names),3),np.nan)
            boot=R/f'PRIVATE/{stem}_mi{mi:03d}_bootstrap.npz';done=0
            if boot.exists():z=np.load(boot);draw=z['draws'];done=int(z['done'])
            for b in range(done,199):
                for name,v in fit(plan[b,inv].astype(float),False).items():draw[b,names.index(name)]=v
                if (b+1)%50==0 or b==198:np.savez_compressed(boot,draws=draw,done=b+1)
            for j,name in enumerate(names):
                child=stem+'_'+name;status=R/f'QA/{child}_mi{mi:03d}.json'
                if name not in point:write(status,dict(status='NOT_ESTIMABLE_POINT',mi=mi));continue
                good=draw[np.isfinite(draw[:,j]).all(1),j]
                U=np.cov(good,rowvar=False) if len(good)>=180 else np.full((3,3),np.nan)
                np.savez_compressed(R/f'PRIVATE/{child}_mi{mi:03d}.npz',q=point[name],U=U)
                write(status,dict(status='ESTIMATED' if len(good)>=180 else 'POINT_ONLY_INSUFFICIENT_BOOTSTRAP',people=p.people,windows=len(p.y),
                  families=len(np.unique(p.family)),observed_windows=int(p.obs.sum()),sources=[t.s for t in tt],bootstrap_valid=len(good),failures=dict(fail)))
            write(jp,dict(status='ATTEMPTS_COMPLETE',methods=names,failures=dict(fail)))
        if mi%5==0:print('OBSERVATION',comp,mi,flush=True)
    for role in ('delta_four_sources','ipw_supported_sources'):
        stem=f'OBSERVATION_{comp}_{role}'
        names=list(ob.DELTAS) if role=='delta_four_sources' else ['matched_unweighted',*ob.IPW]
        for name in names:pooled(stem+'_'+name,['risk_history','risk_reference','risk_difference'],['probability','probability','probability_difference'],dict(module='observation',comparison=comp,role=role,method=name))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('comparison',choices=COMPS);a=ap.parse_args();run(a.comparison)

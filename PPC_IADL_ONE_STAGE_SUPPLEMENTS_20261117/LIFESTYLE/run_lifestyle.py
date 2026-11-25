"""One-stage PPC lifestyle supplements. Original donor assets are read-only."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import runpy
import time
import warnings

for var in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[var] = '1'

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, qr, solve_triangular
from scipy.optimize import linprog
from scipy.special import expit
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
WS = ROOT.parents[1]
V4 = WS/'PPC_IADL_Dynamic_V4_20261116'
V5 = WS/'PPC_IADL_SHARE_V5_20261116'
V6 = WS/'PPC_IADL_V6_Repair_Synthesis_20261116'
OLD = runpy.run_path(str(V4/'CODE/behavior_a2_d_v4.py'))
SH = runpy.run_path(str(V5/'CODE/share_association_v5.py'))
FIRTH = runpy.run_path(str(V6/'CODE/firth_engine_v6.py'))
EST = OLD['EST']
COMP = EST['COMPARE']
SOURCES = ('CHARLS','HRS','ELSA','SHARE')
BEH = ('activity','smoking','alcohol')
BOOT = 199
SPEC = ROOT/'LIFESTYLE_SPEC_CN.md'
LOCK = ROOT/'PREFIT_LOCK.json'
PRIVATE = ROOT/'PRIVATE'
OUT = ROOT/'TABLES'
COUNT_NAMES = {'n','people','families','events','nonevents','n_activity','n_smoking','n_alcohol','n_poolsources',
               'n_behavior_0','n_behavior_1','n_before','n_confirmed_death','n_later_ppc0_1','n_later_ppc_2plus',
               'n_middle_ppc_2plus','n_middle_ppc0_1','n_valid_t2','n_missing_t2','n_unobserved_middle'}


def dump(t, name):
    p = pd.DataFrame(t)
    for c in COUNT_NAMES.intersection(p):
        if c != 'n_poolsources':
            p[c] = p[c].map(lambda x: '' if pd.isna(x) else '<10' if x<10 else f'~{int((x+5)//10)*10}')
    p.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')


def case(d):
    return dict(n=len(d),people=d.person.nunique(),families=d.cluster.nunique(),
                events=d.y.eq(1).sum(),nonevents=d.y.eq(0).sum())


def share_chain(q):
    panel=pd.read_parquet(V5/'PRIVATE/SHARE_panel.parquet',columns=[
        'person','country','wave','month','proxy','route','code_fixed','d_fixed','next_status'])
    assert not panel.duplicated(['person','wave']).any()
    dates=pd.read_csv(V5/'PRIVATE/SHARE_COUNTRY_WAVE_OPPORTUNITIES_exact.csv')
    pairs=set(map(tuple,dates.loc[dates.main_v5_country,['country','wave']].to_numpy()))
    schedules=((4,5,6,7),(5,6,7,8),(6,7,8,9))
    parts=[]
    opportunities=[]
    for a,b,c,d in schedules:
        scope=sorted(set(k for k,_ in pairs if all((k,w) in pairs for w in (a,b,c,d))))
        opportunities.append(dict(schedule=f'{a}->{b}->{c}->{d}', countries=len(scope), country_codes=';'.join(scope)))
        z=q.loc[q.window.eq(f'{a}->{b}->{c}') & q.entry_eligible & q.country.isin(scope) &
                q.route0.eq('regular') & q.route1.eq('regular')].copy()
        z['planned2']=d; z['schedule']=f'{a}->{b}->{c}->{d}'
        parts.append(z)
    e=pd.concat(parts).sort_values(['month1','wave'],kind='stable').drop_duplicates('person').reset_index(drop=True)
    assert e.person.is_unique and e.gap01.gt(0).all()
    fin=panel.rename(columns={c:('planned2' if c=='wave' else c+'_last') for c in panel if c!='person'})
    e=e.merge(fin,on=['person','planned2'],how='left',validate='many_to_one')
    e['gap23']=e.month_last-e.month2
    e['t1_status']=np.select([e.legal2 & e.route2.eq('regular') & e.d_fixed2.le(1),
                             e.legal2 & e.route2.eq('regular') & e.d_fixed2.ge(2),e.legal2],
                            ['comparable_PPC_0_1','comparable_PPC_2plus','living_invalid_PPC'],default=e.status2)
    valid=e.next_status2.eq('living_interview') & e.route_last.eq('regular') & e.proxy_last.eq(0) & e.gap23.gt(0) & e.country_last.eq(e.country)
    e['t2_status']=np.select([valid & e.d_fixed_last.notna(),e.next_status2.eq('living_interview')], ['living_interview','living_invalid_PPC'],
                             default=e.next_status2.fillna('unverified_t2_without_t1'))
    e['y']=e.d_fixed_last.ge(2).astype(float).where(valid&e.d_fixed_last.notna()&e.t1_status.eq('comparable_PPC_0_1'))
    e['p0']=e.code_fixed1; e['kgrp']=e.K1.clip(upper=3)
    for b in BEH:
        orig='drinking' if b=='alcohol' else b
        e[f'pre_{b}']=e[f'{orig}0']; e[f't1_{b}']=e[f'{orig}2']
    flds=['chronic_count0','cesd0','sex1','age1','education1','chronic_count1','cesd1','cog_z1','p0','kgrp','gap01','country']
    assert (e.loc[e.y.notna(),['gap01','gap12','gap23']]>0).all().all()
    return e, opportunities, flds


def arm_valid(d, candidate, decomp):
    if d.arm.nunique()!=2:
        return 'no_both_history_arms_in_shared_context'
    if not all(z.y.nunique()==2 for _,z in d.groupby('arm')):
        return 'outcome_constant_in_arm'
    if decomp and not all(z.mediator.nunique()==2 for _,z in d.groupby('arm')):
        return 'middle_behavior_constant_in_arm'
    if not decomp and any(d[b].nunique()<2 for b in candidate):
        return 'prior_behavior_constant'
    return 'pass'


def prep(publish):
    data_d={}; data_a={}; flow=[]; timing=[]; gate=[]
    qsh=SH['load']()
    share_e,opps,share_cols=share_chain(qsh)
    if publish:
        dump(opps,'SHARE_FOUR_WAVE_ADMINISTRATIVE_OPPORTUNITIES')
    for src in SOURCES:
        if src=='SHARE':
            full=qsh.rename(columns={'activity0':'activity','smoking0':'smoking','drinking0':'alcohol'})
            seq=share_e
        else:
            q,panel=OLD['load'](src)
            overlay=OLD['source_overlay'](src)
            seq=OLD['four_chain'](src,q,panel,overlay)
            full=q.assign(prev_wave=q.window.str.split('->').str[0].astype(int)).merge(
                overlay.rename(columns={'wave':'prev_wave'}),on=['person','prev_wave'],how='left',validate='many_to_one')
        for comp,(a,b) in COMP.items():
            ctx=['wave','p0']+(['kgrp'] if a=='H11' else [])+(['country'] if src=='SHARE' else [])
            if src=='SHARE':
                fullrisk=SH['make_stages'](full,comp)[-1]
            else:
                fullrisk=EST['stages'](full,comp,'primary')[0][-1]
            for labels,covs in [('activity_only',('activity',)),('all_three_local',BEH)]:
                cutoff=fullrisk.dropna(subset=list(covs))
                z=EST['restrict'](cutoff,ctx)
                reason=arm_valid(z,covs,False)
                for stage,dat in [('original_complete_shared',fullrisk),('prior_behavior_complete',cutoff),('matched_shared_context',z)]:
                    for arm in (a,b):
                        g=dat.loc[dat.arm.eq(arm)]
                        flow.append(dict(source=src,comparison=comp,module='A2',specification=labels,stage=stage,arm=arm,
                                         **case(g),**{f'n_{j}':int(g[j].notna().sum()) for j in BEH}))
                gate.append(dict(source=src,comparison=comp,module='A2',specification=labels,gate=reason,**case(z)))
                if reason=='pass':
                    data_a.setdefault((comp,labels),{})[src]=(fullrisk,z,covs)
            for l in BEH:
                if src=='SHARE':
                    i=seq.loc[seq.history.isin((a,b))].copy()
                    i['arm']=i.history; i['hx']=i.arm.eq(a).astype(int)
                    middle=i.loc[i.t1_status.eq('comparable_PPC_0_1')]
                    p=middle.loc[middle[[f'pre_{l}',f't1_{l}']].notna().all(axis=1)].copy()
                    if l=='alcohol':
                        p=p.loc[p.gap12.ge(3)].copy()
                    p['mediator']=p[f't1_{l}'].eq(2 if l=='smoking' else 1).astype(int)
                    z=EST['restrict'](p.dropna(subset=['y',*share_cols]),ctx)
                else:
                    i,middle,p,z=OLD['d_dataset'](src,seq,comp,l)
                for stage,dat in [('earliest_admin_chain',i),('middle_live_ppc_below_2',middle),('both_behavior_measured',p),
                                  ('final_observed_ppc',p.loc[p.y.notna()]),('complete_shared_context',z)]:
                    for arm in (a,b):
                        g=dat.loc[dat.arm.eq(arm)]
                        flow.append(dict(source=src,comparison=comp,module='D',specification=l,stage=stage,arm=arm,
                                         **case(g),n_behavior_0=g.get('mediator',pd.Series(dtype=int)).eq(0).sum(),
                                         n_behavior_1=g.get('mediator',pd.Series(dtype=int)).eq(1).sum()))
                reason=arm_valid(z,l,True)
                comparability='eligible' if l=='activity' or (l=='smoking' and src!='SHARE') else (
                    'not_pooled_different_alcohol_recall_and_frequency' if l=='alcohol' else 'not_pooled_daily_smoking_route')
                gate.append(dict(source=src,comparison=comp,module='D',specification=l,gate=reason,
                                 measurement_comparability=comparability,**case(z)))
                if reason=='pass' and comparability=='eligible':
                    data_d.setdefault((comp,l),{})[src]=z
                if l=='activity':
                    for (s,arm),g in i.groupby(['schedule','arm']):
                        timing.append(dict(source=src,comparison=comp,schedule=s,arm=arm,n=len(g),
                                           n_middle_ppc0_1=g.t1_status.eq('comparable_PPC_0_1').sum(),
                                           n_middle_ppc_2plus=g.t1_status.eq('comparable_PPC_2plus').sum(),
                                           n_unobserved_middle=(~g.t1_status.isin(['comparable_PPC_0_1','comparable_PPC_2plus'])).sum(),
                                           n_valid_t2=g.y.notna().sum(),n_missing_t2=g.y.isna().sum(),
                                           n_confirmed_death=(g.t1_status.eq('confirmed_death')|g.t2_status.eq('confirmed_death')).sum(),
                                           **{f'{c}_median':g[c].where(g[c].gt(0)).median() if g[c].gt(0).sum()>=10 else np.nan
                                              for c in ('gap01','gap12','gap23')}))
    if publish:
        dump(flow,'LIFESTYLE_SUPPORT'); dump(timing,'LIFESTYLE_CHAIN_TIMING'); dump(gate,'LIFESTYLE_PREFIT_GATES')
    return data_a,data_d


def source_design(d,src,comp,kind,extra):
    z=d.assign(gap12=0) if kind=='D' else d
    base=(SH['matrix']()(z,src,comp,'M_severity') if src=='SHARE' else EST['matrix'](z,src,comp,'M_severity'))[0].copy()
    if kind=='D':
        base['prior_chronic']=d.chronic_count0; base['prior_cesd']=d.cesd0
        past=pd.get_dummies(d[f'pre_{extra[0]}'].astype(int),prefix='prior_behavior',drop_first=True,dtype=float)
        base=pd.concat([base,past],axis=1)
    for c in extra if kind=='A2' else ():
        if c=='smoking':
            base=pd.concat([base,pd.get_dummies(d[c].astype(int),prefix='prior_smoking',drop_first=True,dtype=float)],axis=1)
        else:
            base['prior_'+c]=d[c]
    return base.loc[:,base.nunique().gt(1) | base.columns.isin(['Intercept'])]


def assemble(groups,comp,kind,extra):
    keys=list(groups)
    d=pd.concat([groups[k].assign(source=k) for k in keys],ignore_index=True)
    blocks=[source_design(groups[k],k,comp,kind,extra).drop(columns=['hx','Intercept']).add_prefix(k+':') for k in keys]
    col=['Intercept','hx']
    out=[np.ones(len(d)),d.hx.to_numpy(float)]
    for k in keys[1:]:
        out.append(d.source.eq(k).to_numpy(float)); col.append('source:'+k)
    for k,b in zip(keys,blocks):
        block=np.zeros((len(d),len(b.columns)))
        block[d.source.eq(k),:]=b.to_numpy(float)
        out.extend(block.T); col.extend(b.columns.tolist())
    x=np.ascontiguousarray(np.asarray(out).T)
    assert np.isfinite(x).all()
    return d,x,col


def columns(x,ww,force):
    z=x[ww>0]
    # Fast definitive full-rank screen; near-dependence still goes to QR.
    if len(z)>=z.shape[1]:
        sd=np.linalg.norm(z,axis=0)
        zg=z/np.maximum(sd,1e-12)
        ev=np.linalg.eigvalsh(zg.T@zg)
        if ev[0]>ev[-1]*1e-13:
            return np.arange(x.shape[1])
    _,R,p=qr(z,mode='economic',pivoting=True,check_finite=False)
    a=np.abs(R.diagonal()); k=int((a>a[0]*1e-10).sum())
    if k==x.shape[1]:
        return np.arange(k)
    # Give intercept and any shared terms precedence; verify none lose identifiability.
    nuisance=np.setdiff1d(np.arange(x.shape[1]),force)
    nz=z[:,nuisance]; fz=z[:,force]
    if len(nuisance):
        nu,Rn,_=qr(nz,mode='economic',pivoting=True,check_finite=False)
        an=np.abs(Rn.diagonal()); kn=int((an>an[0]*1e-10).sum()) if an[0]>0 else 0
        # The intercept may be duplicated by surviving source dummies; reprioritize it, not the contrasts.
        residual=fz[:,1:]-nu[:,:kn]@(nu[:,:kn].T@fz[:,1:])
        if np.linalg.matrix_rank(residual,tol=np.linalg.norm(fz)*1e-10)<len(force)-1:
            raise ValueError('redraw_shared_effect_lost')
    q0,r0=qr(fz,mode='economic',check_finite=False)
    if np.abs(r0.diagonal()).min()<=1e-10:
        raise ValueError('redraw_required_columns_rank')
    rz=nz-q0@(q0.T@nz)
    _,r1,p1=qr(rz,mode='economic',pivoting=True,check_finite=False)
    retained=np.sort(np.concatenate((np.asarray(force,dtype=int),nuisance[p1[:k-len(force)]])))
    if np.linalg.matrix_rank(z[:,retained])!=k:
        raise ValueError('redraw_relevel_span_unresolved')
    return retained


def mle_warm_exact(x,y,w,start):
    """Frequency-weight QR coordinates; same likelihood and active row span."""
    sw=np.sqrt(w); n=np.sqrt(len(y))
    q,r=qr(x*sw[:,None],mode='economic',check_finite=False)
    z=np.ascontiguousarray(q*(n/sw)[:,None]); b=r@start/n
    for i in range(32):
        e=z@b; p=expit(e); g=z.T@(w*(y-p))
        if np.max(abs(r.T@g/n))<=1e-7:
            return solve_triangular(r,b*n,check_finite=False),i,q
        info=z.T@(z*(w*p*expit(-e))[:,None])
        d=cho_solve(cho_factor(info,check_finite=False),g,check_finite=False)
        h=min(1.,4/max(4.,float(abs(d).max())))
        v=np.sum(w*(y*e-np.logaddexp(0,e)))
        for j in range(12):
            nb=b+h*d; ne=z@nb
            if np.sum(w*(y*ne-np.logaddexp(0,ne)))>=v-1e-9:
                b=nb; break
            h*=0.5
        else:
            raise ValueError('strict_mle_line_search_no_gain')
    raise ValueError('strict_mle_gradient_not_reached')


def model(x,y,w,s,typ,force,warm=None):
    take=w>0
    for k in np.unique(s):
        mask=take & (s==k)
        if np.unique(y[mask]).size!=2:
            raise ValueError('resample_source_endpoint_constant')
    choose=columns(x,w,force); z=x[:,choose]
    z0=z[take]; y0=y[take]; w0=w[take]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if typ=='MLE':
            b=None; qw=None
            if warm is not None:
                try:
                    b,it,qw=mle_warm_exact(z0,y0,w0,warm[choose])
                except (ValueError,np.linalg.LinAlgError):
                    pass
            if b is None:
                fit=sm.GLM(y0,z0,family=sm.families.Binomial(),freq_weights=w0).fit(
                    start_params=None if warm is None else warm[choose],maxiter=60,tol=1e-9)
                b=np.asarray(fit.params)
                if not fit.converged or any('separation' in str(c.message).lower() for c in caught):
                    raise ValueError('mle_separation_or_nonconvergence')
                it=fit.fit_history.get('iteration',np.nan)
        else:
            b,qa=FIRTH['fit'](z0,y0,w0,None if warm is None else warm[choose]); it=qa['iterations']
    p=expit(z0@b)
    if typ=='MLE' and (p.min()<1e-8 or p.max()>1-1e-8):
        for j in range(1,z0.shape[1]):
            u=z0[:,j]
            if np.all((u==0) | (u==1)) and np.unique(y0[u==1]).size==1:
                raise ValueError('mle_certified_pure_indicator')
        if warm is None:
            # Bounded LP certificate, not a fitting objective: common sign of
            # the fitted endpoint margin for every observation with one strict.
            margin=(2*y0-1)[:,None]*z0/np.maximum(abs(z0).max(0),1e-12)
            lint=linprog(-margin.mean(0),A_ub=-margin,b_ub=np.zeros(len(y0)),bounds=(-1,1),method='highs',
                        options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
            if lint.success:
                ev=margin@lint.x
                if ev.min()>=-1e-8 and ev.max()>1e-7 and ev.sum()>1e-6:
                    raise ValueError('mle_certified_quasi_separation_lp')
    score=float(np.max(abs(z0.T@(w0*(y0-p)))))
    scaled=z0/np.maximum(np.sqrt(np.average(z0*z0,axis=0,weights=w0)),1e-12)
    info=(scaled.T*(w0*p*(1-p)))@scaled
    eig=np.linalg.eigvalsh(info)
    condition=float(eig[-1]/eig[0]) if eig[0]>0 else np.inf
    if not np.isfinite(b).all() or not np.isfinite(p).all():
        raise ValueError('nonfinite_coefficients_or_prediction')
    if typ=='MLE':
        if qw is None:
            qw=qr(z0*np.sqrt(w0)[:,None],mode='economic',check_finite=False)[0]
        evw=np.linalg.eigvalsh((qw.T*(p*(1-p)))@qw)*len(y0)
        qr_cond=float(evw[-1]/evw[0]) if evw[0]>0 else np.inf
        if qr_cond>1e11 or evw[0]<1e-10:
            raise ValueError('mle_weak_information')
        if score>1e-5 or abs(b).max()>60:
            raise ValueError('mle_score_or_coefficient_boundary')
    else:
        qr_cond=np.nan
    allb=np.zeros(x.shape[1]); allb[choose]=b
    shift={k:(0. if typ=='MLE' else FIRTH['flic_shift']((x@allb)[take&(s==k)],y[take&(s==k)],w[take&(s==k)])) for k in np.unique(s)}
    log=dict(engine=typ,columns=len(choose),initial_design_columns=x.shape[1],iterations=it,score_max=score,scaled_information_condition=condition,
             qr_information_condition=qr_cond,
             min_prediction=float(p.min()),max_prediction=float(p.max()),active_column_count=len(choose),
             firth_adjusted_score_max_scaled=qa['adjusted_score_max_scaled'] if typ!='MLE' else np.nan)
    return allb,shift,log


def opt(x,y,w,s,force):
    try:
        return model(x,y,w,s,'MLE',force)
    except (ValueError,np.linalg.LinAlgError) as err:
        if str(err).startswith('redraw_'):
            raise
        b,a,q=model(x,y,w,s,'FIRTH_FLIC',force)
        q['ordinary_disposition']=str(err)
        return b,a,q


def eta(x,s,m):
    return x@m[0]+np.array([m[1][k] for k in s])


def rosters(d):
    src=d.source.to_numpy(str)
    cl=(d.source.astype(str)+':'+d.cluster.astype(str)).to_numpy(str)
    return src,cl


def summaries(a,w,s,names):
    out=[]
    for k in ['Pooled',*np.unique(s)]:
        ii=(s==k) if k!='Pooled' else np.ones(len(s),bool)
        out.extend((k,n,float(v)) for n,v in zip(names,np.average(a[ii],axis=0,weights=w[ii])))
    return out


def d_func(x,xy,y,m,s,w,fm,fy):
    e=eta(x,s,fm)-x[:,1]*fm[0][1]
    m0=expit(e); m1=expit(e+fm[0][1])
    o=eta(xy,s,fy)-xy[:,1]*fy[0][1]-m*fy[0][-1]
    p00=expit(o); p01=expit(o+fy[0][-1]); p10=expit(o+fy[0][1]); p11=expit(o+fy[0][1]+fy[0][-1])
    r00=p00+(p01-p00)*m0; r10=p10+(p11-p10)*m0; r11=p10+(p11-p10)*m1
    on=o+xy[:,1]*fy[0][1]
    base=[m1-m0,expit(on+fy[0][-1])-expit(on),r11-r00,r10-r00,r11-r10,r00,r10,r11]
    assert np.allclose(base[2],base[3]+base[4],rtol=0,atol=1e-12)
    return summaries(np.column_stack(base),w,s,('history_to_behavior_rd','behavior_to_ppc_rd','total_rd','direct_component_rd',
                                               'indirect_component_rd','R00','R10','R11'))


def a_func(xa,xb,s,w,fa,fb):
    ee=[eta(x,s,f)-x[:,1]*f[0][1] for x,f in ((xa,fa),(xb,fb))]
    a=[expit(t) for t in ee]; b=[expit(t+f[0][1]) for t,f in zip(ee,(fa,fb))]
    ra=b[0]-a[0]; rb=b[1]-a[1]
    return summaries(np.column_stack((a[0],b[0],ra,a[1],b[1],rb,rb-ra)),w,s,(
        'same_records_reference_risk','same_records_index_risk','same_records_rd','adjusted_reference_risk',
        'adjusted_index_risk','adjusted_rd','adjusted_minus_same_records_rd'))


def baseline_func(x,s,w,f):
    ee=eta(x,s,f)-x[:,1]*f[0][1]
    a,b=expit(ee),expit(ee+f[0][1]); return summaries(np.column_stack((a,b,b-a)),w,s,('reference_risk','index_risk','rd'))


def run_pair(mode,comp,labels,groups,seed,tag):
    rows=[]; logs=[]; fails=Counter(); key=f'{tag}{mode}_{comp}_{labels}'
    base=dict(module=mode,comparison=comp,specification=labels,sources='|'.join(groups),n_poolsources=len(groups))
    if len(groups)<2:
        return [dict(**base,status='fewer_than_2_structurally_supported_sources')],logs
    if mode=='D':
        chain=groups
        d,xa,na=assemble(chain,comp,mode,(labels,)); yy=d.y.to_numpy(float); m=d.mediator.to_numpy(float)
        xb=np.column_stack((xa,m)); yy1=m; nb=na+['SHARED_middle_behavior']
        fn=lambda s,w,fa,fb:d_func(xa,xb,yy,m,s,w,fa,fb)
    else:
        chain={src:q[1] for src,q in groups.items()}
        d,xa,na=assemble(chain,comp,mode,()); _,xb,nb=assemble(chain,comp,mode,next(iter(groups.values()))[2]); yy1=yy=d.y.to_numpy(float)
        fn=lambda s,w,fa,fb:a_func(xa,xb,s,w,fa,fb)
    sa,cl=rosters(d); w0=np.ones(len(d)); fora=[0,1]; forb=[0,1,xb.shape[1]-1] if mode=='D' else [0,1]
    try:
        fa=opt(xa,yy1,w0,sa,fora); fb=opt(xb,yy,w0,sa,forb)
        logs.extend([dict(**base,link=li,**f[2],shared_log_or=float(f[0][1])) for li,f in [('first',fa),('second',fb)]])
        if mode=='D':
            logs[-1]['common_behavior_outcome_log_or']=float(fb[0][-1])
        point=fn(sa,w0,fa,fb)
        seedplan,ids=FIRTH['family_plan'](cl,sa,seed,n=BOOT)
        draws=[]
        for g in seedplan:
            w=g[ids].astype(float)
            try:
                da=model(xa,yy1,w,sa,fa[2]['engine'],fora,fa[0]); db=model(xb,yy,w,sa,fb[2]['engine'],forb,fb[0])
                draws.append([v for _,_,v in fn(sa,w,da,db)])
            except (ValueError,np.linalg.LinAlgError,ZeroDivisionError) as e:
                fails[str(e)[:140]]+=1
        ok=len(draws)>=180
        quantile=np.quantile(draws,[.025,.975],axis=0).T if ok else np.full((len(point),2),np.nan)
        for (s,fn_name,v),(lo,hi) in zip(point,quantile):
            counts=case(d if s=='Pooled' else d.loc[d.source.eq(s)])
            rows.append(dict(**base,standardized_in=s,functional=fn_name,estimate=v,lower=lo,upper=hi,**counts,
                             status='estimated' if ok else 'point_only_bootstrap_below_90pct',
                             bootstrap_attempts=BOOT,bootstrap_success=len(draws),first_engine=fa[2]['engine'],second_engine=fb[2]['engine']))
        np.savez_compressed(PRIVATE/f'{key}_fits.npz',first_beta=fa[0],second_beta=fb[0],first_names=np.asarray(na),
                            second_names=np.asarray(nb),draws=np.asarray(draws))
    except (ValueError,np.linalg.LinAlgError) as e:
        rows.append(dict(**base,status='not_estimable',reason=str(e)[:140]))
    if mode=='A2':
        broad={src:q[0] for src,q in groups.items()}
        try:
            dr,xx,nr=assemble(broad,comp,mode,()); sr,cr=rosters(dr); yr=dr.y.to_numpy(float); wr=np.ones(len(dr))
            fr=opt(xx,yr,wr,sr,fora); pp=baseline_func(xx,sr,wr,fr)
            ss,idx=FIRTH['family_plan'](cr,sr,seed,n=BOOT)
            bst=[]
            for i in ss:
                w=i[idx].astype(float)
                try:
                    ff=model(xx,yr,w,sr,fr[2]['engine'],fora,fr[0]); bst.append([v for _,_,v in baseline_func(xx,sr,w,ff)])
                except (ValueError,np.linalg.LinAlgError,ZeroDivisionError) as e:
                    fails['full_reference:'+str(e)[:125]]+=1
            good=len(bst)>=180
            ci=np.quantile(bst,[.025,.975],axis=0).T if good else np.full((len(pp),2),np.nan)
            for (s,fn_name,v),(lo,hi) in zip(pp,ci):
                counts=case(dr if s=='Pooled' else dr.loc[dr.source.eq(s)])
                rows.append(dict(**base,standardized_in=s,functional='full_sample_'+fn_name,estimate=v,lower=lo,upper=hi,
                                 **counts,status='estimated' if good else 'point_only_bootstrap_below_90pct',
                                 bootstrap_attempts=BOOT,bootstrap_success=len(bst),first_engine=fr[2]['engine']))
            logs.append(dict(**base,link='full_sample_reference',**fr[2],shared_log_or=float(fr[0][1])))
            np.savez_compressed(PRIVATE/f'{key}_broad_ref.npz',beta=fr[0],names=np.asarray(nr),draws=np.asarray(bst))
        except (ValueError,np.linalg.LinAlgError) as e:
            rows.append(dict(**base,functional='full_sample_reference',status='not_estimable',reason=str(e)[:140]))
    if fails:
        logs.extend(dict(**base,link='failed_redraw',reason=k,failed_redraws=v) for k,v in fails.items())
    return rows,logs


def file_hash(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main(mode,only,tag):
    start=time.monotonic(); OUT.mkdir(exist_ok=True); PRIVATE.mkdir(exist_ok=True)
    sig=file_hash(SPEC)
    if mode!='prefit' and json.loads(LOCK.read_text())['spec_sha256']!=sig:
        raise RuntimeError('Spec changed after prefit')
    A,D=prep(mode=='prefit')
    if mode=='prefit':
        LOCK.write_text(json.dumps(dict(spec_sha256=sig,behavior_branches='study sponsor supplied',
                                        design_gate_sha256=file_hash(OUT/'LIFESTYLE_PREFIT_GATES.csv'))),encoding='utf-8')
        return
    if json.loads(LOCK.read_text())['design_gate_sha256']!=file_hash(OUT/'LIFESTYLE_PREFIT_GATES.csv'):
        raise RuntimeError('Source support changed after prefit')
    out=[]; diagnostics=[]
    for ci,comp in enumerate(COMP):
        if only and only!=comp:
            continue
        options=(('activity_only','all_three_local') if mode=='A2' else BEH)
        for ix,k in enumerate(options):
            if mode=='D' and k=='alcohol':
                out.append(dict(module=mode,comparison=comp,specification=k,status='not_pooled_due_to_recall_mismatch'))
            else:
                r,l=run_pair(mode,comp,k,(A if mode=='A2' else D).get((comp,k),{}),20261117+100*ci+ix,tag)
                out.extend(r); diagnostics.extend(l)
            print(mode,comp,k,sorted(set(x['status'] for x in out if x['comparison']==comp and x['specification']==k)),flush=True)
            dump(out,f'{tag}ONE_STAGE_{mode}_RESULTS'); dump(diagnostics,f'{tag}ONE_STAGE_{mode}_DIAGNOSTICS')
    (OUT/f'{tag}ONE_STAGE_{mode}_RECEIPT.json').write_text(json.dumps(dict(branch=mode,compare_only=only,elapsed_seconds=round(time.monotonic()-start,2),
        spec_sha256=sig,n_records_output=len(out),n_requested_bootstrap=BOOT),indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prefit','A2','D'))
    p.add_argument('--only',choices=tuple(COMP))
    p.add_argument('--boundary-review',action='store_true')
    args=p.parse_args()
    if args.only and not args.boundary_review:
        p.error('Use --boundary-review for a separately stored contrast-only run')
    tag='BOUNDARY_REVIEW_' if args.boundary_review else ''
    if args.only=='H10_vs_H00':
        tag+='SECONDARY_'
    main(args.mode,args.only,tag)

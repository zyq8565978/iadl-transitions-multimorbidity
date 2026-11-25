"""Read frozen living observations; join revised native death evidence locally."""
from importlib.util import spec_from_file_location, module_from_spec
import numpy as np
import pandas as pd
from panel_v6_engine import R, V5

SPEC=spec_from_file_location('frozen_v5_panel',V5/'CODE/panel_state.py')
v5=module_from_spec(SPEC)
SPEC.loader.exec_module(v5)
PRIVATE=R/'PRIVATE'
SOURCES=v5.SOURCES
ARMS=v5.ARMS
DAYS=365.2425

def safe_n(x):
    return '0' if x==0 else '<10' if x<10 else str(int(10*((x+5)//10)))

def output(name,source,comp,a):
    dest=R/'TABLES'
    dest.mkdir(parents=True,exist_ok=True)
    a.to_csv(dest/f'PANEL_V6_{name}_{source}_{comp}.csv',index=False,encoding='utf-8-sig')

def frozen_inputs(s):
    return [V5/'PRIVATE/PANEL'/f'{s}_{k}.parquet' for k in ('baseline','observations')]

def module_inputs(s):
    return [PRIVATE/f'death_V6_{s}.parquet']

def input_data(s,c):
    b,o=map(pd.read_parquet,frozen_inputs(s))
    for z in (b,o):
        z['person']=z.person.astype(str); z['wave']=z.wave.astype(str)
    b['country']=b.country.astype(str); b['cluster']=b.cluster.astype(str)
    assert b.person.is_unique and o.person.isin(b.person).all()
    a=v5.fit_reference(b,s,c)
    assert a.person.is_unique
    d=pd.read_parquet(module_inputs(s)[0])
    d['person']=d.person.astype(str)
    assert d.person.is_unique
    return a, o[o.person.isin(a.person)].copy(), d

def date_from_month(n, day=15):
    n=np.asarray(n,float)
    m=pd.Series(pd.NaT,index=range(len(n)),dtype='datetime64[ns]')
    occurs=np.isfinite(n)
    nn=np.rint(n[occurs]).astype(int)-1
    m.loc[occurs]=pd.to_datetime(dict(year=nn//12,month=nn%12+1,day=day)).to_numpy()
    return m.to_numpy()

def anchors(a,o):
    idx=pd.Series(a.index,index=a.person)
    x=o.loc[o.status.isin(['A','B','alive_unknown']) & o.time_years.gt(0) & o.time_years.le(6)].copy()
    x['pid']=x.person.map(idx).astype(int)
    x['idx_month']=a.idx_calendar.to_numpy()[x.pid]
    x['status_date']=date_from_month(x.idx_month+x.time_years*12)
    x['alive_floor']=date_from_month(x.idx_month+x.time_years*12,day=1)
    x['time_years']=((x.status_date-date_from_month(x.idx_month)).dt.days/DAYS)
    x['state']=x.status.map(v5.ST).astype(int)
    assert x.status_date.notna().all()
    clash=x.groupby(['pid','status_date']).state.transform('nunique').gt(1)
    if clash.any():
        raise ValueError('Non-identical real interview states on one month anchor')
    x=x.sort_values(['pid','time_years']).drop_duplicates(['pid','time_years'])
    return x

def death_and_sequences(a,o,d):
    """D known only from an exact event or a bounded interval; alive union is a constraint, not a visit."""
    x=anchors(a,o)
    t=a[['person','idx_calendar']].merge(d,on='person',how='left',validate='one_to_one')
    t['anchor']=date_from_month(t.idx_calendar)
    t['baseline_alive_floor']=date_from_month(t.idx_calendar,day=1)
    t=t.join(x.groupby('pid').agg(last_modeled_live=('status_date','max'),last_real_alive_floor=('alive_floor','max')))
    t['last_actual_alive_bound']=t[['last_alive_date','last_real_alive_floor','baseline_alive_floor']].max(axis=1)
    t['last_model_alive']=t[['last_alive_date','last_modeled_live','anchor']].max(axis=1)
    t['source_conflict']=t.conflict.eq(True)
    e=t.death_exact_date.notna()
    has_upper=~e & t.death_hi_date.notna()
    t['overlap_exact_with_live_anchor']=e & t.death_exact_date.le(t.last_model_alive)
    t['overlap_interval_with_live_anchor']=has_upper & t.death_hi_date.le(t.last_model_alive)
    t['valid_exact']=t.death_confirmed.eq(True)&e&~t.source_conflict&~t.overlap_exact_with_live_anchor
    t['admissible_upper']=has_upper & ~t.overlap_interval_with_live_anchor & ~t.source_conflict & t.death_confirmed.eq(True)
    t['analysis_lo']=t[['death_lo_date','last_model_alive']].max(axis=1)
    t['calibration_lo']=t[['death_lo_date','last_actual_alive_bound']].max(axis=1)
    t['finite_interval']=t.admissible_upper & t.death_hi_date.gt(t.analysis_lo)
    t['ambiguous_fit']=t.overlap_exact_with_live_anchor | t.overlap_interval_with_live_anchor
    rows=[]
    use_living=x[['pid','time_years','state','occasion']].copy()
    use_living['kind']='actual_living_interview'
    for q in t.itertuples():
        latest=q.last_modeled_live if pd.notna(q.last_modeled_live) else q.anchor
        exact=float((q.death_exact_date-q.anchor).days/DAYS) if q.valid_exact else None
        high=float((q.death_hi_date-q.anchor).days/DAYS) if q.finite_interval else None
        # Survived to the later of the independent last-alive bound and, if valid,
        # the lower death bound. An exact later death also proves survival to 6.
        last=q.last_model_alive
        if q.finite_interval:
            last=q.analysis_lo
        if q.valid_exact and exact>6:
            last=q.death_exact_date
        alv=min(6.,float((last-q.anchor).days/DAYS))
        last_visit=float((latest-q.anchor).days/DAYS)
        if alv>last_visit+1e-10:
            rows.append((q.Index,alv,99,np.nan,'known_alive_bound_not_visit'))
        if q.valid_exact and 0<exact<=6:
            rows.append((q.Index,exact,4,np.nan,'exact_death_density'))
        elif q.finite_interval and 0<high<=6:
            rows.append((q.Index,high,3,np.nan,'interval_death_upper'))
    extra=pd.DataFrame(rows,columns=use_living.columns)
    r=pd.concat([use_living,extra],ignore_index=True) if len(extra) else use_living.copy()
    r=r.sort_values(['pid','time_years'],kind='stable').reset_index(drop=True)
    r['dt']=r.time_years-r.groupby('pid',sort=False).time_years.shift().fillna(0)
    assert r.dt.gt(0).all(), 'ILLEGAL_SEQUENCE' 
    r['origin_exact']=r.groupby('pid',sort=False).state.shift().fillna(1).astype(int)
    offsets=np.r_[0,np.cumsum(np.bincount(r.pid,minlength=len(a)))].astype(np.int32)
    return t,r,offsets,x

def end_month_bounds(a,o,s):
    """Conservative group-level administrative end, never the person's actual last visit."""
    # SHARE official-origin grid and all three other V5 source schedules supply original
    # opportunity. Across the other sources, use the minimum of the calendar opportunity
    # already assigned at index and any independently supplied documented wave end.
    cap=a.admin_horizon_years.to_numpy(float)
    f=PRIVATE/f'panel_admin_V6_{s}.csv'
    if f.exists():
        z=pd.read_csv(f,dtype={'country':str})
        assert z.country.is_unique and set(z.columns)>={'country','administrative_end'}
        limit=pd.to_datetime(a.country.map(z.set_index('country').administrative_end))
        real=(limit-pd.Series(date_from_month(a.idx_calendar))).dt.days/DAYS
        cap=np.minimum(cap,real.fillna(0.))
    return cap

def support_masks(a,o,x,s):
    masks={};rows=[]
    cap=end_month_bounds(a,o,s)
    a0=a.assign(admin_horizon_years=cap)
    for T in (2,4):
        good=[]
        for wave in a0.wave.unique():
            admin=a0.wave.eq(wave)&a0.admin_horizon_years.ge(T)
            if s=='CHARLS' and T==4 and str(wave).replace('.0','')=='2018': admin &= False
            wrows=[]
            for h in (0,1):
                sel=admin&a0.h.eq(h)
                z=x.loc[x.pid.isin(a.index[sel])&x.state.isin([1,2])]
                near=z.loc[z.time_years.between(T-1.5,T+1.5)]
                low=bool(near.time_years.le(T).any()); high=bool(near.time_years.ge(T).any())
                wrows.append(dict(year=T,index_wave=wave,h=h,baseline_n=int(sel.sum()),
                                  neighborhood_measured_n=near.pid.nunique(),bracket_below=low,bracket_above=high))
            yes=all(r['neighborhood_measured_n']>=10 and r['bracket_below'] and r['bracket_above'] for r in wrows)
            if yes: good.append(wave)
            for d in wrows:
                d['has_legal_2_sided_wave_support']=yes
                for k in ('baseline_n','neighborhood_measured_n'): d[k]=safe_n(d[k])
                rows.append(d)
        masks[T]=(a0.admin_horizon_years.ge(T)&a0.wave.isin(good)).to_numpy()
    return masks,pd.DataFrame(rows)

def survival_at(t,T):
    """Return model-independent known-dead/known-alive states at a fixed elapsed cutoff."""
    cutoff=t.anchor+pd.to_timedelta(T*DAYS,unit='D')
    am=t.death_confirmed.eq(True)&~t.source_conflict
    exact=am&t.death_exact_date.notna()&t.death_exact_date.gt(t.last_actual_alive_bound)
    interval=am&t.death_hi_date.notna()&t.death_hi_date.gt(t.last_actual_alive_bound)
    known_D=(exact&t.death_exact_date.le(cutoff)) | (interval&t.death_hi_date.le(cutoff))
    known_L=t.last_actual_alive_bound.ge(cutoff) | (exact&t.death_exact_date.gt(cutoff)) | (
      am&t.calibration_lo.ge(cutoff)&t.death_lo_date.notna()&(
          t.death_hi_date.isna() | t.death_lo_date.lt(t.death_hi_date)))
    if (known_D&known_L).any():
        c=known_D&known_L
        known_D &= ~c; known_L &= ~c
    return np.where(known_D,1,np.where(known_L,0,-1))

def compatible_interval_ci(y,clusters):
    """Descriptive moment margin for a fixed eligibility frame; clusters not people are resampled units."""
    y=np.asarray(y,float); m=y.mean()
    if len(y)<2:return m,np.nan
    u=pd.Series(y-m).groupby(np.asarray(clusters)).sum().to_numpy()
    d=len(u)
    return m,float(1.96*np.sqrt((u@u)*d/max(1,d-1))/len(y))

def calibration_survival(a,t,masks,pred=None,occupancy=False):
    plans=[(T,T,mask) for T,mask in masks.items()]
    if occupancy: plans.extend((T,4,masks[4]) for T in [1,2,3,4])
    rows=[]; point={}
    g=(a.country+' / '+a.cluster)
    for T,target,mask in plans:
        status=survival_at(t,T)
        pass_arms=[]
        for h in (0,1):
            sel=mask&a.h.eq(h).to_numpy()
            status_ar=status[sel]; n=len(status_ar)
            k0=int((status_ar==0).sum()); k1=int((status_ar==1).sum()); U=n-k0-k1
            enough=n>=50 and U/n<=.10 if n else False
            q=float(np.average(pred[T][sel,2])) if pred is not None and n else np.nan
            l,u=(k1/n,(k1+U)/n) if n else (np.nan,np.nan)
            good=np.nan; margin_l=margin_u=np.nan
            if n and pred is not None:
                _,margin_l=compatible_interval_ci((status_ar==1)-pred[T][sel,2],g[sel].to_numpy())
                _,margin_u=compatible_interval_ci((status_ar!=0)-pred[T][sel,2],g[sel].to_numpy())
                good=bool(q>=l-margin_l and q<=u+margin_u)
            pass_arms.append(enough and (bool(good) if pred is not None else True))
            # Precise reported bounds must not reveal unreleased small complementary cells.
            disclose=n>=50 and all(k==0 or k>=10 for k in (k0,k1,U))
            obs=lambda v:round(float(v),4) if disclose and pd.notna(v) else np.nan
            rows.append(dict(target=f'constant_{target}_year_eligible',
                             cutoff_year=T,history=('index' if h else 'reference'),
                             N=safe_n(n),known_alive=safe_n(k0),known_death=safe_n(k1),unknown=safe_n(U),
                             lower=obs(l),upper=obs(u),margin_low=obs(margin_l),margin_high=obs(margin_u),
                             fitted_expected_death=obs(q),ascertainment_width_at_most_10pp=enough,
                             compatible_bounds=good,
                             calibration_role='partial_identification_not_independent_external_validation'))
        point[T,target]=all(pass_arms)
    return pd.DataFrame(rows),point

def fit_info(a,t,r):
    temp=[]
    for h in (0,1):
        who=a.index[a.h.eq(h)]
        z=r.loc[r.pid.isin(who)].copy()
        # Bounds at L/U never displace actual observed reversals.
        aobs=z.loc[z.kind.eq('actual_living_interview')&z.state.isin([1,2])].copy()
        aobs['start']=aobs.groupby('pid').state.shift().fillna(1)
        nAB=int((aobs.start.eq(1)&aobs.state.eq(2)).sum())
        nBA=int((aobs.start.eq(2)&aobs.state.eq(1)).sum())
        tt=t.loc[who]
        nD=int(z.state.isin([3,4]).sum())
        row=dict(history=('index' if h else 'reference'),N=len(who),real_AB=nAB,real_BA=nBA,
                 exact_likelihood_rows=int(z.state.eq(4).sum()),interval_likelihood_rows=int(z.state.eq(3).sum()),
                 known_alive_bounds=int(z.kind.eq('known_alive_bound_not_visit').sum()),confirmed_death=int(tt.death_confirmed.eq(True).sum()),
                 conflicting_date=int(tt.source_conflict.sum()),ambiguous_anchor_overlap=int(tt.ambiguous_fit.sum()),
                 no_dated_death_or_outside_fit=int(tt.death_confirmed.eq(True).sum())-nD)
        temp.append(row)
    ready=all(q['real_AB']>=1 and q['real_BA']>=1 and q['exact_likelihood_rows']+q['interval_likelihood_rows']>=2 for q in temp)
    ready=bool(ready and sum(q['real_BA'] for q in temp)>=8)
    return pd.DataFrame([{k:safe_n(v) if k!='history' else v for k,v in z.items()} for z in temp]),ready

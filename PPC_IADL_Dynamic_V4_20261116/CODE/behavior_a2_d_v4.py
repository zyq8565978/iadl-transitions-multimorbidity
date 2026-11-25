"""Prespecified V4 previous-visit behavior block and four-visit paths."""
import os
for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[var] = '1'
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import runpy
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT.parent
V2 = WS/'PPC_IADL_Dynamic_V2_20261115'
V3 = WS/'PPC_IADL_Dynamic_V3_20261116'
CN = WS/'PPC_IADL_CHARLS_Equated_Update_20261116'/'EDUCATION3_DIAGNOSTIC'
PRIV = ROOT/'PRIVATE'/'BEHAVIOR'
TAB = ROOT/'TABLES'
EST = runpy.run_path(str(V2/'CODE'/'fit_ppc.py'))
STATUS = runpy.run_path(str(V2/'CODE'/'prepare_v2.py'))['status']
EST['fit_one'].__globals__['PRIVATE'] = PRIV
BASE_MAT = EST['matrix']
DOMAINS = dict(zip(range(8), [0, 1, 1, 2, 1, 2, 2, 3]))
D_SCHEDULE = {
    'CHARLS': [(2013,2015,2018,2020)],
    'HRS': [(2012,2014,2016,2018),(2014,2016,2018,2020)],
    'ELSA': [(1,2,3,4),(2,3,4,5),(6,7,8,9)],
}
BEHAV = ('activity', 'smoking', 'alcohol')
RAW = WS/'原始数据'
SOURCE = {
    'CHARLS': (RAW/'01_CHARLS_中国'/'02_协调与简化数据'/'harmonized charls'/'H_CHARLS_D_Data.dta',
               'ID', {2013:2,2015:3,2018:4}),
    'HRS': (RAW/'03_HRS_美国'/'02_协调与简化数据'/'RAND HRS Longitudinal File 2020'/'randhrs1992_2020v2.dta',
            'hhidpn', {2012:11,2014:12,2016:13,2018:14,2020:15}),
    'ELSA': (RAW/'02_ELSA_英国'/'02_协调与简化数据'/'Harmonized ELSA'/'h_elsa_g3.dta',
             'idauniq', {w:w for w in range(1,10)}),
}
SEED = 20261116
B = 199


def activity(v, m):
    return pd.Series(np.select([v.eq(1)|m.eq(1), v.eq(0)&m.eq(0)], [1.,0.], default=np.nan),index=v.index)


def source_overlay(src):
    fn,key,waves = SOURCE[src]
    act_v, act_m = {'CHARLS':('vgact_c','mdact_c'), 'HRS':('vgactx','mdactx'), 'ELSA':('vgactx_e','mdactx_e')}[src]
    drink = {'CHARLS':'drinkn_c','ELSA':'drinkd_e','HRS':'drinkd'}[src]
    cols = [f'r{w}{name}' for w in waves.values() for name in (act_v,act_m,'smokev','smoken',drink)
            if not(src=='ELSA' and w==1 and name==drink)]
    raw = pd.read_stata(fn,columns=[key]+cols,convert_categoricals=False)
    if src=='CHARLS':
        ids = raw[key].astype(str).str.strip()
        person = ids.where(ids.str.len().ne(11), ids.str[:-2]+'0'+ids.str[-2:])
    else:
        person = raw[key].astype('Int64').astype(str)
    assert person.notna().all() and not person.duplicated().any()
    values = []
    for wave,w in waves.items():
        p = f'r{w}'
        v,m = raw[p+act_v],raw[p+act_m]
        if src!='CHARLS':
            v,m = [s.le(3).astype(float).where(s.isin(range(1 if src=='HRS' else 2,6))) for s in (v,m)]
        current,ever = raw[p+'smoken'],raw[p+'smokev']
        smoking = np.select([current.eq(1), current.eq(0)&ever.eq(1), current.eq(0)&ever.eq(0)],
                            [2.,1.,0.], default=np.nan)
        d = raw[p+drink] if p+drink in raw else pd.Series(np.nan,index=raw.index)
        alcohol = d.ge(4 if src=='CHARLS' else 1).astype(float).where(d.isin(range(10 if src=='CHARLS' else 8)))
        values.append(pd.DataFrame(dict(person=person,wave=wave,activity=activity(v,m),smoking=smoking,alcohol=alcohol)))
    return pd.concat(values,ignore_index=True)


def load(src):
    p=(CN if src=='CHARLS' else V2)/'PRIVATE'
    panel = pd.read_parquet(p/f'{src}_panel.parquet')
    q = pd.read_parquet(p/f'{src}_windows.parquet')
    for z in (q,panel):
        z['person'] = z.person.astype(str)
    return q,panel


def linkage(src,panel,overlay):
    z=panel[['person','wave','proxy']].merge(overlay,how='left',on=['person','wave'],validate='one_to_one',indicator=True)
    rows=[]
    for wave,g in z.groupby('wave'):
        if wave not in SOURCE[src][2]:continue
        for b in BEHAV:
            rows.append(dict(source=src,wave=wave,field=b,panel_records=len(g),id_linked=int(g._merge.eq('both').sum()),
                             measure_known=int(g[b].notna().sum()),measure_unknown=int(g[b].isna().sum()),
                             positive=int(g[b].eq(2 if b=='smoking' else 1).sum()),
                             mode='ELSA W1 alcohol frequency structurally unavailable' if src=='ELSA' and wave==1 and b=='alcohol'
                             else ('prior PA random half module; missing retained' if src=='CHARLS' and wave in (2013,2015) and b=='activity' else 'as documented')))
    return rows


def mat_a2(d,src,comparison,layer):
    ds,formula,const = BASE_MAT(d,src,comparison,'M_severity')
    if layer=='M_severity_behavior':
        add = pd.get_dummies(d['smoking'].astype(int),prefix='smoking',drop_first=True,dtype=float)
        for b in ('activity','alcohol'):
            if d[b].nunique()>1:add[b] = d[b]
            else:const.append(b)
        ds = pd.concat([ds,add],axis=1)
        formula += ' + C(prior_smoking) + prior_activity + prior_alcohol_frequency'
    return ds,formula,const


def support(z):
    return dict(n=len(z),people=z.person.nunique(),families=z.cluster.nunique(),events=z.y.eq(1).sum(),nonevents=z.y.eq(0).sum())


def a2(src,q,overlay):
    allres,counts=[],[]
    prev = overlay.rename(columns={'wave':'prev_wave'})
    q['prev_wave'] = q.window.str.split('->').str[0].astype(int)
    w = q.merge(prev,on=['person','prev_wave'],how='left',validate='many_to_one')
    orig = pd.read_csv(V3/'TABLES'/f'PPC_RESULTS_{src}.csv')
    spec_col = 'specification' if 'specification' in orig else 'scenario'
    for comp in EST['COMPARE']:
        stages,ctx=EST['stages'](w,comp,'primary')
        full = stages[-1]
        cc=full.dropna(subset=list(BEHAV))
        same=EST['restrict'](cc,ctx)
        assert set(same.index).issubset(full.index) and not same[list(BEHAV)].isna().any().any()
        for desc,z in [('v3_base',full),('complete_behavior',cc),('same_sample',same)]:
            for arm,gg in z.groupby('arm'):
                counts.append(dict(source=src,comparison=comp,stage=desc,history=arm,**support(gg),
                                   **{f'missing_{b}':int(gg[b].isna().sum()) for b in BEHAV},
                                   pairs_structural_alcohol=int((gg.prev_wave.eq(1)&(src=='ELSA')).sum())))
        ref = orig.loc[orig.comparison.eq(comp)&orig[spec_col].eq('primary')&orig.model_layer.eq('M_severity')]
        if len(ref)!=1:raise ValueError(f'{src} {comp}: V3 result key not unique ({len(ref)})')
        keys=['risk_index','risk_reference','risk_difference','risk_difference_lo','risk_difference_hi','odds_ratio','odds_ratio_lo','odds_ratio_hi']
        ans=[]
        for spec,layer,z in [('V3_full_reference','M_severity',full),('A2_same_records_no_behavior','M_severity',same),
                             ('A2_same_records_add_behavior','M_severity_behavior',same)]:
            row=dict(source=src,comparison=comp,analysis=spec,**support(z))
            try:
                if spec=='V3_full_reference':
                    qv=ref.iloc[0]
                    if 'windows' in qv and len(full)!=qv['windows']:raise AssertionError('V3 sample count does not match')
                    r=qv.to_dict()
                else:
                    EST['fit_one'].__globals__['matrix'] = mat_a2
                    r,_,_= EST['fit_one'](z,src,comp,spec,layer)
                row.update(status='estimated',**{k:r[k] for k in keys},rank=r.get('rank'),information_condition=r.get('information_condition'))
            except (ValueError,np.linalg.LinAlgError) as e:
                row.update(status='not_estimable',reason=str(e))
            ans.append(row)
        for row,other in zip(ans[1:],ans):
            if 'risk_difference' in row and 'risk_difference' in other:
                row['risk_difference_change_from_preceding_specification']=row['risk_difference']-other['risk_difference']
        allres.extend(ans)
    return allres,counts


def four_chain(src,q,panel,overlay):
    windows=[]
    for a,b,c,d in D_SCHEDULE[src]:
        triple = f'{a}->{b}->{c}'
        z=q.loc[q.window.eq(triple)&q.entry_eligible].copy()
        assert len(z) and not z.person.duplicated().any()
        z['planned2']=d
        z['schedule']=f'{triple}->{d}'
        z['prev_wave']=a
        windows.append(z)
    e=pd.concat(windows).sort_values(['month1','wave'],kind='stable').drop_duplicates('person').reset_index(drop=True)
    assert not e.person.duplicated().any() and e.gap01.gt(0).all()
    p2=panel[['person','wave','month','code_fixed']].rename(columns={'wave':'planned2','month':'month3','code_fixed':'code_fixed3'})
    p2['seen3']=True
    e=e.merge(p2,on=['person','planned2'],how='left',validate='many_to_one')
    for endpoint,w in [('pre','prev_wave'),('t1','next_wave')]:
        z=overlay.rename(columns={'wave':w,**{b:f'{endpoint}_{b}' for b in BEHAV}})
        e=e.merge(z,on=['person',w],how='left',validate='many_to_one')
    e['gap23']=e.month3-e.month2
    e['t1_status'] = np.select([e.legal2&e.d_fixed2.ge(2),e.legal2&e.d_fixed2.le(1),e.legal2],
                             ['comparable_PPC_2plus','comparable_PPC_0_1','living_invalid_PPC'],
                             default=e.status2)
    e['t2_status']=STATUS(e.next_status2,src)
    if (e.t2_status.eq('confirmed_death')&e.seen3.eq(True)&e.legal2).any():
        raise AssertionError(f'{src}: subsequent respondent marked dead')
    e.loc[e.seen3.eq(True)&~e.gap23.gt(0),'t2_status']='interview_date_invalid'
    e.loc[e.seen3.eq(True)&e.gap23.gt(0),'t2_status']='living_interview'
    d2=e.code_fixed3.map(DOMAINS)
    e.loc[e.t2_status.eq('living_interview')&d2.isna(),'t2_status']='living_invalid_PPC'
    e['y']=d2.ge(2).astype(float).where(e.t1_status.eq('comparable_PPC_0_1')&e.t2_status.eq('living_interview'))
    e['p0']=e.code_fixed1
    e['kgrp']=e.K1.clip(upper=3)
    return e


def d_dataset(src,e,comp,b):
    a,ref=EST['COMPARE'][comp]
    initial=e.loc[e.history.isin([a,ref])].copy()
    initial['arm']=initial.history
    initial['hx']=initial.arm.eq(a).astype(int)
    l = initial.loc[initial.t1_status.eq('comparable_PPC_0_1')].copy()
    min_months={'CHARLS':12,'HRS':3,'ELSA':2}[src] if b=='alcohol' else 0
    p=l.loc[l.gap12.ge(min_months)&l[[f'pre_{b}',f't1_{b}']].notna().all(axis=1)].copy()
    p['mediator']=p[f't1_{b}'].eq(2 if b=='smoking' else 1).astype(int)
    d=p.dropna(subset=['y','chronic_count0','cesd0','sex1','age1','education1','chronic_count1','cesd1','cog_z1','p0','kgrp','gap01'])
    ctx=['wave','p0','kgrp'] if comp=='H11_vs_H01' else ['wave','p0']
    d=EST['restrict'](d,ctx)
    return initial,l,p,d


def d_fit(x,y,w):
    if len(np.unique(y[w>0]))<2:raise ValueError('one outcome level after resample')
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter('always')
        m=sm.GLM(y,x,family=sm.families.Binomial(),freq_weights=w).fit(maxiter=55,tol=1e-9)
    p=np.asarray(m.fittedvalues)
    information=x.T@(x*(w*p*(1-p))[:,None])
    if not m.converged or not np.isfinite(m.params).all() or any('separation' in str(i.message).lower() for i in warns):
        raise ValueError('separation or nonconvergence')
    if np.linalg.eigvalsh(information)[0]<=1e-9 or np.linalg.cond(information)>1e11 or abs(x.T@(w*(y-p))).max()>1e-5:
        raise ValueError('weak information or score')
    return m.params


def paths(d,src,comp,b):
    if d.empty or d.arm.nunique()!=2:raise ValueError('no shared support')
    for arm,z in d.groupby('arm'):
        if z.y.nunique()!=2 or z.mediator.nunique()!=2:raise ValueError('outcome or behavior constant in an arm')
    ds,formula,constants=BASE_MAT(d.assign(gap12=0),src,comp,'M_severity')
    ds['prior_chronic']=d.chronic_count0
    ds['prior_cesd']=d.cesd0
    add=pd.get_dummies(d[f'pre_{b}'].astype(int),prefix='pre',drop_first=True,dtype=float)
    ds=pd.concat([ds,add],axis=1)
    ds=ds.loc[:,(ds.nunique()>1)|ds.columns.to_series().eq('Intercept')]
    j=ds.columns.get_loc('hx')
    x=ds.to_numpy(float)
    y,m=d.y.to_numpy(float),d.mediator.to_numpy(float)
    xy=np.column_stack((x,m))
    if np.linalg.matrix_rank(xy)!=xy.shape[1]:raise ValueError('joint design rank')
    z0,z1=x.copy(),x.copy()
    z0[:,j]=0
    z1[:,j]=1
    codes,clusters=pd.factorize(d.cluster,sort=True)
    g=len(clusters)
    if g<2:raise ValueError('family resampling impossible')
    def calculate(w):
        bm=d_fit(x,m,w)
        by=d_fit(xy,y,w)
        m0,m1=expit(z0@bm),expit(z1@bm)
        p00,p01=expit(z0@by[:-1]),expit(z0@by[:-1]+by[-1])
        p10,p11=expit(z1@by[:-1]),expit(z1@by[:-1]+by[-1])
        r00,r10,r11=[np.average(a,weights=w) for a in (p00+(p01-p00)*m0,p10+(p11-p10)*m0,p10+(p11-p10)*m1)]
        return np.array([np.average(m1-m0,weights=w),np.average(expit(x@by[:-1]+by[-1])-expit(x@by[:-1]),weights=w),
                         r11-r00,r10-r00,r11-r10,r00,r10,r11])
    point=calculate(np.ones(len(d)))
    assert np.isclose(point[2],point[3]+point[4])
    rng=np.random.default_rng(SEED)
    prob=np.full(g,1/g)
    boot=[]
    for _ in range(B):
        try:boot.append(calculate(rng.multinomial(g,prob)[codes]))
        except (ValueError,np.linalg.LinAlgError):pass
    labels=['history_to_behavior_rd','behavior_to_ppc_rd','total_rd','direct_rd','indirect_rd','R00','R10','R11']
    out=dict(status='estimated' if len(boot)>=.90*B else 'point_only_bootstrap_below_90pct',
             bootstrap_attempts=B,bootstrap_success=len(boot),design_terms=';'.join(ds.columns),spec_formula=formula.replace(' + g12',''))
    for i,k in enumerate(labels):
        out[k]=point[i]
        if len(boot)>=.90*B:out[k+'_lo'],out[k+'_hi']=np.quantile(np.asarray(boot)[:,i],[.025,.975])
    return out


def d_analysis(src,e,fit_effects=True):
    res,flow,timing=[],[],[]
    for sched,z in e.groupby('schedule'):
        for field in ('gap01','gap12','gap23'):
            x=z.loc[z[field].gt(0),field].dropna()
            timing.append(dict(source=src,schedule=sched,interval=field,n=len(x),median=x.median(),q1=x.quantile(.25),q3=x.quantile(.75)))
    for comp in EST['COMPARE']:
        for b in BEHAV:
            initial,l,p,d=d_dataset(src,e,comp,b)
            if b=='activity':
                for h in EST['COMPARE'][comp]:
                    sub=initial.loc[initial.arm.eq(h)]
                    for st,g in sub.groupby('t1_status'):
                        flow.append(dict(source=src,comparison=comp,history=h,stage='t1',status=st,n=len(g)))
                    for st,g in sub.loc[sub.t1_status.eq('comparable_PPC_0_1')].groupby('t2_status'):
                        flow.append(dict(source=src,comparison=comp,history=h,stage='t2_among_t1_lt2',status=st,n=len(g)))
                    for schedule,g in sub.groupby('schedule'):
                        flow.append(dict(source=src,comparison=comp,history=h,stage='initial_schedule',status=schedule,n=len(g)))
            for h in EST['COMPARE'][comp]:
                for stage,z in [('t1_lt2',l),('known_behavior_pre_and_t1',p),('complete_shared',d)]:
                    gg=z.loc[z.arm.eq(h)]
                    flow.append(dict(source=src,comparison=comp,candidate=b,history=h,stage=stage,status='included',
                                     **support(gg), t1_behavior_positive=(gg.mediator.eq(1).sum() if 'mediator' in gg else np.nan),
                                     pre_behavior_positive=int(gg[f'pre_{b}'].eq(2 if b=='smoking' else 1).sum()),
                                     pre_behavior_known=int(gg[f'pre_{b}'].notna().sum())))
            row=dict(source=src,comparison=comp,candidate=b,**support(d))
            if fit_effects:
                try:
                    row.update(paths(d,src,comp,b))
                except (ValueError,np.linalg.LinAlgError) as ex:
                    row.update(status='not_estimable',reason=str(ex))
            res.append(row)
    return res,flow,timing


COUNT = {'n','people','families','events','nonevents','measure_known','measure_unknown','positive','panel_records','id_linked',
         'missing_activity','missing_smoking','missing_alcohol','pairs_structural_alcohol','t1_behavior_positive',
         'pre_behavior_positive','pre_behavior_known'}


def round_counts(x):
    if pd.isna(x):return ''
    if x==0:return '0'
    return '<10' if x<10 else f'~{int(np.floor((x+5)/10)*10)}'


def output(items,stem):
    d=pd.DataFrame(items)
    d.to_csv(PRIV/(stem+'_exact_local.csv'),index=False,encoding='utf-8-sig')
    for k in set(d)&COUNT:
        d[k]=d[k].map(round_counts)
    d.to_csv(TAB/(stem+'.csv'),index=False,encoding='utf-8-sig')


def main():
    PRIV.mkdir(parents=True,exist_ok=True)
    fit_effects='--structural' not in sys.argv
    lock=ROOT/'V4_PROTOCOL_LOCK.json'
    spec=ROOT/'REPORTS'/'A2_D_PRE_EFFECT_SPEC_CN.md'
    prereq={'global_lock_sha256':hashlib.sha256(lock.read_bytes()).hexdigest(),
            'behavior_spec_sha256':hashlib.sha256(spec.read_bytes()).hexdigest(),
            'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'utc_checked_before_fit':datetime.now(timezone.utc).isoformat()}
    if fit_effects:
        assert prereq['global_lock_sha256']=='1a991a6020a643788e1a3f1a90a4042cb51797d76d19303c719ebff7f16e14b2'
        (ROOT/'QA'/'D_BEHAVIOR_EXECUTION_LOCK.json').write_text(json.dumps(prereq,indent=2),encoding='utf-8')
    for src in SOURCE:
        q,panel=load(src)
        overlay=source_overlay(src)
        output(linkage(src,panel,overlay),'A2_D_SOURCE_LINK_'+src)
        e=four_chain(src,q,panel,overlay)
        if fit_effects:
            a,ac=a2(src,q,overlay)
            output(a,'A2_RESULTS_'+src)
            output(ac,'A2_SUPPORT_'+src)
        dd,df,dt=d_analysis(src,e,fit_effects)
        if fit_effects:output(dd,'D_RESULTS_'+src)
        output(df,'D_SUPPORT_'+src)
        output(dt,'D_TIMING_'+src)
        print(src, 'completed' if fit_effects else 'structural counts complete',flush=True)


if __name__=='__main__':
    main()

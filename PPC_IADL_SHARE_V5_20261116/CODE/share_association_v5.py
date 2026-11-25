"""Frozen SHARE next-visit comparisons and finite explanation bridges."""
import argparse
import hashlib
import json
import os
import runpy
from datetime import datetime, timezone
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import pandas as pd
from scipy.special import expit
from statsmodels.tools.sm_exceptions import PerfectSeparationError

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT.parent/'PPC_IADL_Dynamic_V2_20261115'
V3 = ROOT.parent/'PPC_IADL_Dynamic_V3_20261116'
EXACT = ROOT/'PRIVATE'/'SHARE_ASSOC'
TABLE = ROOT/'TABLES'
EST = runpy.run_path(str(V2/'CODE/fit_ppc.py'))
V3_HELPERS = runpy.run_path(str(V3/'CODE/ppc_v3.py'))
COMPARE, FIELDS = EST['COMPARE'], EST['FIELDS']
BASE_MAT, FIT = EST['matrix'], EST['fit_one']
OBS_FIT = V3_HELPERS['validated_binomial']
DELTA = V3_HELPERS['DELTA']
BOOT = 199
BASE_INPUT = ROOT/'PRIVATE'/'SHARE_windows.parquet'
FAILURE = (ValueError, ArithmeticError, np.linalg.LinAlgError, PerfectSeparationError)
BEHAV = ('activity0','smoking0','drinking0')


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def verify(expected):
    p = ROOT/'SHARE_ANALYSIS_LOCK.json'
    raw = p.read_bytes()
    if not expected or hashlib.sha256(raw).hexdigest()!=expected:
        raise RuntimeError('SHARE lock absent or SHA-256 not verified')
    lock = json.loads(raw.decode('utf-8-sig'))
    def leaves(v):
        if isinstance(v, dict):
            return [item for x in v.values() for item in leaves(x)]
        if isinstance(v, list):
            return [item for x in v for item in leaves(x)]
        return [v]
    hashes = set(str(v) for v in leaves(lock))
    for f in (ROOT/'SHARE_MODEL_SPEC.md',BASE_INPUT,ROOT/'PRIVATE'/'SHARE_panel.parquet'):
        if digest(f) not in hashes:
            raise RuntimeError('Frozen specification/input digest absent or unmatched')


def ctx(comparison):
    return ['country','wave','p0']+(['kgrp'] if comparison=='H11_vs_H01' else [])


def matrix(extra=()):
    def make(d, src, comp, layer):
        design, formula, constants = BASE_MAT(d, src, comp, layer)
        for v in ('country',)+tuple(extra):
            if d[v].nunique()<=1:
                constants.append(v)
                continue
            if v in ('country','smoking0'):
                cat = pd.get_dummies(d[v].astype(str), prefix=v, drop_first=True, dtype=float)
                design = pd.concat([design,cat],axis=1)
                formula += '+ C('+v+')'
            else:
                design[v] = d[v].astype(float)
                formula += ' + '+v
        return design,formula,constants
    return make


def fitcounts(d):
    stat = dict(**EST['count'](d))
    for h,prefix in ((1,'index'),(0,'reference')):
        a=d.loc[d.hx.eq(h)]
        for key,c in [('windows',len(a)),('people',a.person.nunique()),
                      ('events',a.y.eq(1).sum()),('nonevents',a.y.eq(0).sum()),
                      ('unknown_endpoint',a.y.isna().sum())]:
            stat[prefix+'_'+key]=int(c)
    return stat


def make_stages(q, comp):
    a,b=COMPARE[comp]
    ini=q.loc[q.entry_eligible & q.history.isin((a,b))].copy()
    ini['arm'],ini['hx']=ini.history,ini.history.eq(a).astype(int)
    ini['y'],ini['p0'],ini['kgrp']=ini.y_fixed,ini.code_fixed1,ini.K1.clip(upper=3)
    obs=ini.loc[ini.y.notna()]
    cc=obs.dropna(subset=FIELDS+['country','p0','kgrp','cluster'])
    out=EST['restrict'](cc,ctx(comp))
    assert (out.gap01.gt(0)&out.gap12.gt(0)).all()
    assert ini.loc[ini.y.notna(),'legal2'].all()
    return ini,obs,cc,out


def result(d,comp,spec,layer,extra=(),outcome='PPC'):
    a,b=COMPARE[comp]
    row=dict(source='SHARE',comparison=comp,specification=spec,model_layer=layer,
             index_arm=a,reference_arm=b,
             role='primary_contrast' if comp=='H11_vs_H01' else 'key_secondary_contrast',
             outcome=outcome,**fitcounts(d))
    try:
        if len(d)<4 or d.hx.nunique()!=2:
            raise ValueError('No common background with both histories')
        cells=[row[f'{a}_{b}'] for a in ('index','reference') for b in ('events','nonevents')]
        if (min(cells)<10 and outcome!='PPC'):
            raise ValueError('Sparse subgroup endpoint or non-event in an arm')
        FIT.__globals__['PRIVATE']=EXACT
        FIT.__globals__['matrix']=matrix(extra)
        est,_,_=FIT(d,'SHARE',comp,spec,layer)
        row.update(status='estimated',**est)
        row['common_background']='country × index wave × initial PPC type; plus current difficulty count category for H11/H01'
        row['gap12_months_median']=d.gap12.median()
        row['gap12_months_q1'],row['gap12_months_q3']=d.gap12.quantile([.25,.75])
        row['country_count']=int(d.country.nunique())
        row['included_index_waves']=';'.join(map(str,sorted(d.wave.unique())))
    except FAILURE as ex:
        detail = str(ex) if isinstance(ex,(ValueError,EST['DiagnosticFailure'])) else type(ex).__name__
        # Retain only aggregate diagnostics or fixed error descriptors; do not dump frames/model calls.
        if '\n' in detail or len(detail)>280:
            detail=type(ex).__name__
        row.update(status='not_estimable',detail=detail)
        if isinstance(ex, EST['DiagnosticFailure']):
            row.update({k:v for k,v in ex.diagnostics.items() if k!='warnings'})
    print('SHARE',comp,spec,layer,row['status'],flush=True)
    return row


def coarse(n):
    return '<10' if n<10 else str(int((n+5)//10)*10)


def save_rows(rows, name, primary=False):
    d=pd.DataFrame(rows)
    d.to_csv(EXACT/(name+'_exact.csv'),index=False,encoding='utf-8-sig')
    if 'warnings' in d:
        d=d.drop(columns=['warnings'])
    for i,r in d.iterrows():
        ec=[r.get(k,100) for k in ('index_events','index_nonevents','reference_events','reference_nonevents')]
        protected=min(ec)<10
        for col in d.columns:
            if col in ('windows','people','families','events','nonevents','unknown_endpoint',
                      'target_windows','target_people','observed_windows','incomplete_windows') or (
                    col.startswith(('index_','reference_')) and col.split('_',1)[1] in
                    ('windows','people','events','nonevents','unknown_endpoint')):
                if pd.notna(r[col]) and (not primary or protected or col not in ('windows','people','families')):
                    d[col]=d[col].astype(object)
                    d.at[i,col]=coarse(r[col])
        if protected and r.get('status')=='estimated':
            mask=[c for c in d if c.startswith(('risk_','log_or','odds_ratio','nominal_p'))]
            d.loc[i,mask]=np.nan
            d.at[i,'status']='estimated_small_cell_not_released'
    d.to_csv(TABLE/(name+'.csv'),index=False,encoding='utf-8-sig')


def load():
    q=pd.read_parquet(BASE_INPUT).reset_index(drop=True)
    assert not q.duplicated(['person','window']).any()
    assert q.loc[q.entry_eligible,'d_fixed1'].isin([0,1]).all()
    assert q.loc[q.entry_eligible,['K0','K1']].notna().all().all()
    assert q.loc[q.y_fixed.notna(),'legal2'].all()
    return q


def primary(q, lock_sha):
    results=[]; support=[]
    for comp in COMPARE:
        stages=make_stages(q,comp)
        for d,step in zip(stages,('eligible','observed','complete','shared')):
            support.append(dict(source='SHARE',contrast=comp,stage=step,**fitcounts(d)))
        for layer in ('M_basic','M_severity'):
            results.append(result(stages[-1],comp,'primary',layer))
    save_rows(results,'SHARE_PPC_PRIMARY',primary=True)
    save_rows(support,'SHARE_PPC_MODEL_FLOW')
    record=dict(share_lock_sha256=lock_sha,finished_utc=datetime.now(timezone.utc).isoformat(),
                n_expected=4,n_attempted=len(results),n_estimated=sum(x['status']=='estimated' for x in results),
                rows_sha256=digest(TABLE/'SHARE_PPC_PRIMARY.csv'),
                statuses=[{k:r[k] for k in ('comparison','model_layer','status')} for r in results])
    (ROOT/'QA'/'SHARE_PRIMARY_GATE.json').write_text(json.dumps(record,indent=2),encoding='utf-8')


def pair_support(d,alt,cs):
    keep=d.index.intersection(alt.index)
    while len(keep):
        new=EST['restrict'](d.loc[keep],cs).index.intersection(EST['restrict'](alt.loc[keep],cs).index)
        if new.equals(keep):
            break
        keep=new
    return d.loc[keep],alt.loc[keep]


def sensitivity(q):
    results=[]; audits=[]; prior_flow=[]
    date_format=2020*12+3
    for comp in COMPARE:
        ini,_,cc,main=make_stages(q,comp); cs=ctx(comp)
        specs=[
            ('age50',cc.loc[cc.age1.ge(50)],()),
            ('earliest',cc.loc[cc.first_eligible],()),
            ('pre_COVID_calendar_2020_03',cc.loc[cc.month1.lt(date_format)&cc.month2.lt(date_format)],()),
            ('two_index_prev_PPC_below_2',cc.loc[cc.d_fixed0.lt(2)],()),
        ]
        if 'adl_any1' in cc.columns:
            z=cc.loc[cc.adl_any1.notna()]
            specs.extend([('ADL_same_records_reference',z,()),('ADL_add_same_records',z,('adl_any1',))])
        else:
            audits.append(dict(comparison=comp,branch='ADL',status='required_measure_absent'))
        for name,fields in (('previous_health',('cesd0','cog_z0')),('previous_PPC',('prior_ppc',))):
            g=cc.assign(prior_ppc=cc.d_fixed0.ge(2).astype(float).where(cc.d_fixed0.notna()))
            z=g.dropna(subset=list(fields))
            specs.extend([(name+'_same_records_reference',z,()),(name+'_add_same_records',z,fields)])
        if all(c in cc for c in BEHAV):
            z=cc.dropna(subset=list(BEHAV))
            specs.extend([('prior_behavior_same_records_reference',z,()),
                          ('prior_behavior_add_same_records',z,BEHAV)])
            audits.append(dict(comparison=comp,branch='behavior_block',status='checked_source_mapping',windows=len(z),
                               people=z.person.nunique()))
        else:
            audits.append(dict(comparison=comp,branch='behavior_block',status='required_measure_absent'))
        for spec,z,fields in specs:
            results.append(result(EST['restrict'](z,cs),comp,spec,'M_severity',fields))
        if all(c in q for c in ('d_country1','d_country2','code_country1','cog_z_country1')):
            both=cc.loc[cc.d_country1.le(1)&cc.d_country2.notna()&cc.code_country1.notna()&
                        cc.cog_z_country1.notna()]
            alt=both.assign(y=both.d_country2.ge(2).astype(float),p0=both.code_country1,
                            cog_z1=both.cog_z_country1)
            d,alt=pair_support(both,alt,cs)
            assert d.index.equals(alt.index)
            for spec,g in [('W5_reference_global_same_records',d),('W5_reference_country_same_records',alt)]:
                results.append(result(g,comp,spec,'M_severity'))
        else:
            audits.append(dict(comparison=comp,branch='country_reference',status='required_measure_absent'))
        po=EST['restrict'](cc.loc[cc.p0.eq(1)],cs)
        for bit,path in ((2,'subsequent_psychological'),(4,'subsequent_cognitive')):
            row=po.assign(y=(po.code_fixed2.astype(int)&bit!=0).astype(float))
            results.append(result(row,comp,'physical_only_'+path,'M_severity',outcome=path))
        e=ini.copy()
        e['prior']=np.select([e.d_fixed0.lt(2),e.d_fixed0.ge(2)],['observed_0_1','observed_2plus'],'unknown')
        for (arm,prior),g in e.groupby(['arm','prior'],observed=True):
            prior_flow.append(dict(source='SHARE',comparison=comp,arm=arm,prior_PPC=prior,
                                   windows=len(g),people=g.person.nunique(),families=g.cluster.nunique(),
                                   events=g.y.eq(1).sum(),nonevents=g.y.eq(0).sum(),unknown_endpoint=g.y.isna().sum()))
    save_rows(results,'SHARE_PPC_EXPLANATORY')
    save_rows(audits,'SHARE_PPC_EXPLANATORY_SUPPORT')
    save_rows(prior_flow,'SHARE_PPC_PREVIOUS_STATUS')


def select_target(q,comp):
    ini,_,_,_=make_stages(q,comp)
    # This flag must be built from valid next interview/known route opportunity, not from y.
    if 'selection_opportunity2' not in ini.columns:
        return ini.iloc[:0]
    d=ini.loc[ini.legal2&ini.selection_opportunity2.eq(True)].dropna(
        subset=FIELDS+['p0','kgrp','country','cluster']).copy()
    eligible=EST['restrict'](d.loc[d.y.notna()],ctx(comp))
    return d.loc[pd.MultiIndex.from_frame(d[ctx(comp)]).isin(pd.MultiIndex.from_frame(eligible[ctx(comp)]))].copy()


def local_selection(d,comp):
    hd=dict(source='SHARE',comparison=comp,target_windows=len(d),target_people=d.person.nunique(),
            observed_windows=int(d.y.notna().sum()),incomplete_windows=int(d.y.isna().sum()),
            families=d.cluster.nunique(),target='verified compatible actual next interviews, shared within country')
    if d.empty:
        return [dict(**hd,method='local_selection',status='not_estimable_no_verified_compatible_target')],[]
    design,_,_=matrix()(d,'SHARE',comp,'M_severity')
    r=d.y.notna().to_numpy(); hx=d.hx.to_numpy()
    if any(sum(r&(hx==a))<20 or sum(~r&(hx==a))<10 for a in (0,1)):
        return [dict(**hd,method='local_selection',status='not_estimable_insufficient_complete_or_incomplete_cells')],[]
    keep=design.loc[r].nunique().gt(1)|design.columns.isin(['Intercept'])
    if any(design.loc[:,~keep].nunique().gt(1)):
        return [dict(**hd,method='local_selection',status='not_estimable_unrepresented_compatible_target_level')],[]
    design=design.loc[:,keep]; j=design.columns.get_loc('hx')
    xa=design.drop(columns='g12',errors='ignore').to_numpy(float)
    x=design.to_numpy(float); y=d.y.to_numpy(float)
    x1=x.copy(); x1[:,j]=1
    x0=x.copy(); x0[:,j]=0
    fam,codes=np.unique(d.cluster.astype(str),return_inverse=True)
    reg=d.groupby('cluster').country.nunique()
    if not reg.eq(1).all():
        return [dict(**hd,method='local_selection',status='not_estimable_cross_country_family_partitions')],[]
    country=d.drop_duplicates('cluster').assign(cluster=lambda z:z.cluster.astype(str)).set_index('cluster').loc[fam,'country']
    strata=[np.where(country.to_numpy()==c)[0] for c in country.unique()]
    if len(fam)<2:
        return [dict(**hd,method='local_selection',status='not_estimable_insufficient_families')],[]
    def projected(b,w,da=0,d0=0):
        return np.array([np.average(expit(x1@b+da*~r),weights=w),
                         np.average(expit(x0@b+d0*~r),weights=w)])
    one=np.ones(len(d))
    b=OBS_FIT(x[r],y[r]); points={k:projected(b,one,*dd) for k,dd in DELTA.items()}
    weights=[]; reason=''
    try:
        p=expit(xa@OBS_FIT(xa,r.astype(float)))
        ranges=[]
        for a in (0,1):
            s=hx==a; w=1/p[r&s]; ess=w.sum()**2/(w@w)
            ranges.append((p[s].min(),p[s].max()))
            weights.append(dict(comparison=comp,group='index' if a else 'reference',
                                observed_windows=sum(r&s),incomplete_windows=sum(~r&s),
                                min_probability=p[s].min(),max_probability=p[s].max(),ess=ess))
        if min(p)<.05 or any(a['ess']<.5*a['observed_windows'] for a in weights) or max(a[0] for a in ranges)>min(a[1] for a in ranges):
            reason='Frozen probability, common range or relative effective sample gate failed'
        else:
            points['IPW']=projected(OBS_FIT(x[r],y[r],var=1/p[r]),one)
    except FAILURE:
        reason='Availability or weighted outcome model failed numeric checks'
    sims={k:[] for k in points}
    rng=np.random.default_rng(20261117+list(COMPARE).index(comp))
    for _ in range(BOOT):
        fl=np.zeros(len(fam),dtype=int)
        for ss in strata:
            fl += np.bincount(rng.choice(ss,len(ss),replace=True),minlength=len(fam))
        wt=fl[codes]
        try:
            bb=OBS_FIT(x[r],y[r],freq=wt[r])
        except FAILURE:
            continue
        for k,dd in DELTA.items():
            sims[k].append(projected(bb,wt,*dd))
        if 'IPW' in points:
            try:
                pp=expit(xa@OBS_FIT(xa,r.astype(float),freq=wt))
                if pp[wt>0].min()<.05:
                    continue
                bw=OBS_FIT(x[r],y[r],freq=wt[r],var=1/np.maximum(pp[r],1e-100))
                sims['IPW'].append(projected(bw,wt))
            except FAILURE:
                continue
    rows=[]
    for key,pt in points.items():
        m=len(sims[key]); lo=hi=np.nan
        if m>=.9*BOOT:
            sims_r=np.asarray(sims[key]); lo,hi=np.quantile(sims_r[:,0]-sims_r[:,1],[.025,.975])
        rows.append(dict(**hd,method=key,status='estimated' if m>=.9*BOOT else 'point_only_bootstrap_failed',
                         boots_attempted=BOOT,boots_success=m,risk_index=pt[0],risk_reference=pt[1],
                         risk_difference=pt[0]-pt[1],risk_difference_lo=lo,risk_difference_hi=hi))
    if 'IPW' not in points:
        rows.append(dict(**hd,method='IPW',status='not_estimable',detail=reason))
    return rows,weights


def selection(q):
    out=[]; diag=[]
    for comp in COMPARE:
        d=select_target(q,comp)
        try:
            r,w=local_selection(d,comp)
            out.extend(r);diag.extend(w)
        except FAILURE:
            out.append(dict(source='SHARE',comparison=comp,method='local_selection',status='not_estimable_numeric',
                            target_windows=len(d),observed_windows=int(d.y.notna().sum())))
        print('SHARE',comp,'selection complete',flush=True)
    save_rows(out,'SHARE_PPC_SELECTION')
    if diag:
        save_rows(diag,'SHARE_PPC_SELECTION_WEIGHTS')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--lock-sha256',required=True)
    ap.add_argument('stage',choices=['primary','sensitivity','selection'])
    args=ap.parse_args()
    verify(args.lock_sha256)
    EXACT.mkdir(parents=True,exist_ok=True)
    panel_windows=load()
    if args.stage=='primary':
        primary(panel_windows,args.lock_sha256)
    else:
        gate=ROOT/'QA'/'SHARE_PRIMARY_GATE.json'
        if json.loads(gate.read_text(encoding='utf-8'))['share_lock_sha256']!=args.lock_sha256:
            raise RuntimeError('Primary result dependency not frozen')
        {'sensitivity':sensitivity,'selection':selection}[args.stage](panel_windows)

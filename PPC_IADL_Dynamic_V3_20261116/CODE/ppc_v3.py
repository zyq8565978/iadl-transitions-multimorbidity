"""V3 PPC, targeted measurement qualification, pathways, and observation sensitivity."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import runpy
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT.parent/'PPC_IADL_Dynamic_V2_20261115'
EQ = ROOT.parent/'PPC_IADL_CHARLS_Equated_Update_20261116'
OLD_EST = V2/'CODE/fit_ppc.py'
SOURCES = ('CHARLS', 'HRS', 'ELSA')
TAB = ROOT/'TABLES'
PR = ROOT/'PRIVATE'/'PPC'
QA = ROOT/'QA'
PPC_CONTRACT = ROOT/'PPC_BRANCH_CONTRACT_CN.md'
BOOT = 199
L = np.log(1.5)
DELTA = {'delta0': (0.,0.), 'both_missing_OR1p5': (L,L),
         'H1_OR1p5_H0_OR2over3': (L,-L), 'H1_OR2over3_H0_OR1p5': (-L,L)}
EST = runpy.run_path(str(OLD_EST))
FIT = EST['fit_one']
FIT.__globals__['PRIVATE'] = PR
COMPARE = EST['COMPARE']
FIELDS = EST['FIELDS']
FAILS = (ValueError, np.linalg.LinAlgError, sm.tools.sm_exceptions.PerfectSeparationError)

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def sources(src):
    directory = (EQ/'EDUCATION3_DIAGNOSTIC' if src=='CHARLS' else V2)/'PRIVATE'
    return (directory/f'{src}_windows.parquet', directory/f'{src}_panel.parquet')

def lock_check(src):
    path=QA/'PPC_BRANCH_LOCK.json'
    payload={'contract_sha256': digest(PPC_CONTRACT), 'inherited_estimator_sha256': digest(OLD_EST),
             'inputs': {s: {x.name:digest(x) for x in sources(s)} for s in SOURCES},
             'frozen_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
             'bootstrap_attempts': BOOT, 'delta': DELTA,
             'numpy':np.__version__, 'pandas':pd.__version__, 'statsmodels':sm.__version__}
    if path.exists():
        fixed=json.loads(path.read_text(encoding='utf-8'))
        for k in ('contract_sha256','inherited_estimator_sha256','inputs'):
            assert payload[k]==fixed[k], f'Locked authority changed: {k}'
    else:
        path.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    return payload['inputs'][src]

def release_n(v):
    return '<10' if v<10 else str(int((v+5)//10)*10)

def save(rows, src, name, count_cols=(), redact=None):
    d=pd.DataFrame(rows)
    if d.empty and name=='OBS_WEIGHT_DIAGNOSTICS':
        d=pd.DataFrame(columns=['source','comparison','group','target_windows','observed_windows',
            'min_probability','p01','p50','p99','max_probability','minimum_weight','p95_weight','maximum_weight','ESS'])
    d.to_csv(PR/f'{name}_{src}_exact.csv',encoding='utf-8-sig',index=False)
    if redact is not None:
        mask=redact(d)
        for c in [x for x in d if x not in {'source','comparison','specification','model_layer','role','status','outcome','detail',*count_cols}]:
            d.loc[mask,c]=None
        d.loc[mask & d.status.eq('estimated'),'status']='not_released_small_cell'
    for col in count_cols:
        if col in d:
            d[col]=d[col].map(lambda n: np.nan if pd.isna(n) else release_n(n))
    d.to_csv(TAB/f'{name}_{src}.csv',encoding='utf-8-sig',index=False)

def linked(src):
    w,p=sources(src)
    q=pd.read_parquet(w)
    cols=['person','wave','physical','psychological_v2','cog_fixed','cog_reference_comparable']
    if src=='HRS':
        cols+=['cognition_mode','c3_route','cognitive_orientation_design_not_asked']
    u=pd.read_parquet(p,columns=cols)
    assert not q.duplicated(['person','window']).any()
    assert not u.duplicated(['person','wave']).any()
    u=u.rename(columns={'wave':'next_wave'})
    ix=q.index.copy()
    q=q.merge(u,how='left',on=['person','next_wave'],validate='many_to_one',sort=False)
    assert q.index.equals(ix)
    cmp=q.legal2 & q.y_fixed.notna()
    assert (q.loc[cmp,'code_fixed2']==q.loc[cmp,'physical']+2*q.loc[cmp,'psychological_v2']+4*q.loc[cmp,'cog_fixed']).all()
    if src=='CHARLS':
        e=q.education1_raw
        mapped=e.map({1:0,2:0,3:0,4:1,5:2,6:2,7:2,8:2,9:2,10:2,11:2})
        assert mapped.equals(q.education1)
    adl=ROOT/'PRIVATE'/f'{src}_adl.parquet'
    if adl.exists():
        overlay=pd.read_parquet(adl)
        assert not overlay.duplicated(['person','wave']).any()
        q=q.merge(overlay[['person','wave','adl_any','adl_count','adl_items']],how='left',on=['person','wave'],validate='many_to_one',sort=False)
    else:
        q[['adl_any','adl_count','adl_items']]=np.nan
    assert q.index.equals(ix)
    return q

def reason(q,src):
    s=q.status2.copy()
    living=q.legal2
    missing=living & q.y_fixed.isna()
    s.loc[living & ~missing]='observed_PPC'
    s.loc[missing]='interview_missing_domain_or_item_reason_unresolved'
    s.loc[missing & q.cog_reference_comparable.eq(False)]='planned_noncomparable_cognitive_measurement'
    if src=='HRS':
        s.loc[missing & q.c3_route.eq('structural_skip')]='known_orientation_structural_skip'
        s.loc[missing & q.cognition_mode.eq('web')]='known_web_mode_not_comparable'
        s.loc[missing & q.cognition_mode.eq('unknown')]='unknown_cognitive_mode'
        s.loc[missing & q.c3_route.eq('imputed_value_excluded')]='imputed_score_excluded'
    s.loc[missing & q.proxy2.eq(1)]='proxy_interview_excludes_direct_cognition'
    s.loc[missing & q.proxy2.isna()]='interview_respondent_route_unknown'
    assert s.notna().all()
    return s

def baseline(q,src):
    e=q.loc[q.entry_eligible].copy()
    e['availability']=np.where(e.y_fixed.notna(),'observed','unobserved')
    e['reason']=reason(e,src)
    e['gap12']=e.gap12.where(e.legal2)
    rows=[]; flow=[]; domains=[]
    for (wave,history),g in e.groupby(['wave','history'],observed=True):
        den=len(g)
        for status,n in g.reason.value_counts().items():
            flow.append(dict(source=src,index_wave=wave,history=history,status=status,n=n,N=den,
                             percentage=100*n/den if n>=10 and den-n>=10 else np.nan))
        legal=g.loc[g.legal2]
        for domain,var in [('physical','physical'),('psychological','psychological_v2'),('cognitive','cog_fixed')]:
            n=int(legal[var].isna().sum());denj=len(legal)
            domains.append(dict(source=src,index_wave=wave,history=history,domain=domain,
                                n=n,N=denj,percentage=100*n/denj if n>=10 and denj-n>=10 else np.nan))
    for (h,a),g in pd.concat([e,e.assign(availability='all')]).groupby(['history','availability'],observed=True):
        stub=dict(source=src,history=h,availability=a,N=len(g))
        for c,label in {'age1':'age_years','K1':'IADL_count','chronic_count1':'disease_count',
                        'cesd1':'CES_D_score','cog_z1':'mean_subtest_Z','gap01':'previous_to_index_months',
                        'gap12':'index_to_next_actual_months_legal_interviews_only',
                        'adl_count':'ADL_count_on_verified_items','adl_items':'count_of_ADL_items'}.items():
            val=g[c].dropna()
            rows.append(dict(**stub,measure=label,category='continuous',n=len(val),missing=len(g)-len(val),
                             mean=val.mean() if len(val)>=10 else np.nan,SD=val.std() if len(val)>=10 else np.nan,
                             median=val.median() if len(val)>=10 else np.nan,
                             p25=val.quantile(.25) if len(val)>=10 else np.nan,p75=val.quantile(.75) if len(val)>=10 else np.nan))
        for c,label in {'sex1':'sex_native','education1':'education_native_or_CHARLS_V3',
                        'code_fixed1':'PPC_index_bitcode','proxy1':'proxy_index',
                        'adl_any':'any_ADL_difficulty'}.items():
            z=g[c].astype('string').fillna('missing')
            for cat,n in z.value_counts().items():
                rows.append(dict(**stub,measure=label,category=cat,n=n,
                                 percentage=100*n/len(g) if n>=10 and len(g)-n>=10 else np.nan))
    save(rows,src,'BASELINE_HISTORY',('n','N','missing'))
    save(flow,src,'OBS_FLOW',('n','N'))
    save(domains,src,'OBS_INTERVIEW_MISSING_DOMAINS',('n','N'))
    if src=='HRS':
        e['index_qualification']=np.where(e.age1.ge(65),'index_age_65_plus','index_age_below_65')
        rows=[]
        for (a,h),g in e.groupby(['index_qualification','history'],observed=True):
            for status,n in g.reason.value_counts().items():
                rows.append(dict(source=src,index_qualification=a,history=h,status=status,n=n,N=len(g),
                                 percentage=100*n/len(g) if min(n,len(g)-n)>=10 else np.nan))
        save(rows,src,'OBS_HRS_ORIENTATION_AGE_ROUTE',('n','N'))

def create_fit_rows(d,src,comp,spec,layers,outcome='PPC'):
    for layer in layers:
        out=dict(source=src,comparison=comp,specification=spec,model_layer=layer,outcome=outcome,
                 role='primary_contrast' if comp=='H11_vs_H01' else 'key_secondary_contrast',
                 **EST['count'](d))
        for arm,pfx in zip(COMPARE[comp],('index','reference')):
            a=d.loc[d.arm.eq(arm)]
            out.update({f'{pfx}_windows':len(a),f'{pfx}_events':int(a.y.eq(1).sum()),
                        f'{pfx}_nonevents':int(a.y.eq(0).sum()),f'{pfx}_raw_risk':a.y.mean()})
        try:
            if d.empty or d.arm.nunique()!=2:
                raise ValueError('No shared background with both histories')
            if outcome!='PPC' and min(out[k] for k in ('index_events','reference_events','index_nonevents','reference_nonevents'))<10:
                raise ValueError('Insufficient path events or nonevents in at least one arm')
            result,_,_=FIT(d,src,comp,spec,layer)
            out.update(status='estimated',**result)
        except FAILS as ex:
            out.update(status='not_estimable',detail=str(ex))
            if hasattr(ex,'diagnostics'):
                out.update(ex.diagnostics)
        print(src,comp,spec,layer,out['status'],flush=True)
        yield out

BASE_MATRIX=EST['matrix']
def extend_matrix(fields):
    def func(d,src,comp,layer):
        x,formula,const=BASE_MATRIX(d,src,comp,layer)
        for f in fields:
            if d[f].nunique()>1:
                x[f]=d[f]
                formula+=' + '+f
            else:
                const.append(f)
        return x,formula,const
    return func

def paths(q,src):
    states={0:'no_domain',1:'physical_only',2:'psychological_only',3:'physical_psychological',
            4:'cognitive_only',5:'physical_cognitive',6:'psychological_cognitive',7:'three_domains'}
    rows=[]; fixed=[]
    e=q.loc[q.entry_eligible].copy()
    e['end_state']=reason(e,src)
    e.loc[e.y_fixed.notna(),'end_state']=e.loc[e.y_fixed.notna(),'code_fixed2'].map(states)
    for (p,h),g in e.groupby(['code_fixed1','history'],observed=True):
        for state,n in g.end_state.value_counts().items():
            nobs=int(g.y_fixed.notna().sum())
            obs=state in states.values()
            rows.append(dict(source=src,baseline_state=states[p],history=h,next_state=state,
                             n=n,N=len(g),observed_N=nobs,
                             percentage_all=100*n/len(g) if n>=10 and len(g)-n>=10 else np.nan,
                             percentage_observed=100*n/nobs if obs and n>=10 and nobs-n>=10 else np.nan))
    for comp in COMPARE:
        stages,context=EST['stages'](q,comp,'primary')
        d=EST['restrict'](stages[2].loc[stages[2].p0.eq(1)],context)
        prev_mask=d.index
        for endpoint,bit in [('new_psychological_includes_both',2),('new_cognitive_includes_both',4)]:
            d=d.copy()
            assert d.index.equals(prev_mask)
            d['y']=((d.code_fixed2.astype(int) & bit)!=0).astype(float)
            fixed+=list(create_fit_rows(d,src,comp,endpoint,['M_severity'],endpoint))
    save(rows,src,'PATH_MUTUALLY_EXCLUSIVE',('n','N','observed_N'))
    protective_fit_save(fixed,src,'PATH_ADJUSTED')

def protective_fit_save(rows,src,name):
    cols=('windows','people','families','index_windows','index_events','index_nonevents',
          'reference_windows','reference_events','reference_nonevents')
    # Coarse counts for rare branches including complementary cells; raw risks absent when sparse.
    df=pd.DataFrame(rows)
    small=df[['index_events','index_nonevents','reference_events','reference_nonevents']].min(axis=1)<10
    df.to_csv(PR/f'{name}_{src}_exact.csv',encoding='utf-8-sig',index=False)
    for k in cols:
        df[k]=df[k].astype(object)
        df.loc[small,k]=df.loc[small,k].map(release_n)
    df.loc[small,['index_raw_risk','reference_raw_risk']]=np.nan
    df.to_csv(TAB/f'{name}_{src}.csv',encoding='utf-8-sig',index=False)

def main_ppc(q,src,adl_only=False):
    results=[]; support=[]
    for comp in COMPARE:
        stages,context=EST['stages'](q,comp,'primary')
        shared=stages[-1]
        outputs=[]
        if adl_only:
            d=EST['restrict'](shared.loc[shared.adl_any.notna()],context)
            outputs=[('ADL_common_reference',d,('M_severity',),()),
                     ('ADL_extended_same_records',d,('M_severity',),('adl_any',))]
        else:
            previous=EST['restrict'](shared.dropna(subset=['cesd0','cog_z0']),context)
            outputs=[('primary',shared,('M_basic','M_severity'),()),
                     ('earliest',EST['stages'](q,comp,'earliest')[0][-1],('M_severity',),()),
                     ('history_health_common_reference',previous,('M_severity',),()),
                     ('history_health_extended_same_records',previous,('M_severity',),('cesd0','cog_z0'))]
            if src=='HRS':
                d=EST['stages'](q.loc[q.age1.ge(65)],comp,'primary')[0][-1]
                outputs.append(('index65_measurement_eligible',d,('M_severity',),()))
            sup,_,_=EST['support_rows'](src,comp,'primary',stages,context)
            support+=sup
        for spec,d,ls,extra in outputs:
            FIT.__globals__['matrix']=extend_matrix(extra) if extra else BASE_MATRIX
            results+=list(create_fit_rows(d,src,comp,spec,ls))
        FIT.__globals__['matrix']=BASE_MATRIX
    protective_fit_save(results,src,'PPC_ADL' if adl_only else 'PPC_RESULTS')
    if support:
        save(support,src,'PPC_SUPPORT',('windows','people','families','events','nonevents','unknown_endpoint'))

def validated_binomial(x,y,freq=None,var=None):
    good=np.ones(len(y),bool) if freq is None else freq>0
    xx=x[good]; yy=y[good]
    if len(np.unique(yy))<2 or np.linalg.matrix_rank(xx)!=xx.shape[1]:
        raise ValueError('No full-rank variation')
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        fit=sm.GLM(yy,xx,family=sm.families.Binomial(),
                   freq_weights=None if freq is None else freq[good],
                   var_weights=None if var is None else var[good]).fit(maxiter=50,tol=1e-9)
    if not fit.converged or not np.isfinite(fit.params).all() or any('separation' in str(v.message).lower() for v in ws):
        raise ValueError('Binomial model failed diagnostics')
    p=expit(xx@fit.params)
    wt=np.ones(len(p))
    if freq is not None: wt*=freq[good]
    if var is not None: wt*=var[good]
    info=xx.T@(xx*(wt*p*(1-p))[:,None])
    if np.linalg.eigvalsh(info).min()<=1e-10 or np.linalg.cond(info)>1e12:
        raise ValueError('Weak information or separation')
    return np.asarray(fit.params)

def obs_prepare(q,src,comp):
    # Include only those with verified compatible next visits, not all known/presumed alive.
    target=q.entry_eligible & q.legal2 & q.proxy2.eq(0) & q.cog_reference_comparable.eq(True)
    if src=='HRS':
        target &= (q.age1.ge(65)&q.cognition_mode.isin(['face_to_face','phone','interviewer_era'])
                   &~q.c3_route.eq('structural_skip'))
    d=q.loc[target & q.history.isin(COMPARE[comp])].copy()
    d['y']=d.y_fixed
    d['p0']=d.code_fixed1
    d['kgrp']=d.K1.clip(upper=3)
    d['arm']=d.history
    d['hx']=d.arm.eq(COMPARE[comp][0]).astype(int)
    d=d.dropna(subset=FIELDS+['p0','kgrp'])
    cs=['wave','p0','kgrp'] if comp=='H11_vs_H01' else ['wave','p0']
    observed=EST['restrict'](d.loc[d.y.notna()],cs)
    d=d.loc[pd.MultiIndex.from_frame(d[cs]).isin(pd.MultiIndex.from_frame(observed[cs]))].copy()
    return d

def selection_one(d,src,comp):
    # Inference includes resampling the target distribution, not just outcome-model coefficients.
    x,_,_=BASE_MATRIX(d,src,comp,'M_severity')
    x=x.loc[:,x.loc[d.y.notna()].nunique().gt(1)|x.columns.isin(['Intercept'])]
    j=x.columns.get_loc('hx')
    xc=x.drop(columns='g12',errors='ignore').to_numpy(float)
    x=x.to_numpy(float)
    i=x.copy();i[:,j]=1
    v=x.copy();v[:,j]=0
    y=d.y.to_numpy(float); r=np.isfinite(y)
    hx=d.hx.to_numpy(int)
    _,fam=np.unique(d.cluster.to_numpy(str),return_inverse=True); nf=int(fam.max()+1)
    if nf<2:
        raise ValueError('Fewer than two independent families')
    def eval_y(b,counts,di=0,dv=0):
        p1=expit(i@b+di*~r);p0=expit(v@b+dv*~r)
        return np.array([np.average(p1,weights=counts),np.average(p0,weights=counts)])
    weights={}
    ow_reason=''
    b=validated_binomial(x[r],y[r])
    if any((sum(r&(hx==a))<20 or sum(~r&(hx==a))<10) for a in [0,1]):
        ow_reason='Insufficient known compatible observed or incomplete interviews within one history'
    else:
        try:
            alpha=validated_binomial(xc,r.astype(float))
            pr=expit(xc@alpha)
            for label,mask in [('all',np.ones(len(r),bool)),('index',hx==1),('reference',hx==0)]:
                w=1/pr[r&mask]; p=pr[mask]
                weights[label]=dict(target_windows=int(mask.sum()),observed_windows=int((r&mask).sum()),
                                    min_probability=p.min(),p01=np.quantile(p,.01),p50=np.median(p),
                                    p99=np.quantile(p,.99),max_probability=p.max(),
                                    minimum_weight=w.min(),p95_weight=np.quantile(w,.95),maximum_weight=w.max(),
                                    ESS=w.sum()**2/(w@w))
            if pr.min()<.05 or any(t['ESS']<.5*t['observed_windows'] for t in weights.values()):
                ow_reason='Fixed minimum probability or relative ESS gate failed'
            if min(weights['index']['max_probability'],weights['reference']['max_probability'])<max(weights['index']['min_probability'],weights['reference']['min_probability']):
                ow_reason='No common probability range between history groups'
            if not ow_reason:
                bw=validated_binomial(x[r],y[r],var=1/pr[r])
        except FAILS as ex:
            ow_reason=str(ex)
    supported=not ow_reason
    methods={k:eval_y(b,np.ones(len(r)),*delta) for k,delta in DELTA.items()}
    if supported:methods['IPW']=eval_y(bw,np.ones(len(r)))
    samples={k:[] for k in methods}
    rng=np.random.default_rng(20261116+100*SOURCES.index(src)+list(COMPARE).index(comp))
    for _ in range(BOOT):
        wt=np.bincount(rng.integers(nf,size=nf),minlength=nf)[fam]
        try:
            bb=validated_binomial(x[r],y[r],freq=wt[r])
        except FAILS:
            continue
        for k,delta in DELTA.items():
            samples[k].append(eval_y(bb,wt,*delta))
        if not supported:continue
        try:
            aa=validated_binomial(xc,r.astype(float),freq=wt)
            pp=expit(xc@aa)
            if pp[wt>0].min()<.05:continue
            ww=validated_binomial(x[r],y[r],freq=wt[r],var=1/np.maximum(pp[r],1e-100))
            samples['IPW'].append(eval_y(ww,wt))
        except FAILS:
            continue
    head=dict(source=src,comparison=comp,target_windows=len(d),observed_windows=int(r.sum()),
              incomplete_windows=int((~r).sum()),target_people=int(d.person.nunique()),families=nf,
              target='timely verified compatible self-interviews; complete baseline; shared background')
    vals=[]
    if not supported:
        vals.append(dict(**head,method='IPW',status='not_estimable',detail=ow_reason,boots_attempted=0,boots_success=0))
    for k,p in methods.items():
        m=len(samples[k])
        lo=hi=np.nan
        if m>=.9*BOOT:
            ar=np.asarray(samples[k]);lo,hi=np.quantile(ar[:,0]-ar[:,1],[.025,.975])
        vals.append(dict(**head,method=k,status='estimated' if m>=.9*BOOT else 'point_only_bootstrap_failed',
                         risk_index=p[0],risk_reference=p[1],risk_difference=p[0]-p[1],
                         risk_difference_lo=lo,risk_difference_hi=hi,boots_attempted=BOOT,boots_success=m))
    diag=[dict(source=src,comparison=comp,group=g,**t) for g,t in weights.items()]
    arm=[]
    for a in [0,1]:
        mask=hx==a
        arm.append(dict(source=src,comparison=comp,group='index' if a==1 else 'reference',
                        target_windows=int(mask.sum()),observed_windows=int((mask&r).sum()),
                        incomplete_windows=int((mask&~r).sum()),
                        complete_rate=(mask&r).sum()/mask.sum() if min((mask&r).sum(),(mask&~r).sum())>=10 else np.nan))
    return vals,diag,arm

def selection(q,src):
    rows=[]; diag=[]; arms=[]
    for comp in COMPARE:
        d=obs_prepare(q,src,comp)
        try:
            rr,ww,aa=selection_one(d,src,comp)
            rows+=rr;diag+=ww;arms+=aa
        except FAILS as ex:
            for k in ['IPW',*DELTA]:
                rows.append(dict(source=src,comparison=comp,target_windows=len(d),observed_windows=int(d.y.notna().sum()),
                                 incomplete_windows=int(d.y.isna().sum()),method=k,status='not_estimable',detail=str(ex)))
        print(src,comp,'observation',[(r['method'],r['status']) for r in rows if r['comparison']==comp],flush=True)
    save(rows,src,'OBS_SENSITIVITY',('target_windows','observed_windows','incomplete_windows','target_people','families'))
    save(diag,src,'OBS_WEIGHT_DIAGNOSTICS',('target_windows','observed_windows'))
    save(arms,src,'OBS_TARGET_SUPPORT',('target_windows','observed_windows','incomplete_windows'))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source',choices=SOURCES)
    parser.add_argument('--stage',choices=('lock','main','selection','adl','descriptive'),default='main')
    opt=parser.parse_args()
    PR.mkdir(parents=True,exist_ok=True)
    lock_check(opt.source)
    if opt.stage=='lock':
        raise SystemExit(0)
    q=linked(opt.source)
    if opt.stage=='main':
        baseline(q,opt.source)
        main_ppc(q,opt.source)
        paths(q,opt.source)
    elif opt.stage=='selection':
        selection(q,opt.source)
    elif opt.stage=='adl':
        baseline(q,opt.source)
        main_ppc(q,opt.source,True)
    else:
        baseline(q,opt.source)

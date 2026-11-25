"""V3 strictly post-index longitudinal models; authorised local inputs only."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import patsy
from scipy.special import expit, logit
from scipy.stats import t
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_cluster

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent/'PPC_IADL_Dynamic_V2_20261115'
EQUATED = ROOT.parent/'PPC_IADL_CHARLS_Equated_Update_20261116'
DIAG = EQUATED/'EDUCATION3_DIAGNOSTIC'
PRIVATE = ROOT/'PRIVATE'
DOMAINS = {'physical':'chronic_count','psychological':'cesd_z','cognitive':'cog_z'}
COMPARE = {'H11_minus_H01':('H11','H01'),'H10_minus_H00':('H10','H00')}
LAST = {'CHARLS':(2020,2020,2020),'HRS':(2020,2020,2020),'ELSA':(9,10,9)}
BASE = ['age1','sex1','education1','chronic_count1','cesd_z1','cog_z1','gap01','code_fixed1','K1']
UNITS = {'physical':'disease_count','psychological':'fixed_reference_SD','cognitive':'mean_subtest_Z'}

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def save(rows, name, private=False):
    pd.DataFrame(rows).to_csv(ROOT/('PRIVATE' if private else 'TABLES')/f'{name}.csv',
                              index=False,encoding='utf-8-sig')

def prepare(source):
    folder = DIAG if source == 'CHARLS' else OLD
    panel = pd.read_parquet(folder/'PRIVATE'/f'{source}_panel.parquet')
    windows = pd.read_parquet(folder/'PRIVATE'/f'{source}_windows.parquet')
    eligible = windows.loc[windows.entry_eligible]
    entries = eligible.loc[eligible.first_eligible].copy()
    first = eligible.sort_values(['month1','wave']).drop_duplicates('person')
    pd.testing.assert_frame_equal(entries[['person','wave','month1']].sort_values('person').reset_index(drop=True),
                                  first[['person','wave','month1']].sort_values('person').reset_index(drop=True))
    assert entries.person.is_unique and entries.d_fixed1.le(1).all()
    entries['index_wave'] = entries.wave
    entries['Kgroup'] = entries.K1.clip(upper=3)
    if source == 'CHARLS':
        raw = pd.read_parquet(EQUATED/'PRIVATE/CHARLS_windows.parquet',columns=['education1'])
        mapped = raw.education1.map({**dict.fromkeys([1,2,3],0),4:1,**dict.fromkeys(range(5,12),2)})
        assert np.array_equal(windows.education1,mapped,equal_nan=True)
    entries = entries[['person','cluster','month1','index_wave','history','Kgroup',*BASE]]
    obs = entries.merge(panel[['person','wave','month',*DOMAINS.values()]],
                        on='person',how='left',validate='one_to_many')
    assert not panel.duplicated(['person','wave']).any()
    obs = obs.loc[obs.wave.ge(obs.index_wave)].copy()
    obs['years'] = (obs.month-obs.month1)/12
    assert obs.loc[obs.wave.eq(obs.index_wave),'years'].eq(0).all()
    return panel, entries, obs

def overlap(d, comp):
    context = ['index_wave','code_fixed1']+(['Kgroup'] if comp=='H11_minus_H01' else [])
    support = d.groupby(context,observed=True).hx.nunique()
    return d.loc[pd.MultiIndex.from_frame(d[context]).isin(support.index[support.eq(2)])].copy()

def sample(entries, obs, comp, domain, horizon):
    a,b=COMPARE[comp]
    e=entries.loc[entries.history.isin([a,b])].copy()
    e['hx']=e.history.eq(a).astype(int)
    ec=e.dropna(subset=BASE)
    es=overlap(ec,comp)
    field=DOMAINS[domain]
    d=obs.loc[obs.person.isin(es.person)&obs.years.gt(0)&obs.years.le(horizon)&obs[field].notna()].copy()
    d['hx']=d.history.eq(a).astype(int)
    d=overlap(d,comp)
    return d, [('entry',e),('complete_baseline',ec),('shared_baseline',es),('observed_post_index_shared',d)]

def derived(d):
    return d.assign(age10=(d.age1-65)/10,time2=d.years-2,
                    extra_disease=np.maximum(d.chronic_count1-1,0),gap01_years=d.gap01/12)

def fit(d,source,comp,domain,tag,old_cache=None):
    if d.hx.nunique()!=2 or d.groupby('hx').wave.nunique().min()<2:
        raise ValueError('Both groups require observed data in two or more post-index waves')
    d=derived(d)
    education='education1' if source=='HRS' else 'C(education1)'
    formula=('hx * time2 + age10 + I(age10 ** 2) + C(sex1) + '+education+
             ' + C(code_fixed1) + extra_disease + cesd_z1 + cog_z1 + C(index_wave) + gap01_years')
    if comp=='H11_minus_H01':
        formula+=' + C(Kgroup)'
    design=patsy.dmatrix(formula,d,return_type='dataframe',NA_action='raise')
    X=np.asarray(design)
    bounded=domain=='physical'
    y=d[DOMAINS[domain]].to_numpy(float)/(7 if bounded else 1)
    groups=d.cluster.to_numpy(str)
    ng=len(np.unique(groups))
    if not np.isfinite(y).all() or (bounded and ((y<0)|(y>1)).any()):
        raise ValueError('Illegal response values')
    if np.linalg.matrix_rank(X)!=X.shape[1] or ng<=X.shape[1]:
        raise ValueError('Design rank or independent family support insufficient')
    if old_cache is not None:
        with np.load(old_cache,allow_pickle=False) as z:
            assert np.array_equal(z['X'],X) and np.array_equal(z['y'],y)
            assert np.array_equal(z['cluster'],groups) and np.array_equal(z['person'],d.person.to_numpy(str))
            beta=z['beta']; historical_V=z['covariance']
            p=expit(X@beta)
        fit_warnings=''
        method='reused_checked_identical_education3_checkpoint'
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result=(sm.GLM(y,X,family=sm.families.Binomial()).fit(maxiter=150,tol=1e-10)
                    if bounded else sm.OLS(y,X).fit(method='qr'))
        beta=np.asarray(result.params); p=np.asarray(result.fittedvalues)
        if not getattr(result,'converged',True):
            raise ValueError('Nonconvergence')
        fit_warnings='; '.join(str(w.message) for w in caught)
        method='new_fit'
    w=p*(1-p) if bounded else np.ones(len(p))
    info=X.T@(w[:,None]*X)
    inv=np.linalg.inv(info)
    _,codes=np.unique(groups,return_inverse=True)
    scores=np.zeros((ng,X.shape[1]))
    np.add.at(scores,codes,X*(y-p)[:,None])
    cov=inv@(scores.T@scores)@inv*(ng/(ng-1))*((len(y)-1)/(len(y)-X.shape[1]))
    comparison_cov=historical_V if old_cache is not None else cov_cluster(result,groups,use_correction=True)
    err=float(np.max(np.abs(comparison_cov-cov)))
    condition=float(np.linalg.cond(info))
    score=float(abs(X.T@(y-p)).max())
    eig=float(np.linalg.eigvalsh(cov).min())
    # Numeric boundaries are checked in addition to solver flags; no factor is silently dropped.
    if not np.isfinite(cov).all() or eig < -1e-8 or err>1e-6 or condition>1e12 or score>1e-5:
        raise ValueError(f'Covariance/likelihood diagnostic failed: agreement={err}, condition={condition}, score={score}, covariance_min={eig}')
    critical=float(t.ppf(.975,ng-1))
    np.savez_compressed(PRIVATE/f'LONG_{source}_{comp}_{domain}_{tag}.npz',
                        X=X,y=y,cluster=groups,person=d.person.to_numpy(str),beta=beta,covariance=cov,
                        fitted=p,years=d.years.to_numpy(float),terms=np.asarray(design.columns,dtype=str))
    diagnostic=dict(formula=formula,parameters=X.shape[1],fit_method=method,information_condition=condition,
                    covariance_min_eigenvalue=eig,sandwich_library_error=err,score_max=score,
                    residual_sd=float(np.std(y-p,ddof=1))*(7 if bounded else 1),
                    max_leverage=float((w*np.sum((X@inv)*X,axis=1)).max()),warnings=fit_warnings)
    return d,design.design_info,beta,cov,critical,diagnostic

def infer(value,grad,cov,critical,bounded_mean=False):
    var=float(grad@cov@grad)
    if not np.isfinite(var) or var< -1e-10:
        raise ValueError('Non-finite or negative functional variance')
    se=np.sqrt(max(0,var))
    if bounded_mean:
        p=value/7
        lo,hi=7*expit(logit(p)+np.array([-1,1])*critical*se/(7*p*(1-p)))
    else:
        lo,hi=value+np.array([-1,1])*critical*se
    return dict(estimate=value,se=se,lower=lo,upper=hi)

def standardize(d,di,beta,cov,critical,domain,year,target):
    estimates=[]; gradients=[]
    bounded=domain=='physical'
    for arm in (1,0):
        x=np.asarray(patsy.build_design_matrices([di],target.assign(hx=arm,time2=year-2))[0])
        eta=x@beta
        if bounded:
            p=expit(eta); v=float(7*p.mean()); g=(7*x*(p*(1-p))[:,None]).mean(axis=0)
        else:
            v=float(eta.mean()); g=x.mean(axis=0)
        estimates.append(v); gradients.append(g)
    pairs=[{'functional':'standardized_mean','arm':arm,'scale':UNITS[domain],
            **infer(v,g,cov,critical,bounded)} for arm,v,g in zip((1,0),estimates,gradients)]
    pairs.append({'functional':'standardized_mean_difference','arm':np.nan,'scale':UNITS[domain],
                  **infer(estimates[0]-estimates[1],gradients[0]-gradients[1],cov,critical)})
    return pairs

def gran_count(n):
    return '<10' if n<10 else str(int(np.floor((n+5)/10)*10))

def support_for_year(d, entries, opportunity, year, meta, records):
    valid=[]
    for wave in sorted(entries.index_wave.unique()):
        cap,nominal_months=opportunity[wave]
        w=d.loc[d.index_wave.eq(wave)]
        spans=w.groupby('hx').years.agg(['min','max']).reindex([0,1])
        bracket=bool(spans['min'].le(year+1e-8).all() and spans['max'].ge(year-1e-8).all())
        eligible=cap>=year and bracket
        if eligible:
            valid.append(wave)
        for hx,history in zip((1,0),COMPARE[meta['comparison']]):
            e=entries.loc[entries.index_wave.eq(wave)&entries.history.eq(history)]
            f=w.loc[w.hx.eq(hx)]
            records.append(dict(**meta,time_years=year,index_wave=wave,history=history,
                calendar_elapsed_months_median_difference=nominal_months,planned_horizon_years=cap,
                administratively_open=cap>=year,both_arms_bracket=bracket,reference_wave_included=eligible,
                index_people=len(e),administrative_people=len(e) if cap>=year else 0,
                fitted_people=f.person.nunique(),actual_near_time_people=f.loc[f.years.between(year-.5,year+.5)].person.nunique(),
                actual_min_years=f.years.min(),actual_max_years=f.years.max()))
    return valid

def bridge(d,source,comp,domain,meta):
    oldpath=EQUATED/'PRIVATE'/f'CHARLS_{comp}_{domain}_longitudinal.npz'
    if not oldpath.exists():
        oldpath=oldpath.with_name(oldpath.stem+'_failed.npz')
    with np.load(oldpath,allow_pickle=False) as z:
        assert np.array_equal(z['y'],d[DOMAINS[domain]].to_numpy(float)/(7 if domain=='physical' else 1))
        assert np.array_equal(z['cluster'],d.cluster.to_numpy(str))
        assert np.array_equal(z['years'],d.years.to_numpy(float))
        if 'person' in z.files:
            assert np.array_equal(z['person'],d.person.to_numpy(str))
        else:
            # The original failed fit omitted person from its checkpoint; all preserved outcome, time,
            # family and non-education model columns are verified in their original order below.
            assert domain=='physical' and comp=='H10_minus_H00'
        x11=patsy.dmatrix(
            ('hx * time2 + age10 + I(age10 ** 2) + C(sex1) + C(education1)'
             ' + C(code_fixed1) + extra_disease + cesd_z1 + cog_z1 + C(index_wave) + gap01_years')
             +(' + C(Kgroup)' if comp=='H11_minus_H01' else ''),derived(d),return_type='dataframe')
        keep=[i for i,col in enumerate(x11.columns) if not col.startswith('C(education1)')]
        oldkeep=[i for i,col in enumerate(z['terms']) if not col.startswith('C(education1)')]
        assert list(x11.columns[keep])==list(z['terms'][oldkeep])
        assert np.array_equal(np.asarray(x11)[:,keep],z['X'][:,oldkeep])
    cache=DIAG/'PRIVATE'/f'CHARLS_{comp}_{domain}_longitudinal.npz'
    dd,di,b,V,cr,diag=fit(d,source,comp,domain,'education_bridge',cache if domain=='physical' else None)
    pred=standardize(dd,di,b,V,cr,domain,4.,dd.drop_duplicates('person'))[-1]
    old=pd.read_csv(EQUATED/'TABLES/CHARLS_longitudinal_results.csv')
    old=old.loc[old.comparison.eq(comp)&old.domain.eq(domain)]
    old=old.loc[(old.functional.eq('standardized_mean_difference')&old.time_years.eq(4))|old.status.eq('not_estimable')].iloc[0]
    slope_pos=di.column_names.index('hx:time2')
    slope_var=float(V[slope_pos,slope_pos])
    old_slope=pd.read_csv(EQUATED/'TABLES/CHARLS_longitudinal_results.csv')
    old_slope=old_slope.loc[old_slope.comparison.eq(comp)&old_slope.domain.eq(domain)&
                            old_slope.functional.eq('history_time_interaction')]
    return dict(**meta,status='estimated',same_outcome_family_time_non_education_columns=True,
                person_order_checkpoint_available=oldpath.name.endswith('_longitudinal.npz'),
                education3_fit_method=diag['fit_method'],records=len(d),people=d.person.nunique(),
                original_education11_status=old.status,original_education11_4year=old.get('estimate',np.nan),
                original_education11_4year_lo=old.get('lower',np.nan),original_education11_4year_hi=old.get('upper',np.nan),
                education3_4year=pred['estimate'],education3_4year_lo=pred['lower'],education3_4year_hi=pred['upper'],
                old_slope=old_slope.estimate.iloc[0] if len(old_slope) else np.nan,
                education3_slope=b[slope_pos],education3_slope_se=np.sqrt(slope_var),
                reference_note='Same legacy full-follow-up participant target; not V3 wave-qualified year 4')

def run():
    lock=json.loads((ROOT/'V3_PROTOCOL_LOCK.json').read_text(encoding='utf-8'))
    assert sha(ROOT/'V3_AMENDMENT_AND_ESTIMANDS_CN.md')==lock['amendment_sha256']
    rows=[]; curves=[]; support=[]; landmarks=[]; bridge_rows=[]; obs_rows=[]; source_hashes={}
    for source in LAST:
        panel,entries,obs=prepare(source)
        folder=DIAG if source=='CHARLS' else OLD
        source_hashes[source]={stem:sha(folder/'PRIVATE'/f'{source}_{stem}.parquet') for stem in ['panel','windows']}
        dates=panel.groupby('wave').month.median()
        all_measured=obs.loc[obs.years.between(0,6)]
        for domain,field in DOMAINS.items():
            for key,part in all_measured.loc[all_measured[field].notna()].groupby(['index_wave','wave','history']):
                iw,wv,h=key; n=len(part)
                obs_rows.append(dict(source=source,domain=domain,index_wave=iw,calendar_wave=wv,history=h,
                    n_measured=gran_count(n),time_median=part.years.median() if n>=10 else np.nan,
                    mean=part[field].mean() if n>=10 else np.nan,sd=part[field].std() if n>=10 else np.nan,
                    scale=UNITS[domain]))
        for comp,arms in COMPARE.items():
            for domain,last in zip(DOMAINS,LAST[source]):
                meta=dict(source=source,comparison=comp,domain=domain,index_history=arms[0],reference_history=arms[1],
                          education_version='approved_three_group' if source=='CHARLS' else 'inherited_V2')
                d,stages=sample(entries,obs,comp,domain,6.)
                for stage,frame in stages:
                    for history,g in frame.groupby('history'):
                        support.append(dict(**meta,stage=stage,history=history,records=len(g),
                                            people=g.person.nunique(),families=g.cluster.nunique()))
                if source=='CHARLS':
                    full,_=sample(entries,obs,comp,domain,np.inf)
                    bf=bridge(full,source,comp,domain,meta)
                    bf.update(horizon06_records=len(d),horizon06_people=d.person.nunique())
                    bridge_rows.append(bf)
                wave_gap={w:float(dates[last]-dates[w]) for w in entries.index_wave.unique()}
                opportunity={w:(int(np.floor(g/12+.5)),g) for w,g in wave_gap.items()}
                wave_sets={yr:support_for_year(d,entries,opportunity,yr,meta,landmarks) for yr in (2.,4.,6.)}
                common=dict(**meta,n_records=len(d),n_people=d.person.nunique(),n_families=d.cluster.nunique(),
                            fit_scope='strictly_post_index_0_6y',status='estimated')
                try:
                    dd,di,b,V,cr,diag=fit(d,source,comp,domain,'06')
                    slope_pos=di.column_names.index('hx:time2'); grad=np.eye(len(b))[slope_pos]
                    slope=infer(b[slope_pos],grad,V,cr)
                    p=2*t.sf(abs(slope['estimate']/slope['se']),common['n_families']-1)
                    rows.append(dict(**common,**diag,**slope,time_years=np.nan,functional='history_time_interaction',
                         nominal_p=p,scale='logit_fraction_per_year' if domain=='physical' else UNITS[domain]+'_per_year'))
                    target=dd.drop_duplicates('person')
                    for yr,kept in wave_sets.items():
                        ref=target.loc[target.index_wave.isin(kept)]
                        base=dict(**common,**diag,time_years=yr,target_people=len(ref),
                                  included_index_waves=';'.join(map(str,kept)))
                        if ref.empty:
                            rows.append({**base,'status':'time_not_supported','functional':'standardized_mean_difference',
                                         'scale':UNITS[domain],'detail':'No administratively eligible indexed wave brackets time in both histories'})
                        else:
                            rows.extend({**base,**p} for p in standardize(dd,di,b,V,cr,domain,yr,ref))
                    yr=4. if wave_sets[4.] else 2.
                    ref=target.loc[target.index_wave.isin(wave_sets[yr])]
                    if len(ref):
                        rng=dd.loc[dd.index_wave.isin(wave_sets[yr])].groupby(['index_wave','hx']).years.agg(['min','max'])
                        times=np.arange(np.ceil(rng['min'].max()*4)/4,np.floor(rng['max'].min()*4)/4+.01,.25)
                        for today in times:
                            for pred in standardize(dd,di,b,V,cr,domain,today,ref):
                                curves.append(dict(**common,time_years=today,fixed_reference_landmark=yr,
                                                   target_people=len(ref),**pred))
                except (ValueError,np.linalg.LinAlgError,ZeroDivisionError) as error:
                    rows.append({**common,'status':'not_estimable','detail':str(error),
                                 'functional':'history_time_interaction'})
    result=pd.DataFrame(rows)
    ix=result.functional.eq('history_time_interaction')
    assert ix.sum()==18
    result.loc[ix,'BH_q_18_planned_interactions']=multipletests(result.loc[ix,'nominal_p'].fillna(1),method='fdr_bh')[1]
    result.loc[~result.status.eq('estimated'),'BH_q_18_planned_interactions']=np.nan
    save(result,'LONG_RESULTS_V3'); save(curves,'LONG_CURVES_V3')
    save(bridge_rows,'LONG_EDUCATION_BRIDGE'); save(obs_rows,'LONG_OBSERVED_0_6')
    safe=pd.DataFrame(support); save(safe,'LONG_MODEL_SUPPORT_EXACT',True)
    for c in ['records','people','families']:
        safe[c]=safe[c].map(gran_count)
    save(safe,'LONG_MODEL_SUPPORT')
    safe=pd.DataFrame(landmarks); save(safe,'LONG_TIME_SUPPORT_EXACT',True)
    few=safe.fitted_people.lt(10)
    for c in ['index_people','administrative_people','fitted_people','actual_near_time_people']:
        safe[c]=safe[c].map(gran_count)
    safe.loc[few,['actual_min_years','actual_max_years']]=np.nan
    save(safe,'LONG_TIME_SUPPORT')
    summary=dict(spec_sha256=sha(ROOT/'LONGITUDINAL_SPEC_V3.md'),sources=source_hashes,
                 models_attempted=int(ix.sum()),models_estimable=int((ix&result.status.eq('estimated')).sum()),
                 aggregate_disclosure='Granular counts nearest ten; <10 includes zero. Small observed cell statistics withheld.')
    (ROOT/'TABLES/LONG_RUN_V3.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    run()

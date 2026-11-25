"""Source-locked fixed references and consecutive three-interview structure; no models."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PRIVATE=ROOT/'PRIVATE'
TABLES=ROOT/'TABLES'
for p in (PRIVATE,TABLES): p.mkdir(exist_ok=True)
WINDOWS={'CHARLS':[(2013,2015,2018),(2015,2018,2020)],
         'HRS':[(2012,2014,2016),(2014,2016,2018),(2016,2018,2020)],
         'ELSA':[(1,2,3),(2,3,4),(3,4,5),(7,8,9)]}
ANCHORS={'CHARLS':2013,'HRS':2012,'ELSA':1}
SCORES=['c1','c2','c3','c4']
ITEMS=['x'+c for c in 'MFPSC']
DOMAINS={i:i.bit_count() for i in range(8)}
BANDS=[45,50,55,60,65,70,75,80,np.inf]
HISTORY=['A','B','C','D']

def save(d,name):
    if not isinstance(d,pd.DataFrame):d=pd.DataFrame(d)
    d.to_csv(TABLES/f'{name}.csv',encoding='utf-8-sig',index=False)

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def counts(d):
    return dict(windows=len(d),people=d.person.nunique(),families=d.cluster.nunique())

def cluster_key(d):
    parent={}
    def find(x):
        while x!=parent.setdefault(x,x):
            parent[x]=parent[parent[x]]
            x=parent[x]
        return x
    for p,f in d[['person','family']].itertuples(index=False,name=None):
        a,b=find('person:'+p),find('family:'+f)
        if a!=b:parent[a]=b
    return d.person.map(lambda p:find('person:'+p))

def callable_wave_mask(source,d):
    flag=d['fixed_anchor_comparable'] if 'fixed_anchor_comparable' in d else (
        ~d.wave.isin([6,10]) if source=='ELSA' else pd.Series(True,index=d.index))
    expected=d.wave.isin([2013,2015]) if source=='CHARLS' else (
        d.wave.isin([1,2,3,4,5,7,8,9]) if source=='ELSA' else pd.Series(True,index=d.index))
    if not np.array_equal(flag.fillna(False),expected):
        raise ValueError(f'Wave-level measurement qualification differs from frozen evidence for {source}')
    return expected

def code_domains(d,cog):
    return (d.physical+2*d.psychological+4*cog).astype(float)

def classify(components,limits,valid):
    ok=pd.DataFrame(np.broadcast_to(valid.to_numpy()[:,None],components.shape),index=components.index,columns=SCORES)
    known=components.notna()&limits.notna()&ok
    hit=(components.le(limits)&known).any(axis=1)
    return hit.astype(float).where(hit|known.all(axis=1))

def reference(d,source):
    names=['immediate recall','delayed recall','orientation','animal fluency' if source=='ELSA' else 'serial subtraction']
    waves=callable_wave_mask(source,d)
    anchor=ANCHORS[source]
    r=d.loc[d.wave.eq(anchor)&d.age.ge(45)&d.proxy.eq(0),['age']+SCORES].copy()
    row=r[SCORES].agg(['count','mean','std']).T
    assert row['count'].ge(30).all() and row['std'].gt(0).all()
    row['cutoff']=row['mean']-1.5*row['std']
    observed=d[SCORES].where(d.proxy.eq(0),axis=0)
    full_limits=pd.DataFrame({s:row.loc[s,'cutoff'] for s in SCORES},index=d.index)
    d['cog_fixed']=classify(observed,full_limits,waves)
    d['code_fixed']=code_domains(d,d.cog_fixed)
    for k in SCORES:d['z'+k]=((observed[k]-row.loc[k,'mean'])/row.loc[k,'std']).where(waves)
    r['band']=pd.cut(r.age,BANDS,right=False,labels=False)
    d['band']=pd.cut(d.age,BANDS,right=False,labels=False)
    bands=r.groupby('band')[SCORES].agg(['count','mean','std']).reindex(range(len(BANDS)-1))
    counts_good=bands.xs('count',level=1,axis=1).ge(30).all().all()
    std_good=bands.xs('std',level=1,axis=1).gt(0).all().all()
    band_limits={}
    for k in SCORES:
        band_limits[k]=bands[(k,'mean')]-1.5*bands[(k,'std')]
    band_table=pd.DataFrame(band_limits)
    if counts_good and std_good and source!='CHARLS':
        limits=pd.DataFrame({k:d.band.map(band_table[k]) for k in SCORES})
        d['cog_agefix']=classify(observed,limits,waves)
        d['code_agefix']=code_domains(d,d.cog_agefix)
        age_support='eligible_age_reference'
    else:
        d['cog_agefix']=np.nan;d['code_agefix']=np.nan
        age_support='not_eligible_all_bands_or_source_comparability'
    params=[]; agreement=[]
    for w,g in d.groupby('wave'):
        g=g.loc[g.age.ge(45)]
        valid=bool(waves[d.wave.eq(w)].all())
        for k,task in zip(SCORES,names):
            v=g[k].where(g.proxy.eq(0)).dropna()
            params.append(dict(source=source,apply_wave=w,component=task,reference='fixed_all_anchor_self',
                anchor=anchor,reference_n=int(row.loc[k,'count']),reference_mean=row.loc[k,'mean'],
                reference_sd=row.loc[k,'std'],threshold=row.loc[k,'cutoff'],comparable=valid,
                application_n=len(v),application_mean=v.mean(),application_sd=v.std(),
                application_low=(v.le(row.loc[k,'cutoff']).sum() if valid else np.nan),
                application_age45_n=len(g),agefix_status=age_support))
        oldc=g.cognitive_old if 'cognitive_old' in g else g.code_old.map({i:int(bool(i&4)) for i in range(8)})
        both=oldc.notna()&g.cog_fixed.notna()
        cboth=g.code_old.notna()&g.code_fixed.notna()
        agreement.append(dict(source=source,wave=w,age45_visits=len(g),cog_complete4=g[SCORES].notna().all(axis=1).sum(),
            old_cog_known=oldc.notna().sum(),old_cog_positive=oldc.eq(1).sum(),fixed_cog_known=g.cog_fixed.notna().sum(),
            fixed_cog_positive=g.cog_fixed.eq(1).sum(),fixed_cog_negative=g.cog_fixed.eq(0).sum(),
            paired_cog_known=both.sum(),paired_cog_agreement=(oldc.loc[both]==g.cog_fixed.loc[both]).sum(),
            old_PPC_complete=g.code_old.notna().sum(),fixed_PPC_complete=g.code_fixed.notna().sum(),
            paired_PPC_complete=cboth.sum(),paired_PPC_agreement=(g.code_old.loc[cboth]==g.code_fixed.loc[cboth]).sum(),
            physical_known=g.physical.notna().sum(),physical_positive=g.physical.eq(1).sum(),
            psychological_known=g.psychological.notna().sum(),psychological_positive=g.psychological.eq(1).sum(),
            comparable=valid))
    for band in bands.index:
        for k,task in zip(SCORES,names):
            params.append(dict(source=source,apply_wave=np.nan,component=task,reference='fixed_anchor_age_band_audit',
                age_band=f'{BANDS[band]}+' if band==len(bands)-1 else f'{BANDS[band]}–{BANDS[band+1]-1}',
                anchor=anchor,reference_n=bands.loc[band,(k,'count')],reference_mean=bands.loc[band,(k,'mean')],
                reference_sd=bands.loc[band,(k,'std')],threshold=band_table.loc[band,k],
                comparable=bool(source!='CHARLS' and counts_good and std_good),agefix_status=age_support))
    save(params,f'{source}_cognition_reference')
    save(agreement,f'{source}_cognition_agreement')
    return d,age_support

def order_status(raw):
    q=raw.astype('string').str.lower().fillna('unknown')
    state=pd.Series('survival_unknown',index=raw.index,dtype='string')
    state[q.str.contains('dead|death|deceased')&~q.str.contains('not_dead|unknown|unconfirmed')]='confirmed_death'
    state[q.str.contains('alive|living')&~q.str.contains('unknown|not_known')]='known_alive_not_interviewed'
    state[q.str.contains('admin|no_next')]='administrative_end'
    state[q.str.contains('exit|scope|drop|withdraw')]='out_of_scope_or_sample_exit'
    state[q.isin(['observed','interviewed','completed'])]='interview_registered_missing_link'
    return state

def history(k0,k1):
    keep=k0.notna()&k1.notna()
    h=np.select([k0.eq(0)&k1.eq(0),k0.eq(0)&k1.gt(0),k0.gt(0)&k1.gt(0),k0.gt(0)&k1.eq(0)],
                HISTORY,default='unknown')
    return pd.Series(h,index=k0.index).where(keep,'unknown')

def build(source):
    p=ROOT/source/'PRIVATE/panel.parquet'
    d=pd.read_parquet(p)
    assert not d.duplicated(['person','wave']).any() and d.person.notna().all()
    d['person']=d.person.astype(str);d['family']=d.family.astype('string').fillna(d.person).astype(str)
    d['cluster']=cluster_key(d)
    d['code_old']=d['code4_old'] if 'code4_old' in d else d['code4']
    if 'xM_old' not in d:d['xM_old']=d.xM
    for k in ['age','month','proxy','sex','education','chronic_count','physical','psychological','code_old','cesd','xM_old']+ITEMS+SCORES:
        d[k]=pd.to_numeric(d[k],errors='raise').astype(float)
    assert d[ITEMS].stack().isin([0,1]).all() and d.code_old.dropna().isin(DOMAINS).all()
    d['k_obs']=d[ITEMS].notna().sum(axis=1); d['k_low']=d[ITEMS].eq(1).sum(axis=1)
    d['k_high']=d.k_low+5-d.k_obs
    d['K']=d.k_low.where(d.k_obs.eq(5))
    d['K_old']=d[['xM_old']+ITEMS[1:]].sum(axis=1,min_count=5)
    d,agefix=reference(d,source)
    basefields=['person','cluster','age','month','K','K_old','proxy']+ITEMS
    current=basefields+['sex','education','chronic_count','code_old','code_fixed','code_agefix','next_status','cesd']+['z'+k for k in SCORES]
    future=['person','month','code_old','code_fixed','code_agefix','proxy']
    out=[]; first=[]
    for a,b,c in WINDOWS[source]:
        def take(w,cols,n):
            return d.loc[d.wave.eq(w),cols].rename(columns={k:f'{k}{n}' for k in cols if k!='person'})
        u=take(a,basefields,0); v=take(b,current,1); z=take(c,future,2)
        v['seen1']=True;z['seen2']=True
        t=u.merge(v,on='person',how='left',validate='one_to_one').merge(z,on='person',how='left',validate='one_to_one')
        t['cluster']=t.cluster0
        t['wave']=b
        t['window']=f'{a}->{b}->{c}'
        t['gap01']=t.month1-t.month0
        t['gap12']=t.month2-t.month1
        t['entry']=t.seen1.eq(True)&t.gap01.gt(0)&t.age1.ge(45)
        t['core_age50']=t.age0.ge(50)&t.age1.ge(50)
        t['history']=history(t.K0,t.K1);t['history_old']=history(t.K_old0,t.K_old1)
        t['same_task_overlap']=sum((t[k+'0'].eq(1)&t[k+'1'].eq(1)).astype(int) for k in ITEMS)
        t['status2']=order_status(t.next_status1)
        present=t.seen2.eq(True)
        assert not (present&t.status2.eq('confirmed_death')).any()
        t.loc[present&~t.gap12.gt(0),'status2']='interview_nonpositive_or_unknown_interval'
        t.loc[present&t.gap12.gt(0),'status2']='living_interview_positive_date'
        t['legal2']=t.status2.eq('living_interview_positive_date')
        for tag in ('fixed','old','agefix'):
            x=t[f'code_{tag}1'];y=t[f'code_{tag}2']
            t['d_'+tag+'1']=x.map(DOMAINS)
            t['d_'+tag+'2']=y.map(DOMAINS)
            t['y_'+tag]=t['d_'+tag+'2'].ge(2).astype(float).where(t.legal2&y.notna()&t['d_'+tag+'1'].le(1))
        first.append(dict(source=source,window=t.window.iloc[0],definition='common_structure',
            group='ALL',stage='actual_t0',**counts(t)))
        first.append(dict(source=source,window=t.window.iloc[0],definition='common_structure',
            group='ALL',stage='t1_seen',**counts(t.loc[t.seen1.eq(True)])))
        out.append(t.loc[t.entry].copy())
    return pd.concat(out,ignore_index=True),first,agefix,digest(p)

def audit(source,d,initial):
    flow=list(initial); changes=[]; diffs=[]; kdist=[]
    for w,x in d.groupby('window'):
        both=x.loc[x.history.ne('unknown')]
        for name,block in [('t1_age45_ordered',x),('t0_t1_five_complete',both)]:
            flow.append(dict(source=source,window=w,definition='common_structure',group='ALL',stage=name,**counts(block),
                age50_people=block.loc[block.core_age50,'person'].nunique()))
        for tag in ['fixed','old']:
            d1=f'd_{tag}1';d2=f'd_{tag}2'; ed=f'y_{tag}'
            nominal=f'code_{tag}1'
            human='COG_FIXED_ANCHOR' if tag=='fixed' else 'COG_WAVE_AGE_RELATIVE'
            for name,block in [('t1_PPC_complete',both.loc[both[nominal].notna()]),('t1_PPC_0_or_1',both.loc[both[d1].le(1)])]:
                flow.append(dict(source=source,window=w,definition=human,group='ALL',stage=name,**counts(block)))
            q=both.loc[both[d1].le(1)]
            for a in HISTORY:
                g=q.loc[q.history.eq(a)]
                valid=g[ed].notna()
                n={'events':g[ed].eq(1).sum(),'nonevents':g[ed].eq(0).sum(),
                   'valid_PPC2':valid.sum(),'observed_people':g.loc[valid,'person'].nunique(),
                   'proxy0':g.proxy0.eq(1).sum(),'proxy1':g.proxy1.eq(1).sum(),'proxy2':g.proxy2.eq(1).sum(),
                   't1_chronic_count_mean':g.chronic_count1.mean(),
                   't1_count_1':g.K1.eq(1).sum(),'t1_count_2plus':g.K1.ge(2).sum(),
                   'gap01_median':g.gap01.median(),'gap01_q1':g.gap01.quantile(.25),'gap01_q3':g.gap01.quantile(.75),
                   'gap12_median_observed':g.loc[g.legal2,'gap12'].median(),
                   'gap12_q1_observed':g.loc[g.legal2,'gap12'].quantile(.25),'gap12_q3_observed':g.loc[g.legal2,'gap12'].quantile(.75),
                   't2_PPC_missing_or_unvalidated':(g.legal2&~valid).sum(),
                   't2_not_observed_other':g.status2.isin(['out_of_scope_or_sample_exit','interview_registered_missing_link',
                       'interview_nonpositive_or_unknown_interval']).sum()}
                for label in ['confirmed_death','known_alive_not_interviewed','survival_unknown','administrative_end']:
                    n[label]=g.status2.eq(label).sum()
                assert n['valid_PPC2']+n['t2_PPC_missing_or_unvalidated']+n['t2_not_observed_other']+sum(
                    n[k] for k in ['confirmed_death','known_alive_not_interviewed','survival_unknown','administrative_end'])==len(g)
                flow.append(dict(source=source,window=w,definition=human,group=a,stage='history_riskset_0_or_1',
                    **counts(g),**n))
                ev=g.loc[g[ed].eq(1)]
                for code,j in [('P',1),('Y',2),('C',4)]:
                    before=ev[f'code_{tag}1'].astype(int); after=ev[f'code_{tag}2'].astype(int)
                    changes.append(dict(source=source,window=w,definition=human,group=a,gained=code,
                        progression=len(ev),new_domain=((before&j).eq(0)&(after&j).gt(0)).sum()))
            sec=both.loc[both[d1].eq(2)]
            for a in HISTORY:
                g=sec.loc[sec.history.eq(a)]
                flow.append(dict(source=source,window=w,definition=human,group=a,stage='descriptive_2_to_3',
                    **counts(g),valid_PPC2=(g.legal2&g[d2].notna()).sum(),events=(g.legal2&g[d2].eq(3)).sum(),
                    nonevents=(g.legal2&g[d2].isin([0,1,2])).sum()))
        for a,g in both.groupby('history'):
            for k,h in g.groupby('K1'):
                kdist.append(dict(source=source,window=w,history=a,current_count=int(k),**counts(h),
                    same_task_overlap=(h.same_task_overlap.gt(0).sum() if a=='C' else np.nan)))
        # Selection and transition drift, including events on the shared observed/eligible definition subset.
        diffs.append(dict(source=source,window=w,t1_ordered_n=len(x),history_new_known=len(both),
            history_old_known=x.history_old.ne('unknown').sum(),
            history_changes_both_known=(x.history.ne('unknown')&x.history_old.ne('unknown')&x.history.ne(x.history_old)).sum(),
            fixed_eligible=x.y_fixed.notna().sum(),old_eligible=x.y_old.notna().sum(),
            eligible_both=(x.y_fixed.notna()&x.y_old.notna()).sum(),
            fixed_events=x.y_fixed.eq(1).sum(),old_events=x.y_old.eq(1).sum(),
            fixed_event_both=(x.y_old.notna()&x.y_fixed.eq(1)).sum(),old_event_both=(x.y_fixed.notna()&x.y_old.eq(1)).sum()))
    save(flow,f'{source}_flow');save(changes,f'{source}_domain_gains');save(diffs,f'{source}_measurement_diff')
    save(kdist,f'{source}_task_history_support')

if __name__=='__main__':
    import argparse
    import json
    a=argparse.ArgumentParser()
    a.add_argument('source',choices=WINDOWS)
    source=a.parse_args().source
    p,f,agefix,digest_source=build(source)
    audit(source,p,f)
    p.to_parquet(PRIVATE/f'{source}_three_windows.parquet',index=False)
    status={'source':source,'legal_history_entries':len(p),'distinct_people':p.person.nunique(),
            'fixed_reference_status':'COGNITION_REFERENCE_NOT_SUPPORTABLE' if source=='CHARLS' else 'FIXED_REFERENCE_AVAILABLE',
            'fixed_age_band':agefix,'new_input_sha256':digest_source}
    (TABLES/f'{source}_prep_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status))

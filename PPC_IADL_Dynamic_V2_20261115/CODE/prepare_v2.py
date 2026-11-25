"""V2 source-specific fixed references, risk sets and observation counts (no fits)."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'PPC_IADL_History_ThreeVisit_20261115'
PRIVATE=ROOT/'PRIVATE';TABLES=ROOT/'TABLES'
for p in [PRIVATE,TABLES]:p.mkdir(exist_ok=True)
SCORES=['c1','c2','c3','c4'];ITEMS=['x'+v for v in 'MFPSC']
ANCHOR={'CHARLS':2013,'HRS':2012,'ELSA':1}
WINDOWS={'CHARLS':[(2013,2015,2018),(2015,2018,2020)],'HRS':[(2012,2014,2016),(2014,2016,2018),(2016,2018,2020)],'ELSA':[(1,2,3),(2,3,4),(3,4,5),(6,7,8),(7,8,9)]}
HISTORY=['H00','H01','H11','H10']
DOMAINS={i:i.bit_count() for i in range(8)}
def save(x,name):
    pd.DataFrame(x).to_csv(TABLES/(name+'.csv'),encoding='utf-8-sig',index=False)
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def counts(d):return dict(windows=len(d),people=d.person.nunique(),families=d.cluster.nunique())
def cluster_key(d):
    parent={}
    def find(x):
        while x!=parent.setdefault(x,x):parent[x]=parent[parent[x]];x=parent[x]
        return x
    for p,f in d[['person','family']].itertuples(index=False,name=None):
        a,b=find('p:'+p),find('f:'+f)
        if a!=b:parent[a]=b
    return d.person.map(lambda p:find('p:'+p))
def history(a,b):
    return pd.Series(np.select([a.eq(0)&b.eq(0),a.eq(0)&b.gt(0),a.gt(0)&b.gt(0),a.gt(0)&b.eq(0)],HISTORY,default='unknown'),index=a.index)
def status(raw,source):
    q=raw.astype('string').str.lower().fillna('unknown')
    s=pd.Series('survival_unknown',index=q.index)
    s.loc[q.str.contains('dead|death|deceased')&~q.str.contains('unknown|not_dead|unconfirmed')]='confirmed_death'
    s.loc[q.str.contains('alive|living')&~q.str.contains('unknown|not_known')]='reported_alive_noninterview'
    if source=='HRS':s.loc[s.eq('reported_alive_noninterview')]='alive_or_presumed_alive_noninterview'
    s.loc[q.str.contains('admin|no_next')]='administrative_end'
    s.loc[q.str.contains('exit|scope|withdraw|drop')]='sample_exit'
    s.loc[q.isin(['observed','interviewed','completed'])]='interview_registered_no_link'
    return s
def reference(d,source):
    comparable=(d.wave.isin([2013,2015]) if source=='CHARLS' else d.wave.isin([1,2,3,4,5,7,8,9]) if source=='ELSA' else pd.Series(True,index=d.index))
    r=d.loc[d.wave.eq(ANCHOR[source])&d.age.ge(45)&d.proxy.eq(0)&d[SCORES].notna().all(axis=1),SCORES]
    params=r.agg(['count','mean','std']).T
    assert len(r)>0 and params['std'].gt(0).all()
    complete=d[SCORES].notna().all(axis=1)&d.proxy.eq(0)&comparable
    z=(d[SCORES]-params['mean'])/params['std']
    d['cog_z']=z.mean(axis=1).where(complete)
    d['cog_fixed']=z.lt(-1.5).any(axis=1).astype(float).where(complete)
    d['partial_cognitive_positive']=((z.lt(-1.5).any(axis=1))&~complete&comparable&d.proxy.eq(0))
    d['cog_reference_comparable']=comparable
    for s in SCORES:d['z'+s]=z[s].where(comparable&d.proxy.eq(0))
    cy=d.loc[d.wave.eq(ANCHOR[source])&d.age.ge(45)&d.proxy.eq(0),'cesd'].dropna()
    assert cy.std()>0
    d['cesd_z']=(d.cesd-cy.mean())/cy.std()
    d['psychological_v2']=d.cesd.ge(10 if source=='CHARLS' else 4).astype(float).where(d.cesd.notna())
    old=d['cognitive_old'] if source=='CHARLS' else (d['code4_old'] if source=='HRS' else d.code4).map({i:int(bool(i&4)) for i in DOMAINS})
    d['cog_old']=old
    # Rebuild other domains identically to isolate the historical cognitive algorithm.
    d['code_fixed']=d.physical+2*d.psychological_v2+4*d.cog_fixed
    d['code_old']=d.physical+2*d.psychological_v2+4*old
    rows=[]
    for k in SCORES:
        rows.append(dict(source=source,component=k,reference_wave=ANCHOR[source],reference_n=len(r),mean=params.loc[k,'mean'],sd=params.loc[k,'std'],threshold=params.loc[k,'mean']-1.5*params.loc[k,'std'],reference_type='four_complete_self_age45',sd_ddof=1,weighted=False))
    rows.append(dict(source=source,component='cesd',reference_wave=ANCHOR[source],reference_n=len(cy),mean=cy.mean(),sd=cy.std(),reference_type='complete_scale_self_age45',sd_ddof=1,weighted=False))
    qa=[]
    for w,g in d.groupby('wave'):
        a=g.loc[g.age.ge(45)];at=complete.loc[a.index]
        qa.append(dict(source=source,wave=w,age45_n=len(a),reference_compatible=bool(comparable.loc[a.index].all()),four_complete_direct=int(at.sum()),
            strict_lt_vs_le_changes=int(((z.loc[a.index].lt(-1.5).any(axis=1)!=z.loc[a.index].le(-1.5).any(axis=1))&at).sum()),
            cognitive_known=a.cog_fixed.notna().sum(),cognitive_low=a.cog_fixed.eq(1).sum(),partial_positive_excluded=a.partial_cognitive_positive.sum(),
            fixed_ppc=a.code_fixed.notna().sum(),old_cognitive_ppc=a.code_old.notna().sum(),
            both_classified=(a.code_fixed.notna()&a.code_old.notna()).sum(),disagreement=(a.code_fixed.notna()&a.code_old.notna()&a.code_fixed.ne(a.code_old)).sum()))
    save(rows,source+'_reference');save(qa,source+'_measurement_qa')
    return d
def build(source):
    p=ROOT/source/'PRIVATE/panel.parquet'
    if not p.exists():p=OLD/source/'PRIVATE/panel.parquet'
    d=pd.read_parquet(p)
    d.person=d.person.astype(str);d.family=d.family.astype('string').fillna(d.person).astype(str)
    assert not d.duplicated(['person','wave']).any()
    d['cluster']=cluster_key(d)
    for k in ['age','month','sex','proxy','education','chronic_count','physical','cesd']+SCORES+ITEMS:d[k]=pd.to_numeric(d[k],errors='raise').astype(float)
    assert d[ITEMS].stack().isin([0,1]).all()
    d['k_observed']=d[ITEMS].notna().sum(axis=1);d['k_lower']=d[ITEMS].eq(1).sum(axis=1);d['k_upper']=d.k_lower+5-d.k_observed
    d['K']=d.k_lower.where(d.k_observed.eq(5))
    if 'xM_old' not in d:d['xM_old']=d.xM
    d['K_old']=d[['xM_old']+ITEMS[1:]].sum(axis=1,min_count=5)
    d=reference(d,source)
    d.to_parquet(PRIVATE/f'{source}_panel.parquet',index=False)
    fields=['person','cluster','month','age','sex','education','proxy','K','K_old','k_observed','k_lower','k_upper','code_fixed','code_old','chronic_count','cesd','cesd_z','cog_z','next_status']+ITEMS
    if 'wealth' in d:fields+=['wealth']
    windows=[];flow=[]
    for a,b,c in WINDOWS[source]:
        def take(w,ix):
            z=d.loc[d.wave.eq(w),fields].copy();z['seen']=True
            return z.rename(columns={k:k+str(ix) for k in z if k!='person'})
        q=take(a,0).merge(take(b,1),how='left',on='person',validate='one_to_one').merge(take(c,2),how='left',on='person',validate='one_to_one')
        q['cluster']=q.cluster0;q['wave']=b;q['next_wave']=c;q['window']=f'{a}->{b}->{c}';q['source']=source
        q['gap01']=q.month1-q.month0;q['gap12']=q.month2-q.month1
        q['ordered_entry']=q.seen1.eq(True)&q.gap01.gt(0)&q.age1.ge(45)
        q['history']=history(q.K0,q.K1);q['history_old']=history(q.K_old0,q.K_old1)
        q['same_task_overlap']=sum((q[k+'0'].eq(1)&q[k+'1'].eq(1)).astype(int) for k in ITEMS)
        q['status2']=status(q.next_status1,source)
        assert not (q.seen2.eq(True)&q.status2.eq('confirmed_death')).any()
        q.loc[q.seen2.eq(True)&~q.gap12.gt(0),'status2']='interview_date_invalid'
        q.loc[q.seen2.eq(True)&q.gap12.gt(0),'status2']='living_interview'
        q['legal2']=q.status2.eq('living_interview')
        for tag in ['fixed','old']:
            for j in [0,1,2]:q[f'd_{tag}{j}']=q[f'code_{tag}{j}'].map(DOMAINS)
            q['y_'+tag]=q[f'd_{tag}2'].ge(2).astype(float).where(q.legal2&q[f'd_{tag}2'].notna()&q[f'd_{tag}1'].le(1))
        q['entry_eligible']=q.ordered_entry&q.history.ne('unknown')&q.d_fixed1.le(1)&q[['cesd1','cog_z1','chronic_count1']].notna().all(axis=1)
        for stage,mask in [('tprev_records',pd.Series(True,index=q.index)),('index_present',q.seen1.eq(True)),('ordered_index_age45',q.ordered_entry),('complete_history',q.ordered_entry&q.history.ne('unknown')),('index_PPC_complete',q.ordered_entry&q.history.ne('unknown')&q.code_fixed1.notna()),('eligible_baseline_0_or_1',q.entry_eligible)]:
            flow.append(dict(source=source,window=q.window.iloc[0],stage=stage,history='ALL',**counts(q.loc[mask])))
        windows.append(q.loc[q.ordered_entry].copy())
    q=pd.concat(windows,ignore_index=True)
    q['first_eligible']=False
    first=q.loc[q.entry_eligible].sort_values(['month1','wave']).drop_duplicates('person').index
    q.loc[first,'first_eligible']=True
    for (w,h),g in q.loc[q.entry_eligible].groupby(['window','history']):
        row=dict(source=source,window=w,history=h,stage='eligible_history_riskset',**counts(g),events=g.y_fixed.eq(1).sum(),nonevents=g.y_fixed.eq(0).sum(),endpoint_missing=(g.legal2&g.y_fixed.isna()).sum(),
                 current_k_mean=g.K1.mean(),gap01_median=g.gap01.median(),gap12_median_observed=g.loc[g.legal2,'gap12'].median(),gap12_q1=g.loc[g.legal2,'gap12'].quantile(.25),gap12_q3=g.loc[g.legal2,'gap12'].quantile(.75))
        for v in sorted(q.status2.unique()):row[v]=g.status2.eq(v).sum()
        assert row['events']+row['nonevents']+row['endpoint_missing']+g.status2.ne('living_interview').sum()==len(g)
        flow.append(row)
    transitions=[]
    for (w,h,start),g in q.loc[q.history.ne('unknown')&q.code_fixed1.notna()].groupby(['window','history','code_fixed1']):
        endpoint=g.code_fixed2.map(lambda x:str(int(x)) if pd.notna(x) else 'PPC_unobserved')
        endpoint=endpoint.where(g.legal2,g.status2)
        for end,ix in endpoint.groupby(endpoint).groups.items():
            z=g.loc[ix];transitions.append(dict(source=source,window=w,history=h,start_code=int(start),end_code=end,**counts(z)))
    save(flow,source+'_flow');save(transitions,source+'_transitions')
    q.to_parquet(PRIVATE/f'{source}_windows.parquet',index=False)
    ss={'source':source,'input':str(p.relative_to(ROOT.parent)),'input_sha256':digest(p),'windows':len(q),'eligible_windows':int(q.entry_eligible.sum()),'earliest_people':int(q.first_eligible.sum()),'valid_primary_endpoints':int(q.loc[q.entry_eligible,'y_fixed'].notna().sum())}
    (TABLES/f'{source}_preparation.json').write_text(json.dumps(ss,indent=2),encoding='utf-8')
    print(json.dumps(ss))
if __name__=='__main__':
    import sys
    build(sys.argv[1])

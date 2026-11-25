"""Bounded HRS V2 source correction; no outcome-association fitting."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
WORK = OUT.parents[1]
OLD = WORK/'PPC_IADL_History_ThreeVisit_20261115/HRS/PRIVATE/panel.parquet'
PPC = WORK/'PPC_CuCoM_IJNS_Upgrade_20261115/HRS/PRIVATE/HRS_person_wave_domain_source.parquet'
SKILL = Path.home()/'.codex/skills/seven-aging-pig-skill'
sys.dont_write_bytecode = True
sys.path.insert(0,str(SKILL/'scripts'))
import data

DISEASES = ['hibp','diab','cancr','lung','heart','strok','arthr']
YEARS = [2012,2014,2016,2018,2020]

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def disease_code(raw,q,kind):
    """Current-wave raw report and its documented wording; never future disputes."""
    a=pd.to_numeric(raw,errors='coerce');q=pd.to_numeric(q,errors='coerce')
    d=a.map({0:0.,1:1.,3:1.,4:0.,6:1.})
    if kind=='strok':d.loc[a.eq(2)]=1.
    # RAND code 6 records a confirmed prior diagnosis with no new/current disease.
    # No-since-last response when Q explicitly documents previous condition is not never.
    d.loc[a.eq(0)&q.isin([6,15,18,19])]=1.
    d.loc[raw.astype('string').str.lower().eq('.y')]=1.
    # .p (preload yes but no corroborating prior report), DK, RF, and other skips stay NA.
    return d

def main():
    (OUT/'PRIVATE').mkdir(parents=True,exist_ok=True)
    hashes={str(p.relative_to(WORK)):sha(p) for p in [OLD,PPC]}
    panel=pd.read_parquet(OLD)
    original=panel.copy()
    fields=['hhidpn']+[f'r{w}{d}{suffix}' for w in range(11,16) for d in DISEASES for suffix in ['', 'q']]
    native=data.native_info('F12656')
    assert set(fields).issubset(native['column_names'])
    raw=data.load_table('F12656',columns=fields,restore_tags=True)
    raw['person']=pd.to_numeric(raw.hhidpn).astype('Int64').astype('string')
    raw=raw.set_index('person',verify_integrity=True)
    domain=pd.read_parquet(PPC)
    domain=domain.loc[domain.current_age45plus].drop(columns='wave').rename(columns={'person_id':'person','year_nominal':'wave'})
    panel=panel.merge(domain[['person','wave','wealth','iwstat','death_year','death_month']],on=['person','wave'],validate='one_to_one')
    panel['chronic_count_old']=panel.chronic_count
    panel['physical_old']=panel.physical
    panel['psychological_old']=panel.psychological
    panel['psychological']=panel.cesd.ge(4).astype(float).where(panel.cesd.notna())
    routes=[];cross=[]
    for year,w in zip(YEARS,range(11,16)):
        ix=panel.index[panel.wave.eq(year)]
        r=raw.reindex(panel.loc[ix,'person']).set_axis(ix)
        for d in DISEASES:
            field=f'r{w}{d}'; qfield=field+'q'
            panel.loc[ix,'d_'+d]=disease_code(r[field],r[qfield],d)
            panel.loc[ix,'raw_d_'+d]=r[field].astype('string')
            panel.loc[ix,'q_d_'+d]=pd.to_numeric(r[qfield],errors='coerce')
            for (answer,question),n in r.groupby([field,qfield],dropna=False).size().items():
                routes.append(dict(wave=year,disease=d,raw=str(answer),wording=str(question),n=int(n)))
            cross.append(dict(cohort='HRS',wave=year,construct='physical_'+d,source='RAND HRS 1992–2020 V2 F12656',field=field,route_field=qfield,
                rule='0 no; 1/3 yes; 4 dispute-and-no no; 6 prior diagnosis yes; stroke2 TIA yes; documented prior-diagnosis Q6/15/18/19 plus no-new yes; .Y yes; other unknown',
                evidence='RAND 2020 V2 pp353–394, especially389–391; E-field future correction described431',
                no_future_revision=True))
    disease=panel[['d_'+d for d in DISEASES]]
    panel['chronic_known_n']=disease.notna().sum(axis=1)
    panel['chronic_count']=disease.sum(axis=1,min_count=7)
    # V2 P is identifiable from any confirmed positive; a negative needs all seven.
    panel['physical']=pd.Series(np.nan,index=panel.index)
    panel.loc[disease.eq(1).any(axis=1),'physical']=1.
    panel.loc[disease.eq(0).all(axis=1),'physical']=0.
    panel['physical_partial_positive']=disease.eq(1).any(axis=1)&panel.chronic_count.isna()
    assert len(panel)==len(original) and not panel.duplicated(['person','wave']).any()
    for c in original.columns.difference(['physical','psychological','chronic_count','chronic_known_n']):
        pd.testing.assert_series_equal(panel[c],original[c],check_dtype=False)
    ref=panel.loc[panel.wave.eq(2012)&panel.proxy.eq(0)&panel.age.ge(45)&panel[['c1','c2','c3','c4']].notna().all(axis=1)]
    rows=[]
    for c in ['c1','c2','c3','c4']:
        mean=ref[c].mean();sd=ref[c].std(ddof=1);z=(panel[c]-mean)/sd
        rows.append(dict(cohort='HRS',reference_wave=2012,reference_rule='self45plus_four_direct_complete_before_IADL_or_followup_selection',component=c,n_reference=len(ref),mean=mean,sd=sd,raw_threshold=mean-1.5*sd,comparison='<',ddof=1,weighted=False,boundary_equal_n=int(z.eq(-1.5).sum())))
    cesdref=panel.loc[panel.wave.eq(2012)&panel.proxy.eq(0)&panel.age.ge(45),'cesd'].dropna()
    rows.append(dict(cohort='HRS',reference_wave=2012,reference_rule='self45plus_complete_CESD8_before_IADL_or_followup_selection',component='cesd',n_reference=len(cesdref),mean=cesdref.mean(),sd=cesdref.std(ddof=1),raw_threshold=np.nan,comparison='CESD>=4',ddof=1,weighted=False,boundary_equal_n=np.nan))
    summary=[]
    for y,p in panel.groupby('wave'):
        both=p.chronic_count.notna()&p.chronic_count_old.notna()
        summary.append(dict(wave=int(y),n=len(p),n_four_direct=int(p[['c1','c2','c3','c4']].notna().all(axis=1).sum()),
           n_chronic_complete=int(p.chronic_count.notna().sum()),n_cesd_complete=int(p.cesd.notna().sum()),n_wealth=int(p.wealth.notna().sum()),
           n_chronic_old_new_different=int((both&p.chronic_count.ne(p.chronic_count_old)).sum()),n_chronic_now_unknown=int((p.chronic_count.isna()&p.chronic_count_old.notna()).sum()),
           n_physical_changed_among_both=int((p.physical.notna()&p.physical_old.notna()&p.physical.ne(p.physical_old)).sum())))
    pd.DataFrame(rows).to_csv(OUT/'HRS_REFERENCE_QA_V2.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv(OUT/'HRS_SOURCE_COVERAGE_V2.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(routes).to_csv(OUT/'PRIVATE/HRS_DISEASE_EXACT_ROUTES.csv',index=False)
    pd.DataFrame(cross).to_csv(OUT/'HRS_DISEASE_CROSSWALK_V2.csv',index=False,encoding='utf-8-sig')
    iadl=pd.read_csv(WORK/'PPC_IADL5_Exposure_Revision/HRS/FIVE_ITEM_CROSSWALK.csv')
    iadl.loc[iadl.item.eq('M'),'primary_mapping']='Direct 1 yes;5 no;6 health-confirmed yes;7+G0515 no;7+G0511+G0521 yes; unresolved/nonhealth unknown'
    iadl.loc[iadl.item.eq('M'),'medication_nonuse_or_phone_routing']='Use observed hypothetical ability response, not actual use; V2 source remains same contemporaneous G051/G052'
    iadl.to_csv(OUT/'HRS_IADL_CROSSWALK_V2.csv',index=False,encoding='utf-8-sig')
    support=[]
    for y in YEARS[:-1]:
        a=panel.loc[panel.wave.eq(y)]
        b=panel.loc[panel.wave.eq(y+2),['person','age','c3','cognition_mode']].rename(columns={'age':'next_age','c3':'next_orientation','cognition_mode':'next_mode'})
        k=a.merge(b,on='person',how='left',validate='one_to_one')
        for group,sel in [('all',pd.Series(True,index=k.index)),('current_four_complete_under65',k[['c1','c2','c3','c4']].notna().all(axis=1)&k.age.lt(65)),('current_four_complete_ge65',k[['c1','c2','c3','c4']].notna().all(axis=1)&k.age.ge(65))]:
            z=k.loc[sel]
            support.append(dict(wave=y,group=group,n=len(z),next_observed=int(z.next_status.eq('observed').sum()),next_recorded_living_nonresponse=int(z.next_status.eq('known_alive_noninterview').sum()),next_dead=int(z.next_status.eq('dead').sum()),next_orientation_valid=int(z.next_orientation.notna().sum()),next_interviewed_under65=int(z.next_age.lt(65).sum()),next_web=int(z.next_mode.eq('web').sum())))
    support=pd.DataFrame(support)
    support.to_csv(OUT/'PRIVATE/HRS_OBSERVATION_SUPPORT_EXACT_V2.csv',index=False)
    support.loc[~support.group.eq('current_four_complete_under65')|support.wave.eq(2016)].to_csv(OUT/'HRS_OBSERVATION_SUPPORT_V2.csv',index=False,encoding='utf-8-sig')
    panel.to_parquet(OUT/'PRIVATE/panel.parquet',index=False)
    assert all(sha(WORK/k)==v for k,v in hashes.items())
    (OUT/'HRS_SOURCE_QA_V2.json').write_text(json.dumps(dict(old_inputs_unchanged=True,hashes=hashes,output_panel_sha256=sha(OUT/'PRIVATE/panel.parquet'),rows=len(panel),people=int(panel.person.nunique()),reference_n=len(ref),preserved_oldcode4=True,no_models=True),ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()

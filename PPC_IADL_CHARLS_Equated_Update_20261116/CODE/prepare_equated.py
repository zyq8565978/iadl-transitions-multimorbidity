"""Published fixed CHARLS memory equating; preserve all V2 inputs read-only."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent/'PPC_IADL_Dynamic_V2_20261115'
MEASURE = ROOT.parent/'PPC_CHARLS_COGNITION_RECONCILIATION_20261115/MEASUREMENT'
PRIVATE, TABLES = ROOT/'PRIVATE', ROOT/'TABLES'
SCORES = ['c1','c2','c3','c4']
DOMAINS = {i:i.bit_count() for i in range(8)}


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def counts(d):
    return dict(windows=len(d), people=d.person.nunique(), families=d.cluster.nunique())


def main():
    for directory in (PRIVATE,TABLES,ROOT/'MEASUREMENT'):
        directory.mkdir(parents=True,exist_ok=True)
    inputs=[OLD/'PRIVATE/CHARLS_panel.parquet',OLD/'PRIVATE/CHARLS_windows.parquet',
            OLD/'TABLES/CHARLS_reference.csv',OLD/'FROZEN_PROTOCOL_V2.md',MEASURE/'eqt_crosswalk_table.xlsx']
    before={str(path):digest(path) for path in inputs}
    p=pd.read_parquet(inputs[0]); q=pd.read_parquet(inputs[1])
    ref=pd.read_csv(inputs[2]).set_index('component')
    xwalk=pd.read_excel(inputs[4])
    assert digest(inputs[4])=='c2951e4ddead427348e3ad1a86e4e1c8ef47e5bc31ec41bef912e9b2090be798'
    assert not p.duplicated(['person','wave']).any()
    assert not q.duplicated(['person','window']).any()
    original=p[SCORES].copy()
    valid=p[SCORES].notna().all(axis=1)&p.proxy.eq(0)
    zraw=(p[SCORES]-ref.loc[SCORES,'mean'])/ref.loc[SCORES,'sd']
    p['cog_z_uncalibrated']=zraw.mean(axis=1).where(valid)
    p['cog_uncalibrated']=zraw.lt(-1.5).any(axis=1).astype(float).where(valid)
    p['code_uncalibrated']=p.physical+2*p.psychological_v2+4*p.cog_uncalibrated
    later=p.wave.isin([2018,2020])
    for field,test in [('c1','imrc'),('c2','dlrc')]:
        p[field+'_raw']=p[field]
        mapping=xwalk.loc[xwalk.test.eq(test)].set_index('original_score').eqt_score
        assert set(mapping.index)==set(range(11))
        assert p.loc[later,field].dropna().isin(range(11)).all()
        p.loc[later,field]=p.loc[later,field].map(mapping)
    pd.testing.assert_frame_equal(original.loc[~later],p.loc[~later,SCORES])
    assert p[SCORES].isna().equals(original.isna())
    z=(p[SCORES]-ref.loc[SCORES,'mean'])/ref.loc[SCORES,'sd']
    p['cog_reference_comparable']=p.wave.isin([2013,2015,2018,2020])
    valid &= p.cog_reference_comparable
    p['cog_z']=z.mean(axis=1).where(valid)
    p['cog_fixed']=z.lt(-1.5).any(axis=1).astype(float).where(valid)
    p['partial_cognitive_positive']=z.lt(-1.5).any(axis=1)&~valid&p.proxy.eq(0)&p.cog_reference_comparable
    p['code_fixed']=p.physical+2*p.psychological_v2+4*p.cog_fixed
    for field in SCORES:
        p['z'+field]=z[field].where(p.cog_reference_comparable&p.proxy.eq(0))
    lookup=p.set_index(['person','wave'])
    wave0=q.window.str.split('->').str[0].astype(int)
    for j,wave in [(0,wave0),(1,q.wave),(2,q.next_wave)]:
        key=pd.MultiIndex.from_arrays([q.person,wave])
        for field in ['cog_z','code_fixed','cog_z_uncalibrated','code_uncalibrated']:
            q[field+str(j)]=lookup[field].reindex(key).to_numpy()
        for tag in ['fixed','uncalibrated']:
            q[f'd_{tag}{j}']=q[f'code_{tag}{j}'].map(DOMAINS)
    for tag in ['fixed','uncalibrated']:
        q['y_'+tag]=q[f'd_{tag}2'].ge(2).astype(float).where(q.legal2&q[f'd_{tag}2'].notna()&q[f'd_{tag}1'].le(1))
    q['entry_eligible']=q.ordered_entry&q.history.ne('unknown')&q.d_fixed1.le(1)&q[['cesd1','cog_z1','chronic_count1']].notna().all(axis=1)
    q['first_eligible']=False
    first=q.loc[q.entry_eligible].sort_values(['month1','wave']).drop_duplicates('person').index
    q.loc[first,'first_eligible']=True
    assert q.entry_eligible.sum()==10024 and q.first_eligible.sum()==7738
    assert q.loc[q.entry_eligible,'y_fixed'].notna().sum()==6421
    assert not q.loc[~q.legal2,'y_fixed'].notna().any()
    p.to_parquet(PRIVATE/'CHARLS_panel.parquet',index=False)
    q.to_parquet(PRIVATE/'CHARLS_windows.parquet',index=False)
    shutil.copy2(inputs[2],TABLES/'CHARLS_reference.csv')
    shutil.copy2(inputs[4],ROOT/'MEASUREMENT/eqt_crosswalk_table.xlsx')
    xwalk.to_csv(ROOT/'MEASUREMENT/published_equating_crosswalk.csv',index=False,encoding='utf-8-sig')
    flow=[]; details=[]; measurement=[]
    for window,g in q.groupby('window'):
        for stage,mask in [('ordered_index_age45',g.ordered_entry),('complete_history',g.history.ne('unknown')),
                           ('index_PPC_complete',g.history.ne('unknown')&g.code_fixed1.notna()),
                           ('eligible_baseline_0_or_1',g.entry_eligible),
                           ('observed_PPC_endpoint',g.entry_eligible&g.y_fixed.notna()),
                           ('earliest_eligible',g.first_eligible)]:
            flow.append(dict(source='CHARLS',window=window,stage=stage,**counts(g.loc[mask])))
        for history,h in g.loc[g.entry_eligible].groupby('history'):
            row=dict(source='CHARLS',window=window,history=history,**counts(h),events=int(h.y_fixed.eq(1).sum()),
                nonevents=int(h.y_fixed.eq(0).sum()),endpoint_missing=int((h.legal2&h.y_fixed.isna()).sum()),
                gap01_median=h.gap01.median(),gap12_median_observed=h.loc[h.legal2,'gap12'].median())
            for state in sorted(q.status2.unique()):row[state]=int(h.status2.eq(state).sum())
            assert row['events']+row['nonevents']+row['endpoint_missing']+h.status2.ne('living_interview').sum()==len(h)
            details.append(row)
    for wave,g in p.groupby('wave'):
        a=g.loc[g.age.ge(45)]
        measurement.append(dict(source='CHARLS',wave=wave,age45_n=len(a),four_test_scorable=int(a.cog_z.notna().sum()),
            fixed_ppc=int(a.code_fixed.notna().sum()),raw_fixed_ppc=int(a.code_uncalibrated.notna().sum()),
            fixed_vs_raw_cognitive_changed=int((a.cog_fixed.notna()&a.cog_uncalibrated.notna()&a.cog_fixed.ne(a.cog_uncalibrated)).sum()),
            minimum_immediate=a.c1.min(),maximum_immediate=a.c1.max(),minimum_delayed=a.c2.min(),maximum_delayed=a.c2.max()))
    pd.DataFrame(flow).to_csv(TABLES/'CHARLS_flow.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(details).to_csv(PRIVATE/'CHARLS_flow_history_exact.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(measurement).to_csv(TABLES/'CHARLS_measurement_qa.csv',index=False,encoding='utf-8-sig')
    summary=dict(source='CHARLS',preparation='fixed_published_equating',eligible_windows=int(q.entry_eligible.sum()),
        earliest_people=int(q.first_eligible.sum()),valid_primary_endpoints=int(q.loc[q.entry_eligible,'y_fixed'].notna().sum()),
        valid_endpoint_unique_people=int(q.loc[q.entry_eligible&q.y_fixed.notna(),'person'].nunique()),
        old_inputs_unchanged=all(digest(Path(path))==value for path,value in before.items()),input_sha256=before,
        panel_sha256=digest(PRIVATE/'CHARLS_panel.parquet'),windows_sha256=digest(PRIVATE/'CHARLS_windows.parquet'),
        imputation_performed=False,association_fits_in_preparation=0)
    assert summary['old_inputs_unchanged']
    (TABLES/'CHARLS_preparation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='input_sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

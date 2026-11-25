"""Source-only CHARLS V2 checks; no association fit and no old output writes."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
WORK = HERE.parents[1]
SKILL = Path.home() / '.codex/skills/seven-aging-pig-skill'
sys.dont_write_bytecode = True
sys.path.insert(0, str(SKILL / 'scripts'))
import data as pig

OLD = WORK / 'PPC_IADL_History_ThreeVisit_20261115/CHARLS/PRIVATE/panel.parquet'
RAW = WORK / '原始数据/01_CHARLS_中国/01_官方原始数据'
SEVEN = (1, 3, 4, 5, 7, 8, 13)
FIDS = {2013: 'F08831', 2015: 'F08868', 2018: 'F08889', 2020: 'F08908'}
INPUTS = []


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_native(fid, prefixes):
    path = RAW / pig.table_info(fid)['source'].split('charls_raw_data/')[1]
    with pd.io.stata.StataReader(path) as reader:
        labels = reader.variable_labels()
    columns = [k for k in labels if k == 'ID' or any(k.startswith(v) for v in prefixes)]
    x = pd.read_stata(path, columns=columns, convert_categoricals=False)
    ids = x.pop('ID').astype('string').str.strip()
    ids = ids.mask(ids.str.len().eq(11), ids.str[:-2] + '0' + ids.str[-2:])
    x.index = pd.Index(ids, name='person')
    assert x.index.is_unique
    INPUTS.append(dict(source=path.relative_to(WORK).as_posix(), fid=fid, sha256=sha(path)))
    return x


def value(d, name):
    return d[name].astype(float) if name in d else pd.Series(np.nan, index=d.index)


def disease(d, year, k):
    if year == 2020:
        nk = k + (k >= 13)
        a, b = value(d, f'da003_{nk}_'), value(d, f'da002_{nk}_')
        return a.map({1: 1., 2: 0.}).fillna(pd.Series(1., index=d.index).where(b.isin([1, 2, 3])))
    a = value(d, f'da007_{k}_')
    yes, no = a.eq(1), a.eq(2)
    if year in (2013, 2015):
        preload, agree, new = (value(d, f'zda007_{k}_'), value(d, f'da007_w2_1_{k}_'),
                              value(d, f'da007_w2_2_{k}_'))
        yes |= preload.eq(1) & ~agree.eq(2) & ~new.eq(2)
        yes |= new.eq(1)
        if k not in (1, 5, 11):
            yes |= preload.eq(2) & agree.eq(2)
        no |= new.eq(2)
    else:
        preload, change = value(d, f'zdiagnosed_{k}_'), value(d, f'da010_w2_2_{k}_')
        yes |= preload.eq(1) & ~change.eq(99) & ~a.eq(2)
        no |= change.eq(99) & ~a.eq(1)
    return pd.Series(np.select([yes & ~no, no & ~yes], [1., 0.], default=np.nan), index=d.index)


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    p = pd.read_parquet(OLD)
    oldhash = sha(OLD)
    INPUTS.append(dict(source=OLD.relative_to(WORK).as_posix(), fid='prior_clean_panel', sha256=oldhash))
    assert not p.duplicated(['person', 'wave']).any()
    audit = []
    for year, fid in FIDS.items():
        raw = read_native(fid, ['da007', 'zda007', 'da010_w2_2', 'zdiagnosed', 'da003', 'da002', 'dc'])
        q = p.loc[p.wave.eq(year)].set_index('person')
        raw = raw.reindex(q.index)
        conditions = pd.concat([disease(raw, year, k) for k in SEVEN], axis=1)
        count = conditions.sum(axis=1, min_count=7)
        phys = pd.Series(np.select([conditions.eq(1).any(axis=1), conditions.eq(0).all(axis=1)],
                                  [1., 0.], default=np.nan), index=q.index)
        np.testing.assert_allclose(count.to_numpy(), q.chronic_count.to_numpy(float), equal_nan=True)
        np.testing.assert_allclose(phys.to_numpy(), q.physical.to_numpy(float), equal_nan=True)
        mood_source = read_native('F08884', ['dc']) if year == 2018 else raw
        mood_source = mood_source.reindex(q.index)
        first = 16 if year == 2020 else 9
        items = [value(mood_source, f'dc{j:03}').where(lambda v: v.between(1, 4))
                 for j in range(first, first + 10)]
        cesd = pd.concat([4-v if j in (4, 7) else v-1 for j, v in enumerate(items)], axis=1).sum(axis=1, min_count=10)
        cesd = cesd.where(q.proxy.eq(0))
        np.testing.assert_allclose(cesd.to_numpy(), q.cesd.to_numpy(float), equal_nan=True)
        a = q.age.ge(45)
        four = q[['c1', 'c2', 'c3', 'c4']].notna().all(axis=1)
        audit.append(dict(wave=year, age45_n=int(a.sum()), self45_n=int((a&q.proxy.eq(0)).sum()),
            iadl5_complete_n=int((a&q[['xM','xF','xP','xS','xC']].notna().all(axis=1)).sum()),
            physical7_complete_n=int((a&count.notna()).sum()), physical_domain_known_n=int((a&phys.notna()).sum()),
            cesd10_complete_self_n=int((a&cesd.notna()).sum()), four_cognitive_complete_self_n=int((a&four).sum()),
            physical_native_match=True, cesd_native_match=True, cognitive_2013_comparable=year <= 2015))
    refs = []
    ref = p.loc[p.wave.eq(2013)&p.age.ge(45)&p.proxy.eq(0)&p[['c1','c2','c3','c4']].notna().all(axis=1)]
    for col, name in zip(['c1','c2','c3','c4'], ['immediate_recall','delayed_recall','orientation','serial7']):
        mean, sd = ref[col].mean(), ref[col].std(ddof=1)
        assert sd > 0
        z = (p[col]-mean)/sd
        eligible = p.wave.isin([2013,2015]) & p.age.ge(45)&p.proxy.eq(0)&p[['c1','c2','c3','c4']].notna().all(axis=1)
        refs.append(dict(domain='cognition', component=col, name=name, reference_wave=2013,
            reference_n=len(ref), mean=mean, sd=sd, ddof=1, cutoff=mean-1.5*sd,
            strict_lt_vs_le_boundary_changes=int((eligible & z.eq(-1.5)).sum()),
            target_wave_scope='2013;2015', reference_definition='self45plus_all_four_complete_unweighted'))
    mood = p.loc[p.wave.eq(2013)&p.age.ge(45)&p.proxy.eq(0)&p.cesd.notna(),'cesd']
    refs.append(dict(domain='psychological', component='cesd', name='CESD10', reference_wave=2013,
        reference_n=len(mood), mean=mood.mean(), sd=mood.std(ddof=1), ddof=1, cutoff=np.nan,
        strict_lt_vs_le_boundary_changes=np.nan, target_wave_scope='2013;2015;2018;2020',
        reference_definition='self45plus_complete_cesd10_unweighted'))
    pd.DataFrame(audit).to_csv(HERE/'CHARLS_FIELD_COVERAGE_V2.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(refs).to_csv(HERE/'CHARLS_REFERENCE_PARAMETERS_V2.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(INPUTS).to_csv(HERE/'CHARLS_SOURCE_HASHES_V2.csv', index=False, encoding='utf-8-sig')
    # Pairwise repeat support is not an exposure-outcome estimate or an analysis sample.
    repeated=[]
    for start, end in [(2013,2015),(2015,2018),(2018,2020),(2015,2020)]:
        a=p.loc[p.wave.eq(start),['person','month','age','chronic_count','cesd']]
        b=p.loc[p.wave.eq(end),['person','month','chronic_count','cesd']]
        ab=a.merge(b,on='person',suffixes=('_start','_end'),validate='one_to_one')
        ab=ab.loc[ab.age.ge(45)&ab.month_end.gt(ab.month_start)]
        for field in ['chronic_count','cesd']:
            d=ab.loc[ab[[field+'_start',field+'_end']].notna().all(axis=1)]
            elapsed=(d.month_end-d.month_start)/12
            repeated.append(dict(start=start,end=end,domain=field,people=len(d),
                interval_years_q25=elapsed.quantile(.25),interval_years_median=elapsed.median(),interval_years_q75=elapsed.quantile(.75)))
    pd.DataFrame(repeated).to_csv(HERE/'CHARLS_REPEAT_DOMAIN_SUPPORT_V2.csv', index=False, encoding='utf-8-sig')
    assert oldhash == sha(OLD)
    status=dict(source_native_count_and_cesd_checks_pass=True, old_panel_unchanged=True,
        old_panel_sha256=oldhash, new_panel_needed=False, four_complete_reference_n=len(ref),
        fixed_2013_ppc_windows_supported=0, reason='2018/2020 recall regimen incompatible with fixed 2013 four-test metric',
        physical_and_psychological_followup_supported=True, effect_fits=0)
    (HERE/'CHARLS_MEASUREMENT_QA_V2.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False))


if __name__ == '__main__':
    main()

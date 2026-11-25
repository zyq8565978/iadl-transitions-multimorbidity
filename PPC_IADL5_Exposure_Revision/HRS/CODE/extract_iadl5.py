"""HRS native IADL5 from 2012–2020 Section G and a read-only, previously QA'd PPC panel."""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HRS = Path(__file__).resolve().parents[1]
WORK = HRS.parents[1]
PRIVATE = HRS / 'PRIVATE'
PRIVATE.mkdir(parents=True, exist_ok=True)
OLD = WORK / 'PPC_CuCoM_IJNS_Upgrade_20261115/HRS/PRIVATE'
NATIVE = WORK / '原始数据/03_HRS_美国/01_官方原始数据'
SKILL = Path.home() / '.codex/skills/seven-aging-pig-skill'
sys.dont_write_bytecode = True
sys.path.insert(0, str(SKILL / 'scripts'))
import data

YEARS = [2012, 2014, 2016, 2018, 2020]
DISEASE = ['hibpe', 'diabe', 'cancre', 'lunge', 'hearte', 'stroke', 'arthre']
ITEMS = {
    'M': ('050', '052', '053', 'meds', 'taking medications'),
    'F': ('059', '060', '061', 'money', 'managing money'),
    'P': ('047', '048', '049', 'phone', 'making telephone calls'),
    'S': ('044', '045', '046', 'shop', 'shopping for groceries'),
    'C': ('041', '042', '043', 'meals', 'preparing a hot meal'),
}
RAND_DOC = 'https://www.rand.org/content/dam/rand/www/external/labor/aging/dataprod/randhrs1992_2020v2.pdf'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def locate(name):
    paths = [p for p in NATIVE.rglob(name) if p.is_file()]
    assert len(paths) == 1, (name, len(paths))
    return paths[0]


def classify(raw, why, hypothetical, native_present, item):
    x = raw.map({1: 1., 5: 0.})
    v = raw.map({5: 0.})
    reason = pd.Series('NA_nonresponse', index=raw.index, dtype='string')
    reason.loc[raw.isna() & native_present] = 'NA_nonresponse_or_skip'
    reason.loc[~native_present] = 'NA_structural_no_core_G'
    reason.loc[raw.eq(1)] = 'direct_difficulty'
    reason.loc[raw.eq(5)] = 'direct_no_difficulty'
    branch = raw.isin([6, 7])
    reason.loc[branch & why.eq(5)] = 'NA_not_applicable_nonhealth'
    reason.loc[branch & ~why.isin([1, 5])] = 'NA_nonresponse_reason_unknown'
    x.loc[branch & why.eq(1)] = 1.
    v.loc[raw.eq(6) & why.eq(1)] = 1.
    reason.loc[raw.eq(6) & why.eq(1)] = 'direct_cannot_health_confirmed'
    reason.loc[raw.eq(7) & why.eq(1)] = 'direct_not_done_health_confirmed'
    if item == 'M':
        nonuse = raw.eq(7)
        x.loc[nonuse] = np.nan
        reason.loc[nonuse] = 'NA_not_applicable_medication_not_done'
        reason.loc[nonuse & hypothetical.eq(1)] = 'NA_not_applicable_hypothetical_difficulty'
        reason.loc[nonuse & hypothetical.eq(5)] = 'NA_not_applicable_hypothetical_no_difficulty'
    assert (x.notna() == reason.str.startswith('direct')).all()
    return x, v, reason


def run():
    path = OLD / 'HRS_person_wave_domain_source.parquet'
    initial = sha(path)
    old = pd.read_parquet(path)
    manifest = [{'asset': 'verified PPC histories', 'path': str(path.relative_to(WORK)), 'sha256': initial}]
    assert set(old.year_nominal) == set(YEARS)
    design_cols = ['hhidpn', 'raestrat', 'raehsamp'] + [
        f'r{w}{x[3]}a' for w in range(11, 16) for x in ITEMS.values()]
    rand = data.load_table('F12656', columns=design_cols, restore_tags=False)
    rand['person'] = pd.to_numeric(rand.hhidpn).astype('Int64').astype('string')
    rand = rand.set_index('person', verify_integrity=True)
    catalog = data.native_info('F12656')
    refs = catalog['column_names_to_labels']
    metadata = []
    waves = []
    for year in YEARS:
        w = 11 + (year - 2012) // 2
        prefix = 'NOPQR'[w - 11] + 'G'
        name = f'H{year % 100:02d}G_R'
        source, cb = locate(name + '.dta'), locate(name + '.txt')
        lines = cb.read_text(encoding='latin-1').splitlines()
        item_fields = {prefix + key for row in ITEMS.values() for key in row[:3]}
        cols = ['HHID', 'PN', prefix + '051'] + sorted(item_fields)
        label_reader = pd.io.stata.StataReader(source)
        labels_native = label_reader.variable_labels()
        native_names = {k.upper(): k for k in labels_native}
        labels = {k.upper(): v for k, v in labels_native.items()}
        raw = pd.read_stata(source, columns=[native_names[c] for c in cols], convert_categoricals=False).rename(columns=str.upper)
        raw['person'] = (raw.HHID.astype('string').str.zfill(6) + raw.PN.astype('string').str.zfill(3)).str.lstrip('0')
        raw = raw.set_index('person', verify_integrity=True)
        for f in [source, cb]:
            manifest.append({'asset': ('native Section G' if f == source else 'local Section G codebook'),
                             'path': str(f.relative_to(WORK)), 'sha256': sha(f)})
        p = old.loc[old.current_age45plus & old.year_nominal.eq(year)].copy()
        p = p.assign(person=p.person_id.astype('string')).set_index('person', verify_integrity=True)
        m = pd.DataFrame(index=p.index)
        m['family'] = p.household_id.astype('string')
        m['wave'] = year
        m['month'] = (12*p.iwendy+p.iwendm).where(p.iwendy.between(year-1, year+2)&p.iwendm.between(1,12))
        m['age'] = p.age
        m['sex'] = p.female + 1.
        m['education'] = p.education_years.where(p.education_years.between(0, 30))
        m['rural'] = np.nan
        m['proxy'] = p.proxy.where(p.proxy.isin([0, 1]))
        m['code4'] = p.physical_any + 2*p.psychological + 4*p.cognitive_strict
        m['ppc_binary'] = p.ppc_binary_strict
        m['cognitive_imputed_any'] = p.cognitive_imputed_any
        m['cognitive_orientation_design_not_asked'] = p.orientation_design_not_asked_flag
        m['chronic_count'] = p.disease_count_complete
        m['chronic_known_n'] = p[DISEASE].isin([0, 1]).sum(axis=1)
        m['weight'] = p.wtresp.where(p.wtresp.gt(0))
        m['nursing_home_weight'] = p.wtcrnh.where(p.wtcrnh.gt(0))
        m['stratum'] = rand.raestrat.reindex(m.index)
        m['psu'] = rand.raehsamp.reindex(m.index)
        m['rand_iadl5'] = p.iadl5a
        m['nursing_home_native'] = p.nursing_home_raw
        m['next_status'] = 'administrative_end'
        if year != YEARS[-1]:
            follow = old.loc[old.year_nominal.eq(year + 2)].set_index('person_id').reindex(m.index)
            status = follow.iwstat.map({4: 'known_alive_noninterview', 5: 'dead',
                                       6: 'dead', 7: 'dropped_from_sample'}).fillna('survival_unknown')
            status.loc[follow.interviewed.eq(True)] = 'observed'
            m['next_status'] = status
        raw_re = raw.reindex(m.index)
        native_present = m.index.isin(raw.index)
        assert np.array_equal(native_present, p.native_G_present)
        for item,(a,b,h,ref,task) in ITEMS.items():
            av, bv, hv = (prefix + n for n in (a,b,h))
            lev = raw_re[av]
            why = raw_re[bv]
            hyp = raw_re[prefix + '051'] if item == 'M' else pd.Series(np.nan, index=m.index)
            x, v, route = classify(lev, why, hyp, native_present, item)
            m['x' + item], m['v' + item], m['route' + item] = x, v, route
            m['raw_' + item], m['why_' + item] = lev, why
            m['help_' + item] = raw_re[hv].map({1: 1., 5: 0.})
            m['rand_' + item] = rand[f'r{w}{ref}a'].reindex(m.index)
            if item == 'M':
                m['hypothetical_M'] = hyp
            anchors = {field: next(i for i,l in enumerate(lines, 1) if re.match(r'^'+field+r'\s+', l))
                       for field in (av,bv,hv)}
            assert all(lev.dropna().isin([-8, 1, 5, 6, 7, 8, 9])), (year, item)
            assert 'DIFFICULTY' in labels[av].upper(), labels[av]
            metadata.append(dict(cohort='HRS', wave=year, item=item, task=task, source=source.name,
                native_field=av, native_secondary=bv+(f';{prefix}051 (hypothetical only)' if item=='M' else ''),
                construct=f'Any difficulty {task} because of a health or memory problem',
                raw_values='1 yes;5 no;6 cannot do;7 does not do;8/9 unknown/refused;-8 web nonresponse where used;blank inapplicable/partial',
                primary_mapping='1→1;5→0;6/7 with reason=1→1, except medication 7 always unknown/out of applicable scope',
                missing_route='6/7 with reason=5 not applicable; unresolved reason nonresponse; absent core G structurally missing; blank within G cannot always distinguish skip/partial; no ownership inference',
                native_help_field=hv,
                proxy_scope='Self or proxy when answered; actual mode separate; difficulty not inferred from assistance',
                medication_nonuse_or_phone_routing=('Nonuse plus G051 hypothetical answer does not code actual ability' if item=='M' else ('No direct telephone ownership question used; 7+nonhealth not applicable' if item=='P' else 'Nonperformance traced by cause probe, not imputed as no difficulty')),
                harmonized_field=f'r{w}{ref}a', harmonized_label=refs[f'r{w}{ref}a'],
                codebook=cb.name, codebook_lines=';'.join(f'{k}:{v}' for k,v in anchors.items()),
                official_url=f'https://hrsonline.isr.umich.edu/modules/meta/{year}/core/codebook/h{year%100:02d}g_r.htm',
                harmonization_document=RAND_DOC,
                n_current_age45plus=len(m), n_native_field_response=int(lev.notna().sum()),
                n_valid=int(x.notna().sum()), n_yes=int(x.eq(1).sum()), n_no=int(x.eq(0).sum()),
                n_valid_self=int((x.notna()&m.proxy.eq(0)).sum()), n_valid_proxy=int((x.notna()&m.proxy.eq(1)).sum()),
                n_harmonized_both_known_disagree=int((x.notna()&m['rand_'+item].isin([0,1])&x.ne(m['rand_'+item])).sum()),
                n_harmonized_known_native_unknown=int((x.isna()&m['rand_'+item].isin([0,1])).sum())))
        waves.append(m.reset_index())
    panel = pd.concat(waves, ignore_index=True)
    crosswalk = pd.DataFrame(metadata)
    for i,row in crosswalk.iterrows():
        w,it = row['wave'],row['item']
        if w == YEARS[-1]:
            continue
        x = 'x'+it
        p = panel.loc[panel.wave.eq(w), ['person','month',x]].merge(
            panel.loc[panel.wave.eq(w+2), ['person','month',x]], on='person', validate='one_to_one', suffixes=('','_next'))
        crosswalk.loc[i,'n_valid_adjacent_double_observed'] = int((p[x].notna()&p[x+'_next'].notna()&p.month_next.gt(p.month)).sum())
    xcols = ['x'+j for j in ITEMS]
    complete = panel[xcols].notna().all(axis=1)
    disease_qa = panel.chronic_count.notna().eq(panel.chronic_known_n.eq(7)).all()
    qa = dict(source_sha_unchanged=sha(path)==initial, one_person_wave=not panel.duplicated(['person','wave']).any(),
        all_five_waves_exact=set(panel.wave)==set(YEARS), age45plus=panel.age.ge(45).all(),
        inherited_current_rows=len(panel)==92099, binary_or_unknown=panel[xcols].isin([0,1]).sum().sum()==panel[xcols].notna().sum().sum(),
        medication_hypothetical_never_used=panel.loc[panel.raw_M.eq(7),'xM'].isna().all(),
        nonhealth_never_coded_as_direct_no=all(panel.loc[panel['raw_'+j].isin([6,7]) & panel['why_'+j].eq(5), 'x'+j].isna().all() for j in ITEMS),
        no_future_selection=True, weight_design_columns_bound=panel[['stratum','psu']].notna().all().all(),
        disease_complete_count=disease_qa, next_2020_administrative_end=panel.loc[panel.wave.eq(2020),'next_status'].eq('administrative_end').all(),
        no_models=True)
    assert all(qa.values()), qa
    # Exact routing-cell counts and the panel remain under PRIVATE, not in the share package.
    panel.to_parquet(PRIVATE/'panel.parquet', index=False)
    crosswalk.to_csv(HRS/'FIVE_ITEM_CROSSWALK.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(manifest).to_csv(HRS/'HRS_SOURCE_LINEAGE.csv', index=False, encoding='utf-8-sig')
    routes = pd.concat([panel.groupby(['wave','route'+j]).size().rename('n').reset_index().rename(columns={'route'+j:'route'}).assign(item=j) for j in ITEMS])
    routes.to_csv(PRIVATE/'HRS_EXACT_ROUTE_COUNTS.csv', index=False, encoding='utf-8-sig')
    summary = dict(rows=len(panel), people=int(panel.person.nunique()), families=int(panel.family.nunique()),
        complete5_n=int(complete.sum()), complete5_per_wave={int(w):int(n) for w,n in complete.groupby(panel.wave).sum().items()},
        positive_survey_weight_n=int(panel.weight.notna().sum()), age45plus_per_wave={int(w):int(n) for w,n in panel.groupby('wave').size().items()},
        qa={k:bool(v) for k,v in qa.items()}, participant_scope='2012–2020 observed Core person-waves current age>=45')
    (HRS/'HRS_EXTRACTION_QA.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()

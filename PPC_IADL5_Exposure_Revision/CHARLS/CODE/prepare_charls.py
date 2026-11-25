"""Read-only native CHARLS IADL crosswalk and private panel; no association fitting."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
WORK = HERE.parents[1]
RAW = WORK / '原始数据/01_CHARLS_中国/01_官方原始数据'
PPC = WORK / 'PPC_CuCoM_IJNS_Upgrade_20261115/CHARLS/PRIVATE/charls_person_panel.parquet'
HARM = RAW.parent / '02_协调与简化数据/harmonized charls/H_CHARLS_D_Data.dta'
SKILL = Path.home() / '.codex/skills/seven-aging-pig-skill'
sys.dont_write_bytecode = True
sys.path.insert(0, str(SKILL / 'scripts'))
import data as pig

WAVES = (2011, 2013, 2015, 2018, 2020)
FILES = dict(zip(WAVES, ('F08791', 'F08831', 'F08868', 'F08889', 'F08908')))
ITEMS = dict(M='Taking medications at the right dose/time', F='Managing money or assets', P='Making telephone calls', S='Shopping for groceries', C='Preparing hot meals')
OLD = dict(M='db020', F='db019', P='db035', S='db018', C='db017')
NEW = dict(M='db020', F='db022', P='db018', S='db016', C='db014')
HITEMS = dict(M='medsa', F='moneya', P='phonea', S='shopa', C='mealsa')
INHERITED = dict(M='meds', F='finance', P='phone', S='shop', C='meals')
DOCS = {
    2011: ('CHARLS_codebook.pdf', '679–680'),
    2013: ('CHARLS_Wave2_Main_Questionnaire.pdf', '99, 103–106'),
    2015: ('CHARLS_2015_Questionnaire.pdf', '80, 82, 84–86'),
    2018: ('CHARLS_2018_Household_Questionnaire.pdf', '95–96, 99–100'),
    2020: ('CHARLS_2020_Questionnaire_Chinese.pdf', '34–40'),
}
HARM_URL = 'https://charls.charlsdata.com/Public/ashelf/public/uploads/document/harmonized_charls/application/Harmonized_CHARLS_D.pdf'
W5_URL = 'https://charls.charlsdata.com/Public/ashelf/public/uploads/document/2020-charls-wave5/application/CHARLS_2020_Questionnaire_English.pdf'
SCREEN = ('db001', 'db003', 'db004', 'db005', 'db006', 'db007', 'db008', 'db009')
MANIFEST = []


def source(path, role):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    MANIFEST.append(dict(path=path.relative_to(WORK).as_posix(), role=role, sha256=digest, bytes=path.stat().st_size))


def id_index(frame):
    ids = frame.pop('ID').astype('string').str.strip()
    ids = ids.mask(ids.str.len().eq(11), ids.str[:-2] + '0' + ids.str[-2:])
    assert ids.str.fullmatch(r'\d{12}').all() and ids.is_unique
    frame.index = pd.Index(ids.astype(str), name='person')
    return frame


def csv(frame, name):
    pd.DataFrame(frame).to_csv(HERE / f'{name}.csv', index=False, encoding='utf-8-sig')


def main():
    (HERE / 'PRIVATE').mkdir(exist_ok=True)
    source(PPC, 'prior approved CHARLS participant-wave PPC, covariates, actual dates and next-wave vital/interview status; immutable')
    source(HARM, 'Gateway Harmonized CHARLS D W1–W4; item and aggregate benchmarks only')
    panel = pd.read_parquet(PPC)
    hcols = ['ID'] + [f'r{j}{k}' for j in range(1, 5) for k in [*HITEMS.values(), 'iadla', 'iadlza', 'iadlzam']
                          if j > 1 or k not in ['phonea', 'iadla', 'iadlza', 'iadlzam']]
    harm = id_index(pd.read_stata(HARM, columns=hcols, convert_categoricals=False))
    blocks, audit, summaries, cards = [], [], [], []
    for j, wave in enumerate(WAVES, 1):
        native_map = NEW if wave == 2020 else OLD
        path = RAW / pig.table_info(FILES[wave])['source'].split('charls_raw_data/')[1]
        qfile, qpages = DOCS[wave]
        source(path, f'{FILES[wave]} native Health Status and Functioning')
        source(RAW / str(wave) / qfile, f'official documentation; PDF pages {qpages}')
        with pd.io.stata.StataReader(path) as reader:
            labels = reader.variable_labels()
            values = reader.value_labels()
        d = id_index(panel.loc[panel.wave.eq(wave)].copy())
        help_cols = {k:(f'db{int(v[2:])+1:03}' if wave == 2020 else v+'_w2') for k,v in native_map.items()} if wave != 2011 else {}
        native_cols = [c for c in [*native_map.values(), *help_cols.values(), *SCREEN] if c in labels]
        raw = id_index(pd.read_stata(path, columns=['ID', *dict.fromkeys(native_cols)], convert_categoricals=False))
        assert set(raw.index) == set(d.index)
        assert ('db035' not in raw) if wave == 2011 else native_map['P'] in raw
        raw = raw.reindex(d.index)
        screened = raw[list(SCREEN)].eq(1).all(axis=1) if wave < 2020 else pd.Series(False, index=d.index)
        block = d[['family', 'wave', 'age', 'sex', 'education', 'code4', 'ppc4', 'proxy', 'rural', 'next_status', 'weight', 'community']].rename(columns={'ppc4':'ppc_binary', 'community':'psu'}).copy()
        block['month'] = d.current_date_month.astype('Int64')
        block['chronic_count'] = d.phys_count7.where(d.phys_missing7.eq(0))
        block['chronic_confirmed_count'] = d.phys_count7
        block['chronic_unknown'] = d.phys_missing7
        block['education_band'] = d.education.map({**dict.fromkeys([1,2,3], 'below_primary'), 4:'primary', **dict.fromkeys(range(5,12), 'middle_plus')})
        block['stratum'] = pd.NA
        age45 = block.age.ge(45)
        for item, target in ITEMS.items():
            var = native_map[item]
            fielded = var in raw
            v = raw[var] if fielded else pd.Series(np.nan, index=d.index)
            valid = v.isin([1, 2, 3, 4])
            phone_not_applicable = v.eq(5) & (item == 'P')
            x = v.ge(2).astype(float).where(valid)
            hvar = f'r{j}{HITEMS[item]}' if j <= 4 and fielded else None
            h = harm[hvar].reindex(d.index) if hvar else pd.Series(np.nan, index=d.index)
            benchmark = valid & h.notna()
            # Harmonized D permits 0 for mobility-based routing. Inferred status
            # requires BOTH a verified raw negative screen and a coordinated 0.
            routed0 = v.isna() & h.eq(0) & screened if fielded else pd.Series(False, index=d.index)
            np.testing.assert_array_equal(v.loc[valid].to_numpy(), d.loc[valid, 'a_'+INHERITED[item]].to_numpy())
            assert not (h.notna() & phone_not_applicable).any()
            reason = np.select([valid, phone_not_applicable, routed0, v.notna()],
                ['direct_1_4', 'not_applicable_no_phone', 'confirmed_inferred_0', 'nonresponse_or_invalid_code'],
                default='unknown_blank' if fielded else 'structural_not_fielded')
            block['v'+item] = v.where(valid)
            block['x'+item] = x
            block['raw'+item] = v
            block['xi'+item] = x.mask(routed0, 0)
            block['route'+item] = reason
            block['h'+item] = d['h_'+INHERITED[item]]
            h_answer = help_cols.get(item)
            h_native = raw[h_answer].map({1:1., 2:0.}) if h_answer in raw else pd.Series(np.nan,index=d.index)
            np.testing.assert_array_equal(h_native.to_numpy(), block['h'+item].to_numpy())
            benchmark_notes = 'Comparison only; partial sums and any coordination-inferred zero not primary'
            labels_for_item = values.get(var, {})
            cards.append(dict(cohort='CHARLS', wave=wave, item=item, target=target, native_file_id=FILES[wave],
                native_variable=var if fielded else '', native_label=labels.get(var, ''),
                native_value_rule='1=no difficulty; 2–4=any difficulty; telephone 5=not applicable' if fielded else 'telephone not fielded',
                local_value_label_5=labels_for_item.get(5, ''),
                primary_mapping='1→0; 2/3/4→1; other/missing distinct, not zero' if fielded else 'structural missing',
                coordinated_variable=hvar or '', coordinated_note=benchmark_notes if hvar else 'No corresponding D release item',
                role_duration='Ability due to health/memory; 2013–2018 IADL intro excludes expected duration below 3 months; exact 2020 item wording differs',
                conditional_routing='No observed explicit not-taking/drug-use gate; ability answer is not proof of actual use; no phone explicit only where source code 5 observed',
                proxy_rule='Preserve inherited function respondent mode; no proxy reclassification to self',
                questionnaire=qfile, pdf_pages=qpages, official_reference=W5_URL if wave == 2020 else HARM_URL,
                n_current_natives=len(block), n_age45=int(age45.sum()), n_observed=int((age45 & valid).sum()),
                n_self_observed=int((age45 & valid & block.proxy.eq(0)).sum()),
                n_proxy_observed=int((age45 & valid & block.proxy.eq(1)).sum()),
                help_variable=h_answer if h_answer in raw else '', n_same_task_help_observed=int((age45 & h_native.notna()).sum()),
                n_explicit_no_phone=int((age45 & phone_not_applicable).sum()),
                n_confirmed_inferred_zero=int((age45 & routed0).sum()),
                n_coordinated_zero_raw_blank_unverified=int((age45 & v.isna() & h.eq(0) & ~routed0).sum()),
                n_native_harmonized_overlap=int((age45 & benchmark).sum()),
                n_native_harmonized_disagreement=int((age45 & benchmark & x.ne(h)).sum()),
                support_role='four_item_history_only' if wave == 2011 else 'main_five_item_window'))
            for code, count in v.loc[age45].value_counts(dropna=False).items():
                audit.append(dict(wave=wave, item=item, raw_value=code, n=int(count), status='direct' if pd.notna(code) and code in [1,2,3,4] else ('no_phone' if item == 'P' and code == 5 else 'not_eligible_as_binary')))
        xcols = ['x'+i for i in ITEMS]
        complete = block[xcols].notna().all(axis=1)
        active4 = ['x'+i for i in ITEMS if i != 'P']
        complete4 = block[active4].notna().all(axis=1)
        harm5 = harm[f'r{j}iadlza'].reindex(d.index) if j in (2,3,4) else pd.Series(np.nan, index=d.index)
        harm3 = harm[f'r{j}iadla'].reindex(d.index) if j in (2,3,4) else pd.Series(np.nan, index=d.index)
        harm_missing = harm[f'r{j}iadlzam'].reindex(d.index) if j in (2,3,4) else pd.Series(np.nan, index=d.index)
        n = dict(wave=wave, age45_n=int(age45.sum()), complete_five_n=int((age45 & complete).sum()),
            complete_four_n=int((age45 & complete4).sum()), ppc_complete_n=int((age45 & d.code4.notna()).sum()),
            complete_five_and_ppc_n=int((age45 & complete & d.code4.notna()).sum()),
            complete_seven_diseases_n=int((age45 & block.chronic_count.notna()).sum()),
            harmonized_five_nonmissing=int((age45 & harm5.notna()).sum()),
            harmonized_sum_available_with_one_or_more_items_missing=int((age45 & harm5.notna() & harm_missing.gt(0)).sum()),
            direct_complete_harmonized_five_both=int((age45 & complete & harm5.notna()).sum()),
            direct_complete_harmonized_five_disagree=int((age45 & complete & harm5.notna() & harm5.ne(block[xcols].sum(axis=1))).sum()),
            direct_complete_harmonized_three_disagree=int((age45 & complete & harm3.notna() & harm3.ne(block[['xM', 'xF', 'xP']].sum(axis=1))).sum()),
            complete_five_with_confirmed_inferred_zero_n=int((age45 & block[['xi'+i for i in ITEMS]].notna().all(axis=1)).sum()),
            n_with_positive_current_wave_weight=int((age45 & block.weight.gt(0)).sum()),
            rural_documented_n=int((age45 & block.rural.notna()).sum()),
            question_scope='four-item history only' if wave==2011 else 'five-item')
        summaries.append(n)
        blocks.append(block.reset_index())
    long = pd.concat(blocks, ignore_index=True)
    assert not long.duplicated(['person', 'wave']).any()
    assert len(long) == 96137 and long.person.nunique() == 25762
    assert long.family.notna().all()
    assert long['month'].notna().all()
    assert long.loc[long.wave.eq(2011), 'xP'].isna().all()
    assert long.loc[long.routeP.eq('not_applicable_no_phone'), ['xP','xiP','vP']].isna().all().all()
    pairs, stages = {}, []
    cols = ['person', 'wave', 'month', 'age', 'code4', 'next_status', *['x'+i for i in ITEMS]]
    for t, u in zip(WAVES[:-1], WAVES[1:]):
        now = long.loc[long.wave.eq(t), cols]
        nxt = long.loc[long.wave.eq(u), cols].drop(columns=['next_status','age','wave'])
        p = now.merge(nxt, on='person', suffixes=('', '_next'), how='left', validate='one_to_one', indicator=True)
        assert (p._merge.eq('both') == p.next_status.eq('interviewed')).all()
        p = p.loc[p.age.ge(45)].copy()
        timed = p.next_status.eq('interviewed') & p.month_next.sub(p.month).gt(0)
        pairs[t] = p.loc[timed]
        stages.append(dict(index_wave=t, next_wave=u, current_age45_n=len(p), next_interviewed=int(p.next_status.eq('interviewed').sum()),
            next_confirmed_death=int(p.next_status.eq('dead').sum()), next_known_alive_no_interview=int(p.next_status.eq('alive_not_interviewed').sum()),
            next_vital_unknown=int(p.next_status.eq('life_unknown').sum()), positive_date_pairs=int(timed.sum()),
            complete_PPC_both_n=int((timed & p[['code4','code4_next']].notna().all(axis=1)).sum()),
            complete_five_and_PPC_both_n=int((timed & p[['code4','code4_next', *['x'+i for i in ITEMS]]].notna().all(axis=1)).sum()),
            interval_q25=float(p.loc[timed,'month_next'].sub(p.loc[timed,'month']).quantile(.25)),
            interval_median=float(p.loc[timed,'month_next'].sub(p.loc[timed,'month']).median()),
            interval_q75=float(p.loc[timed,'month_next'].sub(p.loc[timed,'month']).quantile(.75))))
    for row in cards:
        if row['wave'] in pairs:
            p = pairs[row['wave']]
            x = 'x'+row['item']
            both = p[[x,x+'_next']].notna().all(axis=1)
            row['n_valid_adjacent_item_pair'] = int(both.sum())
            row['n_valid_adjacent_item_and_PPC_pair'] = int((both & p[['code4','code4_next']].notna().all(axis=1)).sum())
            row['next_wave'] = WAVES[WAVES.index(row['wave'])+1]
        else:
            row['n_valid_adjacent_item_pair'] = np.nan
            row['n_valid_adjacent_item_and_PPC_pair'] = np.nan
            row['next_wave'] = 'administrative_end_in_local_window'
    for file in MANIFEST:
        with (WORK/file['path']).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == file['sha256']
    long.to_parquet(HERE / 'PRIVATE/charls_iadl5_panel.parquet', index=False)
    csv(cards, 'FIVE_ITEM_CROSSWALK')
    csv(summaries, 'CHARLS_WAVE_ITEM_COMPLETENESS')
    csv(stages, 'CHARLS_ADJACENT_STATUS_SUPPORT')
    csv(MANIFEST, 'CHARLS_SOURCE_PROVENANCE')
    pd.DataFrame(audit).to_csv(HERE/'PRIVATE/raw_code_aggregate_internal.csv', index=False)
    qa = dict(status='PASS', native_links_equal_exact_health_keys=True, inherited_valid_ability_reproduced=True,
        repeated_person_wave_unique=True, original_hashes_unchanged=True, no_phone_not_zero=True,
        ages_kept_unconditional_in_private_panel=True, wave1_phone_never_assigned=True,
        actual_month_fields_complete=True, adjacent_status_key_reproduced=True,
        person_waves=len(long), people=int(long.person.nunique()), model_fits=0)
    (HERE/'CHARLS_EXTRACTION_QA.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
    print(json.dumps(qa, ensure_ascii=False))


if __name__ == '__main__':
    main()

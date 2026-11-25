"""Read-only ELSA V2 source verification; no association model is fitted."""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
WORK = OUT.parents[1]
PRIOR = WORK / 'PPC_IADL_History_ThreeVisit_20261115/ELSA/PRIVATE/panel.parquet'
RAW = WORK / '原始数据/02_ELSA_英国'
X = ['xM', 'xF', 'xP', 'xS', 'xC']
COG = ['c1', 'c2', 'c3', 'c4']
DISEASE = dict(hibpe='hibpf', diabe='diabf', cancre='cancrf', lunge='lungf',
               hearte='heartf', stroke='strokf', arthre='arthrf')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def native_psced(wave):
    fn = ('wave_6_elsa_data_v2.dta' if wave == 6
          else 'wave_10_elsa_data_eul_v1.dta')
    path = RAW / f'01_官方原始数据/wave{wave}' / fn
    with pd.io.stata.StataReader(path) as r:
        labels = r.variable_labels()
    lookup = {k.lower(): k for k in labels}
    fields = [lookup[f'psced{x}'] for x in 'abcdefgh']
    cols = [lookup['idauniq']] + fields
    if wave == 6:
        cols += [k for k in labels if re.fullmatch('headl(?:[a-z]{2}|96)', k.lower())]
    data = pd.read_stata(path, columns=cols, convert_categoricals=False)
    data.columns = data.columns.str.lower()
    data['person'] = data.pop('idauniq').astype('Int64').astype('string')
    assert data.person.is_unique
    data = data.set_index('person')
    items = data[[f'psced{x}' for x in 'abcdefgh']]
    valid = items.isin([1, 2]).all(axis=1)
    score = pd.DataFrame({f: items[f].eq(2 if f[-1] in 'df' else 1).astype(float)
                          .where(items[f].isin([1, 2])) for f in items})
    total = score.sum(axis=1, min_count=8).where(valid)
    return path, data, total, items.notna().sum(axis=1), items.isin([1, 2]).sum(axis=1)


def main():
    for folder in ['PRIVATE', 'TABLES', 'source']:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    before = sha(PRIOR)
    old = pd.read_parquet(PRIOR)
    p = old.copy()
    assert not p.duplicated(['person', 'wave']).any()
    checks = []
    inputs = [PRIOR]
    # G3 pp274-275 explicitly identify retroactive recoding. Flag 6 says this
    # wave's original YES was reset to NO after a later interview dispute.
    gpath = RAW/'02_协调与简化数据/Harmonized ELSA/h_elsa_g3.dta'
    inputs.append(gpath)
    g = pd.read_stata(gpath, columns=['idauniq']+[f'r{w}{v}' for w in range(1,10)
                      for v in [*DISEASE, *DISEASE.values()]], convert_categoricals=False).copy()
    g['person'] = g.pop('idauniq').astype('Int64').astype('string')
    g = g.set_index('person')
    p['physical_g3_retrospective'] = p.physical
    p['chronic_count_g3_retrospective'] = p.chronic_count
    flags = []
    for w in range(1,10):
        ix = p.index[p.wave.eq(w)]
        d = g.reindex(p.loc[ix,'person'])
        d.index = ix
        disease = pd.DataFrame(index=ix)
        for key, flag in DISEASE.items():
            e, f = d[f'r{w}{key}'], d[f'r{w}{flag}']
            restore = f.eq(6) & e.eq(0)
            disease[key] = e.where(e.isin([0,1])).mask(restore, 1.)
            p.loc[ix, 'disease_'+key] = disease[key]
            flags.append(dict(wave=w, condition=key, later_dispute_restored=int(restore.sum()),
                              current_dispute_kept=int(f.eq(3).sum()),
                              later_dispute_missing_kept=int((f.eq(6)&e.isna()).sum()),
                              later_dispute_positive_kept=int((f.eq(6)&e.eq(1)).sum())))
        p.loc[ix, 'chronic_count'] = disease.sum(axis=1,min_count=7)
        p.loc[ix, 'chronic_count_lower'] = disease.eq(1).sum(axis=1)
        p.loc[ix, 'chronic_observed'] = disease.notna().sum(axis=1)
        p.loc[ix, 'physical'] = np.where(disease.eq(1).any(axis=1),1.,
                                        np.where(disease.eq(0).all(axis=1),0.,np.nan))
    pd.DataFrame(flags).to_csv(OUT/'PRIVATE/ELSA_DISEASE_DISPUTE_DETAIL_V2.csv',index=False,encoding='utf-8-sig')
    public_flags = pd.DataFrame(flags)[['wave','condition','later_dispute_restored','current_dispute_kept']]
    # Rare flag/source inconsistencies stay local; no small-cell count is exported.
    public_flags = public_flags.groupby('wave')[['later_dispute_restored','current_dispute_kept']].sum().reset_index()
    public_flags.to_csv(OUT/'TABLES/ELSA_DISEASE_DISPUTE_QA_V2.csv',index=False,encoding='utf-8-sig')
    checks.append(dict(check='Restore only G3 future-dispute flag6; retain current flag3 and original missingness',status='PASS',
                       n=sum(r['later_dispute_restored'] for r in flags)))
    for w in [6, 10]:
        path, native, total, _, known = native_psced(w)
        inputs.append(path)
        sel = p.wave.eq(w)
        wpanel = p.loc[sel].set_index('person')
        rawsum = total.reindex(wpanel.index).where(wpanel.proxy.eq(0))
        if w == 6:
            assert np.array_equal(rawsum.to_numpy(), wpanel.cesd.to_numpy(), equal_nan=True)
            checks.append(dict(check='W6 native complete CESD8 equals prior G3 score', status='PASS', n=len(wpanel)))
            n = native.reindex(wpanel.index)
            items = n[[k for k in n if k.startswith('headl')]]
            none = items.headl96.eq(1)
            some = items.drop(columns='headl96').eq(1).any(axis=1)
            valid_list = (none ^ some) & ~items.isin([-9, -8, -4, -3]).any(axis=1)
            fields = dict(xM='headlme', xF='headlmo', xP='headlph', xS='headlsh', xC='headlpr')
            for target, src in fields.items():
                direct = n[src].eq(1).astype(float).where(valid_list & n[src].isin([0, 1]))
                assert np.array_equal(direct.to_numpy(), wpanel[target].to_numpy(), equal_nan=True)
                checks.append(dict(check=f'W6 native {src} equals {target}', status='PASS', n=len(wpanel)))
        else:
            # This extends only an explicitly verified continuous domain, never W10 PPC.
            p.loc[sel, 'cesd'] = rawsum.to_numpy()
            p.loc[sel, 'cesd_missing'] = (8-known.reindex(wpanel.index)).where(wpanel.proxy.eq(0)).to_numpy()
            p.loc[sel, 'psychological'] = rawsum.ge(4).astype(float).where(rawsum.notna()).to_numpy()
            p.loc[sel, 'cesd_source_v2'] = 'W10 PScedA-H; full 8; self only; D/F reverse'
            checks.append(dict(check='W10 only complete self CESD8 populated; no PPC or cognition supplied', status='PASS', n=int(rawsum.notna().sum())))
    # V2 source refinement does not overwrite the old derived fields/code4.
    unchanged = [c for c in old if c not in ['cesd', 'cesd_missing', 'psychological',
                    'physical', 'chronic_count', 'chronic_count_lower', 'chronic_observed']]
    pd.testing.assert_frame_equal(p[unchanged], old[unchanged])
    assert p.loc[p.wave.eq(10), ['code4', 'chronic_count', *COG]].isna().all().all()
    assert sha(PRIOR) == before
    p.to_parquet(OUT / 'PRIVATE/panel.parquet', index=False)

    refbase = p.loc[p.wave.eq(1) & p.age.ge(45) & p.proxy.eq(0)].copy()
    ref = refbase.loc[refbase[COG].notna().all(axis=1)]
    reference = []
    for field in COG:
        z = ref[field]
        reference.append(dict(source='ELSA', reference_wave=1, reference_population='self age>=45 all four cognitive tests complete',
                              field=field, n=len(z), mean=z.mean(), sd=z.std(ddof=1), cutoff=z.mean()-1.5*z.std(ddof=1),
                              weighted=False, ddof=1))
    for population, d in [('self age>=45 observed complete CESD8', refbase), ('four-cognitive-complete subset; diagnostic only', ref)]:
        z = d.cesd.dropna()
        reference.append(dict(source='ELSA', reference_wave=1, reference_population=population, field='cesd',
                              n=len(z), mean=z.mean(), sd=z.std(ddof=1), cutoff=np.nan, weighted=False, ddof=1))
    pd.DataFrame(reference).to_csv(OUT / 'TABLES/ELSA_REFERENCE_V2.csv', index=False, encoding='utf-8-sig')
    # Counts below are structural only, no exposure-by-outcome associations.
    q = p.copy()
    q['complete4'] = q[COG].notna().all(axis=1)
    q['complete5'] = q[X].notna().all(axis=1)
    q['cog_low_v2'] = np.nan
    for f in COG:
        row = next(r for r in reference if r['field'] == f)
        q[f'z_{f}'] = (q[f]-row['mean'])/row['sd']
    q.loc[q.complete4, 'cog_low_v2'] = q.loc[q.complete4, [f'z_{f}' for f in COG]].lt(-1.5).any(axis=1).astype(float)
    q['ppc_v2'] = q.physical + 2*q.psychological + 4*q.cog_low_v2
    base = q.loc[q.wave.eq(7)].copy()
    prev = q.loc[q.wave.eq(6)].set_index('person')
    nxt = q.loc[q.wave.eq(8)].set_index('person')
    base = base.join(prev[['month', 'complete5', 'IADL5_COUNT']], on='person', rsuffix='_prev')
    base = base.join(nxt[['month', 'ppc_v2']], on='person', rsuffix='_next')
    structural = base.age.ge(45) & base.complete5 & base.complete5_prev.eq(True) & base.month.gt(base.month_prev)
    pre = structural & base.ppc_v2.notna() & base.cesd.notna() & base.chronic_count.notna()
    popcnt = lambda s: s.map({i: int(i).bit_count() for i in range(8)})
    risk = pre & popcnt(base.ppc_v2).le(1)
    summary = [
        dict(metric='W6 age>=45 valid five tasks', n=int((q.wave.eq(6)&q.age.ge(45)&q.complete5).sum())),
        dict(metric='W6-W7 IADL history and ordered dates, W7 age>=45', n=int(structural.sum())),
        dict(metric='W7 PPC plus continuous baseline domain severity valid; no future requirement', n=int(pre.sum())),
        dict(metric='W6-W7-W8 primary baseline 0/1 risk before future status', n=int(risk.sum())),
        dict(metric='Above risk with W8 valid PPC and ordered actual date', n=int((risk&base.ppc_v2_next.notna()&base.month_next.gt(base.month)).sum())),
        dict(metric='W1 self age>=45 all four cognition reference', n=len(ref)),
        dict(metric='W1 self age45-49 all four cognition reference', n=int(ref.age.lt(50).sum())),
        dict(metric='W10 age>=45 self complete CESD8', n=int((q.wave.eq(10)&q.age.ge(45)&q.cesd.notna()).sum())),
    ]
    pd.DataFrame(summary).to_csv(OUT/'TABLES/ELSA_W6_HISTORY_SUPPORT_V2.csv', index=False, encoding='utf-8-sig')
    coverage=q.loc[q.age.ge(45)].groupby('wave').agg(n=('person','size'),complete_IADL5=('complete5','sum'),
                        complete_cognition4=('complete4','sum'),valid_cesd=('cesd','count'),valid_seven_count=('chronic_count','count'),
                        legal_date=('month','count'))
    coverage.reset_index().to_csv(OUT/'TABLES/ELSA_DOMAIN_COVERAGE_V2.csv',index=False,encoding='utf-8-sig')
    boundary = []
    for w,g in q.groupby('wave'):
        zs=g[[f'z_{f}' for f in COG]]
        boundary.append(dict(wave=w,exact_threshold_cells=int(zs.eq(-1.5).sum().sum()),
                             complete_case_changed_lt_vs_le=int(((zs.lt(-1.5).any(axis=1)) != (zs.le(-1.5).any(axis=1))).where(g.complete4,False).sum())))
    pd.DataFrame(boundary).to_csv(OUT/'TABLES/ELSA_BOUNDARY_QA_V2.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(checks).to_csv(OUT/'TABLES/ELSA_SOURCE_QA_V2.csv',index=False,encoding='utf-8-sig')
    prior_cross = pd.read_csv(WORK/'PPC_IADL5_Exposure_Revision/ELSA/TABLES/FIVE_ITEM_CROSSWALK.csv')
    cross = []
    for row in prior_cross.to_dict('records'):
        cross.append(dict(source='ELSA', wave=row['wave'], construct='IADL '+row['item'],
            fields=row['original_variable'], scoring=row['coding'], eligibility=row['proxy'],
            limitations='Valid checklist report, not objectively verified capacity', evidence=row['documentation']))
    for w in range(1,11):
        cross.append(dict(source='ELSA',wave=w,construct='CESD8',fields=f'r{w}cesd; r{w}cesdm' if w<=9 else 'PScedA-H',
            scoring='Complete 8-item sum 0..8; D/F reverse; psychological threshold >=4',eligibility='Self interview; all 8 valid',
            limitations='W10 continuous psychological domain only; no W10 PPC supplied' if w==10 else 'No partial sum or proxy substitution',
            evidence='G3 published score; W6 native reproduction and questionnaire p421' if w<=9 else 'W10 questionnaire p446; native labels and explicit recoding'))
        if w<=9:
            cross.append(dict(source='ELSA',wave=w,construct='Seven-disease count',
                fields='; '.join(f'r{w}{s}/{f}' for s,f in DISEASE.items()),
                scoring='G3 diagnosis; undo only documented future-dispute flag6 to original YES; retain current flag3; complete seven sum; no cummax',
                eligibility='All seven known for count; any known positive only for binary physical',
                limitations='W3-7 cancer also includes leukemia/lymphoma absent W8-9; W9 two heart-dispute fields unavailable; not clinical full severity',evidence='G3 printed pp273-275; old retrospective columns preserved'))
            for c, suffix in zip(COG,['imrc','dlrc','orient','verbf']):
                cross.append(dict(source='ELSA',wave=w,construct=c,
                    fields='STRUCTURALLY_ABSENT' if w==6 and c=='c4' else f'r{w}{suffix}',
                    scoring='Unweighted fixed W1 four-complete reference; sample SD ddof=1; z<-1.5; composite requires all four',
                    eligibility='Self, actual valid score; four observed tests for primary cognition',
                    limitations='W6 cannot form full cognition; can supply history IADL/physical/CESD' if w==6 else 'Fixed reference does not remove practice or interview-mode effects',
                    evidence='G3 printed pp459-475; prior raw-score source QA'))
    pd.DataFrame(cross).to_csv(OUT/'TABLES/ELSA_MEASUREMENT_CROSSWALK_V2.csv',index=False,encoding='utf-8-sig')
    inputs += [RAW/'04_文档资料/问卷/elsa_w6_2012_2013_原始问卷.pdf',RAW/'04_文档资料/问卷/elsa_w10_2021_2023_原始问卷.pdf']
    pd.DataFrame([dict(path=str(f.relative_to(WORK)),sha256=sha(f)) for f in inputs]).to_csv(OUT/'TABLES/ELSA_INPUTS_V2.csv',index=False,encoding='utf-8-sig')
    (OUT/'TABLES/ELSA_DECISION_V2.json').write_text(json.dumps(dict(w6_history_allowed=True,w6_ppc_allowed=False,
        w10_complete_cesd_allowed=True,w10_cognition_and_seven_count_not_validated=True,w10_ppc_allowed=False,
        no_association_models_fitted=True,retrospective_disease_flag6_restored=True,
        old_file_sha256_unchanged=before,new_panel_sha256=sha(OUT/'PRIVATE/panel.parquet')),
        indent=2),encoding='utf-8')
    print(pd.DataFrame(reference).to_string(index=False))
    print(pd.DataFrame(summary).to_string(index=False))
    print(coverage.to_string())


if __name__ == '__main__':
    main()

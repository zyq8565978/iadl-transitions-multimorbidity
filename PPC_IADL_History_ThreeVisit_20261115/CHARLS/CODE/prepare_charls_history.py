"""Build CHARLS source interface for three-visit protocol audit; never fit an effect model."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
WORK = HERE.parents[1]
SKILL = Path.home()/'.codex/skills/seven-aging-pig-skill'
sys.dont_write_bytecode = True
sys.path.insert(0, str(SKILL/'scripts'))
import data as pig

IADL = WORK/'PPC_IADL5_Exposure_Revision/CHARLS/PRIVATE/charls_iadl5_panel.parquet'
OLDPPC = WORK/'PPC_CuCoM_IJNS_Upgrade_20261115/CHARLS/PRIVATE/charls_person_panel.parquet'
RAW = WORK/'原始数据/01_CHARLS_中国/01_官方原始数据'
SOURCES = []
FIDS = {2013:'F08831',2015:'F08868',2018:'F08884',2020:'F08908'}
DOCS = {2013:'CHARLS_Wave2_Main_Questionnaire.pdf',2015:'CHARLS_2015_Questionnaire.pdf',
        2018:'CHARLS_2018_Household_Questionnaire.pdf',2020:'CHARLS_2020_Questionnaire_Chinese.pdf'}
EPOCH = {2011:'W1_only_no_phone',2013:'auditory_single_form_conditional_reread',
         2015:'auditory_single_form',2018:'visual_or_oral_three_explicit_learning_trials',
         2020:'oral_or_visual_three_explicit_learning_trials'}


def source(p, role):
    with p.open('rb') as f:
        digest=hashlib.file_digest(f,'sha256').hexdigest()
    SOURCES.append(dict(path=p.relative_to(WORK).as_posix(),role=role,sha256=digest,bytes=p.stat().st_size))


def word_score(d, prefix):
    selected = sum(d[prefix+str(j)].eq(j).astype(int) for j in range(1,11))
    none = d[prefix+'11'].eq(11)
    bad = pd.Series(False,index=d.index)
    for j in (12,13):
        if prefix+str(j) in d:
            bad |= d[prefix+str(j)].eq(j)
    return selected.astype(float).where((selected.gt(0)|none)&~bad&~(selected.gt(0)&none))


def main():
    (HERE/'PRIVATE').mkdir(exist_ok=True)
    source(IADL,'immutable prior IADL5 item panel')
    source(OLDPPC,'immutable prior four-component PPC source panel')
    cols = ['ID','wave','proxy','code4','physical','psychological','cognitive_4component','cesd10',
            'recall_immediate1','recall_delayed','orientation','executive','executive_steps_observed']
    a = pd.read_parquet(IADL)
    b = pd.read_parquet(OLDPPC,columns=cols).rename(columns={'ID':'person','proxy':'proxy_old','code4':'code4_old'})
    p = a.merge(b,on=['person','wave'],how='inner',validate='one_to_one')
    assert len(a)==len(b)==len(p) and p['person'].nunique()==25762
    np.testing.assert_array_equal(p.proxy.to_numpy(float),p.proxy_old.to_numpy(float))
    np.testing.assert_array_equal(p.code4.to_numpy(float),p.code4_old.to_numpy(float))
    for k,v in dict(c1='recall_immediate1',c2='recall_delayed',c3='orientation',c4='executive',cesd='cesd10').items():
        p[k] = p[v].where(p.proxy.eq(0))
    # Preserve actual arithmetic result, but do not use partial serial scores as if 5 comparisons were administered.
    p['c4'] = p['c4'].where(p.executive_steps_observed.eq(5))
    p['cognitive_old'] = p.cognitive_4component
    p['cognition_protocol_epoch'] = p.wave.map(EPOCH)
    p['fixed_anchor_comparable'] = p.wave.isin([2013,2015])
    p['fixed_primary_triplet_allowed'] = False  # both specified triplets cross the verified 2018 recall protocol change
    p['functional_assistance_mode_native'] = np.nan
    shape=[]
    pres = {2013:('dc006_1_s','dc027s'),2015:('dc006s','dc027s'),
            2018:('dc028_w4_s','dc047_w4_s','dc029_w4_s','dc030_w4_s'),
            2020:('dc012_s','dc028_s','dc013_s','dc014_s')}
    for wave,fid in FIDS.items():
        path=RAW/pig.table_info(fid)['source'].split('charls_raw_data/')[1]
        source(path, f'native documented direct cognition, {fid}, {wave}')
        source(RAW/str(wave)/DOCS[wave],f'official questionnaire {wave}')
        with pd.io.stata.StataReader(path) as reader:
            names=reader.variable_labels()
        want=['ID']+list(dict.fromkeys(c for c in names if any(c.startswith(pr) for pr in pres[wave]) or
                                       (wave==2013 and c=='dc006_1') or (wave==2020 and c in ['dc001','dc002','dc003','dc004','dc005'])))
        raw=pd.read_stata(path,columns=want,convert_categoricals=False)
        raw['person']=raw['ID'].astype('string')
        raw=raw.set_index('person',verify_integrity=True)
        pp=p.loc[p.wave.eq(wave)].set_index('person',verify_integrity=True)
        raw=raw.reindex(pp.index)
        age=pp.age.ge(45)
        self_=pp.proxy.eq(0)
        mode_file = path
        if wave == 2018:
            mode_file=RAW/pig.table_info('F08889')['source'].split('charls_raw_data/')[1]
            source(mode_file, '2018 native function-module assistance mode, read separately from Cognition')
        mode_key = 'db045' if wave == 2020 else 'db032'
        mode=pd.read_stata(mode_file,columns=['ID',mode_key],convert_categoricals=False).set_index('ID',verify_integrity=True)[mode_key]
        p.loc[p.wave.eq(wave),'functional_assistance_mode_native'] = p.loc[p.wave.eq(wave),'person'].map(mode)
        first,delay = [word_score(raw,pr) for pr in pres[wave][:2]]
        # Preserve the old first- and delayed-word scoring source exactly for the audit, with direct known self-only intake.
        np.testing.assert_allclose(first.where(self_),pp.c1,equal_nan=True)
        np.testing.assert_allclose(delay.where(self_),pp.c2,equal_nan=True)
        if wave==2020:
            # 'Don't know' (997) separated from correct/incorrect; don't let it become unheard failure across eras.
            orient = raw[['dc001','dc002','dc003','dc004','dc005']]
            sc = orient.eq(1).sum(axis=1).astype(float).where(orient.isin([1,2]).all(axis=1)&self_)
            p.loc[p.wave.eq(wave),'c3'] = p.loc[p.wave.eq(wave),'person'].map(sc)
        def selected_total(prob):
            return int((age&self_&prob.notna()).sum())
        recent = dict(wave=wave, ages45plus_self=int((age&self_).sum()), immediate_direct_n=selected_total(first),
                      delayed_direct_n=selected_total(delay),
                      immediate_mean=float(first.loc[age&self_].mean()), delayed_mean=float(delay.loc[age&self_].mean()),
                      second_separate_trial_n=selected_total(word_score(raw,pres[wave][2])) if len(pres[wave])>2 else np.nan,
                      third_separate_trial_n=selected_total(word_score(raw,pres[wave][3])) if len(pres[wave])>3 else np.nan,
                      separate_repeat_scores=True if len(pres[wave])>2 else False,
                      fixed_2013_measurement_gate='different_recall_protocol' if wave>=2018 else 'early_recall_declared_reference_epoch')
        if wave==2013:
            recent['conditional_raw_reread_gt1_n']=int((age&self_&raw.dc006_1.between(2,3)).sum())
        shape.append(recent)
    keep=['person','family','wave','month','age','sex','education','education_band','chronic_count','code4','ppc_binary',
          'next_status','proxy','rural','weight','psu','c1','c2','c3','c4','cesd','physical','psychological','cognitive_old',
          'executive_steps_observed','cognition_protocol_epoch','functional_assistance_mode_native',
          'fixed_anchor_comparable','fixed_primary_triplet_allowed']
    keep += [prefix+item for prefix in ['x','route','v'] for item in 'MFPSC']
    p=p[keep].copy()
    assert not p.duplicated(['person','wave']).any()
    for j,maximum in {'c1':10,'c2':10,'c3':5,'c4':5,'cesd':30}.items():
        assert p[j].dropna().between(0,maximum).all()
        assert not p.loc[~p.proxy.eq(0),j].notna().any()
    assert not p.loc[p.wave.eq(2011),'xP'].notna().any()
    assert not p.loc[p.executive_steps_observed.ne(5),'c4'].notna().any()
    for name in ['physical','psychological','cognitive_old']:
        assert p[name].dropna().isin([0,1]).all()
    assert p.loc[p.code4.notna(),['physical','psychological','cognitive_old']].notna().all(axis=1).all()
    pd.testing.assert_series_equal(p.code4.astype(float), (p.physical+2*p.psychological+4*p.cognitive_old).rename('code4'), check_dtype=False)
    # Write only the new local interface and aggregate diagnostics; original files inspected above stay read-only.
    p.to_parquet(HERE/'PRIVATE/panel.parquet',index=False)
    for name,q in [('SOURCE_LINEAGE',SOURCES),('RECALL_PROTOCOL_EMPIRICAL',shape)]:
        pd.DataFrame(q).to_csv(HERE/(name+'.csv'),index=False,encoding='utf-8-sig')
    supports=[]
    for w,d in p.loc[p.age.ge(45)].groupby('wave'):
        supports.append(dict(wave=w,eligible_n=len(d),known_self=int(d.proxy.eq(0).sum()),
            known_no_function_assistance_n=int(d.functional_assistance_mode_native.eq(1).sum()) if w in FIDS else np.nan,
            continuous_complete_n=int(d[['c1','c2','c3','c4']].notna().all(axis=1).sum()),
            count_seven_complete_n=int(d.chronic_count.notna().sum()),
            **{k+'_n':int(d[k].notna().sum()) for k in ['c1','c2','c3','c4','cesd','physical','psychological','cognitive_old','code4']}))
    pd.DataFrame(supports).to_csv(HERE/'DIRECT_SCORE_SUPPORT.csv',index=False,encoding='utf-8-sig')
    status=dict(cohort='CHARLS',reference_status='COGNITION_REFERENCE_NOT_SUPPORTABLE',
        reference_candidate=2013,reason='Required 2018/2020 transitions cross a substantively changed word-recall regimen, not simply a mean shift',
        old_definition_unchanged=True,new_effect_models_fitted=0,read_only_sources=len(SOURCES),person_waves=len(p),
        people=int(p.person.nunique()),prespecified_primary_windows_pass=0,
        direct_scores='observed only, excluding explicit/unknown full interview proxy; partial module assistance flag preserved; complete 5 arithmetic comparisons; early vs late word-recall not equated')
    (HERE/'CHARLS_SOURCE_QA.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status))


if __name__=='__main__':main()

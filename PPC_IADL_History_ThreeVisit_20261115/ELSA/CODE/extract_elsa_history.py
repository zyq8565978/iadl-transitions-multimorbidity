"""Add observed source test scores to unchanged previous ELSA IADL5 panel. No fitting."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
WORK = OUT.parents[1]
RAW = WORK/'原始数据/02_ELSA_英国'
G3 = RAW/'02_协调与简化数据/Harmonized ELSA/h_elsa_g3.dta'
PDF = RAW/'02_协调与简化数据/Harmonized ELSA/5050_harmonized_elsa_g3_2002-2019.pdf'
W6 = RAW/'01_官方原始数据/wave6/wave_6_elsa_data_v2.dta'
PRIOR = WORK/'PPC_IADL5_Exposure_Revision/ELSA/PRIVATE/person_wave.parquet'
COG = {'c1':('imrc',10),'c2':('dlrc',10),'c3':('orient',4),'c4':('verbf',100)}
SEVEN = ['hibpe','diabe','cancre','lunge','hearte','stroke','arthre']
COLS = [*COG,'cesd','cesd_missing','physical','psychological']


def sha(path):
    with path.open('rb') as fd:
        return hashlib.file_digest(fd,'sha256').hexdigest()


def ranged(s,lo,hi):
    return s.where(s.between(lo,hi)&s.mod(1).eq(0))


def main():
    for name in ['PRIVATE','TABLES']:
        (OUT/name).mkdir(exist_ok=True)
    old_sha=sha(PRIOR)
    old=pd.read_parquet(PRIOR)
    assert old[['person','wave']].notna().all().all() and not old.duplicated(['person','wave']).any()
    with pd.io.stata.StataReader(G3) as rd:
        labels=rd.variable_labels()
    with pd.io.stata.StataReader(W6) as rd:
        w6labs=rd.variable_labels()
    structures=[k for k,v in w6labs.items() if k.casefold().startswith('cf') and any(q in (k+' '+v).lower() for q in ['animal','verbal fluency','cfani'])]
    assert not structures and 'r6verbf' not in labels, structures
    suffix=[x[0] for x in COG.values()]+['cesd','cesdm']+SEVEN
    g=pd.read_stata(G3,columns=['idauniq']+[f'r{w}{s}' for w in range(1,10) for s in suffix if f'r{w}{s}' in labels],convert_categoricals=False)
    g['person']=g.pop('idauniq').astype('Int64').astype('string')
    assert g.person.notna().all() and g.person.is_unique
    g=g.set_index('person')
    pieces=[]; sources=[]
    for w,p in old.groupby('wave',sort=False):
        p=p.set_index('person').copy()
        p['old_code4']=p.code4
        if w<=9:
            assert p.index.isin(g.index).all()
            o=g.reindex(p.index)
            r=f'r{w}'
            def raw(name): return o.get(r+name,pd.Series(np.nan,index=o.index))
            isself=p.proxy.eq(0)
            for dest,(s,hi) in COG.items():
                assert (r+s in o)==(w!=6 or s!='verbf')
                p[dest]=ranged(raw(s),0,hi).where(isself)
                sources.append(dict(wave=w,output=dest,field=r+s if r+s in labels else 'STRUCTURAL_ABSENCE',
                                    label=labels.get(r+s,'')))
            nmiss=ranged(raw('cesdm'),0,8)
            z=ranged(raw('cesd'),0,8).where(isself)
            p['cesd_missing']=nmiss.where(isself)
            p['cesd']=z.where(nmiss.eq(0))
            p['cesd_lower']=z.where(nmiss.notna())
            p['cesd_upper']=(z+nmiss).clip(upper=8).where(nmiss.notna())
            # Preserve the established complete CES-D8 threshold rule; bounds are diagnostics only.
            p['psychological']=p['cesd'].ge(4).astype(float).where(p['cesd'].notna())
            dz=o[[r+c for c in SEVEN]].where(lambda b:b.isin([0,1]))
            ph=pd.Series(np.where(dz.eq(1).any(axis=1),1.,np.where(dz.eq(0).all(axis=1),0.,np.nan)),index=p.index)
            assert np.array_equal(ph.to_numpy(),p.physical.to_numpy(),equal_nan=True)
            assert np.array_equal(p.psychological.to_numpy(),p.psych.to_numpy(),equal_nan=True)
            assert np.array_equal(dz.sum(axis=1,min_count=7).to_numpy(),p.chronic_count.to_numpy(),equal_nan=True)
            for name,field in [('cesd','cesd'),('cesd_missing','cesdm')]+[(s,s) for s in SEVEN]:
                sources.append(dict(wave=w,output=name,field=r+field,label=labels[r+field]))
        else:
            p[[*COG,'cesd','cesd_missing','cesd_lower','cesd_upper','psychological']]=np.nan
            assert not p.primary_ppc_wave.any(), 'W10 cannot enter strict endpoint'
        pieces.append(p.reset_index())
    new=pd.concat(pieces,ignore_index=True)
    pd.testing.assert_frame_equal(old.reset_index(drop=True),new[old.columns].reset_index(drop=True),check_dtype=False)
    assert sha(PRIOR)==old_sha and new.loc[new.proxy.ne(0),list(COG)].isna().all().all()
    assert new.loc[new.wave.eq(6),'c4'].isna().all() and new.loc[new.wave.eq(10),COLS].isna().all().all()
    ref=new.loc[new.wave.eq(1)&new.age.ge(45)&new.proxy.eq(0)].copy()
    ref['age_group']=(ref.age//5*5).clip(upper=80)
    refout=[]
    for lab,group in [('all_45plus',ref),*((str(int(a)),b) for a,b in ref.groupby('age_group'))]:
        for field in COG:
            x=group[field].dropna()
            refout.append(dict(reference_wave=1,age_group=lab,test=field,self_interviews=len(group),valid_n=len(x),
                               mean=x.mean(),sd=x.std(ddof=1),low=x.mean()-1.5*x.std(ddof=1),min=x.min(),max=x.max()))
    refs=pd.DataFrame(refout)
    assert refs.valid_n.ge(30).all() and refs.sd.gt(0).all(), 'W1 reference support insufficient'
    coverage=new.loc[new.age.ge(45)].groupby('wave').agg(eligible_interviews=('person','size'),self_interviews=('proxy',lambda p:p.eq(0).sum()),
                                    **{f'valid_{x}':(x,'count') for x in COLS+['old_code4']})
    new.to_parquet(OUT/'PRIVATE/panel.parquet',index=False)
    coverage.reset_index().to_csv(OUT/'TABLES/ELSA_SCORE_COVERAGE.csv',index=False,encoding='utf-8-sig')
    refs.to_csv(OUT/'TABLES/ELSA_W1_REFERENCE_SUPPORT.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(sources).to_csv(OUT/'TABLES/ELSA_SOURCE_FIELDS.csv',index=False,encoding='utf-8-sig')
    manifest=pd.DataFrame([dict(path=p.relative_to(WORK).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in [G3,PDF,W6,PRIOR]])
    manifest.to_csv(OUT/'TABLES/ELSA_INPUT_HASHES.csv',index=False,encoding='utf-8-sig')
    (OUT/'TABLES/ELSA_EXTRACTION_QA.json').write_text(json.dumps(dict(old_columns_preserved=True,old_eligible_no_change=True,
        exact_source_physical_chronic_psychological_reproduction=True,w6_fluency_structural=True,w10_not_used_strict=True,
        no_predictors_observed_from_proxy_cognitive=True,reference_tests_supported=True,no_fit_invoked=True,rows=len(new),
        old_columns_sha256=old_sha,new_panel_sha256=sha(OUT/'PRIVATE/panel.parquet')),indent=2),encoding='utf-8')
    print(coverage.to_string());print(refs.loc[refs.age_group.eq('all_45plus')].to_string(index=False));print('Minimum W1 bin N:',refs.valid_n.min())


if __name__=='__main__':
    main()

"""Native ELSA IADL5 and G3 PPC scope; approved local inputs only, no fitting."""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
RAW = OUT.parents[1] / '原始数据/02_ELSA_英国'
CORE = RAW / '01_官方原始数据'
G3 = RAW / '02_协调与简化数据/Harmonized ELSA/h_elsa_g3.dta'
D = 'https://doc.ukdataservice.ac.uk/doc/5050/mrdoc/pdf/5050_harmonized_elsa_g3_2002-2019.pdf'
STUDY = 'https://www.elsa-project.ac.uk/study-documentation'
FILES = {1:'wave_1_core_data_v3',2:'wave_2_core_data_v4',3:'wave_3_elsa_data_v4',
         4:'wave_4_elsa_data_v3',5:'wave_5_elsa_data_v4',6:'wave_6_elsa_data_v2',
         7:'wave_7_elsa_data',8:'wave_8_elsa_data_eul_v2',9:'wave_9_elsa_data_eul_v1',
         10:'wave_10_elsa_data_eul_v1'}
ITEMS = {'M':('Taking medications',11,'headlme','medsa'),
         'F':('Managing money',13,'headlmo','moneya'),
         'P':('Making telephone calls',10,'headlph','phonea'),
         'S':('Shopping for groceries',9,'headlsh','shopa'),
         'C':('Preparing a hot meal',8,'headlpr','mealsa')}
X = ['x'+q for q in ITEMS]
DISEASES = ['hibpe','diabe','cancre','lunge','hearte','stroke','arthre']
COG = {'imrc':10,'dlrc':10,'orient':4,'verbf':100}
PPC_WAVES = {1,2,3,4,5,7,8,9}
QUESTION_PAGES = {1:38,2:154,3:97,4:105,5:95,6:94,7:93,8:95,9:92,10:117}
MANIFEST, DICT, CROSS, PPC_NORMS, ROUTES = [], [], [], [], []


def load(path, columns, pattern=None, must=()):
    with pd.io.stata.StataReader(path) as rd:
        labels = rd.variable_labels()
        vals = rd.value_labels()
    keys = {k.lower():k for k in labels}
    if not set(must) <= keys.keys():
        raise ValueError(f'{path.name} missing {sorted(set(must)-keys.keys())}')
    chosen = sorted({keys[x] for x in ['idauniq']+columns if x in keys} |
                    {k for lower,k in keys.items() if pattern and re.search(pattern,lower)})
    ds = pd.read_stata(path,columns=chosen,convert_categoricals=False)
    ds.columns = ds.columns.str.lower()
    assert ds.idauniq.notna().all() and ds.idauniq.is_unique, path.name
    ds['idauniq'] = ds.idauniq.astype('Int64').astype('string')
    with path.open('rb') as f:
        digest = hashlib.file_digest(f,'sha256').hexdigest()
    MANIFEST.append(dict(relative_source=str(path.relative_to(RAW)), bytes=path.stat().st_size,
                         sha256=digest, rows=len(ds), selected_columns=len(chosen)))
    for k in chosen:
        DICT.append(dict(product=path.name, field=k, label=labels[k],
                         native_codes=json.dumps({str(c):label for c,label in vals.get(k,{}).items()},ensure_ascii=False)))
    return ds.set_index('idauniq'),{k.lower():k for k in chosen}


def valid(x, lo, hi):
    return x.where(x.between(lo,hi))


def count_bound(x):
    known = x.eq(1).sum(axis=1)
    return known, known + x.isna().sum(axis=1)


def classify(native, wave, out, known_names):
    if wave <= 2:
        items = [n for n in native if re.fullmatch(r'headb\d{2}',n)]
        assert len(items)==(14 if wave==1 else 13)
        z = native[items]
        none = z.eq(96).any(axis=1)
        any_mention = z.isin(range(1,14)).any(axis=1)
        not_asked = z.eq(-1).all(axis=1) | (out.proxy.eq(1) if wave==1 else False)
        nonresponse = z.isin([-8,-9]).any(axis=1)
    else:
        items = [n for n in native if re.fullmatch(r'headl(?:[a-z]{2}|96)',n)]
        assert 'headl96' in items
        z = native[items]
        none = z.headl96.eq(1)
        any_mention = z.drop(columns='headl96').eq(1).any(axis=1)
        not_asked = z.isin([-1,-2]).all(axis=1)
        nonresponse = z.isin([-9,-8,-4,-3]).any(axis=1)
    valid_list = (none ^ any_mention) & ~not_asked & ~nonresponse
    out['list_evidenced'] = valid_list.astype(int)
    out['list_conflict'] = (none & any_mention).astype(int)
    for suffix,(label,code,field,ref) in ITEMS.items():
        if wave <= 2:
            selected = z.eq(code).any(axis=1)
            not_app = pd.Series(False,index=z.index)
            actual_field = f'HEADB01–HEADB{len(items)}; option {code}'
            native_codes = f'{code}=task selected; 96=none; -1=not applicable/trailing spare; -8/−9 unknown/refusal'
        else:
            if suffix=='P' and wave in [4,5]:
                field='headlte'
            selected = native[field].eq(1)
            not_app = native[field].isin([-1,-2])
            actual_field = known_names[field]
            native_codes = '1=selected, 0=not mentioned, negative codes=source missing or out of route'
        okay = valid_list if wave<=2 else valid_list & native[field].isin([0,1])
        out['x'+suffix] = selected.astype(float).where(okay)
        out['route'+suffix] = np.select(
            [out.proxy.eq(1)&(wave==1),none&any_mention,not_asked|not_app,nonresponse,okay&selected,okay&none,okay],
            ['not_asked_proxy_W1','contradictory_list','not_applicable_unspecified','source_nonresponse',
             'selected_difficulty','explicit_list_none','unselected_evidenced_list'],
            default='no_valid_list_evidence')
        CROSS.append(dict(source='ELSA',wave=wave,item=suffix,task=label,core=FILES[wave]+'.dta',
            original_variable=actual_field,native_codes=native_codes,coordinated_check=f'R{wave}{ref.upper()}' if wave<=9 else '',
            eligibility_and_question='Multi-select list of health/memory-related difficulty; disorders expected <3 months excluded',
            coding='Selected=1; not selected=0 only with valid list evidence; unknown/off-route preserved',
            proxy='W1 proxy structurally omitted; available W2 onward where actually answered',
            nonuse='No task-specific not-doing, phone access or medication use response category in this item',
            severity='No comparable ordinal assistance/dependence level in this question',
            documentation=f'W{wave} official interview questionnaire PDF page {QUESTION_PAGES[wave]}; {STUDY}; '+
                          (f'G3 printed pp. 167–169; {D}' if wave<=9 else 'W10 native Stata value labels')))


def wave_ppc(out, wave, h):
    r = f'r{wave}'
    dz = h[[r+c for c in DISEASES]].where(lambda x:x.isin([0,1]))
    div = dz.notna().sum(axis=1)
    out['chronic_count'] = dz.sum(axis=1,min_count=7)
    out['chronic_observed'] = div
    out['chronic_count_lower'] = dz.eq(1).sum(axis=1)
    out['physical'] = np.where(dz.eq(1).any(axis=1),1.,np.where(dz.eq(0).all(axis=1),0.,np.nan))
    cesd = valid(h[r+'cesd'],0,8).where(h[r+'cesdm'].eq(0)&out.proxy.eq(0))
    out['psych'] = cesd.ge(4).astype(float).where(cesd.notna())
    band = (out.age//5*5).clip(upper=80).where(out.age.ge(45))
    norms = pd.DataFrame(index=out.index)
    for item,upper in COG.items():
        score = valid(h[r+item],0,upper).where(out.proxy.eq(0)) if r+item in h else pd.Series(np.nan,index=h.index)
        dist = score.groupby(band).agg(['count','mean','std'])
        dist['cutoff'] = (dist['mean'] - 1.5*dist['std']).where(dist['count'].ge(30)&dist['std'].gt(0))
        cutoff = band.map(dist.cutoff)
        norms[item] = score.le(cutoff).astype(float).where(score.notna()&cutoff.notna())
        for age,row in dist.iterrows():
            PPC_NORMS.append(dict(wave=wave,age_band_lower=age,component=item,**row.to_dict()))
    for flag,cols in [('cog',['imrc','dlrc','orient','verbf']),('cog3_diagnostic',['imrc','dlrc','orient'])]:
        z=norms[cols]
        out[flag] = np.where(z.eq(1).any(axis=1),1.,np.where(z.eq(0).all(axis=1),0.,np.nan))
    out['code4'] = out.physical + 2*out.psych + 4*out.cog
    out['code3_diagnostic'] = out.physical + 2*out.psych + 4*out.cog3_diagnostic
    lo,hi = count_bound(out[['physical','psych','cog']])
    out['ppc_binary'] = np.where(lo.ge(2),1.,np.where(hi.lt(2),0.,np.nan))


def household_clusters(panel):
    parent = {i:i for i in panel.person.unique()}
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]]
            a=parent[a]
        return a
    for ids in panel.loc[panel.hh.notna()].groupby(['wave','hh'],sort=False).person:
        people = ids[1].tolist()
        lead = find(people[0])
        for other in people[1:]:
            parent[find(other)] = lead
    return panel.person.map(find)


def main():
    for folder in ['PRIVATE','TABLES']:
        (OUT/folder).mkdir(exist_ok=True)
    hcols = ['ragender','radyear'] + [f'inw{w}' for w in range(1,10)]
    suffix = DISEASES+list(COG)+['agey','proxy','iwstat','iwindy','iwindm','cesd','cesdm','cwtresp','lwtresp','strat','clust']
    hcols += [f'r{w}{s}' for w in range(1,10) for s in suffix+[u[3] for u in ITEMS.values()]]
    h, _ = load(G3,hcols,must=['idauniq','ragender']+[f'inw{w}' for w in range(1,10)])
    assert 'r6verbf' not in h and all(f'r{w}verbf' in h for w in PPC_WAVES)
    discordant = {w:int(h[f'inw{w}'].eq(1).ne(h[f'r{w}iwstat'].eq(1)).sum()) for w in range(1,10)}
    blocks, n10_status = [], None
    for w in range(1,11):
        r = f'r{w}'
        folder = CORE/f'wave{w}'
        fn = folder/(FILES[w]+'.dta')
        n, nn = load(fn,[f'idahhw{w}','indager','indsex','iintdatm','iintdaty',f'w{w}indout'],
                     pattern=r'^headb\d{2}$' if w<=2 else r'^headl(?:[a-z]{2}|96)$')
        if w == 10:
            n10_status = n.w10indout
            d = n.loc[n.w10indout.isin([11,13,21,23,24,25])].copy()
        else:
            ids = h.index[h[f'inw{w}'].eq(1)&h[f'r{w}iwstat'].eq(1)]
            assert ids.isin(n.index).all(), f'Unlinked G3 interview at wave {w}'
            d = n.reindex(ids)
        hh = d[f'idahhw{w}'].where(lambda z:z.gt(0)).astype('Int64').astype('string')
        o = pd.DataFrame(dict(wave=w,hh=hh,primary_ppc_wave=w in PPC_WAVES),index=d.index)
        if w < 10:
            g = h.reindex(d.index)
            o['age'] = valid(g[r+'agey'].replace(-7,90),0,120)
            o['sex'] = g.ragender.where(g.ragender.isin([1,2]))
            o['proxy'] = g[r+'proxy'].where(g[r+'proxy'].isin([0,1]))
            year,month=valid(g[r+'iwindy'],2002,2023),valid(g[r+'iwindm'],1,12)
            for outcol,scol in [('weight','cwtresp'),('longitudinal_weight','lwtresp'),('stratum','strat'),('psu','clust')]:
                raw=g.get(r+scol,pd.Series(np.nan,index=g.index))
                if outcol in ('stratum','psu'):
                    o[outcol]=raw.astype('string').str.strip().replace('',pd.NA)
                else:
                    z=pd.to_numeric(raw,errors='coerce')
                    o[outcol]=z.where(z.gt(0))
            edu = [p for p in folder.glob('wave*ifs*d*.dta') if not p.name.endswith('_CN.dta')]
            assert len(edu)==1, f'IFS path error: {[p.name for p in edu]}'
            ed,_ = load(edu[0],['edqual'],must=['idauniq','edqual'])
            o['education'] = ed.edqual.reindex(d.index).map({7:0,3:1,4:1,5:1,1:2,2:2,6:3})
            wave_ppc(o,w,g)
        else:
            o['age'] = valid(d.indager.replace(-7,90),0,120)
            o['sex'] = d.indsex.where(d.indsex.isin([1,2]))
            o['proxy'] = d.w10indout.isin([13,23,25]).astype(float)
            year,month=valid(d.iintdaty,2021,2023),valid(d.iintdatm,1,12)
            for col in ['weight','longitudinal_weight','stratum','psu','education','chronic_count','chronic_observed',
                        'chronic_count_lower','physical','psych','cog','cog3_diagnostic','code4','code3_diagnostic','ppc_binary']:
                o[col]=np.nan
        o['month'] = year*12+month
        o['rural'] = np.nan
        classify(d,w,o,nn)
        known_n=o[X].notna().sum(axis=1)
        lo,hi=count_bound(o[X]); o['IADL5_observed_n']=known_n
        o['IADL5_lower']=lo; o['IADL5_upper']=hi; o['IADL5_COUNT']=lo.where(known_n.eq(5))
        for s in ITEMS:
            if w<=9:
                o['g3_'+s]=g[f'{r}{ITEMS[s][3]}'].where(lambda z:z.isin([0,1]))
            else:o['g3_'+s]=np.nan
        for s in ITEMS:
            o['v'+s]=np.nan  # no common ordinal severity in list instrument
        blocks.append(o.rename_axis('person').reset_index())
    panel = pd.concat(blocks,ignore_index=True)
    panel['family'] = household_clusters(panel).astype('string')
    panel['next_status'] = 'administrative'
    state_map={0:'out_of_scope',1:'interviewed',4:'alive_not_interviewed',5:'dead',6:'dead',7:'sample_exit',9:'survival_unknown'}
    for w in range(1,10):
        sel=panel.wave.eq(w)
        ids=panel.loc[sel,'person']
        if w<=8:
            target=h[f'r{w+1}iwstat'].map(state_map)
            target=target.mask(target.eq('interviewed')&h[f'inw{w+1}'].ne(1),'survival_unknown')
        else:
            target=n10_status.map({11:'interviewed',13:'interviewed',21:'interviewed',23:'interviewed',24:'interviewed',25:'interviewed',95:'dead'})
        panel.loc[sel,'next_status'] = ids.map(target).fillna('survival_unknown')
    panel['person']=panel.person.astype('string')
    assert not panel.duplicated(['person','wave']).any()
    assert panel.family.notna().all()
    assert panel.loc[panel.wave.eq(6),'cog'].dropna().eq(1).all()
    assert panel.loc[panel.wave.eq(10),'code4'].isna().all()
    assert panel.loc[panel.wave.eq(1)&panel.proxy.eq(1),X].isna().all().all()
    assert panel.loc[panel.list_conflict.eq(1),X].isna().all().all()
    assert panel.loc[panel.IADL5_observed_n.lt(5),'IADL5_COUNT'].isna().all()
    assert panel['IADL5_lower'].le(panel['IADL5_upper']).all()
    panel.to_parquet(OUT/'PRIVATE/person_wave.parquet', index=False)
    p = panel.loc[panel.age.ge(45)].copy()
    for row in CROSS:
        w,s=row['wave'],row['item']
        a=p.loc[p.wave.eq(w)].set_index('person')
        nxt=p.loc[p.wave.eq(w+1)].set_index('person')
        z=a.join(nxt[['month','x'+s,'code4']],how='left',rsuffix='_n')
        x,ref=a['x'+s],a['g3_'+s]
        mask=x.notna()&ref.notna()
        row.update(n_age45plus=len(a),actual_year_min=int(((a.month.dropna()-1)//12).min()),
                   actual_year_max=int(((a.month.dropna()-1)//12).max()),item_observed=int(x.notna().sum()),
                   next_same_item_date_valid_n=int((z['x'+s].notna()&z['x'+s+'_n'].notna()&z.month_n.gt(z.month)).sum()),
                   same_record_G3_observed_both_n=int(mask.sum()),
                   G3_disagreements_n=int(x.ne(ref).where(mask,False).sum()) if w<=9 else np.nan)
        for r,nr in a['route'+s].value_counts().items():
            ROUTES.append(dict(wave=w,item=s,route=r,n=nr))
    cov=p.groupby('wave').agg(interviewed_age45plus=('person','size'),complete_IADL5=('IADL5_COUNT','count'),
            complete_PPC_code4=('code4','count'),PPC_binary_determined=('ppc_binary','count'),
            complete_chronic_count=('chronic_count','count'),legal_date=('month','count'),
            proxy_known=('proxy','count'),education_known=('education','count'),
            route_contradictions=('list_conflict','sum'),design_psu_valid=('psu','count'),positive_cwt=('weight','count'))
    cov['complete_IADL5_rate']=cov.complete_IADL5/cov.interviewed_age45plus
    for name, data in [('FIVE_ITEM_CROSSWALK',pd.DataFrame(CROSS)),('source_manifest',pd.DataFrame(MANIFEST)),
                       ('source_fields',pd.DataFrame(DICT)),('cognitive_cutoffs',pd.DataFrame(PPC_NORMS)),
                       ('source_routes_internal',pd.DataFrame(ROUTES)),('wave_coverage',cov.reset_index())]:
        data.to_csv(OUT/'TABLES'/f'{name}.csv',index=False,encoding='utf-8-sig')
    qa=dict(person_wave_keys_unique=True,w1_proxy_not_recoded_negative=True,contradictions_unknown=True,
            incomplete_count_preserved=True,w6_fluency_absent=True,w6_negative_cognition_not_fabricated=True,
            w10_PPC_unresolved_excluded=True,inw_iwstat_discordances=discordant,older_asset_files_not_modified=True,models_fitted=0,
            visits_age45plus=len(p),participants_age45plus=int(p.person.nunique()),
            family_components_age45plus=int(p.family.nunique()),largest_family_component=int(p.groupby('family').person.nunique().max()))
    (OUT/'TABLES/source_QA.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print(cov[['interviewed_age45plus','complete_IADL5','complete_PPC_code4','legal_date','education_known']].to_string())
    print(json.dumps(qa))


if __name__=='__main__':
    main()

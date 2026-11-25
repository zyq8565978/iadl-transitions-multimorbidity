"""Read-only input verification for the targeted November 25 closure."""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='1'
from pathlib import Path
import hashlib,json,zipfile,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; W=ROOT.parent
MI=W/'PPC_IADL_ALL_METHODS_MI_20261118'; PUB=W/'PPC_IADL_FINAL_MANUSCRIPT_20261117'
IPD=W/'PPC_IADL_ZHOU_ADAPTATION_20261117'
for n in ('PRIVATE','QA','TABLES','REPORT','FIGURES','SUBMISSION'):
    (ROOT/n).mkdir(exist_ok=True)
sys.path.insert(0,str(MI/'CODE'))
import common as base
STATES={0:'No affected domain',1:'Physical only',2:'Psychological only',4:'Cognitive only',
        3:'Physical and psychological only',5:'Physical and cognitive only',6:'Psychological and cognitive only',7:'All three domains'}
TARGETS=(3,5,6,7)
def dump(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify_release():
    receipt=json.loads((MI/'QA/RELEASE_RECEIPT.json').read_text())
    path=MI/receipt['zip_name']; assert sha(path)==receipt['sha256'] and receipt['sha256'] in (ROOT/'FINAL_SCOPE_LOCK.md').read_text(encoding='utf-8')
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        manifest=z.read('RELEASE_MANIFEST.json'); assert manifest==(MI/'QA/RELEASE_MANIFEST.json').read_bytes()
        rows=json.loads(manifest); assert len(z.namelist())==len(rows)+1==receipt['members']
        local=0
        for v in rows:
            data=z.read(v['path']); assert len(data)==v['bytes'] and hashlib.sha256(data).hexdigest()==v['sha256'],v['path']
            candidate=(W/v['path'].removeprefix('REPRODUCE/')) if v['path'].startswith('REPRODUCE/') else MI/v['path']
            if candidate.is_file():
                assert sha(candidate)==v['sha256'],str(candidate.relative_to(W)); local+=1
            elif v['path'].startswith('REPRODUCE/') and v['path'].endswith(('.py','.R')):
                raise ValueError('Source reproduction dependency missing: '+v['path'])
    hashed={}
    for name in ['HISTORICAL_HASHES','INPUT_HASHES']:
        registry=json.loads((MI/f'QA/{name}.json').read_text())
        for f,v in registry.items():
            assert sha(Path(f))==v,Path(f).name
        hashed[name]=len(registry)
    base.effect_gate()
    dump(ROOT/'QA/authority.json',dict(archive_sha256=receipt['sha256'],zip_members=receipt['members'],
         manifest_members=len(rows),local_matches=local,original_registries=hashed,original_MI_gate=True,
         contract_sha256=sha(ROOT/'FINAL_SCOPE_LOCK.md')))
def panel(s):
    p=(W/'PPC_IADL_SHARE_V5_20261116/PRIVATE/SHARE_panel.parquet' if s=='SHARE' else MI/f'PRIVATE/{s}_canonical_panel.parquet')
    return pd.read_parquet(p,columns=['person','wave','month','physical','psychological_v2','cog_fixed','code_fixed','chronic_count','cesd','cog_z'])
def load_risk(comp):
    return base.old.load(comp,'M_severity')
def audit():
    verify_release()
    research=[]; family=[]; cross=[]; formal=[]; parts=[]; frozen_refs={}; indices={}; ds={}
    for i,comp in enumerate(base.COMPS):
        d,y,x,src=load_risk(comp); ds[comp]=d
        assert (len(d),d.person.nunique())==[(2883,2340),(59333,34246)][i]
        code=d.code_fixed2.to_numpy(int); ppc=np.isin(code,TARGETS)
        assert np.array_equal(ppc,y) and np.array_equal(y,d.y) and np.isin(code,list(STATES)).all()
        assert d.code_fixed1.isin([0,1,2,4]).all() and d.month2.gt(d.month1).all()
        assert d.groupby('person').family.nunique().eq(1).all()
        indices[comp]=base.keys(d)
        frozen_refs[comp]=hashlib.sha256(pd.util.hash_pandas_object(d[['source','person','window','hx','code_fixed1','code_fixed2','y']],index=False).to_numpy().tobytes()).hexdigest()
        for s,g in d.groupby('source'):
            p=panel(s);p['person']=s+'|'+p.person.astype(str);p=p.set_index(['person','wave'],verify_integrity=True)
            for n,wave in ((1,'wave'),(2,'next_wave')):
                nodes=pd.MultiIndex.from_arrays([g.person,g[wave].astype(int)]);a=p.reindex(nodes)
                derived=(a.physical+2*a.psychological_v2+4*a.cog_fixed).to_numpy()
                np.testing.assert_equal(derived,g['code_fixed'+str(n)].to_numpy())
                np.testing.assert_equal(a.code_fixed.to_numpy(),derived)
                np.testing.assert_allclose(a.month,g['month'+str(n)],atol=0,rtol=0)
            # Per-variable earliest actually observed future anchor within 72 months, as in the inherited MI code.
            indexed=p.reset_index();entry=g[['person','wave','month1','month2','next_wave']]
            pairs=entry.merge(indexed,on='person',suffixes=('','_future'))
            pairs=pairs.loc[pairs.month.gt(pairs.month1)&pairs.month.le(pairs.month1+72)].sort_values('month')
            for v in ('chronic_count','cesd','cog_z','code_fixed'):
                fst=pairs.dropna(subset=[v]).drop_duplicates(['person','wave'])
                cross.append(dict(comparison=comp,source=s,variable=v,windows=len(g),future_aux_present=len(fst),
                                  earliest_aux_is_scheduled=int(fst.wave_future.eq(fst.next_wave).sum()),
                                  earliest_aux_is_same_month=int(fst.month.eq(fst.month2).sum())))
            for origin,ag in [('all',g),('physical_only',g[g.code_fixed1.eq(1)])]:
                e=ag.code_fixed2.to_numpy(int)
                psy=np.isin(e,[2,3,6,7]);cog=np.isin(e,[4,5,6,7])
                assert np.array_equal(psy,(e&2)>0) and np.array_equal(cog,(e&4)>0) and np.array_equal(psy&cog,np.isin(e,[6,7]))
                research.append(dict(comparison=comp,source=s,origin=origin,windows=len(ag),PPC=int(np.isin(e,TARGETS).sum()),psychological=int(psy.sum()),cognitive=int(cog.sum()),both=int((psy&cog).sum())))
        for k in TARGETS:
            z=ppc&(code==k); s_calib=all(0<int(z[g.index].sum())<len(g) for _,g in d.groupby('source'))
            support=min(int(((d.hx==h)&(z==v)).sum()) for h in (0,1) for v in (False,True))>=10
            formal.append(dict(comparison=comp,code=k,windows=len(d),people=d.person.nunique(),families=d.family.nunique(),
                events=int(z.sum()),arm_support=bool(support),source_calibration_possible=bool(s_calib)))
            for (s,h),g in d.assign(outcome=z).groupby(['source','history']):
                for v,gg in g.groupby('outcome'):
                    family.append(dict(comparison=comp,code=k,source=s,history=h,outcome=bool(v),windows=len(gg),families=gg.family.nunique()))
        cols=['source','history','code_fixed1','code_fixed2']
        parts.append(d.groupby(cols).size().reset_index(name='n'))
    allrows=pd.concat(ds.values(),ignore_index=True)
    assert not allrows.duplicated(['person','window']).any() and (len(allrows),allrows.person.nunique(),int(allrows.y.sum()))==(62216,35494,10160)
    counts=pd.concat(parts).groupby(['source','history','code_fixed1','code_fixed2']).n.sum().reset_index()
    assert counts.loc[counts.code_fixed2.isin(TARGETS),'n'].sum()==10160 and counts.n.sum()==62216
    pd.DataFrame(formal).to_csv(ROOT/'PRIVATE/support.csv',index=False)
    counts.to_csv(ROOT/'PRIVATE/state_counts.csv',index=False)
    pd.DataFrame(family).to_csv(ROOT/'PRIVATE/family_support.csv',index=False)
    pd.DataFrame(research).to_csv(ROOT/'PRIVATE/margins.csv',index=False)
    pd.DataFrame(cross).to_csv(ROOT/'TABLES/MI_AUXILIARY_SCHEDULE_AUDIT.csv',index=False)
    cs=pd.DataFrame([dict(code=k,physical=(k&1)>0,psychological=(k&2)>0,cognitive=(k&4)>0,state=v,cross_domain=k in TARGETS) for k,v in STATES.items()])
    cs.to_csv(ROOT/'TABLES/PPC_STATE_CODEBOOK.csv',index=False)
    # MI files hold only three completed covariates, not another set of outcomes. Exact keyed joins below test all 50.
    old_hash={s:sha(MI/f'PRIVATE/{s}_node_binding.parquet') for s in base.SOURCES}
    rows=[]
    for m in range(1,51):
        complete=base.completed(m)
        for comp,keys in indices.items():
            join=complete.loc[keys];assert len(join)==len(keys) and join.index.equals(pd.Index(keys))
            assert join[['smoking','alcohol','bmi']].notna().all().all()
            rows.append(dict(mi=m,comparison=comp,matched_windows=len(join),fixed_member_outcome_hash=frozen_refs[comp]))
    pd.DataFrame(rows).to_csv(ROOT/'QA/MI_50_membership.csv',index=False)
    dump(ROOT/'QA/binding.json',dict(pass_all=True,windows=62216,people=35494,PPC_events=10160,other_states=52056,
         participants_both=int(len(set(ds[base.COMPS[0]].person)&set(ds[base.COMPS[1]].person))),
         domain_bits_and_scheduled_native_dates_exact=True,MI=50,source_nodes_sha256=old_hash,
         state_partition_and_physical_origin_margins_exact=True))
    print('Verified 62,216 windows / 35,494 people / 10,160 events and 50 keyed overlays.',flush=True)
if __name__=='__main__':audit()

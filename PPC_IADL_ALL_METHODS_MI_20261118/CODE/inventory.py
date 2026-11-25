"""Read-only inventory of existing analysis populations, never participant output."""
import os
for v in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'): os.environ[v]='1'
from pathlib import Path
import sys, importlib.util, json, hashlib
import pandas as pd
import numpy as np
R=Path(__file__).resolve().parents[1]; W=R.parent
for n in ('PRIVATE','QA','TABLES','REPORT','FIGURES'): (R/n).mkdir(exist_ok=True)
S=W/'PPC_IADL_ONE_STAGE_SUPPLEMENTS_20261117'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod

def main():
    rows=[]; fields={}; frames=[]
    def add(module,source,d,person='person',wave='wave',family='cluster'):
        fields[module+'|'+source]=list(d.columns)
        z=pd.DataFrame({'source':source,'person':d[person].astype(str),
                        'wave':pd.to_numeric(d[wave],errors='raise').astype(int),
                        'history':d['history'].astype(str)})
        for pref in (source+'|',source+':'): z.person=z.person.str.removeprefix(pref)
        z['key']=source+'|'+z.person+'|'+z.wave.astype(str)
        assert z.history.isin(['H00','H01','H10','H11']).all()
        assert z.groupby('key').history.nunique().eq(1).all()
        z=z.drop_duplicates('key');z['module']=module;frames.append(z)
        old=pd.read_parquet(W/f'PPC_IADL_LONGITUDINAL_MI_20261117/PRIVATE/{source}_node_binding.parquet',columns=['key'])
        rows.append(dict(module=module,source=source,records=len(d),people=z.person.nunique(),nodes=len(z),
                         old_MI_nodes=int(z.key.isin(old.key).sum()),new_nodes=int((~z.key.isin(old.key)).sum())))
    sys.path.insert(0,str(W/'PPC_IADL_ZHOU_ADAPTATION_20261117'))
    import run_ipd as ipd
    for comp in ipd.COMPS:
        d,y,x,src=ipd.load(comp,'M_severity')
        for s,g in d.groupby('source'): add('primary_'+comp,s,g)
    co=load('allmi_cont',S/'CONTINUOUS/run_continuous.py')
    raw=dict(co.inputs())
    for s,(p,e,z) in raw.items():
        p.to_parquet(R/'PRIVATE'/f'{s}_canonical_panel.parquet',index=False)
        fields['panel|'+s]=list(p.columns)
        for comp in co.OLD['COMPARE']:
            for dom in co.DOMAINS:
                d,_,_=co.initialize(s,raw[s],comp,dom,False)
                add('continuous_'+comp+'_'+dom,s,d,wave='index_wave')
    sys.path.insert(0,str(W/'PPC_IADL_ZHOU_ADAPTATION_20261117/CODE'))
    import association
    d,k=association.read_locked(W/'PPC_IADL_ZHOU_ADAPTATION_20261117')
    for comp in association.CONTRASTS:
        z,_=association.cohort_overlap(d,comp)
        for s,g in z.groupby('source'):add('trajectory_'+comp[0]+'_'+comp[1],s,g,wave='index_wave')
    rv=load('allmi_rev',S/'REVERSIBLE/run.py')
    for comp in ipd.COMPS:
        items=rv.source_data(comp.replace('_vs_','_'),False)
        for item in items:
            a=item['a'];add('reversible_'+comp,item['source'],a)
    ls=load('allmi_life',S/'LIFESTYLE/run_lifestyle.py')
    aa,dd=ls.prep(False)
    for (comp,spec),groups in aa.items():
        for s,(full,z,covs) in groups.items():add('A2_'+comp+'_'+spec,s,full)
    for (comp,spec),groups in dd.items():
        for s,z in groups.items():add('D_'+comp+'_'+spec,s,z)
    ob=load('allmi_obs',S/'OBSERVATION/run_observation.py')
    for s in ipd.SOURCES:
        q=ob.S['load']() if s=='SHARE' else ob.O['linked'](s)
        if isinstance(q,tuple):q=q[0]
        for comp in ipd.COMPS:
            d=ob.S['select_target'](q,comp) if s=='SHARE' else ob.O['obs_prepare'](q,s,comp)
            add('observation_'+comp,s,d)
    pd.DataFrame(rows).to_csv(R/'TABLES/MODULE_MEMBERSHIP_INVENTORY.csv',index=False)
    joined=pd.concat(frames,ignore_index=True)
    assert joined.groupby('key').history.nunique().eq(1).all(),'History differs across frozen method bindings'
    joined.to_parquet(R/'PRIVATE/module_nodes.parquet',index=False)
    (R/'QA/FIELD_INVENTORY.json').write_text(json.dumps(fields,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()

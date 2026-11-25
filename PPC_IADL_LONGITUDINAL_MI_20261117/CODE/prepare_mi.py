"""Build unique source/person/visit records; frozen risk sets remain unchanged."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'): os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]; W=R.parent
for sub in ('PRIVATE','TABLES','QA','REPORT','FIGURES'): (R/sub).mkdir(exist_ok=True)
sys.path.insert(0,str(W/'PPC_IADL_ZHOU_ADAPTATION_20261117'))
import run_ipd as old
Q=W/'PPC_IADL_ZHOU_COVARIATE_QC_20261117'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    hist=json.loads((W/'PPC_IADL_COVARIATE_SAMPLE_RECOVERY_AUDIT_20261117/QA/AUDIT_STATUS.json').read_text())['source_sha256']
    for p,h in hist.items():assert sha(W/p)==h,('old_bound_hash_changed',p)
    frozen=json.loads((W/'PPC_IADL_ZHOU_ADAPTATION_20261117/ONE_STAGE_LOCK.json').read_text())
    for p,h in frozen.items():assert sha(W/p)==h,('frozen_original_hash_changed',p)
    sources={s:[] for s in old.SOURCES}; binding={}; support=[]
    for comp in old.COMPS:
        d,y,x,src=old.load(comp,'M_severity')
        frames=[]
        for s in old.SOURCES:
            z=pd.read_parquet(Q/'PRIVATE'/f'{s}_{comp}_bound.parquet')
            z['source']=s;z['comparison']=comp
            z['person']=s+'|'+z.person.astype(str)
            z['family']=s+'|'+z.cluster.astype(str)
            z['key']=z.person+'|'+z.wave.astype(str)
            frames.append(z);sources[s].append(z)
        a=pd.concat(frames,ignore_index=True)
        assert np.array_equal(d.person,a.person) and np.array_equal(d.window,a.window)
        assert np.array_equal(y,a.y) and np.array_equal(d.hx,a.hx)
        d['key']=a.key.to_numpy();d.to_parquet(R/'PRIVATE'/f'{comp}_rows.parquet',index=False)
        np.savez_compressed(R/'PRIVATE'/f'{comp}_frozen_design.npz',x=x,y=y,src=src)
        binding[comp]={'windows':len(d),'people':int(d.person.nunique()),'families':int(d.family.nunique()),
            'events':int(y.sum()),'member_sha256':hashlib.sha256(d[['person','window']].to_csv(index=False).encode()).hexdigest()}
    for s,frames in sources.items():
        a=pd.concat(frames,ignore_index=True)
        assert not a.duplicated(['person','wave']).any()
        nodes=[]
        for suf,t in [('t',1),('pre',0)]:
            n=pd.DataFrame({'key':a.person+'|'+a['wave' if t==1 else 'prior_wave_qc'].astype(str),
                'person':a.person,'family':s+'|'+a['cluster'+str(t)].astype(str),
                'wave':a['wave' if t==1 else 'prior_wave_qc'],'analysis_node':float(t==1),
                'month':a['month'+str(t)],'Yaux':a.y,'outcome_gap':(a.month2-a['month'+str(t)])/12,
                'history':a.history if t==1 else 'unknown_prior',
                'gap01':a.gap01 if t==1 else np.nan,'gap12':a.gap12 if t==1 else np.nan})
            for v in ('age','sex','education','proxy','K','chronic_count','cesd','cog_z','code_fixed'):
                n[v]=a[v+str(t)]
            n['country']=a.country if 'country' in a else 'single'
            for v in ('smoking','alcohol','bmi'):
                n[v]=a[v+'_'+suf]
                n[v+'_reason']=a[v+'_reason_'+suf]
                n['lag_'+v]=a[v+'_pre'] if t==1 else np.nan
            nodes.append(n)
        allnodes=pd.concat(nodes,ignore_index=True)
        conflicts={v:int((allnodes.groupby('key')[v].nunique(dropna=True)>1).sum()) for v in ('smoking','alcohol','bmi','month','age','sex')}
        assert not any(conflicts.values()),(s,conflicts)
        # Current-node definition has priority over prior-only auxiliary representations.
        n=allnodes.drop_duplicates('key',keep='first').reset_index(drop=True)
        n['node_id']=np.arange(len(n))
        n.to_parquet(R/'PRIVATE'/f'{s}_node_binding.parquet',index=False)
        meta=n[['wave','analysis_node','smoking_reason','alcohol_reason','bmi_reason']].copy()
        meta['lag_wave']=n.key.map(a.set_index('key').prior_wave_qc)
        meta.to_csv(R/'PRIVATE'/f'{s}_node_meta.csv',index=False)
        z=pd.DataFrame({'person_id':pd.factorize(n.person,sort=True)[0]+1,
                        'family_id':pd.factorize(n.family,sort=True)[0]+1,
                        'smoking':n.smoking,'alcohol':n.alcohol,'bmi':n.bmi})
        cov={}
        for v in ('age','K','chronic_count','cesd','cog_z','gap01','gap12','outcome_gap'):
            q=pd.to_numeric(n[v],errors='coerce');med=float(q.median());sd=float(q.std()) or 1.
            cov[v]=(q.fillna(med)-med)/sd
            if q.isna().any():cov[v+'_unavailable']=q.isna().astype(float)
        cov['age_squared']=cov['age']**2
        cov['calendar_year']=(n.month-n.month.median())/12
        cov['analysis_node']=n.analysis_node
        cov['Yaux']=n.Yaux
        for v in ('sex','education','proxy','country','code_fixed','history'):
            cc=pd.get_dummies(n[v].astype(str),prefix=v,drop_first=True,dtype=float)
            for j,c in enumerate(cc):cov[f'{v}_{j}']=cc[c]
        for v in ('smoking','alcohol','bmi'):
            q=n['lag_'+v];med=float(q.median());sd=float(q.std()) or 1.
            cov['lag_'+v]=(q.fillna(med)-med)/sd
            cov['lag_'+v+'_unknown']=q.isna().astype(float)
        base=list(cov)
        for v in base:
            if v.startswith(('history_','code_fixed_')) or v in ('age','K','chronic_count','cesd','cog_z'):
                cov['Y_by_'+v]=n.Yaux*cov[v]
        for v,q in cov.items():
            if np.std(q)>1e-10:z[v]=q
        assert z.drop(columns=['smoking','alcohol','bmi']).notna().all().all()
        # No free wave dummy: whole-wave absent values have no identifiable wave intercept.
        z.to_csv(R/'PRIVATE'/f'{s}_mi_input.csv',index=False)
        for (wave,idx),g in n.groupby(['wave','analysis_node']):
            row={'source':s,'wave':int(wave),'analysis_node':int(idx),'nodes':len(g)}
            for v in ('smoking','alcohol','bmi'):
                row[v+'_observed']=int(g[v].notna().sum());row[v+'_missing']=int(g[v].isna().sum())
                row[v+'_structural']=int((g[v].isna() & (g[v+'_reason'].eq('structurally_unavailable')|g[v+'_reason'].eq('.q'))).sum())
            support.append(row)
        print(s,'unique_nodes',len(n),'persons',n.person.nunique(),'analysis_nodes',int(n.analysis_node.sum()),'predictors',z.shape[1]-5,flush=True)
    pd.DataFrame(support).to_csv(R/'PRIVATE/source_wave_support_exact.csv',index=False)
    release=pd.DataFrame(support).astype(object)
    def band(v):
        n=int(v)
        if n==0:return '0'
        for upper,label in ((10,'<10'),(50,'10-49'),(100,'50-99'),(500,'100-499'),(1000,'500-999'),(5000,'1000-4999')):
            if n<upper:return label
        return '5000+'
    for c in release.columns[3:]:release[c]=release[c].map(band)
    release.to_csv(R/'TABLES/source_wave_support.csv',index=False,encoding='utf-8-sig')
    (R/'QA/input_binding.json').write_text(json.dumps({'original_hashes':{**frozen,**hist},'risk_sets':binding,
        'unique_person_wave_pass':True,'no_outcome_imputed':True,'no_new_members':True},indent=2),encoding='utf-8')
    print(json.dumps(binding,indent=2))

if __name__=='__main__':main()

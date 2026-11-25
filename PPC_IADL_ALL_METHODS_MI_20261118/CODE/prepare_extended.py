"""Source-specific enlarged observed covariate nodes and outcome auxiliaries."""
from inventory import R,W,S,load
import sys,json,hashlib
import numpy as np
import pandas as pd
Q=W/'PPC_IADL_ZHOU_COVARIATE_QC_20261117'
PREV=W/'PPC_IADL_LONGITUDINAL_MI_20261117'
SOURCES=('CHARLS','HRS','ELSA','SHARE')

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def numeric_aux(z,values,name):
    v=pd.to_numeric(values,errors='coerce').replace([np.inf,-np.inf],np.nan)
    med=float(v.median()) if v.notna().any() else 0.
    sd=float(v.std()) if v.notna().sum()>1 else 1.
    if not np.isfinite(sd) or sd<1e-10:sd=1.
    if v.notna().any() and v.nunique()>1:z[name]=(v.fillna(med)-med)/sd
    if v.isna().any() and v.notna().any():z[name+'_unavailable']=v.isna().astype(float)

def main():
    membership=pd.read_parquet(R/'PRIVATE/module_nodes.parquet')
    assert membership.groupby('key').history.nunique().eq(1).all()
    frozen_history=membership.drop_duplicates('key').set_index('key').history
    qc=load('allmi_covariates',Q/'CODE/prepare_covariates.py')
    # Import provides read helpers only. New evidence and all output paths stay local.
    qc.R=R;qc.P=R/'PRIVATE';qc.Q=R/'QA';qc.T=R/'TABLES'
    panels={};nodes={};aux={}
    for s in SOURCES:
        p=pd.read_parquet((W/'PPC_IADL_SHARE_V5_20261116/PRIVATE/SHARE_panel.parquet') if s=='SHARE' else R/f'PRIVATE/{s}_canonical_panel.parquet')
        p['person']=p.person.astype(str);p['wave']=p.wave.astype(int)
        assert not p.duplicated(['person','wave']).any()
        p['key']=s+'|'+p.person+'|'+p.wave.astype(str)
        panels[s]=p
        wanted=membership.loc[membership.source.eq(s),'key'].unique()
        ix=p.set_index('key',verify_integrity=True)
        assert set(wanted)<=set(ix.index),(s,'missing frozen index nodes')
        schedule=sorted(p.wave.unique());prev=dict(zip(schedule[1:],schedule[:-1]))
        base=ix.loc[wanted].copy();base['prior_wave']=base.wave.map(prev)
        prior_keys=s+'|'+base.person+'|'+base.prior_wave.astype('Int64').astype(str)
        old=pd.read_parquet(PREV/f'PRIVATE/{s}_node_binding.parquet')
        keys=list(dict.fromkeys([*wanted,*prior_keys[prior_keys.isin(ix.index)],*old.key]))
        assert set(keys)<=set(ix.index)
        n=ix.loc[keys].reset_index();n['analysis_node']=n.key.isin(wanted).astype(float)
        n['person']=s+'|'+n.person
        n['family']=s+'|'+n.cluster.astype(str)
        n['lag_wave']=n.wave.map(prev)
        n['lag_key']=n.person+'|'+n.lag_wave.astype('Int64').astype(str)
        n['frozen_history']=n.key.map(frozen_history)
        assert n.loc[n.analysis_node.eq(1),'frozen_history'].isin(['H00','H01','H10','H11']).all()
        if 'country' not in n:n['country']='single'
        o=pd.read_parquet(Q/f'PRIVATE/{s}_covariates.parquet')
        if s in ('CHARLS','HRS'):
            f,key,ww=qc.files[s];extra_wave=2011 if s=='CHARLS' else 2010
            qc.files[s]=(f,key,{extra_wave:1 if s=='CHARLS' else 10})
            extra=qc.harmonized(s)
            o=pd.concat([o,extra],ignore_index=True)
        o['key']=s+'|'+o.person.astype(str)+'|'+o.wave.astype(int).astype(str)
        oo=o.drop(columns=['person','wave']).set_index('key',verify_integrity=True)
        nc=n[['key','person','family','wave','month','analysis_node','lag_wave','lag_key','frozen_history','country','age','sex','education','proxy','K','chronic_count','cesd','cog_z','code_fixed']].copy()
        for c in ('smoking','alcohol','bmi_raw','height','weight','bmi_month','bmi_mode','smoking_reason','alcohol_reason','bmi_reason'):
            nc[c]=nc.key.map(oo[c])
        good=nc.bmi_raw.between(10,80)&nc.height.between(.9,2.3)&nc.weight.between(20,300)
        nurse=nc.bmi_mode.eq('same_wave_nurse_measurement')
        good &= ~nurse | (nc.bmi_month-nc.month).abs().le(6)
        nc['bmi']=nc.bmi_raw.where(good)
        nc.loc[nurse&~good,'bmi_reason']='nurse_date_or_physical_range_not_admissible'
        for v in ('smoking','alcohol'):
            nc.loc[nc.proxy.ne(0),v]=np.nan
            nc.loc[nc.proxy.ne(0),v+'_reason']='proxy_no_direct_measurement'
        nc.loc[nc.bmi_mode.eq('structurally_unavailable'),'bmi_reason']='structurally_unavailable'
        # Preserve prior timing-qualified measurements exactly. Only previously
        # unextracted early historical waves receive newly verified values.
        old=old.set_index('key',verify_integrity=True)
        use=nc.key.isin(old.index)&~nc.wave.isin([2011] if s=='CHARLS' else [2010] if s=='HRS' else [])
        for v in ('smoking','alcohol','bmi'):
            nc.loc[use,v]=nc.loc[use,'key'].map(old[v])
            nc.loc[use,v+'_reason']=nc.loc[use,'key'].map(old[v+'_reason'])
        assert nc.groupby('person').family.nunique().max()==1
        nodes[s]=nc
        # Actual future domain measurements are auxiliary only, not filled outcomes.
        rec=[]
        groups={k:v.sort_values('month') for k,v in p.groupby('person',sort=False)}
        for row in nc.itertuples():
            g=groups[row.person.split('|',1)[1]]
            fut=g.loc[g.month.gt(row.month)&g.month.le(row.month+72)]
            out={'key':row.key}
            for var in ('chronic_count','cesd','cog_z','code_fixed'):
                f=fut.loc[fut[var].notna(),['month',var]]
                out['future_'+var+'_n']=len(f)
                if len(f):
                    out['future_'+var+'_first']=float(f.iloc[0][var]);out['future_'+var+'_mean']=float(f[var].mean())
                    out['future_'+var+'_first_gap']=(float(f.iloc[0].month)-row.month)/12
                if len(f)>1 and f.month.nunique()>1 and var!='code_fixed':
                    t=(f.month.to_numpy(float)-row.month)/12;t-=t.mean()
                    out['future_'+var+'_slope']=float(t@f[var].to_numpy(float)/(t@t))
            rec.append(out)
        aux[s]=pd.DataFrame(rec).set_index('key')
        print(s,'union nodes',len(nc),'people',nc.person.nunique(),flush=True)
    # Same existing index histories and binary PPC outcomes.
    sys.path.insert(0,str(W/'PPC_IADL_ZHOU_ADAPTATION_20261117'))
    import run_ipd as ipd
    for comp in ipd.COMPS:
        d,*_=ipd.load(comp,'M_severity')
        for s,g in d.groupby('source'):
            key=g.person+'|'+g.wave.astype(int).astype(str)
            for c in ('hx','y','gap01','gap12'):
                aux[s].loc[key,'primary_'+c]=g[c].to_numpy()
    # Actual fixed classes, not posterior uncertainty correction or reclassification.
    sys.path.insert(0,str(W/'PPC_IADL_ZHOU_ADAPTATION_20261117/CODE'))
    import association
    d,k=association.read_locked(W/'PPC_IADL_ZHOU_ADAPTATION_20261117')
    for s,g in d.groupby('source'):
        key=g.person.str.replace(s+':',s+'|',regex=False)+'|'+g.index_wave.astype(int).astype(str)
        take=key.isin(aux[s].index)
        for j in range(1,k+1):aux[s].loc[key[take],f'locked_class_{j}']=g.loc[take,'profile'].eq(j).astype(float).to_numpy()
    rv=load('allmi_reversible_prep',S/'REVERSIBLE/run.py')
    for comp in ipd.COMPS:
        for item in rv.source_data(comp.replace('_vs_','_'),False):
            s=item['source'];a=item['a'];rr=item['rows']
            key=s+'|'+a.person.astype(str)+'|'+a.wave.astype(float).astype(int).astype(str)
            if not set(key)<=set(aux[s].index):raise AssertionError('panel membership mismatch')
            for state in (1,2,3,4,99):
                cnt=rr.loc[rr.state.eq(state)].groupby('pid').size().reindex(a.index,fill_value=0)
                aux[s].loc[key,'panel_state_'+str(state)+'_n']=cnt.to_numpy()
            last=rr.groupby('pid').dt.sum().reindex(a.index,fill_value=0)
            aux[s].loc[key,'panel_last_time']=last.to_numpy()
    life=load('allmi_lifestyle_prep',S/'LIFESTYLE/run_lifestyle.py')
    aa,dd=life.prep(False)
    for (comp,kind),groups in dd.items():
        for s,g in groups.items():
            key=s+'|'+g.person.astype(str)+'|'+g.wave.astype(int).astype(str)
            for c in ('mediator','y'):aux[s].loc[key,'fourvisit_'+kind+'_'+c]=g[c].to_numpy(float)
    summary=[];revision_checks=[]
    for s,n in nodes.items():
        ax=aux[s].reindex(n.key).reset_index(drop=True)
        linked=n.loc[n.analysis_node.eq(1)&n.lag_key.isin(n.key),['key','lag_key','frozen_history']]
        assert linked.lag_key.is_unique
        n['next_index_history']=n.key.map(linked.set_index('lag_key').frozen_history)
        # Explicit pre-history covariate models also need their later, observed
        # analysis history and outcome context in the imputation model only.
        for c in aux[s]:
            if c=='primary_y' or c.startswith('fourvisit_'):
                lookup=pd.Series(linked.key.map(aux[s][c]).to_numpy(),index=linked.lag_key)
                ax['next_index_'+c]=n.key.map(lookup).to_numpy()
        for v in ('smoking','alcohol','bmi'):
            n['lag_'+v]=n.lag_key.map(n.set_index('key')[v])
        archived=R/'ATTEMPTS/MI_v1_incomplete_history_auxiliary/PRIVATE'
        if archived.exists():
            prior=pd.read_parquet(archived/f'{s}_node_binding.parquet')
            fixed=['key','person','family','wave','analysis_node','smoking','alcohol','bmi']
            pd.testing.assert_frame_equal(n[fixed],prior[fixed],check_dtype=False,check_exact=True)
            old_input=pd.read_csv(archived/f'{s}_mi_input.csv',usecols=['primary_hx_unavailable'])
            missing_before=int((n.analysis_node.eq(1)&old_input.primary_hx_unavailable.eq(1)).sum())
            revision_checks.append(dict(source=s,analysis_nodes=int(n.analysis_node.sum()),
                analysis_history_auxiliary_missing_before=missing_before,
                analysis_history_auxiliary_missing_after=int((n.analysis_node.eq(1)&n.frozen_history.isna()).sum()),
                observed_covariates_and_membership_unchanged=True))
        n['node_id']=np.arange(len(n));n.to_parquet(R/f'PRIVATE/{s}_node_binding.parquet',index=False)
        n[['wave','analysis_node','lag_wave','smoking_reason','alcohol_reason','bmi_reason']].to_csv(R/f'PRIVATE/{s}_node_meta.csv',index=False)
        z=pd.DataFrame({'person_id':pd.factorize(n.person,sort=True)[0]+1,'family_id':pd.factorize(n.family,sort=True)[0]+1,
                        **{v:n[v] for v in ('smoking','alcohol','bmi')}})
        for c in ('age','K','chronic_count','cesd','cog_z'):numeric_aux(z,n[c],c)
        z['age_squared']=z.age**2;z['calendar_year']=(n.month-n.month.median())/12
        z['analysis_node']=n.analysis_node
        for history in ('H01','H10','H11'):
            z['iadl_history_'+history]=n.frozen_history.eq(history).astype(float)
        z['iadl_history_unavailable']=n.frozen_history.isna().astype(float)
        for history in ('H01','H10','H11'):
            z['next_index_history_'+history]=n.next_index_history.eq(history).astype(float)
        z['next_index_history_unavailable']=n.next_index_history.isna().astype(float)
        for c in ('sex','education','proxy','country','code_fixed'):
            dm=pd.get_dummies(n[c].astype(str),drop_first=True,dtype=float)
            for j,col in enumerate(dm):z[f'{c}_{j}']=dm[col]
        for v in ('smoking','alcohol','bmi'):numeric_aux(z,n['lag_'+v],'lag_'+v)
        for c in ax:numeric_aux(z,ax[c],c)
        # Preserve nonlinear outcome/history associations approximately without
        # learning or selecting predictors based on an exposure-effect statistic.
        # The same known exposure history is needed for every method's members,
        # including those outside the original next-visit PPC risk set.
        h=n.frozen_history.isin(['H10','H11']).astype(float).where(n.frozen_history.notna())
        for c in ('future_chronic_count_first','future_cesd_first','future_cog_z_first','primary_y'):
            if c in ax:numeric_aux(z,h*ax[c],c+'_by_history')
        if 'next_index_primary_y' in ax:
            hh=n.next_index_history.isin(['H10','H11']).astype(float).where(n.next_index_history.notna())
            numeric_aux(z,hh*ax.next_index_primary_y,'next_index_y_by_history')
        assert z.drop(columns=['smoking','alcohol','bmi']).notna().all().all()
        constant=[c for c in z if c not in ('person_id','family_id','smoking','alcohol','bmi') and z[c].nunique()<=1]
        z=z.drop(columns=constant)
        z.to_csv(R/f'PRIVATE/{s}_mi_input.csv',index=False)
        summary.append(dict(source=s,nodes=len(n),people=n.person.nunique(),index_nodes=int(n.analysis_node.sum()),predictors=len(z.columns)-5,
             analysis_nodes_with_frozen_history=int((n.analysis_node.eq(1)&n.frozen_history.notna()).sum()),
             history_unavailable_auxiliary_only_nodes=int(n.frozen_history.isna().sum()),
             **{v+'_missing':int(n[v].isna().sum()) for v in ('smoking','alcohol','bmi')}))
    pd.DataFrame(summary).to_csv(R/'TABLES/EXTENDED_MI_INPUT_SUPPORT.csv',index=False)
    if revision_checks:
        pd.DataFrame(revision_checks).to_csv(R/'TABLES/MI_HISTORY_COVERAGE_CORRECTION.csv',index=False)
        (R/'QA/MI_INPUT_REVISION_QA.json').write_text(json.dumps(dict(
            revised_before_any_effect_fit=True,pass_all=True,sources=revision_checks),indent=2))
    pd.DataFrame(qc.cross).to_csv(R/'TABLES/ADDITIONAL_HISTORICAL_FIELD_BINDING.csv',index=False)
    oldlock=json.loads((PREV/'QA/input_binding.json').read_text())['original_hashes']
    for p,h in oldlock.items():assert sha(W/p)==h
    frozen={str(PREV/'ANALYSIS_LOCK.md'):sha(PREV/'ANALYSIS_LOCK.md'),**{str(W/p):h for p,h in oldlock.items()}}
    (R/'QA/HISTORICAL_HASHES.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    (R/'QA/INPUT_HASHES.json').write_text(json.dumps({str(p):sha(p) for p in (R/'PRIVATE').glob('*_mi_input.csv')},indent=2),encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False))

if __name__=='__main__':main()

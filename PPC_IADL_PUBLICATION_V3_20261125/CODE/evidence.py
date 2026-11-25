"""Read original inputs locally; create *only* aggregate editorial data. Never fit or impute."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; WORK=ROOT.parent
V2=WORK/'PPC_IADL_PUBLICATION_V2_20261125'; A=V2/'AGGREGATES'
M=WORK/'PPC_IADL_ALL_METHODS_MI_20261118'; L=WORK/'PPC_IADL_LONGITUDINAL_MI_20261117'
OUT=ROOT/'SOURCE_DATA'; AR=ROOT/'ARCHIVE'; SUB=ROOT/'SUBMISSION'; FIG=ROOT/'FIGURES'
for folder in (OUT,AR,SUB,FIG,ROOT/'RENDER_QA'): folder.mkdir(exist_ok=True)
sys.path.insert(0,str(V2/'CODE'))
import baseline as bl
C=bl.cohorts; HS=bl.hs; H=bl.hist
loadj=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def csv(name,rows,where=OUT):pd.DataFrame(rows).to_csv(where/(name+'.csv'),index=False,encoding='utf-8-sig')
def safe(n):return n==0 or n>=10
def nfmt(n):return f'{int(n):,}'
def pct(q,n):
    assert safe(q) and safe(n-q),'Coarsen this partition before disclosure'
    return f'{q:,} ({100*q/n:.1f})'

def baseline_data():
    d=bl.get_first(); d['domains']=d.code_fixed1.replace({4:2}); d['age3']=pd.cut(d.age1,[45,70,80,np.inf],right=False,labels=False)
    frames=[d]+[d[d.history.eq(h)] for h in HS]; by=[d[d.source.eq(s)] for s in C]
    def cell(v,g,mode,lev=None):
        x=g[v].dropna()
        if mode=='cat':return pct(int(x.eq(lev).sum()),len(x))
        if mode=='iqr':return f'{x.median():g} ({x.quantile(.25):g}–{x.quantile(.75):g})'
        places=2 if v=='cog_z1' else 1
        return f'{x.mean():.{places}f} ({x.std():.{places}f})'.replace('-','−')
    def splitcounts(v,gs):return [sum(g[v].eq(2)) for g in gs]
    domain_level='domains' if all(safe(k) for k in splitcounts('domains',frames+by)) else 'single'
    d['single']=d.code_fixed1.ne(0).astype(int)
    if domain_level=='single':
        frames=[d]+[d[d.history.eq(h)] for h in HS];by=[d[d.source.eq(s)] for s in C]
    lev=[(0,'None'),(1,'Physical only'),(2,'Psychological or cognitive only')] if domain_level=='domains' else [(0,'None'),(1,'One affected domain')]
    merged={label:[cell(domain_level,g,'cat',i) for g in frames] for i,label in lev}
    rows=[['Participants, n',*[nfmt(len(g)) for g in by]]]
    def add(v,label,mode='mean',k=None):rows.append([label,*[cell(v,g,mode,k) for g in by]])
    def cats(v,ls):
        for k,label in ls:add(v,label,'cat',k)
    add('age1','Age, years, mean (SD)')
    cats('age3',enumerate(['45–69 years','70–79 years','≥80 years']))
    cats('sex1',[(1,'Men'),(2,'Women')]);add('smoking','Current smoking','cat',1);add('alcohol','Weekly or recent alcohol use','cat',1)
    add('bmi','Body mass index, kg/m², mean (SD)');cats('bmicat',enumerate(['BMI <18.5','BMI 18.5–<25','BMI 25–<30','BMI ≥30']))
    add('K0','Preceding IADL difficulty count, median (IQR)','iqr');add('K1','Index IADL difficulty count, median (IQR)','iqr')
    cats(domain_level,lev);add('chronic_count1','Physical condition count, median (IQR)','iqr')
    for v in bl.dx:add(v,v,'cat',1)
    add('sym','Depressive symptoms, % of scale maximum, mean (SD)');add('cog_z1','Cognition, mean subtest Z, mean (SD)')
    cores=['age1','sex1','education1','K0','K1','code_fixed1','chronic_count1','sym','cog_z1',*bl.dx]
    assert d[cores].notna().all().all()
    vars=['smoking','alcohol','bmi']; missing=[];source=[]
    for label,g in zip(['All',*H],frames):
        for v in vars:
            o=g[v].notna().sum();assert safe(o) and safe(len(g)-o)
            missing.append(dict(group=label,variable=v,total=len(g),observed=o,missing=len(g)-o))
    protected=set()
    for s,g in zip(C,by):
        for v in vars:
            mis=g[v].isna(); st=mis & g[v+'_reason'].eq('structurally_unavailable')
            vals=[int((~mis).sum()),int((mis&~st).sum()),int(st.sum())]
            assert sum(vals)==len(g)
            if not all(map(safe,vals)):protected.add((s,v))
            source.append(dict(source=s,variable=v,total=len(g),observed=vals[0],ordinary_missing=vals[1],whole_wave=vals[2]))
    # One non-disclosable marginal can also be recovered from the pooled total and three other sources.
    # With at least two nonzero, coarsened source marginals the pooled total cannot recover one cell.
    for v in vars:
        small=[r for r in source if r['variable']==v and (r['source'],v) in protected]
        if len(small)==1:
            others=[r for r in source if r['variable']==v and (r['source'],v) not in protected and r['ordinary_missing']>=10]
            protected.add((min(others,key=lambda r:r['ordinary_missing'])['source'],v))
    for row in source:
        cols=['observed','ordinary_missing','whole_wave']; t=row['total']; vec=np.array([row[k] for k in cols]); pc=100*vec/t
        coarse=(row['source'],row['variable']) in protected
        if coarse:
            # Largest remainder percent; true structural zeros are determined by the assessment schedule.
            r=np.floor(pc).astype(int);r[np.argsort(-(pc-r))[:100-r.sum()]]+=1
            nonzero_wave=vec[2]>0
            for k,q,val in zip(cols,r,vec):
                if k=='whole_wave' and not nonzero_wave:row[k]='0'
                else:
                    lo=int(val//50*50);row[k]=f'{lo:,}–{min(lo+49,t):,}'
                row[k+'_plot_pct']=int(q)
        else:
            for k,q in zip(cols,pc):row[k+'_plot_pct']=float(q)
        row['disclosure']=('50-person bands; whole-percentage bars' if coarse else 'exact')
    for i,v in enumerate(('smoking','alcohol')):
        if any((s,v) in protected for s in C):
            lab=['Current smoking','Weekly or recent alcohol use'][i]; r=next(r for r in rows if r[0]==lab)
            r[0]+=', observed %';r[1:]=[f'{100*g[v].mean():.1f}%' for g in by]
    csv('first_index_history_denominators',missing);csv('first_index_source_coverage',source)
    numerical=[]
    for dimension,labels,groups in [('history',['All',*H],frames),('cohort',C,by)]:
        for label,g in zip(labels,groups):
            for v in ['age1','bmi','sym','cog_z1','K0','K1','chronic_count1']:
                x=g[v].dropna(); assert len(x)>=10
                if v in ['K0','K1','chronic_count1']:stats={'median':x.median(),'Q1':x.quantile(.25),'Q3':x.quantile(.75)}
                else:stats={'mean':x.mean(),'SD':x.std()}
                numerical.extend(dict(dimension=dimension,group=label,measure=v,observed=len(x),statistic=k,value=float(z)) for k,z in stats.items())
    csv('baseline_numeric_aggregates',numerical)
    dump(OUT/'baseline_display.json',dict(merged=merged,by_source=rows))
    y=pd.concat([bl.cm.old.load(k,'M_severity')[0][['code_fixed2','y']] for k in bl.cc])
    assert y.code_fixed2.isin(range(8)).all() and np.array_equal(y.code_fixed2.isin([3,5,6,7]),y.y)
    z=y.code_fixed2.value_counts().reindex([0,1,2,4,3,5,6,7]); assert len(y)==62216 and z.iloc[4:].sum()==10160 and z.min()>=10
    csv('next_visit_states',[dict(state=int(k),windows=n,percent=100*n/len(y)) for k,n in z.items()])
    educ={}
    for s,g in zip(C,by):
        x=g.education1
        if s in ['CHARLS','ELSA']:
            ns=x.value_counts().sort_index();assert ns.min()>=10
            educ[s]=[pct(int(v),len(x)) for v in ns]
        else:educ[s]=cell('education1',g,'mean' if s=='HRS' else 'iqr')
    dump(OUT/'local_education_display.json',educ)

def continuous():
    rows=[]
    for comp in ['H11_minus_H01','H10_minus_H00']:
        for dom in ['physical','psychological','cognitive']:
            stem=f'CONTINUOUS_{comp}_{dom}_index_included'
            q=[loadj(M/f'QA/{stem}_mi{i:03d}.json') for i in range(1,51)]
            assert all(a['status']=='ESTIMATED' for a in q)
            for k in ['people','reference_people','windows','post_records','families','curve_range']:
                assert all(v[k]==q[0][k] for v in q),k
            a=q[0]; assert a['curve_range']==[3,4] and a['windows']==a['people']+a['post_records'] and 0<a['reference_people']<=a['people']
            rows.append(dict(comparison=comp,domain=dom,fit_people=a['people'],reference_people=a['reference_people'],measurements=a['windows'],post_measurements=a['post_records'],families=a['families'],imputations=len(q),reference_low=3,reference_high=4))
    csv('continuous_fit_reference',rows)

def smoking():
    n=pd.read_parquet(M/'PRIVATE/CHARLS_node_binding.parquet'); assert n.node_id.tolist()==list(range(len(n))) and n.key.is_unique
    root=WORK/'原始数据/01_CHARLS_中国'; p=next((root/'02_协调与简化数据').rglob('H_CHARLS_D_Data.dta'))
    h=pd.read_stata(p,columns=['ID',*[f'r{i}{v}' for i in [2,3,4] for v in ['smoken','smokev']]],convert_categoricals=False)
    def pid(a):
        z=a.astype(str).str.strip();return 'CHARLS|'+z.where(z.str.len().ne(11),z.str[:-2]+'0'+z.str[-2:])
    h.index=pid(h.ID); assert h.index.is_unique
    native=[];n['ever']=np.nan
    for w,j in [(2013,2),(2015,3),(2018,4)]:
        idx=n.wave.eq(w); u=n.loc[idx];s=u.person.map(h[f'r{j}smoken']); direct=u.proxy.eq(0)
        legal=s.where(direct & s.isin([0,1])); np.testing.assert_allclose(legal,u.smoking,equal_nan=True)
        n.loc[idx,'ever']=u.person.map(h[f'r{j}smokev'])
        rawpath=next((root/'01_官方原始数据'/str(w)).rglob('Health_Status_and_Functioning.dta'))
        with pd.read_stata(rawpath,iterator=True) as r:fields=r.variable_labels()
        qs=[k for k in ['da059','da061','da061_w3','da061_w4'] if k in fields]
        a=pd.read_stata(rawpath,columns=['ID',*qs],convert_categoricals=False);a.index=pid(a.ID)
        assert a.index.is_unique
        # The two current-status items are alternative routed questions. A negative ever screen means not current.
        q=a[[k for k in qs if '061' in k]].where(lambda v:v.isin([1,2,3]))
        current=q.bfill(axis=1).iloc[:,0].map({1:1.,2:0.,3:0.});current=current.fillna(a.da059.map({2:0.}))
        raw=u.person.map(current);both=raw.notna() & u.smoking.notna() & direct
        native.append(dict(wave=w,fields='/'.join(qs),direct_concordant=bool(raw[both].eq(u.smoking[both]).all()),
                           comparable=int(both.sum()),harmonized_missing_native_response_present=bool((u.smoking.isna()&direct&raw.notna()).any())))
    observed=n.smoking.notna(); missing=~observed; assert (observed.sum(),missing.sum())==(16294,1242)
    mats=np.column_stack([pd.read_csv(M/f'PRIVATE/CHARLS_mi_{i:03d}.csv',usecols=['smoking']).smoking for i in range(1,51)])
    assert np.isin(mats,[0,1]).all() and np.all(mats[observed]==n.loc[observed,'smoking'].to_numpy()[:,None])
    n['filled']=mats.mean(axis=1)
    # Link a genuinely preceding wave on the same person; missing earlier reports remain unknown.
    previous=n.lag_key.map(n.set_index('key').smoking)
    available=n.set_index('key').month
    assert (n.loc[previous.notna(),'lag_key'].map(available)<n.loc[previous.notna(),'month']).all()
    # Use documented sex where present; publish the two known-sex groups only if no small
    # missing-sex complement would be recoverable. Otherwise release conditional rates only.
    groups={'wave':n.wave.astype(str),'sex':pd.to_numeric(n.sex,errors='coerce').map({1:'men',2:'women'}).fillna('unknown'),
            'prior_smoking':previous.map({0:'not current',1:'current'}).fillna('not observed')}
    rows=[]
    for kind,vals in groups.items():
        for level in vals.unique():
            sel=vals.eq(level); o=sel&observed;m=sel&missing
            # Exact group counts are released only when no small marginal or binary observed outcome is exposed.
            if not all(safe(int(k)) for k in (o.sum(),m.sum(),n.loc[o,'smoking'].sum(),o.sum()-n.loc[o,'smoking'].sum())):
                # Withhold the whole breakdown rather than permit complementary disclosure.
                rows=[r for r in rows if r['stratum']!=kind]
                if kind=='sex':
                    for lev in ('men','women'):
                        oo=vals.eq(lev)&observed; mm=vals.eq(lev)&missing
                        if min(oo.sum(),mm.sum())>=10:
                            rows.append(dict(stratum=kind,level=lev,observed_n=None,missing_n=None,observed_pct=round(100*n.loc[oo,'smoking'].mean(),1),imputed_pct=round(100*n.loc[mm,'filled'].mean(),1)))
                break
            rows.append(dict(stratum=kind,level=level,observed_n=o.sum(),missing_n=m.sum(),observed_pct=100*n.loc[o,'smoking'].mean() if o.any() else None,imputed_pct=100*n.loc[m,'filled'].mean() if m.any() else None))
    csv('CHARLS_smoking_composition',rows)
    mask=pd.read_csv(M/'TABLES/CHARLS_validation_1_r4_validation.csv').query("variable=='smoking'")
    reason=n.loc[missing,'smoking_reason'].fillna('unavailable').astype(str);rn=reason.value_counts()
    reasons={k:int(v) for k,v in rn.items() if v>=10};remaining=int(rn.sum()-sum(reasons.values()))
    if 0<remaining<10:
        k=min(reasons,key=reasons.get);remaining+=reasons.pop(k)
    if remaining:reasons['other_combined']=remaining
    dist=pd.read_csv(A/'MI_DISTRIBUTION_SUMMARY.csv').iloc[0]
    np.testing.assert_allclose([n.smoking.mean(),mats[missing].mean(),mats[missing].mean(axis=0).min(),mats[missing].mean(axis=0).max()],
        [dist.observed_mean,dist.mean_of_imputed_means,dist.minimum_imputed_mean,dist.maximum_imputed_mean])
    full={}
    for s in C:
        fs=[loadj(M/f'QA/{s}_full_{i}_r4_status.json') for i in range(1,11)]
        assert all(v['observed_values_unchanged'] and v['complete'] and v['legal_values'] for v in fs)
        full[s]=dict(chains=sum(x['m'] for x in fs),iterations=sorted(set(x['iterations'] for x in fs)))
    dump(AR/'smoking_and_MI_check.json',dict(nodes=len(n),observed=int(observed.sum()),missing=int(missing.sum()),observed_pct=100*n.smoking.mean(),imputed_pct=100*mats[missing].mean(),
        full_node_pct=100*mats.mean(),masked_n=int(mask.n.iloc[0]),masked_observed_pct=100*mask.observed_mean.mean(),masked_imputed_pct=100*mask.imputed_mean.mean(),
        reasons=reasons,raw_route_comparison=native,locked_harmonized_direct_nodes_match=True,run_records=full,
        missing_ever_yes_pct=round(100*n.loc[missing,'ever'].eq(1).mean(),1),missing_proxy_pct=round(100*n.loc[missing,'proxy'].ne(0).mean(),1),
        male_observed_pct=round(100*pd.to_numeric(n.loc[observed,'sex']).eq(1).mean(),0),male_missing_pct=round(100*pd.to_numeric(n.loc[missing,'sex']).eq(1).mean(),0)))

def source_qa():
    import limited_checks as chk
    chk.Q=AR;chk.run()
    archived=loadj(V2/'QA/V2_limited_checks.json');actual=loadj(AR/'V2_limited_checks.json');assert actual==archived
    anchors=pd.read_csv(A/'Table2_full_precision.csv');assert len(anchors)>0
if __name__=='__main__':
    for f in [continuous,smoking,baseline_data,source_qa]:
        f(); print(f.__name__,'complete',flush=True)

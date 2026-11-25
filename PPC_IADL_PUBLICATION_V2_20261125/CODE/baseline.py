"""First observed eligible index only. Capture original seven disease items; never release unit records."""
from inputs import *
import numpy as np,pandas as pd,pyarrow.parquet as pq,re,runpy
sys.path.insert(0,str(M/'CODE')); import common as cm
dx=['Hypertension','Diabetes','Cancer','Chronic lung disease','Heart disease','Stroke','Arthritis']
hvar=['hibpe','diabe','cancre','lunge','hearte','stroke','arthre']
def diseases(s,waves):
    old=W/'PPC_CuCoM_IJNS_Upgrade_20261115'
    if s=='CHARLS':
        vv=[f'disease_{i}' for i in (1,3,4,5,7,8,13)]
        a=pd.read_parquet(old/'CHARLS/PRIVATE/charls_person_panel.parquet',columns=['ID','wave',*vv]);pid=a.pop('ID').astype(str).str.strip()
        a['person']=pid.mask(pid.str.len().eq(11),pid.str[:-2]+'0'+pid.str[-2:]);a=a.rename(columns=dict(zip(vv,dx)))
    elif s in ('HRS','ELSA'):
        vv=(['d_'+x for x in ['hibp','diab','cancr','lung','heart','strok','arthr']] if s=='HRS' else ['disease_'+x for x in hvar])
        a=pd.read_parquet(M/f'PRIVATE/{s}_canonical_panel.parquet',columns=['person','wave',*vv]).rename(columns=dict(zip(vv,dx)))
    else:
        native=runpy.run_path(str(W/'PPC_IADL_SHARE_V5_20261116/CODE/share_native_build.py'));pieces=[]
        for w in waves:
            path=native['source'](w,'ph');cols=[k for k in pq.read_schema(path).names if re.fullmatch(r'ph006d(?:[0-9]+|no|ot)',k)]
            p=native['native'](w,'ph',cols);good=native['choices'](p,'ph006')[1]
            b=p[[f'ph006d{i}' for i in (2,5,10,6,1,4)]].copy();b.columns=dx[:-1]
            b['Arthritis']=p.ph006d8 if w==4 else np.select([p.ph006d19.eq(1)|p.ph006d20.eq(1),p.ph006d19.eq(0)&p.ph006d20.eq(0)],[1,0],np.nan)
            pieces.append(b.where(good,axis=0).rename_axis('person').reset_index().assign(wave=w))
        a=pd.concat(pieces)
    a=a[a.wave.isin(waves)].copy();a['person']=s+'|'+a.person.astype(str); a[dx]=a[dx].where(a[dx].isin([0,1]))
    assert not a.duplicated(['person','wave']).any()
    return a[['person','wave',*dx]]
def grp(s):return {'kind':'group','v':[s]}
def row(*v):return {'kind':'data','v':list(map(str,v))}
def get_first():
    b=pd.concat([cm.old.load(k,'M_severity')[0] for k in cc]).sort_values(['person','month1','window']).drop_duplicates('person').copy()
    ns=pd.concat([pd.read_parquet(M/f'PRIVATE/{s}_node_binding.parquet',columns=['key',*[v+y for v in ('smoking','alcohol','bmi') for y in ('','_reason')]]) for s in cohorts]).set_index('key',verify_integrity=True)
    for c in ns:b[c]=ns.loc[cm.keys(b),c].to_numpy()
    d=pd.concat([diseases(s,sorted(b.loc[b.source.eq(s),'wave'].unique())) for s in cohorts])
    b=b.merge(d,on=['person','wave'],how='left',validate='one_to_one');assert b[dx].notna().all().all() and np.array_equal(b[dx].sum(axis=1),b.chronic_count1)
    b['sym']=100*b.cesd1/b.source.map(dict(CHARLS=30,HRS=8,ELSA=8,SHARE=12))
    b['agecat']=pd.cut(b.age1,[45,55,65,75,85,np.inf],right=False,labels=False);b['bmicat']=pd.cut(b.bmi,[-np.inf,18.5,25,30,np.inf],right=False,labels=False)
    b['kcat']=b.K1.clip(upper=3);b['dxcat']=b.chronic_count1.clip(upper=3)
    assert len(b)==35494 and b.history.value_counts().reindex(hs).tolist()==[32986,880,967,661]
    return b
def build():
    b=get_first(); availability=[]; proof=[]; coverage=[];safe='Not shown'
    def splits(t):return [t]+[t[t.history.eq(h)] for h in hs]
    def fpart(k,n):return '—' if n==0 else f'{k:,} ({100*k/n:.1f})'
    pooled,src=[],{s:[] for s in cohorts}
    def append(v,lab,mode,levels=None,scope=None):
        # Protect the two-dimensional margins as well as the cells. An affected set of source x history columns
        # is withheld for the entire variable; both any revealing percentages and complementary categories go with it.
        frames=[('All',b)]+[(s,b[b.source.eq(s)]) for s in cohorts] if scope is None else [(scope,b[b.source.eq(scope)])]
        matrices={s:np.array([[int(g[v].eq(k).sum()) if k is not None else int(g[v].isna().sum()) for g in splits(fr)] for k,_ in (levels or [])+[(None,'')]]) for s,fr in frames} if levels else {}
        masks={s:np.zeros((len(levels or [])+1,5),bool) for s,fr in frames}
        if levels:
            glob=matrices.get('All')
            if glob is not None:
                for i,j in np.argwhere((glob>0)&(glob<10)):
                    if j==0:
                        for mask in masks.values():mask[:,:]=True
                        continue
                    opts=[]
                    for ii in range(len(glob)):
                        if ii==i or (v=='code_fixed1' and {i,ii}!={2,3}):continue
                        for jj in range(1,5):
                            if jj==j:continue
                            zz=glob[np.ix_([i,ii],[j,jj])];room=min(zz[0,0],zz[1,1])+min(zz[0,1],zz[1,0])
                            if room:opts.append((min(20,room),ii,jj))
                    if opts:
                        _,ii,jj=max(opts)
                        for mask in masks.values():mask[np.ix_([i,ii],[j,jj])]=True
                    else:
                        for mask in masks.values():mask[:,1:]=True
            sc=[s for s in matrices if s!='All']
            if len(sc)>1:
                cubes=np.stack([matrices[s][:,1:] for s in sc],axis=1)
                old=np.zeros_like(cubes,dtype=bool)
                for i,j,k in np.argwhere((cubes>0)&(cubes<10)):
                    options=[]
                    for i2 in range(len(cubes)):
                        if i2==i:continue
                        for j2 in range(4):
                            if j2==j:continue
                            for k2 in range(4):
                                if k2==k:continue
                                idx=np.ix_([i,i2],[j,j2],[k,k2]);cube=cubes[idx];ev=np.indices((2,2,2)).sum(axis=0)%2==0
                                room=int(cube[ev].min()+cube[~ev].min())
                                if room>0:options.append((min(10,room),-int((~old[idx]).sum()),i2,j2,k2))
                    if options:
                        *_,i2,j2,k2=max(options);old[np.ix_([i,i2],[j,j2],[k,k2])]=True
                    else:
                        # Structural zero patterns may rule out a circulation preserving all margins.
                        old[:,:,:]=True
                        for s in sc:masks[s][:,0]=True
                for j,s in enumerate(sc):masks[s][:,1:]|=old[:,j,:]
                if any(np.any((matrices[s][:,0]>0)&(matrices[s][:,0]<10)) for s in sc):
                    for s in sc:masks[s][:,:]=True
            elif sc and np.any((matrices[sc[0]]>0)&(matrices[sc[0]]<10)):masks[sc[0]][:,:]=True
        for s,t in frames:
            result=pooled if s=='All' else src[s];sub=splits(t)
            if mode in ('mean','median'):
                def f(g):
                    x=g[v].dropna();p=2 if v=='cog_z1' else 1
                    if len(x)<10:return safe
                    if mode=='mean':return f'{x.mean():.{p}f} ({x.std():.{p}f})'.replace('-','−')
                    return f'{x.median():g} ({x.quantile(.25):g}–{x.quantile(.75):g})'
                result.append(row(lab,*map(f,sub)))
            else:
                if len(levels)>1:result.append(grp(lab))
                for i,(k,l) in enumerate(levels):
                    vals=[safe if masks[s][i,j] else fpart(matrices[s][i,j],len(g[v].dropna())) for j,g in enumerate(sub)]
                    result.append(row(l if len(levels)>1 else lab,*vals))
                    for j,g in enumerate(sub):
                        raw=matrices[s][i,j];assert raw<=g[v].notna().sum()
                        if not masks[s][i,j]:assert raw==0 or raw>=10
                if levels and len(levels)>1:assert np.array_equal(matrices[s][:-1,:].sum(0),[g[v].notna().sum() for g in sub])
    for name,t in [('All',b),*[(s,b[b.source.eq(s)]) for s in cohorts]]:
        target=pooled if name=='All' else src[name];target.append(row('Participants, n',*[f'{len(g):,}' for g in splits(t)]))
    append('age1','Age, years, mean (SD)','mean');append('agecat','Age group, years, n (%)','cat',list(enumerate(['45–54','55–64','65–74','75–84','≥85'])))
    append('sex1','Sex, n (%)','cat',[(1,'Men'),(2,'Women')])
    for s,lev in [('CHARLS',['Less than primary','Primary','Junior high or above']),('ELSA',['No qualification','Secondary qualification','Higher education','Foreign or other qualification'])]:append('education1','Education, n (%)','cat',list(enumerate(lev)),s)
    append('education1','Education, years, mean (SD)','mean',scope='HRS');append('education1','Education, ISCED level, median (IQR)','median',scope='SHARE')
    for v,l in [('smoking','Current smoking, n (%)'),('alcohol','Weekly or comparable recent alcohol use, n (%)')]:append(v,l,'cat',[(0,'No'),(1,'Yes')])
    append('bmi','Body mass index, kg/m², mean (SD)','mean');append('bmicat','Body mass index, kg/m², n (%)','cat',list(enumerate(['<18.5','18.5–<25.0','25.0–<30.0','≥30.0'])))
    append('K0','IADL difficulty count at preceding visit, median (IQR)','median');append('K1','IADL difficulty count at index, median (IQR)','median')
    append('kcat','IADL difficulty count at index, n (%)','cat',list(enumerate(['0 difficulties','1 difficulty','2 difficulties','≥3 difficulties'])))
    append('code_fixed1','Health domains affected at index, n (%)','cat',[(0,'None'),(1,'Physical only'),(2,'Psychological only'),(4,'Cognitive only')])
    append('chronic_count1','Physical condition count, median (IQR)','median');append('dxcat','Physical condition count, n (%)','cat',list(enumerate(['0 conditions','1 condition','2 conditions','≥3 conditions'])))
    for table in [pooled,*src.values()]:table.append(grp('Individual physical conditions, n (%)'))
    for v in dx:
        # Binary complement also participates in disclosure checks. Retain only the positive row.
        append(v,v,'cat',[(1,v),(0,'Negative')])
        for table in [pooled,*src.values()]:del table[-1];del table[-2]
    append('sym','Depressive symptoms, % of scale maximum, mean (SD)','mean');append('cog_z1','Cognitive performance, mean subtest Z, mean (SD)','mean')
    core=['age1','sex1','K0','K1','code_fixed1','chronic_count1',*dx,'sym','cog_z1','education1'];assert b[core].notna().all().all()
    scopes=[('Four cohorts',b)]+[(s,b[b.source.eq(s)]) for s in cohorts]
    for name,t in scopes:
        layers=[('All patterns',t)]+[(h,t[t.history.eq(k)]) for h,k in zip(hist,hs)]
        if name!='Four cohorts':layers += [(str(int(w)),g) for w,g in t.groupby('wave')]
        for n,g in layers:
            for v in ['Core measurements','Current smoking','Alcohol use','Body mass index']:
                c={'Current smoking':'smoking','Alcohol use':'alcohol','Body mass index':'bmi'}.get(v)
                m=g[c].isna() if c else pd.Series(False,index=g.index);st=m&g[c+'_reason'].eq('structurally_unavailable') if c else m
                availability.append(dict(source=name,stratum=n,variable=v,total=len(g),observed=int((~m).sum()),ordinary_missing=int((m&~st).sum()),structural=int(st.sum())))
    # Remove linked marginal partitions as a set if they would reveal a sparse missingness reason.
    # Keep observed denominators separately whenever its total complement is itself disclosable.
    d=pd.DataFrame(availability)
    for v,g in d.groupby('variable'):
        nonglobal=g.source.ne('Four cohorts');wave=g.stratum.str.fullmatch(r'\d+');histrows=g.stratum.ne('All patterns')&~wave
        for part in ['observed','ordinary_missing','structural']:
            bad=(g[part].between(1,9))|((g.total-g[part]).between(1,9))
            suppress=pd.Series(False,index=g.index)
            if (bad&~nonglobal).any():suppress|=True
            else:
                if (bad&nonglobal&~histrows&~wave).any():suppress|=nonglobal
                if (bad&nonglobal&histrows).any():suppress|=nonglobal&histrows
                if (bad&wave).any():suppress|=wave&g.source.isin(g.loc[bad&wave,'source'])
            d.loc[g.index,'mask_'+part]=suppress
    # When one part in a three-part partition is masked, protect a complement too.
    for i,x in d.iterrows():
        if x.mask_ordinary_missing or x.mask_structural:d.loc[i,['mask_ordinary_missing','mask_structural']]=True
        if x.mask_observed:d.loc[i,['mask_ordinary_missing','mask_structural']]=True
    for rr in d.itertuples():
        v=[rr.variable,rr.source,rr.stratum,f'{rr.total:,}']+[safe if getattr(rr,'mask_'+k) else fpart(getattr(rr,k),rr.total) for k in ['observed','ordinary_missing','structural']]
        coverage.append(row(*v))
    pd.DataFrame([r['v'] for r in coverage],columns=['Measure','Cohort','Pattern or index wave','Denominator','Observed, n (%)','Other missing, n (%)','Whole-wave nonmeasurement, n (%)']).to_csv(A/'Table1_denominators.csv',index=False)
    put(A/'baseline_rows.json',{'T1':pooled,'S3':sum(([grp(s)]+src[s] for s in cohorts),[]),'S4':coverage})
    put(Q/'baseline.json',{'people':len(b),'groups':b.history.value_counts().reindex(hs).tolist(),'availability':b[['smoking','alcohol','bmi']].count().to_dict(),'individual_seven_sum_verified':len(b),'all_other_core_measured':True,'suppression':'Small positive cells and complementary sources or margins protected. Missingness partitions never converted to zero.'})
    print('First-index baseline and seven diagnosis items verified:',len(b),'people; availability rows',len(coverage))
if __name__=='__main__':build()

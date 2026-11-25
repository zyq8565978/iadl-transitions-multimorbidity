"""Outcome-blind source rules; descriptive support only. No new effect fitting."""
import os
for v in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'): os.environ[v]='1'
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
R=Path(__file__).resolve().parents[1]; W=R.parent
sys.path.insert(0,str(W/'PPC_IADL_V6_Repair_Synthesis_20261116/CODE'))
from association_v6 import original,SOURCES,COMPS
P=R/'PRIVATE'; P.mkdir(exist_ok=True)
Q=R/'QA'; Q.mkdir(exist_ok=True)
T=R/'TABLES'; T.mkdir(exist_ok=True)
RAW=W/'原始数据'
files={
'CHARLS':(RAW/'01_CHARLS_中国/02_协调与简化数据/harmonized charls/H_CHARLS_D_Data.dta','ID',{2013:2,2015:3,2018:4}),
'HRS':(RAW/'03_HRS_美国/02_协调与简化数据/RAND HRS Longitudinal File 2020/randhrs1992_2020v2.dta','hhidpn',{2012:11,2014:12,2016:13,2018:14}),
'ELSA':(RAW/'02_ELSA_英国/02_协调与简化数据/Harmonized ELSA/h_elsa_g3.dta','idauniq',{i:i for i in range(1,9)})}
hashes={}; cross=[]; checks=[]

def sha(p):
    if str(p) not in hashes:
        with p.open('rb') as f: hashes[str(p)]=hashlib.file_digest(f,'sha256').hexdigest()
    return hashes[str(p)]

def ids(x,s):
    if s=='CHARLS':
        a=x.astype(str).str.strip(); return a.where(a.str.len().ne(11),a.str[:-2]+'0'+a.str[-2:])
    return x.astype('Int64').astype(str)

def read(p,cols):
    sha(p)
    return pd.read_stata(p,columns=cols,convert_categoricals=False,convert_missing=True)

def num(x): return pd.to_numeric(x,errors='coerce').astype(float)

def reasons(x):
    return x.map(lambda v: getattr(v,'string','ordinary_missing') if not isinstance(v,(int,float,np.number)) else ('ordinary_missing' if pd.isna(v) else 'recorded'))

def crossrow(s,w,name,p,fields,rule,question,recall,bmimode='',note=''):
    cross.append(dict(source=s,wave=w,variable=name,source_file=str(p),source_sha256=sha(p),fields=fields,
       coding=rule,question_or_definition=question,recall=recall,time_binding='same native assessment wave; index and prior linked separately',
       missing='DK/refusal/proxy/not administered/invalid not recoded to zero',bmi_mode=bmimode,limitation=note))

def elsa_file(w,kind):
    folder=RAW/f'02_ELSA_英国/01_官方原始数据/wave{w}'
    pp=[p for p in folder.glob('*.dta') if '_CN' not in p.name and
        (('core_data' in p.name or 'elsa_data' in p.name) if kind=='core' else 'nurse_data' in p.name)]
    assert len(pp)==1,(w,kind,pp)
    return pp[0]

def harmonized(s):
    p,key,waves=files[s]
    with pd.read_stata(p,iterator=True) as f: names=f.variable_labels()
    needed=[key]+[f'r{w}{tail}' for w in waves.values() for tail in ('smoken','smokev','drinkn_c' if s=='CHARLS' else 'drinkd','bmi' if s=='HRS' else 'mbmi','height' if s=='HRS' else 'mheight','weight' if s=='HRS' else 'mweight') if f'r{w}{tail}' in names]
    a=read(p,needed); who=ids(a[key],s); assert who.is_unique
    out=[]
    for wave,w in waves.items():
        pre=f'r{w}'
        z=pd.DataFrame({'person':who,'wave':wave})
        z['smoking']=num(a[pre+'smoken']).where(num(a[pre+'smoken']).isin([0,1]))
        z['smoking_reason']=reasons(a[pre+'smoken'])
        if s=='ELSA':
            f=elsa_file(w,'core')
            with pd.read_stata(f,iterator=True) as rr: lbl=rr.variable_labels()
            if 'scako' in lbl:
                raw=read(f,['idauniq','scako']); raw['person']=ids(raw.idauniq,s); raw['alcohol_raw']=num(raw.scako)
                raw['alcohol_reason']=reasons(raw.scako)
                raw['alcohol']=raw.alcohol_raw.le(4).astype(float).where(raw.alcohol_raw.isin(range(1,9)))
                z=z.merge(raw[['person','alcohol','alcohol_raw','alcohol_reason']],on='person',how='left',validate='one_to_one')
                crossrow(s,wave,'alcohol',f,'scako','1–4=weekly; 5–8=less than weekly','Frequency of alcoholic drinks during last 12 months','12 months')
            else:
                z['alcohol']=np.nan; z['alcohol_raw']=np.nan; z['alcohol_reason']='structurally_unavailable'
                crossrow(s,wave,'alcohol',f,'scako absent','No surrogate substituted','Usual weekly alcohol frequency unavailable','not available')
        else:
            field=pre+('drinkn_c' if s=='CHARLS' else 'drinkd')
            v=num(a[field]); z['alcohol_raw']=v
            z['alcohol']=v.ge(4 if s=='CHARLS' else 1).astype(float).where(v.between(0,9 if s=='CHARLS' else 7))
            z['alcohol_reason']=reasons(a[field])
            crossrow(s,wave,'alcohol',p,field,'>=4 of 0–9' if s=='CHARLS' else '>=1 day of 0–7','Usual weekly drinking frequency','12 months; maximum frequency across beverage types' if s=='CHARLS' else '3 months')
        crossrow(s,wave,'smoking',p,pre+'smoken','1=current, 0=not current','Smoking now; native never-smoker routing retained','current')
        field=pre+('bmi' if s=='HRS' else 'mbmi')
        z['bmi_raw']=num(a[field]) if field in a else np.nan
        z['bmi_reason']=reasons(a[field]) if field in a else 'structurally_unavailable'
        z['height']=num(a[pre+('height' if s=='HRS' else 'mheight')]) if pre+('height' if s=='HRS' else 'mheight') in a else np.nan
        z['weight']=num(a[pre+('weight' if s=='HRS' else 'mweight')]) if pre+('weight' if s=='HRS' else 'mweight') in a else np.nan
        z['bmi_month']=np.nan
        z['bmi_mode']=('self_report_current_weight_prior_height' if s=='HRS' else 'same_wave_physical_measurement' if s=='CHARLS' else 'same_wave_nurse_measurement' if w in (2,4,6) else 'core_weight_prior_measured_height' if w==8 else 'structurally_unavailable')
        if s=='ELSA' and w in (2,4,6):
            f=elsa_file(w,'nurse')
            with pd.read_stata(f,iterator=True) as rr: ln=rr.variable_labels(); lookup={k.lower():k for k in ln}
            cols=[lookup[k] for k in ('idauniq','vismon','visyear')]
            dn=read(f,cols); dn.columns=['idauniq','m','y']; dn['person']=ids(dn.idauniq,s)
            dn['bmi_month']=12*num(dn.y)+num(dn.m)
            dn.loc[~num(dn.m).between(1,12)|~num(dn.y).between(2000,2025),'bmi_month']=np.nan
            z=z.drop(columns='bmi_month').merge(dn[['person','bmi_month']],on='person',how='left',validate='one_to_one')
        crossrow(s,wave,'bmi',p,field if field in a else 'not collected','kg/m2; [10,80], height [0.9,2.3] m, weight [20,300] kg; categories <18.5/<25/<30/>=30',
                 'Current assessment weight divided by measured/reported height squared','assessment wave',z.bmi_mode.iloc[0],
                 'ELSA nurse month must be within 6 months of core and before next assessment; W8 current core weight uses prior measured height' if s=='ELSA' else
                 'CHARLS 2018 has no physical-examination BMI release; separate physical-exam date unavailable in 2013/2015 native files, same fieldwork wave only' if s=='CHARLS' else
                 'RAND uses earlier height for continuing respondents; current weight required, no past BMI carried forward')
        out.append(z)
    return pd.concat(out,ignore_index=True)

def share():
    store=Path.home()/'.codex/skills/seven-aging-pig-skill/data-store'
    cat=pd.read_csv(store/'catalog/available_tables.csv',encoding='utf-8-sig').fillna('')
    def native(w,module,cols):
        f=cat.loc[cat.database_name.eq('share')&cat.parquet.str.endswith(f'__sharew{w}_rel9-0-0_{module}.parquet')]
        assert len(f)==1
        p=store/f.iloc[0].parquet; sha(p)
        return pd.read_parquet(p,columns=['mergeid',*cols]).rename(columns={'mergeid':'person'}),p
    panel=pd.read_parquet(W/'PPC_IADL_SHARE_V5_20261116/PRIVATE/SHARE_panel.parquet',columns=['person','wave','month','proxy'])
    sp=RAW/'07_SHARE_欧洲/02_协调与简化数据/Harmonized SHARE/H_SHARE_f2.dta'
    sm=read(sp,['mergeid']+[f'r{w}smoken' for w in (4,5,6,7,8)])
    sm['person']=sm.mergeid.astype(str)
    out=[]
    for w in (4,5,6,7,8):
        z=panel.loc[panel.wave.eq(w)].copy(); z['person']=z.person.astype(str)
        ss=sm[['person',f'r{w}smoken']].copy()
        ss['smoking']=num(ss[f'r{w}smoken']).where(num(ss[f'r{w}smoken']).isin([0,1]))
        ss['smoking_reason']=reasons(ss[f'r{w}smoken'])
        z=z.merge(ss[['person','smoking','smoking_reason']],on='person',how='left',validate='one_to_one')
        v='br010_' if w<6 else 'br039_'
        a,bp=native(w,'br',[v,'br017_']); a['person']=a.person.astype(str)
        legal=a.br017_.isin([1,2]); vv=a[v]
        a['alcohol']=vv.le(4).astype(float).where(legal&vv.between(1,7)) if w<6 else vv.eq(1).astype(float).where(legal&vv.isin([1,5]))
        a['alcohol_raw']=vv; a['alcohol_reason']=np.where(a.alcohol.notna(),'recorded',np.where(~legal,'no_direct_module','native_negative_missing'))
        z=z.merge(a[['person','alcohol','alcohol_raw','alcohol_reason']],on='person',how='left',validate='one_to_one')
        invalid_mode=~z.person.isin(a.loc[legal,'person'])
        z.loc[invalid_mode,'smoking']=np.nan
        z.loc[invalid_mode,'smoking_reason']='no_direct_module'
        a,hp=native(w,'gv_health',['bmi']); a['person']=a.person.astype(str); a=a.rename(columns={'bmi':'bmi_raw'})
        z=z.merge(a,on='person',how='left',validate='one_to_one')
        a,pp=native(w,'ph',['ph012_','ph013_']); a['person']=a.person.astype(str)
        a['weight']=a.ph012_.where(a.ph012_.gt(0)); a['height_reported']=a.ph013_.where(a.ph013_.gt(0))/100
        z=z.merge(a[['person','weight','height_reported']],on='person',how='left',validate='one_to_one')
        z['height']=np.sqrt(z.weight/z.bmi_raw.where(z.bmi_raw.gt(0)))
        both=z.height_reported.notna()&z.bmi_raw.gt(0)&z.weight.notna()
        error=(z.loc[both,'weight']/z.loc[both,'height_reported']**2-z.loc[both,'bmi_raw']).abs()
        checks.append(dict(source='SHARE',wave=w,check='official_bmi_against_current_raw_height_weight',n=int(both.sum()),max_absolute_error=float(error.max()) if len(error) else None))
        z['bmi_reason']=np.where(z.bmi_raw.gt(0),'recorded','native_missing_or_unasked')
        z['bmi_month']=z.month; z['bmi_mode']='official_gv_health_current_self_report'
        crossrow('SHARE',w,'smoking',sp,f'r{w}smoken; BR001/BR002/BR022 official generation','1=current; 0=official routed non-current; all special missing including .q remain missing','Smoke at present; official route through ever-daily smoking and cessation','current',note='Native-defined source-specific covariate, not universally ascertained any-current tobacco. Negative daily-history screen is NOT described as never smoking. W6 continuing respondents and W7 unasked routes remain .q; no manual carry-forward. Official harmonization supersedes older three-category cache only for this new QC overlay.')
        crossrow('SHARE',w,'alcohol',bp,v,'1–4=weekly, 5–7=less than weekly' if w<6 else '1=yes recent drinking, 5=no recent drinking','Usual frequency' if w<6 else 'At least one alcoholic drink in last 7 days','3 months' if w<6 else '7 days',note='Source-specific approximate covariate in W6–8; not weekly habitual drinking')
        crossrow('SHARE',w,'bmi',hp,'bmi; PH012/PH013 verified','same fixed BMI thresholds; negative native codes missing','Official BMI from weight kg and height cm','same assessment',z.bmi_mode.iloc[0],note='Official generated BMI; not manually imputed. Check actual raw height availability.')
        out.append(z.drop(columns=['proxy','month']))
    return pd.concat(out,ignore_index=True)

def bind(s,d,o):
    d=d.copy(); d['row_order']=np.arange(len(d)); d['prior_wave_qc']=d.window.str.split('->').str[0].astype(int)
    for suf,wcol,mc,nextmc in [('t','wave','month1','month2'),('pre','prior_wave_qc','month0','month1')]:
        cols=['person','wave','smoking','alcohol','bmi_raw','height','weight','bmi_month','bmi_mode','smoking_reason','alcohol_reason','bmi_reason']
        t=o[cols].rename(columns={'wave':wcol,**{c:c+'_'+suf for c in cols if c not in ('person','wave')}})
        d=d.merge(t,on=['person',wcol],how='left',validate='many_to_one',sort=False)
        b='bmi_raw_'+suf; h='height_'+suf; wt='weight_'+suf
        valid=d[b].between(10,80)&d[h].between(.9,2.3)&d[wt].between(20,300)
        nurse=d['bmi_mode_'+suf].eq('same_wave_nurse_measurement')
        gap=d['bmi_month_'+suf]-d[mc]
        timeok=~nurse | (gap.abs().le(6)&d['bmi_month_'+suf].lt(d[nextmc]))
        d['bmi_time_gap_'+suf]=gap.where(nurse)
        d['bmi_'+suf]=d[b].where(valid&timeok)
        d.loc[nurse&~timeok,'bmi_reason_'+suf]='nurse_date_missing_outside_6months_or_not_before_next_assessment'
        d.loc[d[b].notna()&~valid,'bmi_reason_'+suf]='invalid_or_out_of_prespecified_physical_range'
        d['bmi_cat_'+suf]=pd.cut(d['bmi_'+suf],[-np.inf,18.5,25,30,np.inf],right=False,labels=[0,1,2,3]).astype(float)
        # No proxy answers promoted to a measured behavior.
        prox=d['proxy1' if suf=='t' else 'proxy0'].ne(0)
        d.loc[prox,['smoking_'+suf,'alcohol_'+suf]]=np.nan
    assert np.array_equal(d.row_order,np.arange(len(d)))
    return d

def publish_count(n): return '<10' if 0<int(n)<10 else str(int(n))

def main():
    support=[]; missing=[]; modes=[]; audits=[]
    for s in SOURCES:
        o=share() if s=='SHARE' else harmonized(s)
        assert not o.duplicated(['person','wave']).any()
        o.to_parquet(P/f'{s}_covariates.parquet',index=False)
        for comp in COMPS:
            a,d,stem=original(s,comp,'M_severity'); sha(stem.with_suffix('.npz')); sha(stem.with_name(stem.name+'_rows.parquet'))
            z=bind(s,d,o); z.to_parquet(P/f'{s}_{comp}_bound.parquet',index=False)
            for timing in ('t','pre'):
                steps=[('original',np.ones(len(z),bool)),('smoking',z[f'smoking_{timing}'].notna()),
                       ('smoking_alcohol',z[[f'smoking_{timing}',f'alcohol_{timing}']].notna().all(axis=1)),
                       ('three_covariates',z[[f'smoking_{timing}',f'alcohol_{timing}',f'bmi_{timing}']].notna().all(axis=1))]
                for step,mask in steps:
                    for (wave,arm),g in z.loc[mask].groupby(['wave','arm']):
                        support.append(dict(source=s,comparison=comp,timing=timing,stage=step,wave=wave,arm=arm,windows=len(g),people=g.person.nunique(),families=g.cluster.nunique(),events=int(g.y.sum()),nonevents=int((1-g.y).sum()),mean_age=g.age1.mean(),mean_chronic=g.chronic_count1.mean(),mean_ppc_domains=g.d_fixed1.mean()))
                for variable in ('smoking','alcohol','bmi'):
                    for (wave,arm,reason),g in z.groupby(['wave','arm',variable+'_reason_'+timing],dropna=False):
                        missing.append(dict(source=s,comparison=comp,timing=timing,variable=variable,wave=wave,arm=arm,reason=reason,windows=len(g)))
                for (wave,mode),g in z.groupby(['wave','bmi_mode_'+timing]):
                    v=g['bmi_time_gap_'+timing].dropna()
                    modes.append(dict(source=s,comparison=comp,timing=timing,wave=wave,mode=mode,windows=len(g),bmi_valid=int(g['bmi_'+timing].notna().sum()),nurse_date_n=len(v),nurse_gap_median=v.median(),nurse_gap_min=v.min(),nurse_gap_max=v.max()))
            print(s,comp,len(z),'index CC',z[['smoking_t','alcohol_t','bmi_t']].notna().all(axis=1).sum(),'prior CC',z[['smoking_pre','alcohol_pre','bmi_pre']].notna().all(axis=1).sum(),flush=True)
    for name,rows in [('MISSINGNESS_AND_SELECTION',support),('MISSINGNESS_REASONS',missing),('BMI_TIME_AND_MODE',modes),('BMI_FORMULA_CHECKS',checks)]:
        df=pd.DataFrame(rows); df.to_csv(P/f'{name}_exact.csv',index=False,encoding='utf-8-sig')
        for col in set(df)&{'windows','people','families','events','nonevents','n','bmi_valid','nurse_date_n'}:
            df[col]=df[col].map(publish_count)
        # Values that would describe fewer than ten records are not released.
        if 'windows' in df:
            small=df.windows.eq('<10')
            for col in ('mean_age','mean_chronic','mean_ppc_domains','nurse_gap_median','nurse_gap_min','nurse_gap_max'):
                if col in df: df.loc[small,col]=np.nan
        df.to_csv(T/f'{name}.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(cross).to_csv(R/'LIFESTYLE_BMI_SOURCE_CROSSWALK.csv',index=False,encoding='utf-8-sig')
    (Q/'SOURCE_INPUT_HASHES.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DESCRIPTIVE PREPARATION COMPLETE; NO NEW EFFECT FITS')

if __name__=='__main__': main()

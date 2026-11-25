"""Create the new local HRS input without editing the frozen two-visit pilot."""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1]
WORK = OUT.parents[1]
PRIVATE = OUT/'PRIVATE'
OLD = WORK/'PPC_IADL5_Exposure_Revision/HRS/PRIVATE/panel.parquet'
PPC = WORK/'PPC_CuCoM_IJNS_Upgrade_20261115/HRS/PRIVATE/HRS_person_wave_domain_source.parquet'
RAW = WORK/'原始数据/03_HRS_美国'
CORE = RAW/'01_官方原始数据'
WAVES = dict(zip([2012,2014,2016,2018,2020], 'NOPQR'))
DOMAIN = ['physical_any','psychological','cesd8_complete','cognitive_strict']
COG = ['imrc','dlrc','mo','dy','yr','dw','ser7']
COGVAL = {'c1':'imrc','c2':'dlrc','c3':'orientation','c4':'ser7'}
TASKS = list('MFPSC')


def locate(folder,name):
    found = [p for p in folder.rglob(name) if p.is_file()]
    assert len(found)==1, (name,len(found))
    return found[0]


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def field_table(path,fields):
    labels = pd.io.stata.StataReader(path).variable_labels()
    native = {c.upper():c for c in labels}
    t = pd.read_stata(path, columns=[native[c] for c in ['HHID','PN']+fields],
                     convert_categoricals=False).rename(columns=str.upper)
    t['person'] = (t.HHID.astype('string').str.zfill(6)+t.PN.astype('string').str.zfill(3)).str.lstrip('0')
    return t.drop(columns=['HHID','PN']).set_index('person', verify_integrity=True)


def eqna(a,b):
    return np.array_equal(np.asarray(a,dtype=float),np.asarray(b,dtype=float),equal_nan=True)


def main():
    PRIVATE.mkdir(parents=True,exist_ok=True)
    oldhash, ppchash = sha(OLD),sha(PPC)
    prior = pd.read_parquet(OLD)
    pp = pd.read_parquet(PPC,columns=['person_id','year_nominal','current_age45plus','proxy']+DOMAIN+COG+['f'+v for v in COG])
    pp = pp.loc[pp.current_age45plus].rename(columns={'person_id':'person','year_nominal':'wave','proxy':'pp_proxy'})
    panel = prior.merge(pp.drop(columns='current_age45plus'),on=['person','wave'],validate='one_to_one')
    assert len(panel)==len(prior)==len(pp)==92099
    assert eqna(panel.proxy,panel.pp_proxy)
    assert eqna(panel.code4,panel.physical_any+2*panel.psychological+4*panel.cognitive_strict)
    panel['code4_old']=panel.code4
    panel['xM_old']=panel.xM
    panel['routeM_old']=panel.routeM
    panel['physical']=panel.physical_any
    panel['cesd']=panel.cesd8_complete
    assert panel.cesd.dropna().between(0,8).all()
    manifest=[{'kind':'old frozen item panel','source':str(OLD.relative_to(WORK)),'sha256':oldhash},
              {'kind':'inherited PPC domains and cognition flags','source':str(PPC.relative_to(WORK)),'sha256':ppchash}]
    evidence=[]; summary=[]; routes=[]; cognition=[]; waves=[]
    for y,prefix in WAVES.items():
        p=panel[panel.wave.eq(y)].copy().set_index('person',verify_integrity=True)
        g_fields=[f'{prefix}G{i:03d}' for i in [50,51,52]]
        gdta=locate(CORE,f'H{y%100}G_R.dta'); gdoc=locate(CORE,f'H{y%100}G_R.txt')
        ddta=locate(CORE,f'H{y%100}D_R.dta'); ddoc=locate(CORE,f'H{y%100}D_R.txt')
        for source in [gdta,gdoc,ddta,ddoc]:manifest.append({'kind':'HRS core release','source':str(source.relative_to(WORK)),'sha256':sha(source)})
        g_lines=gdoc.read_text(encoding='latin-1').splitlines()
        d_lines=ddoc.read_text(encoding='latin-1').splitlines()
        line=lambda lines,unn: next(i for i,l in enumerate(lines,1) if re.match(r'^'+unn+r'\s+',l,re.I))
        g_r=field_table(gdta,g_fields).reindex(p.index)
        a,h,w=(g_r[n] for n in g_fields)
        assert eqna(a,p.raw_M) and eqna(h,p.hypothetical_M) and eqna(w,p.why_M)
        # These routing guards are checked against every actual source document, not imputed from frequencies.
        h_ordinal,w_ordinal=line(g_lines,g_fields[1]),line(g_lines,g_fields[2]); txt=' '.join(g_lines[w_ordinal-13:w_ordinal])
        assert 'G050_DiffMeds = CANTDO' in txt and 'G051_DiffMeds2 <> NO' in txt and 'G051_DiffMeds2 <>' in txt
        hyp_no=a.eq(7)&h.eq(5); hyp_yes=a.eq(7)&h.eq(1)&w.eq(1)
        p.loc[hyp_no,'xM']=0.;p.loc[hyp_yes,'xM']=1.
        p.loc[hyp_no,'routeM']='hypothetical_no_difficulty';p.loc[hyp_yes,'routeM']='hypothetical_difficulty_health_confirmed'
        p.loc[a.eq(7)&h.eq(1)&w.eq(5),'routeM']='NA_not_applicable_hypothetical_no_health_cause'
        p.loc[a.eq(7)&h.eq(1)&~w.isin([1,5]),'routeM']='NA_nonresponse_hypothetical_cause_unknown'
        assert eqna(p.loc[~a.eq(7),'xM'],p.loc[~a.eq(7),'xM_old'])
        assert not p.loc[a.eq(7)&~(hyp_yes|hyp_no),'xM'].notna().any()
        assert p['xM_old'].notna().sum()+int((hyp_no|hyp_yes).sum())==p.xM.notna().sum()
        for keys,count in pd.DataFrame({'g050':a,'g051':h,'g052':w,'proxy':p.proxy}).assign(
                g050=pd.Series(np.where(a.eq(7),'nonuse','other'),index=p.index)).loc[a.eq(7)].groupby(
                ['g050','g051','g052','proxy'],dropna=False).size().items():
            routes.append(dict(wave=y,g050=keys[0],g051=keys[1],g052=keys[2],proxy=keys[3],n=count))
        d_nums=[142,143,144,145,146,150,151,152,153,154,174,177,184,187]
        dates=line(d_lines,f'{prefix}D150'); d_route=' '.join(d_lines[dates-10:dates])
        assert '>= 65' in d_route and 'REIWR' in d_route
        if y>=2018: assert 'IWERADMIN' in d_route
        cols=[f'{prefix}D{i}' for i in d_nums]+([f'{prefix}DMODE'] if y>=2018 else [])
        d=field_table(ddta,cols).reindex(p.index)
        d.columns=[s[2:] if s!='MODE' else s for s in d.columns]
        iwer=d['MODE'].isin([1,2]) if y>=2018 else pd.Series(True,index=p.index)
        p['cognition_mode'] = np.where(y<2018,'interviewer_era',d['MODE'].map({1:'phone',2:'face_to_face',3:'web'}).fillna('unknown') if y>=2018 else 'unknown')
        direct={}
        for c,count,none in [('c1','174','177'),('c2','184','187')]:
            direct[c]=d[count].where(d[count].between(0,10))
            direct[c]=direct[c].mask(direct[c].isna()&d[none].eq(1),0.)
        direct['c3']=d[[str(k) for k in range(151,155)]].apply(lambda s:s.map({1:1.,5:0.})).sum(axis=1,min_count=4)
        ss=d[[str(k) for k in range(142,147)]].apply(lambda v:v.where(v.between(0,997)))
        prev=ss.T.shift(1).T;prev['142']=100
        direct['c4']=((prev-ss).eq(7)).sum(axis=1).where(ss.notna().all(axis=1))
        refs = dict(c1=['imrc'],c2=['dlrc'],c3=['mo','dy','yr','dw'],c4=['ser7'])
        for c,raw in refs.items():
            obs=p[['f'+v for v in raw]].eq(0).all(axis=1)
            rand=p[raw].sum(axis=1,min_count=len(raw))
            plausible=obs & p.proxy.eq(0) & iwer & rand.notna()
            native_match=direct[c].eq(rand)
            p[c] = direct[c].where(plausible & native_match)
            p[c+'_route']=np.select([p[c].notna(),p.proxy.ne(0),~iwer,
                p[['f'+v for v in raw]].eq(2).any(axis=1),p[['f'+v for v in raw]].eq(1).any(axis=1),
                plausible&direct[c].isna(),plausible&~native_match],
                ['direct','proxy','noninterviewer_or_unknown_mode','structural_skip','imputed_value_excluded','native_incomplete','native_replication_discordance'],default='other_unknown')
            assert (p[c].isna()|p[c].eq(rand)).all()
            for lo,hi in [(45,50),(50,65),(65,120)]:
                age=p.age.ge(lo)&p.age.lt(hi)
                cognition.append(dict(wave=y,component=c,score=COGVAL[c],age_from=lo,age_to_exclusive=hi,n=int(age.sum()),
                    n_direct=int((age&p[c].notna()).sum()),n_flagzero_in_interviewer_self=int((age&plausible).sum()),
                    n_flagzero_native_disagreement=int((age&plausible&direct[c].notna()&~native_match).sum()),
                    n_flagzero_native_incomplete=int((age&plausible&direct[c].isna()).sum()),
                    n_proxy=int((age&p.proxy.eq(1)).sum()),n_design_skip=int((age&p[c+'_route'].eq('structural_skip')).sum())))
        both=p[['x'+j for j in TASKS]].notna().all(axis=1)
        bothold=p[['xM_old']+['x'+j for j in 'FPSC']].notna().all(axis=1)
        summary.append(dict(wave=y,respondents=len(p),iadl5_old=int(bothold.sum()),iadl5_new=int(both.sum()),
              raw_nonuse=int(a.eq(7).sum()),hypothetical_no=int(hyp_no.sum()),hypothetical_yes_health=int(hyp_yes.sum()),
              old_code4_known=int(p.code4_old.notna().sum()),full_direct_cognition=int(p[list(COGVAL)].notna().all(axis=1).sum()),
              cesd_complete=int(p.cesd.notna().sum()),mode_web=int(p.cognition_mode.eq('web').sum())))
        evidence.append(dict(wave=y,G050=g_fields[0],G051=g_fields[1],G052=g_fields[2],
           codebook_g051=h_ordinal,codebook_g052=w_ordinal,
           G051_G052_supported='G050=7 routes to G051; G051=1 permits G052; G051=5 skips probe; G052=1 health/memory confirmed',
           orientation_probe=f'{prefix}D150 through D154',orientation_codebook_line=dates,
           orientation_route='>=65 OR not reinterview'+('; interviewer administered only' if y>=2018 else ''),
           raw_cognition='D174 immediate, D184 delayed (+ explicit none recall flags), D151–154 date orientation, D142–146 serial 7s',
           imputation_rules='Use original Section D valid score only where mode-compatible RAND f flag equals 0 and native calculation agrees; exclude imputations and no-administration',
           modern_route='sample-conditional observed self-response; no equality of mode, age gate or practice effects claimed'))
        waves.append(p.reset_index())
    final=pd.concat(waves,ignore_index=True)
    assert len(final)==len(prior) and not final.duplicated(['person','wave']).any() and sha(OLD)==oldhash and sha(PPC)==ppchash
    assert eqna(final.code4,final.code4_old)
    for name in ['randhrs1992_2020v2.pdf','COGIMP9220_dd.pdf','COGIMP9220A_R.txt']:
        q=locate(RAW,name);manifest.append({'kind':'local official documentation','source':str(q.relative_to(WORK)),'sha256':sha(q)})
    for name,vals in [('HRS_SOURCE_EVIDENCE.csv',evidence),('HRS_VINTAGE_LINEAGE.csv',manifest),('HRS_WAVE_MEASUREMENT_SUMMARY.csv',summary)]:
        pd.DataFrame(vals).to_csv(OUT/name,index=False,encoding='utf-8-sig')
    pd.DataFrame(routes).to_csv(PRIVATE/'MEDS_EXACT_ROUTE_COUNTS.csv',index=False)
    pd.DataFrame(cognition).to_csv(PRIVATE/'COGNITION_MODE_VALIDATION.csv',index=False)
    final.to_parquet(PRIVATE/'panel.parquet',index=False)
    checks={'wave_source_rows':len(final),'unique_people':int(final.person.nunique()),'new_old_membership_equal':True,
       'raw_G050_G051_G052_equal_prior_linkage':True,'old_outputs_unchanged':True,'native_and_nonimputed_flags_required':True,
       'no_models':True,'new_primary_cognition_scores_only_no_new_thresholds':True}
    (OUT/'SOURCE_QA.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(cognition).groupby(['wave','component'],as_index=False)[
           ['n_direct','n_flagzero_native_disagreement','n_flagzero_native_incomplete']].sum().to_string(index=False))


if __name__=='__main__':main()

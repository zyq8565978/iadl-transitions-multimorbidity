"""SHARE R9.0.0 measurement and risk-set build; no association or effect calculation."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
STORE = Path.home()/'.codex/skills/seven-aging-pig-skill/data-store'
CAT = pd.read_csv(STORE/'catalog/available_tables.csv', encoding='utf-8-sig').fillna('')
CAT = CAT.loc[CAT.database_name.eq('share') & CAT.parquet.ne('')].copy()
CAT = CAT.set_index('parquet', drop=False)
PRIVATE = ROOT/'PRIVATE'
TABLES = ROOT/'TABLES'
QA = ROOT/'QA'
for directory in (PRIVATE,TABLES,QA):
    directory.mkdir(exist_ok=True)
WAVES = range(4,10)
TRIPLES = [(4,5,6),(5,6,7),(6,7,8),(7,8,9)]
ITEMS = {'xM':11, 'xF':13, 'xP':10, 'xS':9, 'xC':8}
ADL = [1,3,4,5,6]
DOMAINS = {i:i.bit_count() for i in range(8)}
SCORES = ['c1','c2','c3','c4']
CN = {11:'Austria',12:'Germany',13:'Sweden',14:'Netherlands',15:'Spain',16:'Italy',
      17:'France',18:'Denmark',19:'Greece',20:'Switzerland',23:'Belgium',25:'Israel',
      28:'Czech Republic',29:'Poland',30:'Ireland',31:'Luxembourg',32:'Hungary',
      33:'Portugal',34:'Slovenia',35:'Estonia',47:'Croatia',48:'Lithuania',51:'Bulgaria',
      53:'Cyprus',55:'Finland',57:'Latvia',59:'Malta',61:'Romania',63:'Slovakia'}
SOURCE_IDS = {}


def source(w, name, variant=''):
    stub = f'__sharew{w}{variant}_rel9-0-0_{name}.parquet'
    row = CAT.loc[CAT.parquet.str.endswith(stub)]
    if len(row) != 1:
        raise RuntimeError(f'Nonunique/missing native source {stub}: {len(row)}')
    r = row.iloc[0]
    SOURCE_IDS[f'{w}{variant}_{name}'] = (r.file_id,r.parquet)
    return STORE/r.parquet


def native(w, name, cols=None, variant=''):
    p = source(w,name,variant)
    names = pq.read_schema(p).names
    fields = ['mergeid']+list(cols) if cols is not None else names
    if not set(fields).issubset(names):
        raise RuntimeError(f'{name} W{w}{variant}: missing {sorted(set(fields)-set(names))}')
    d = pd.read_parquet(p,columns=fields)
    d['mergeid'] = d.mergeid.astype(str)
    if d.mergeid.duplicated().any() or d.mergeid.str.len().lt(3).any():
        raise RuntimeError(f'Invalid key in W{w}{variant} {name}')
    return d.set_index('mergeid',verify_integrity=True)


def choices(frame, prefix):
    fields = [c for c in frame if re.fullmatch(rf'{prefix}d(?:[0-9]+|no|ot)',c)]
    values = frame[fields]
    none = values[prefix+'dno'].eq(1)
    yes = values.drop(columns=prefix+'dno').eq(1).any(axis=1)
    all_legal = values.isin([0,1]).all(axis=1)
    return fields,all_legal & (none ^ yes),none & yes,all_legal & ~none & ~yes


def binary(x):
    return x.astype(float).where(x.isin([0,1]))


def year_month(y,m):
    return (12*y+m).where(y.between(2000,2030)&m.between(1,12)).astype(float)


def table(out,name):
    out.to_csv(TABLES/f'{name}.csv',encoding='utf-8-sig',index=False)


def rounded(n):
    return '<10' if n<10 else str(int((n+5)//10)*10)


def public_counts(d,cols):
    x=d.copy()
    for c in cols:
        x[c]=x[c].map(lambda v: rounded(v) if pd.notna(v) else None)
    return x


def assemble(w):
    base=native(w,'cv_r',[f'hhid{w}','country','language','gender','age_int','interview',
                         'int_year','int_month','deceased'])
    visit=base.loc[base.interview.eq(1)].copy()
    visit=visit.rename(columns={f'hhid{w}':'family','gender':'sex','age_int':'age'})
    visit['month']=year_month(visit.int_year,visit.int_month)
    visit['age']=visit.age.where(visit.age.between(18,115)).astype(float)
    visit['sex']=visit.sex.where(visit.sex.isin([1,2])).astype(float)
    visit['wave']=w
    visit['regular']=True
    v=native(w,'technical_variables',['mn101_']+(['mn103_'] if w==7 else []))
    visit=visit.join(v,validate='one_to_one')
    visit['route']=np.select([visit.mn103_.eq(0),visit.mn103_.eq(1)],['regular','SHARELIFE'],
                              default='unknown') if w==7 else 'regular'
    visit['mode']='regular_main_CAPI'
    visit['pre_pandemic']=(visit.month < (2020*12+3)).where(visit.month.notna())
    visit['country_code']=visit.country.astype(int)
    visit['country']=visit.country_code.map(CN)
    assert visit.country.notna().all()
    edu=native(w,'gv_isced',['isced1997_r']).reindex(visit.index).isced1997_r
    visit['education']=edu.where(edu.between(0,6)).astype(float)
    pfile=source(w,'ph')
    ph=native(w,'ph',[x for x in pq.read_schema(pfile).names
                     if re.fullmatch(r'ph(?:049|006)d(?:[0-9]+|no|ot)',x) or x=='ph054_'])
    ph=ph.reindex(visit.index)
    check, good, bad, blank=choices(ph,'ph049')
    visit['ph_response']=ph.ph054_.where(ph.ph054_.isin([1,2,3])).astype(float)
    visit['proxy']=visit.ph_response.map({1:0.,2:0.,3:1.})
    contributes=visit.ph_response.isin([1,2])
    visit['ph049_legal']=good
    visit['ph049_conflict']=bad
    visit['ph049_allzero']=blank
    for item,i in ITEMS.items():
        visit[item]=binary(ph[f'ph049d{i}']).where(good & contributes)
    visit['iadl5_complete_any_respondent']=ph[[f'ph049d{i}' for i in ITEMS.values()]].isin([0,1]).all(axis=1)&good
    adl=ph[[f'ph049d{i}' for i in ADL]]
    visit['adl_count']=adl.sum(axis=1,min_count=len(ADL)).where(good & contributes)
    visit['adl_any']=visit.adl_count.gt(0).astype(float).where(visit.adl_count.notna())
    visit['iadl_partial_positive']=(~good & ~bad & contributes &
                                   ph[[f'ph049d{i}' for i in ITEMS.values()]].eq(1).any(axis=1))
    _,d_good,d_bad,d_blank=choices(ph,'ph006')
    diseases={str(i):binary(ph[f'ph006d{i}']) for i in [1,2,4,5,6,10]}
    if w==4:
        diseases['arthritis']=binary(ph['ph006d8'])
    else:
        a,b=ph['ph006d19'],ph['ph006d20']
        diseases['arthritis']=pd.Series(np.select([a.eq(1)|b.eq(1),a.eq(0)&b.eq(0)],
                                                  [1.,0.],default=np.nan),index=ph.index)
    dx=pd.DataFrame(diseases).where(~d_bad & ~d_blank,axis=0)
    visit['ph006_legal']=d_good
    visit['ph006_conflict']=d_bad
    visit['chronic_count']=dx.sum(axis=1,min_count=7).where(d_good)
    visit['physical']=pd.Series(np.select([dx.eq(1).any(axis=1),
                                          dx.eq(0).all(axis=1)&d_good],[1.,0.],default=np.nan),index=visit.index)
    # Four directly administered tests; preserve official scores only with native raw route evidence.
    rawcols=[f'cf{k:03d}_' for k in (3,4,5,6,108,109,110,111,112)]
    word1=[f'cf{k}tot' for k in range(104,108)]
    word2=[f'cf{k}tot' for k in range(113,117)]
    cf=native(w,'cf',rawcols+word1+word2).reindex(visit.index)
    euros=[f'euro{i}' for i in range(1,13)]
    h=native(w,'gv_health',['cf008tot','cf016tot','numeracy2','orienti','eurod','eurodcat']+euros)
    h=h.reindex(visit.index)
    m=native(w,'mh',['mh002_','mh004_','mh010_','mh013_','mh017_']).reindex(visit.index)
    orient=cf[rawcols[:4]]
    seven=cf[rawcols[4:]]
    direct=contributes & visit.route.ne('unknown')
    health_direct=direct & visit.route.eq('regular')
    r1=cf[word1].apply(lambda f:f.where(f.between(0,10)))
    r2=cf[word2].apply(lambda f:f.where(f.between(0,10)))
    one1=r1.notna().sum(axis=1).eq(1)
    one2=r2.notna().sum(axis=1).eq(1)
    same_form=r1.notna().to_numpy() & r2.notna().to_numpy()
    same_form=pd.Series(same_form.sum(axis=1)==1,index=visit.index)
    good_o=orient.isin([1,2]).all(axis=1)
    good_s=(seven.ge(0)&seven.le(9999999)).all(axis=1)
    visit['c1']=h.cf008tot.where(direct & one1 & h.cf008tot.between(0,10)&h.cf008tot.eq(r1.sum(axis=1,min_count=1)))
    visit['c2']=h.cf016tot.where(direct & one2 & h.cf016tot.between(0,10)&h.cf016tot.eq(r2.sum(axis=1,min_count=1)))
    bothword=visit[['c1','c2']].notna().all(axis=1)
    visit.loc[bothword & ~same_form,['c1','c2']]=np.nan
    visit['c3']=h.numeracy2.where(health_direct & h.numeracy2.between(0,5) & good_s)
    visit['c4']=h.orienti.where(health_direct & h.orienti.between(0,4) & good_o &
                               h.orienti.eq(orient.eq(1).sum(axis=1)))
    visit['numeric2_excluded_raw_incomplete']=h.numeracy2.between(0,5)&~good_s
    visit['orient_raw_any']=orient.notna().any(axis=1)
    visit['orient_structural']=w in (4,5) and True
    visit['orient_structural']=visit.orient_structural & visit.mn101_.eq(1) & ~visit.orient_raw_any
    visit['cog_reference_comparable']=health_direct
    euro_parts=h[euros].isin([0,1]).all(axis=1)
    monitor=m.ge(0).any(axis=1)
    visit['eurod']=h.eurod.where(health_direct & h.eurod.between(0,12)&euro_parts & monitor &
                                 h.eurod.eq(h[euros].sum(axis=1,min_count=12)))
    visit['psychological_v2']=visit.eurod.ge(4).astype(float).where(visit.eurod.notna())
    visit['cesd']=visit.eurod  # inherited model ABI; not CES-D in interpretation/output.
    visit['eurod_case_audit']=visit.psychological_v2.notna() & ~visit.psychological_v2.eq(h.eurodcat)
    visit['multi_immediate_forms']=r1.notna().sum(axis=1).gt(1)
    visit['multi_delayed_forms']=r2.notna().sum(axis=1).gt(1)
    visit['word_form_mismatch']=bothword & ~same_form
    bfields=['br001_','br002_','br015_','br016_','br017_','br010_' if w<6 else 'br039_']
    b=native(w,'br',bfields).reindex(visit.index)
    br_self=b.br017_.isin([1,2])
    yes=b[['br015_','br016_']].isin([1,2]).any(axis=1)
    no=b[['br015_','br016_']].isin([3,4]).all(axis=1)
    visit['activity']=pd.Series(np.select([yes,no],[1.,0.],default=np.nan),index=visit.index).where(br_self)
    visit['br001']=b.br001_.where(br_self & b.br001_.isin([1,5]))
    visit['br002']=b.br002_.where(br_self & b.br002_.isin([1,5]))
    if w<6:
        visit['drinking']=b.br010_.isin([1,2,3,4]).astype(float).where(br_self & b.br010_.between(1,7))
        visit['drink_instrument']='BR010_last_3months_weekly_frequency'
    else:
        visit['drinking']=np.nan
        visit['drank7d']=b.br039_.eq(1).astype(float).where(br_self & b.br039_.isin([1,5]))
        visit['drink_instrument']='BR039_any_last_7_days_not_eligible_for_weekly_frequency_block'
    visit['seen']=True
    return visit.reset_index(names='person'),base


def cluster(d):
    parent={}
    def find(x):
        while x != parent.setdefault(x,x):
            parent[x]=parent[parent[x]]
            x=parent[x]
        return x
    for person, country, family in d[['person','country','family']].itertuples(index=False,name=None):
        if pd.isna(family) or len(str(family))<3:
            find('P:'+person)
        else:
            a,b=find('P:'+person),find('H:'+country+':'+str(family))
            if a!=b:
                parent[a]=b
    return d.person.map(lambda p:find('P:'+p))


def scoring(d):
    scores=d[SCORES]
    complete=scores.notna().all(axis=1)&d.cog_reference_comparable
    r=d.loc[d.wave.eq(5)&d.age.ge(45)&complete]
    sy=d.loc[d.wave.eq(5)&d.age.ge(45)&d.eurod.notna(),'eurod']
    ps=r[SCORES].agg(['mean','std'])
    assert len(r)>100 and ps.loc['std'].gt(0).all() and sy.std()>0
    z=(scores-ps.loc['mean'])/ps.loc['std']
    for name in SCORES:
        d['z'+name]=z[name].where(d.cog_reference_comparable)
    d['cog_z']=z.mean(axis=1).where(complete)
    d['cog_fixed']=z.lt(-1.5).any(axis=1).astype(float).where(complete)
    d['partial_cognitive_positive']=z.lt(-1.5).any(axis=1)&~complete&d.cog_reference_comparable
    d['cesd_z']=(d.eurod-sy.mean())/sy.std()
    report=[]
    params=lambda g,p,k,metric:dict(reference=k,measure=metric,reference_wave=5,reference_n=len(g),
                                     mean=p.loc['mean',metric],sd=p.loc['std',metric],
                                     threshold=p.loc['mean',metric]-1.5*p.loc['std',metric],
                                     target='age45_CAPI_direct_and_four_complete_before_window',ddof=1)
    report.extend(params(r,ps,'SHARE_W5',x) for x in SCORES)
    report.append(dict(reference='SHARE_W5',measure='EURO_D',reference_wave=5,reference_n=len(sy),
                       mean=sy.mean(),sd=sy.std(),threshold=4,
                       target='age45_CAPI_direct_EURO_D_complete_before_window',ddof=1))
    d[['cog_z_country','cog_country']]=np.nan
    for k,g in r.groupby('country',sort=True):
        a=g[SCORES].agg(['mean','std'])
        if len(g)<100 or not a.loc['std'].gt(0).all():
            continue
        report.extend(params(g,a,k,x) for x in SCORES)
        m=d.country.eq(k)&complete
        zz=(d.loc[m,SCORES]-a.loc['mean'])/a.loc['std']
        d.loc[m,'cog_z_country']=zz.mean(axis=1)
        d.loc[m,'cog_country']=zz.lt(-1.5).any(axis=1).astype(float)
    for suffix,c in [('fixed','cog_fixed'),('country','cog_country')]:
        d[f'code_{suffix}']=d.physical+2*d.psychological_v2+4*d[c]
        d[f'd_{suffix}']=d[f'code_{suffix}'].map(DOMAINS)
    table(pd.DataFrame(report),'SHARE_W5_REFERENCE_PARAMETERS')


def death_link():
    regular=[f'deadoralive_w{w}' for w in WAVES]
    covcols=['interview_w3','deceased_year','deceased_month','deceased_age',
             'deadoralive_w8ca','deadoralive_w9ca']+regular
    allw=native('X','gv_allwaves_cv_r',covcols)
    events=[]
    for w in WAVES:
        z=native(w,'xt',['xt008_','xt009_','xt010_']).reset_index(names='person')
        z['wave']=w
        events.append(z)
    xt=pd.concat(events,ignore_index=True)
    xt['exact_month']=year_month(xt.xt009_,xt.xt008_)
    xt['valid_year']=xt.xt009_.where(xt.xt009_.between(2000,2030))
    ag=xt.groupby('person',sort=False).agg(xt_first_wave=('wave','min'),xt_year_lo=('valid_year','min'),
               xt_year_hi=('valid_year','max'),xt_month_lo=('exact_month','min'),xt_month_hi=('exact_month','max'))
    u=allw.join(ag,how='left',validate='one_to_one')
    time_rank=u[regular].eq(2).to_numpy()
    wvalues=np.asarray(list(WAVES))
    first=np.where(time_rank.any(axis=1),wvalues[time_rank.argmax(axis=1)],np.nan)
    u['first_dead_wave']=pd.DataFrame({'cv':first,'xt':u.xt_first_wave},index=u.index).min(axis=1)
    u['death_confirmed']=u.first_dead_wave.notna()|u.deadoralive_w8ca.eq(2)|u.deadoralive_w9ca.eq(2)
    cvyear=u.deceased_year.where(u.deceased_year.between(2000,2030))
    cvmonth=year_month(u.deceased_year,u.deceased_month)
    year_all=pd.concat([cvyear,u.xt_year_lo,u.xt_year_hi],axis=1)
    month_all=pd.concat([cvmonth,u.xt_month_lo,u.xt_month_hi],axis=1)
    ymin,ymax=year_all.min(axis=1),year_all.max(axis=1)
    mmin,mmax=month_all.min(axis=1),month_all.max(axis=1)
    disagreement_year=ymin.ne(ymax)&ymin.notna()
    disagreement_month=mmin.ne(mmax)&mmin.notna()
    u['death_date_conflict']=disagreement_year|disagreement_month
    good_date=u.death_confirmed & ~u.death_date_conflict
    u['deceased_month_lo']=mmin.combine_first(ymin*12+1).where(good_date)
    # Exclusive right edge: first day of the month following the complete reported interval.
    u['deceased_month_hi']=(mmax+1).combine_first(ymax*12+13).where(good_date)
    u['death_date_precision']=np.select([u.death_date_conflict,good_date&mmin.notna(),
                                        good_date&ymin.notna(),u.death_confirmed],
                                       ['conflicting','calendar_month','calendar_year','wave_ascertainment_only'],
                                       default='not_confirmed')
    return u


def panel():
    parts=[]
    covers={}
    for w in WAVES:
        a,b=assemble(w)
        parts.append(a)
        covers[w]=b
    d=pd.concat(parts,ignore_index=True).sort_values(['person','month','wave'],na_position='last')
    d=d.reset_index(drop=True)
    assert not d.duplicated(['person','wave']).any()
    d['cluster']=cluster(d)
    ever=d.br001.replace({1:1.,5:0.}).groupby(d.person,sort=False).ffill()
    init_notdaily=d.mn101_.eq(0)&d.br001.eq(5)&d.br002.isna()
    d['smoking']=np.select([d.br002.eq(1),d.br002.eq(5)&ever.eq(1),
                            (d.br002.eq(5)&ever.eq(0))|init_notdaily],[2.,1.,0.],default=np.nan)
    d['K']=d[list(ITEMS)].sum(axis=1,min_count=5)
    scoring(d)
    dates=d.loc[d.month.notna()].groupby(['country','wave'],sort=True).month.agg(['size','min','max']).reset_index()
    dates=dates.rename(columns={'size':'main_interviews_dated','min':'first_regular_month','max':'last_regular_month'})
    w5=set(dates.loc[dates.wave.eq(5),'country'])
    dates['w5_participating_country']=dates.country.isin(w5)
    dates['regular_wave_available']=True
    lookup={(r.country,int(r.wave)):r.last_regular_month for r in dates.itertuples(index=False)}
    triples={(c,w0,w1,w2) for c in w5 for w0,w1,w2 in TRIPLES
             if all((c,w) in lookup for w in (w0,w1,w2))}
    main={c for c,_,_,_ in triples}
    dates['main_v5_country']=dates.country.isin(main)
    table(public_counts(dates,['main_interviews_dated']),'SHARE_COUNTRY_WAVE_OPPORTUNITIES')
    dates.to_csv(PRIVATE/'SHARE_COUNTRY_WAVE_OPPORTUNITIES_exact.csv',index=False)
    d['scope_main']=d.country.isin(main)
    d['next_wave']=d.wave.map({w:w+1 for w in range(4,9)})
    death=death_link()
    last=d.groupby('person',sort=False).month.max().rename('last_regular_main_month')
    death=death.join(last,how='left',validate='one_to_one')
    death['chronology_conflict']=death.deceased_month_hi.le(death.last_regular_main_month)&death.death_confirmed
    cross=[c for c in death if c.startswith('deadoralive_')]+['deceased_month_lo','deceased_month_hi',
        'death_date_precision','death_confirmed','death_date_conflict','first_dead_wave','chronology_conflict',
        'interview_w3','xt_first_wave']
    death[cross].reset_index(names='person').to_parquet(PRIVATE/'SHARE_deaths.parquet',index=False)
    d=d.merge(death[cross],left_on='person',right_index=True,how='left',validate='many_to_one')
    d['next_status']='administrative_end'
    d['last_opportunity_month']=d.country.map(dates.groupby('country').last_regular_month.max())
    d['w7_default_regular']=d.interview_w3.eq(1)
    for w in range(4,9):
        sel=d.wave.eq(w)
        cvnow=covers[w+1]
        a=cvnow.interview.reindex(d.loc[sel,'person']).to_numpy()
        c=d.loc[sel]
        status=c[f'deadoralive_w{w+1}']
        available=c.country.map({cn:m for (cn,ww),m in lookup.items() if ww==w+1})
        dl=c.deceased_month_hi.le(available+1)
        conflict=c.chronology_conflict.fillna(False)
        dead=(status.eq(2)|c.xt_first_wave.le(w+1)|dl) & c.death_confirmed.fillna(False)
        label=np.select([available.isna(),a==1,dead&~conflict,dead&conflict,
                         status.eq(1),status.isin([-101,-102,-103,-109]),a==2],
                        ['no_country_regular_opportunity','living_interview','confirmed_death',
                         'vital_status_conflict','reported_alive_noninterview','sampling_no_opportunity',
                         'confirmed_death'],default='survival_unknown')
        d.loc[sel,'next_status']=label
    d=d.sort_values(['person','wave'],ignore_index=True)
    return d,triples,lookup


def history(a,b):
    return np.select([a.eq(0)&b.eq(0),a.eq(0)&b.gt(0),a.gt(0)&b.gt(0),a.gt(0)&b.eq(0)],
                     ['H00','H01','H11','H10'],default='unknown')


def windows(d,triples,lookup):
    base_fields=['person','country','cluster','month','age','sex','education','proxy','ph_response',
                 'route','mode','pre_pandemic','K','physical','psychological_v2','cog_fixed','cog_country',
                 'code_fixed','d_fixed','code_country','d_country','chronic_count','cesd','eurod','cesd_z',
                 'cog_z','cog_z_country','next_status','adl_count','adl_any','activity','smoking','drinking',
                 'drink_instrument','seen','cog_reference_comparable','orient_structural','w7_default_regular']+list(ITEMS)
    out=[]
    for w0,w1,w2 in TRIPLES:
        scope=[c for c,a,b,cc in triples if (a,b,cc)==(w0,w1,w2)]
        def at(w,s):
            g=d.loc[d.wave.eq(w)&d.country.isin(scope),base_fields]
            return g.rename(columns={c:f'{c}{s}' for c in g if c!='person'})
        q=at(w0,0).merge(at(w1,1),on='person',how='inner',validate='one_to_one')
        q=q.loc[q.country0.eq(q.country1)].merge(at(w2,2),on='person',how='left',validate='one_to_one')
        q['source']='SHARE'
        q['country']=q.country1
        q['cluster']=q.cluster1
        q['wave']=w1
        q['prior_wave']=w0
        q['next_wave']=w2
        q['window']=f'{w0}->{w1}->{w2}'
        q['gap01']=q.month1-q.month0
        q['gap12']=q.month2-q.month1
        q['ordered_entry']=q.gap01.gt(0)&q.age1.ge(45)
        q['history']=history(q.K0,q.K1)
        q['status2']=q.next_status1
        q.loc[q.seen2.eq(True)&~q.gap12.gt(0),'status2']='interview_date_invalid'
        q['legal2']=q.status2.eq('living_interview')&q.seen2.eq(True)&q.gap12.gt(0)&q.country2.eq(q.country1)
        q['entry_eligible']=(q.ordered_entry & q.history.ne('unknown') & q.d_fixed1.le(1) &
                             q[['cesd1','cog_z1','chronic_count1']].notna().all(axis=1))
        q['y_fixed']=q.d_fixed2.ge(2).astype(float).where(q.legal2&q.d_fixed2.notna()&q.d_fixed1.le(1))
        # Alternate reference kept on exact main outcome/entry rows; selection conventions belong in the modeling layer.
        q['y_country']=q.d_country2.ge(2).astype(float).where(q.legal2&q.d_country2.notna())
        q['structural2']=q.legal2 & ((q.route2.eq('SHARELIFE')&q.d_fixed2.isna()) | q.orient_structural2.eq(True))
        q['potential_compatible_route2']=q.w7_default_regular1.eq(True) if w2==7 else True
        q['selection_opportunity2']=q.legal2 & q.route2.eq('regular') & q.proxy2.eq(0) & ~q.structural2
        q['future_status']=np.select([q.y_fixed.notna(),q.structural2,q.legal2&q.proxy2.eq(1),q.legal2],
                         ['observed_ppc','structural_module_unasked','proxy_only_invalid_PPC','interview_missing_domain'],
                                     default=q.status2)
        q['pre_pandemic_chain']=q.pre_pandemic0.eq(True)&q.pre_pandemic1.eq(True)&q.pre_pandemic2.eq(True)
        q['country_field_end_month2']=q.country.map({c:lookup[(c,w2)] for c in scope})
        out.append(q.loc[q.ordered_entry].copy())
    q=pd.concat(out,ignore_index=True)
    q['first_eligible']=False
    i=q.loc[q.entry_eligible].sort_values(['month1','wave'],kind='stable').drop_duplicates('person').index
    q.loc[i,'first_eligible']=True
    assert q.loc[q.first_eligible,'person'].is_unique
    assert not q.duplicated(['person','window']).any()
    return q


def audit(d,q,triples):
    measures={'iadl5_complete':'K','immediate_recall':'c1','delayed_recall':'c2','serial_seven':'c3',
              'time_orientation':'c4','four_tests':'cog_z','eurod':'eurod','seven_disease_count':'chronic_count',
              'physical_domain':'physical','PPC_complete':'code_fixed','ADL5':'adl_count',
              'activity':'activity','smoking':'smoking','drinking':'drinking'}
    summary=[]
    for (w,c,r),g in d.groupby(['wave','country','route'],sort=True):
        buf=dict(wave=w,country=c,route=r,mode='regular_main_CAPI',people=len(g),has_main_country=bool(g.scope_main.all()))
        for new,variable in measures.items():
            buf[new]=int(g[variable].notna().sum())
        for k,l in [(1,'respondent_only_PH'),(2,'joint_PH'),(3,'proxy_only_PH')]:
            buf[l]=int(g.ph_response.eq(k).sum())
        buf['unknown_PH_response']=int(g.ph_response.isna().sum())
        for c2 in ['ph049_conflict','ph049_allzero','ph006_conflict','orient_structural','numeric2_excluded_raw_incomplete',
                   'eurod_case_audit','multi_immediate_forms','multi_delayed_forms','word_form_mismatch']:
            buf[c2]=int(g[c2].sum())
        summary.append(buf)
    a=pd.DataFrame(summary)
    a.to_csv(PRIVATE/'SHARE_MEASUREMENT_BY_COUNTRY_ROUTE_exact.csv',index=False)
    cols=[c for c in a if c not in ['wave','country','route','mode','has_main_country']]
    table(public_counts(a,cols),'SHARE_MEASUREMENT_BY_COUNTRY_ROUTE')
    z=q.loc[q.entry_eligible].copy()
    flow=[]
    for (w,c,h),g in z.groupby(['window','country','history'],sort=True):
        flow.append(dict(window=w,country=c,history=h,eligible_windows=len(g),people=g.person.nunique(),
                         families=g.cluster.nunique(),observed_PPC=int(g.y_fixed.notna().sum()),
                         confirmed_death=int(g.future_status.eq('confirmed_death').sum()),
                         no_interview=int((~g.legal2&~g.future_status.eq('confirmed_death')).sum()),
                         living_PPC_missing=int((g.legal2&g.y_fixed.isna()).sum()),structural=int(g.structural2.sum())))
    f=pd.DataFrame(flow)
    f.to_csv(PRIVATE/'SHARE_WINDOW_SUPPORT_PREFIT_exact.csv',index=False)
    table(public_counts(f,[c for c in f if c not in ['window','country','history']]),
          'SHARE_WINDOW_SUPPORT_PREFIT')
    # Aggregate only; audit is deliberately blind to the PPC event count by history.
    cohort=pd.DataFrame([dict(source='SHARE',baseline_windows=len(z),baseline_people=z.person.nunique(),
                       observed_windows=int(z.y_fixed.notna().sum()),
                       observed_people=z.loc[z.y_fixed.notna(),'person'].nunique())])
    table(cohort,'SHARE_COHORT_FLOW_PREFIT')
    planned=pd.DataFrame(sorted(triples),columns=['country','prior_wave','index_wave','next_wave'])
    planned['next_wave_rule']='exact scheduled standard SHARE wave; no bypass across a missing personal interview'
    table(planned,'SHARE_MAIN_COUNTRY_WINDOWS')
    death=[]
    sample=d.loc[d.scope_main,'person'].nunique()
    v=d.loc[d.scope_main,['person']].drop_duplicates().merge(pd.read_parquet(PRIVATE/'SHARE_deaths.parquet'),
            on='person',how='left',validate='one_to_one')
    for label,g in v.groupby('death_date_precision',dropna=False):
        death.append(dict(precision=label,people=len(g),chronology_conflicts=int(g.chronology_conflict.eq(True).sum()),
                          covered_main_persons=sample))
    table(public_counts(pd.DataFrame(death),['people','chronology_conflicts','covered_main_persons']),
          'SHARE_DEATH_LINKAGE_AUDIT')
    data=[]
    for key,(fid,rel) in sorted(SOURCE_IDS.items()):
        with (STORE/rel).open('rb') as fh:
            h=hashlib.file_digest(fh,'sha256').hexdigest()
        data.append(dict(native_role=key,catalog_file_id=fid,source_release='9.0.0',sha256=h))
    table(pd.DataFrame(data),'SHARE_NATIVE_INPUT_HASHES')
    checks=dict(native_persistent_exposure_outcome_effects_computed=False,first_index_future_free=True,
        regular_w8_w9_only=True,no_corona_state_substitution=True,unique_person_wave=True,
        unique_person_window=True,fixed_reference_from_W5_before_window=True,
        n_panel_rows=len(d),n_country_main=int(z.country.nunique()),n_qualifying_windows=len(z),
        n_qualifying_persons=int(z.person.nunique()),n_observed_ppc_windows=int(z.y_fixed.notna().sum()))
    (QA/'SHARE_NATIVE_BUILD_QA.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks))


def main():
    d,t,a=panel()
    q=windows(d,t,a)
    assert q.loc[~q.legal2,'y_fixed'].isna().all()
    assert not ((d.wave.eq(7)&d.route.eq('SHARELIFE')) & d.code_fixed.notna()).any()
    assert int(q.first_eligible.sum())==q.loc[q.entry_eligible,'person'].nunique()
    d.to_parquet(PRIVATE/'SHARE_panel.parquet',index=False)
    q.to_parquet(PRIVATE/'SHARE_windows.parquet',index=False)
    audit(d,q,t)


if __name__=='__main__':
    main()

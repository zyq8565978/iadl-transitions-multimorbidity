"""V7 pre-fit reconstruction; historical products are read-only and rows stay PRIVATE."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
W = R.parent
V3 = W / 'PPC_IADL_Dynamic_V3_20261116'
V4 = W / 'PPC_IADL_Dynamic_V4_20261116'
V5 = W / 'PPC_IADL_SHARE_V5_20261116'
V6 = W / 'PPC_IADL_V6_Repair_Synthesis_20261116'
V2 = W / 'PPC_IADL_Dynamic_V2_20261115'
EQ = W / 'PPC_IADL_CHARLS_Equated_Update_20261116' / 'EDUCATION3_DIAGNOSTIC'
SOURCES = {'CHARLS': (EQ, 30), 'HRS': (V2, 8), 'ELSA': (V2, 8), 'SHARE': (V5, 12)}
ARCHIVES = {'SAFE_DELIVERY_V3.zip': V3, 'SAFE_DELIVERY_V4.zip': V4, 'SAFE_DELIVERY_V5_SHARE.zip': V5}
SEED = ['person','cluster','wave','month1','history','age1','sex1','education1','K1',
        'code_fixed1','chronic_count1','cesd_z1','cog_z1']
DOMAIN = ['chronic_count','cesd_z','cog_z','cesd']
N_INPUTS = []


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def watched(p):
    N_INPUTS.append({'path':p.relative_to(W).as_posix(),'sha256':sha(p)})
    return p


def load(p, columns):
    return pd.read_parquet(watched(p), columns=columns)


def check_archive_identity():
    p = watched(V6 / 'V6_EXECUTION_LOCK.json')
    authority = json.loads(p.read_text(encoding='utf-8-sig'))
    for name, folder in ARCHIVES.items():
        assert sha(folder / name) == authority['approved_archives'][name]['sha256'], name
        watched(folder / name)
    for name, signature in authority['v6'].items():
        assert sha(V6 / name)==signature, name
        watched(V6 / name)
    for p in (V3/'TRAJECTORY/TRAJECTORY_PROTOCOL_CN.md',V3/'TABLES/TRAJECTORY_SUPPORT.csv',
              V4/'REPORTS/C_TRAJECTORY_REPORT_CN.md',V5/'SHARE_ANALYSIS_FREEZE.md',
              V5/'SHARE_NATIVE_MEASUREMENT_CN.md',V6/'REPORTS/DEATH_NATIVE_SOURCES_V6_CN.md'):
        watched(p)


def scope(x):
    return x.loc[(x.wave.eq(x.index_wave) & x.months.eq(0)) |
                 (x.wave.gt(x.index_wave) & x.months.between(1,72))].copy()


def select_repeaters(z):
    c = z.dropna(subset=DOMAIN).copy()
    assert not c.duplicated(['person','months']).any(), 'ambiguous simultaneous visits'
    size = c.groupby('person').size()
    at_index = c.loc[c.months.eq(0),'person']
    people = size.index[size.ge(3) & size.index.isin(at_index)]
    return c.loc[c.person.isin(people)].sort_values(['person','months']).copy()


def flagged_death_v6(z, source):
    dt = load(V6/'PRIVATE'/f'death_V6_{source}.parquet',
              ['person','death_confirmed','death_hi_date','conflict'])
    assert dt.person.is_unique
    dt['death_hi_month'] = dt.death_hi_date.dt.year*12 + dt.death_hi_date.dt.month
    dt = dt.set_index('person')
    high = z.person.map(dt.death_hi_month)
    valid_bound = z.person.map(dt.death_confirmed).eq(True) & ~z.person.map(dt.conflict).eq(True)
    return valid_bound & z.month.gt(high)


def source_inputs(source, base, max_score):
    geocols = ['country'] if source=='SHARE' else []
    flags = ['entry_eligible','first_eligible']
    w = load(base/'PRIVATE'/f'{source}_windows.parquet', [*SEED,*flags,*geocols])
    e = w.loc[w.entry_eligible & w.first_eligible, [*SEED,*geocols]].copy()
    independent = w.loc[w.entry_eligible].sort_values(['month1','wave']).drop_duplicates('person')
    assert e.person.is_unique and set(zip(e.person,e.wave,e.month1)) == set(zip(independent.person,independent.wave,independent.month1))
    e = e.rename(columns={'wave':'index_wave'})
    native = ['scope_main','first_dead_wave','death_confirmed','death_date_conflict','deceased_month_hi'] if source=='SHARE' else ['next_status']
    p = load(base/'PRIVATE'/f'{source}_panel.parquet', ['person','wave','month',*geocols,*DOMAIN,*native])
    assert not p.duplicated(['person','wave']).any()
    if source=='SHARE':
        p = p.loc[p.scope_main.eq(True)]
        old = pd.Series(dtype=float)
    else:
        old = p.loc[p.next_status.astype(str).str.lower().str.contains('dead|death',regex=True)].groupby('person').wave.min()
    z = scope(e.merge(p,on=['person',*geocols],validate='one_to_many').assign(
        months=lambda x: x.month-x.month1))
    if source=='SHARE':
        dead = z.first_dead_wave.notna() & z.wave.ge(z.first_dead_wave)
        dated = z.death_confirmed.eq(True) & ~z.death_date_conflict.eq(True) & z.month.gt(z.deceased_month_hi)
        natural = dead | dated
    else:
        natural = z.wave.gt(z.person.map(old))
    prior_release = select_repeaters(z.loc[~natural])
    new_flag = flagged_death_v6(z,source)
    x = select_repeaters(z.loc[~natural & ~new_flag])
    if source != 'SHARE':
        reference = pd.read_csv(V3/'TABLES/TRAJECTORY_SUPPORT.csv')
        ref = reference.loc[reference.source.eq(source)&reference.history.eq('ALL')].iloc[0]
        assert int(ref.included_people)==prior_release.person.nunique()
        assert int(ref.included_records)==len(prior_release)
    assert np.isfinite(x[DOMAIN]).all().all()
    assert x.chronic_count.between(0,7).all() and x.chronic_count.eq(x.chronic_count.round()).all()
    assert x.cesd.between(0,max_score).all() and x.cesd.eq(x.cesd.round()).all()
    long = x[['person','months','chronic_count','cesd','cog_z']].rename(columns={
        'chronic_count':'chronic','cesd':'symptoms','cog_z':'cognition'})
    long['time']=long.pop('months')/12
    long['chronic']=long.chronic.astype(int)
    long['symptoms']=long.symptoms.astype(int)
    long['symptoms_max']=max_score
    long['source']=source
    if source!='SHARE':
        e['country']={'CHARLS':'China','HRS':'United States','ELSA':'England'}[source]
    i = e.loc[e.person.isin(x.person)].rename(columns={'age1':'age','sex1':'sex','education1':'education',
        'code_fixed1':'code_fixed','chronic_count1':'chronic_index','cesd_z1':'symptom_z_index',
        'cog_z1':'cognition_index'})
    i['Kgroup']=i.pop('K1').clip(upper=3)
    i['education']=i.education.astype('string')
    i['country']=i.country.astype('string')
    i['source']=source
    i['n_visits']=i.person.map(x.groupby('person').size())
    i['last_time']=i.person.map(x.groupby('person').months.max())/12
    i['cluster']=source + ':' + i.cluster.astype('string')
    hist = []
    for cat in ['ALL','H00','H01','H10','H11']:
        pick = i if cat=='ALL' else i.loc[i.history.eq(cat)]
        before = e if cat=='ALL' else e.loc[e.history.eq(cat)]
        hist.append({'source':source,'history':cat,'eligible_people':len(before),'selected_people':len(pick),
                     'complete_visits':int(pick.n_visits.sum()),'people_4plus':int(pick.n_visits.ge(4).sum()),
                     'n_index_waves':pick.index_wave.nunique(),'latest_complete_time_years':pick.last_time.max()})
    for y in (long,i):
        y['person']=source + ':' + y.person.astype('string')
    assert long.person.notna().all() and i.person.notna().all() and i.cluster.notna().all()
    a = {'source':source,'historical_native_screen_records':int(natural.sum()),
         'additional_v6_confirmed_after_bound_records':int((new_flag&~natural).sum()),
         'reconstructed_prior_repeaters':prior_release.person.nunique(),
         'reconstructed_prior_rows':len(prior_release),'final_repeaters':len(i),'final_rows':len(long)}
    return long,i,hist,a


def protect(n):
    return '<10' if 0 < n < 10 else int((n+5)//10*10)


def main():
    for p in ['PRIVATE','QA','TABLES']:
        (R/p).mkdir(exist_ok=True)
    check_archive_identity()
    panel=[]; indices=[]; totals=[]; audit=[]
    for source,(directory,max_score) in SOURCES.items():
        long,i,rows,notes=source_inputs(source,directory,max_score)
        panel.append(long); indices.append(i); totals.extend(rows); audit.append(notes)
    long = pd.concat(panel,ignore_index=True).sort_values(['source','person','time'])
    ind = pd.concat(indices,ignore_index=True).sort_values(['source','person'])
    cont = long.groupby('person').time.agg(['min','size','max'])
    assert ind.person.is_unique and set(ind.person)==set(cont.index)
    assert cont['min'].eq(0).all() and cont['max'].between(0,6).all() and cont['size'].ge(3).all()
    assert not long.duplicated(['person','time']).any()
    assert long.groupby('person').source.nunique().eq(1).all()
    long.to_parquet(R/'PRIVATE/long.parquet',index=False)
    ind.to_parquet(R/'PRIVATE/index.parquet',index=False)
    pd.DataFrame(audit).to_csv(R/'PRIVATE/prefit_lineage_exact.csv',index=False)
    total = pd.DataFrame(totals)
    total.to_csv(R/'PRIVATE/prefit_support_exact.csv',index=False)
    public = total.copy()
    detail = public.history.ne('ALL') | public.source.eq('SHARE')
    for field in ['eligible_people','selected_people','complete_visits','people_4plus']:
        public[field] = public[field].astype(object)
        public.loc[detail,field]=public.loc[detail,field].map(protect)
        public.loc[~detail&public[field].map(lambda v:isinstance(v,int) and 0<v<10),field]='<10'
    # Exact CHARLS records compared with its released historical benchmark; all smaller
    # SHARE depth cells and corresponding totals may not be recovered by subtraction.
    public.loc[detail,'latest_complete_time_years']=np.nan
    public.to_csv(R/'TABLES/PREFIT_SUPPORT.csv',index=False)
    pd.DataFrame(N_INPUTS).to_csv(R/'QA/PREFIT_INPUT_HASHES.csv',index=False)
    print(public.loc[public.history.eq('ALL')].to_string(index=False),flush=True)


if __name__ == '__main__':
    main()

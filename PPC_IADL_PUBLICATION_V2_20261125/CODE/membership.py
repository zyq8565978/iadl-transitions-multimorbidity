"""Recreate the exact model-estimation rows but never invoke an estimator or save individual records."""
from inputs import *
import os
for v in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[v]='1'
sys.path.insert(0,str(M/'CODE'))
import continuous_suite as f
import pandas as pd,numpy as np
def ids(s,v):return {s+':'+str(k).removeprefix(s+'|').removeprefix(s+':') for k in v}
def run():
    authority()
    raw=dict(f.co.inputs()); risk=[f.base.old.load(k,'M_severity')[0] if hasattr(f,'base') else f.old.load(k,'M_severity')[0] for k in cc]
    primary={s:set().union(*[ids(s,d.loc[d.source.eq(s),'person']) for d in risk]) for s in cohorts}
    tr=pd.read_parquet(Z/'PRIVATE/index.parquet',columns=['person','source','n_visits'])
    traj={s:ids(s,tr.loc[tr.source.eq(s),'person']) for s in cohorts}
    out=[]; union={s:set() for s in cohorts};unobs={s:set() for s in cohorts}; allsets={}
    for comp in cc:
        k=comp.replace('_vs_','_minus_')
        for dom in dm:
            stem=f'CONTINUOUS_{k}_{dom}_index_included';ref=pd.read_csv(M/'TABLES'/f'{stem}.csv').iloc[0];part=[]
            for s in cohorts:
                d=f.co.initialize(s,raw[s],k,dom,True)[0]
                people=ids(s,d.person); original=raw[s][1]; assert original.code_fixed1.isin([0,1,2,4]).all()
                assert d.years.between(0,6).all() and d.years.eq(0).sum()==len(people) and d.loc[d.years.gt(0),'person'].nunique()==len(people)
                assert not d.duplicated(['person','wave']).any()
                union[s]|=people;allsets[(s,k,dom)]=people
                unobs[s].update(zip(d.person.astype(str),d.wave.astype(int)))
                part.append(dict(cohort=s,comparison=k,domain=dom,people=len(people),assessments=len(d),post_index=int(d.years.gt(0).sum()),overlap_main=len(people&primary[s]),overlap_trajectory=len(people&traj[s])))
            total={'cohort':'All','comparison':k,'domain':dom,**{field:sum(z[field] for z in part) for field in list(part[0])[3:]}}
            assert (total['people'],total['assessments'])==(ref.people,ref.windows)
            for mi in range(1,51):
                j=rdj(M/f'QA/{stem}_mi{mi:03}.json'); assert (j['people'],j['windows'],j['post_records'])==(total['people'],total['assessments'],total['post_index'])
            out+=[total,*part]
    pd.DataFrame(out).to_csv(A/'continuous_model_membership.csv',index=False)
    flow=[]
    for s in cohorts:
        a,b,c=primary[s],traj[s],union[s]
        flow.append(dict(cohort=s,main=len(a),trajectory=len(b),continuous=len(c),continuous_measurements=len(unobs[s]),main_trajectory=len(a&b),main_continuous=len(a&c),trajectory_continuous=len(b&c),all_three=len(a&b&c),main_only=len(a-b-c),trajectory_only=len(b-a-c),continuous_only=len(c-a-b),main_trajectory_only=len((a&b)-c),main_continuous_only=len((a&c)-b),trajectory_continuous_only=len((b&c)-a),any_branch=len(a|b|c)))
    all_={'cohort':'All',**{k:sum(r[k] for r in flow) for k in list(flow[0])[1:]}}
    assert [all_[k] for k in ['main','trajectory','main_trajectory']]==[35494,17820,17544] and all_['continuous_only']>0
    # Export exact Venn partition overall. Do not expose sparse cohort-specific Venn or its complements.
    pd.DataFrame([all_]).to_csv(A/'three_branch_overlap.csv',index=False)
    pd.DataFrame([{k:v for k,v in rr.items() if k in ['cohort','main','trajectory','continuous','continuous_measurements']} for rr in [all_,*flow]]).to_csv(A/'three_branch_support.csv',index=False)
    print(json.dumps(all_),flush=True)
if __name__=='__main__':run()

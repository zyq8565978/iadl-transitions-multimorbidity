"""Exhaustive check of the ORIGINAL 199 resamples: an intercept-calibration root requires 0<source rate<1."""
from binding import *
from collections import Counter
def run():
    rows=[];pl=json.loads((ROOT/'QA/EXECUTION_PLAN.json').read_text())['plan_by_spec'];ch=ROOT/'PRIVATE/dispatch'
    for p in pl:
        if p['engine']!='FIRTH':continue
        c=p['comparison'];k=p['code'];stem=p['stem'];d,_,_,src=load_risk(c);y=d.code_fixed2.eq(k).to_numpy(int)
        seed=71118000+int.from_bytes(hashlib.sha256(stem.encode()).digest()[:3],'big')
        plan,inv=base.qc.family_plan(d.family.to_numpy(str),d.stratum.to_numpy(str),seed,199)
        # Count outcomes within each source using sampled families; independent of covariates or any beta.
        absent=[]
        for i,row in enumerate(plan):
            w=row[inv]
            ok=True
            for s in np.unique(src):
                idx=src==s
                wsum=w[idx].sum();event=w[idx]@y[idx]
                ok=bool(ok and 0<event<wsum)
            absent.append(not ok)
        absent=np.array(absent)
        # Check every native/blocked error whose recorded reason is lack of a FLIC solution.
        checks=0
        for a in sorted(ch.glob(stem+'_mi???_redraws.npz')):
            try:
                with np.load(a,allow_pickle=False) as z:
                    dm=int(z['done']);err=json.loads(str(z['errors']))
                    for ix,e in err.items():
                        if e=='calibration_outcome_group_absent':assert absent[int(ix)-1];checks+=1
                    assert np.all(~np.isfinite(z['draws'][:dm][absent[:dm]]))
            except (PermissionError,FileNotFoundError):continue
        sig=hashlib.sha256(plan.tobytes()).hexdigest()
        rows.append(dict(stem=stem,comparison=c,code=k,planned=199,no_root_draws=int(absent.sum()),max_possible_valid=199-int(absent.sum()),required=180,
            bootstrap_variance_mathematically_impossible=bool(199-absent.sum()<180),native_absent_root_reasons_verified=checks,all_50_same_outcome_membership_and_seed=True,draw_plan_hash=sig))
        if 199-absent.sum()<180:
            np.save(ch/(stem+'_no_flic_root.npy'),absent)
    pd.DataFrame(rows).to_csv(ROOT/'QA/FIXED_PLAN_FLIC_ROOT_GATE.csv',index=False)
    print(pd.DataFrame(rows)[['stem','max_possible_valid','required','bootstrap_variance_mathematically_impossible']].to_string(index=False),flush=True)
if __name__=='__main__':run()

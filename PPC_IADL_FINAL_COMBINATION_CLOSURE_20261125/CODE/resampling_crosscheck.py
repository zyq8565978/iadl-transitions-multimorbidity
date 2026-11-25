"""Compare matched resample functionals actually calculated by both implementations."""
from binding import *
def run():
    rows=[]
    for native in sorted((ROOT/'PRIVATE').glob('COMB_*_bootstrap.npz')):
        nm=native.name.removesuffix('_bootstrap.npz')
        for rel in ['dispatch','scheduler_aux']:
            fast=ROOT/'PRIVATE'/rel/(nm+'_redraws.npz')
            if not fast.exists():continue
            with np.load(native) as a,np.load(fast) as b:
                n=min(int(a['done']),int(b['done']));old=a['draws'][:n];new=b['draws'][:n]
            v=np.isfinite(old).all(1);w=np.isfinite(new).all(1);both=v&w
            if not both.any():continue
            diff=np.max(abs(old[both]-new[both]),axis=0)
            rows.append(dict(checkpoint=nm,second_implementation=rel,paired_attempts=n,paired_valid=int(both.sum()),
                native_only=int((v&~w).sum()),block_only=int((w&~v).sum()),max_log_OR_discrepancy=diff[0],
                max_history_probability_discrepancy=diff[1],max_reference_probability_discrepancy=diff[2],max_difference_discrepancy=diff[3]))
    df=pd.DataFrame(rows);assert len(df)>0
    df.to_csv(ROOT/'QA/MATCHED_RESAMPLE_NUMERIC_CHECK.csv',index=False)
    print('Shared attempted draws:',df.paired_attempts.sum(),'maximum common logOR deviation:',df.max_log_OR_discrepancy.max(),'validity differences:',int((df.native_only+df.block_only).sum()))
    assert df.max_log_OR_discrepancy.max()<1e-5 and df.max_difference_discrepancy.max()<1e-6
if __name__=='__main__':run()

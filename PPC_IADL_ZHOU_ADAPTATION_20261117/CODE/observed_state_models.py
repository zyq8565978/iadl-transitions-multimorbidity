"""Discrete observed-visit multistate probabilities in the frozen V6 risk sets."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_ipd import COMPS, load
from ridge import fit
import numpy as np
import pandas as pd
from scipy.linalg import qr


def independent_columns(x):
    """Drop only columns redundant in the actual origin-specific risk set."""
    fixed = x[:, :2]
    u = np.linalg.qr(fixed, mode='reduced')[0]
    z = x[:, 2:] - u @ (u.T @ x[:, 2:])
    z /= np.maximum(np.linalg.norm(z, axis=0), 1.)
    _, r, pivot = qr(z, mode='economic', pivoting=True)
    j = np.r_[0, 1, np.sort(2 + pivot[:(abs(r.diagonal()) > 1e-8).sum()])]
    return x[:, j]


def group(c):
    return np.select([c == 0, np.isin(c, [1,2,4])], [0,1], default=2)


def candidate(d, x, origin, to, labels, contrast, detail):
    n = len(d)
    arms = d.hx.to_numpy(dtype=int)
    q = len(labels)
    tab = np.stack([np.bincount(to[arms==h], minlength=q) for h in (0,1)])
    if np.any(tab.sum(axis=1) < 10):
        return [dict(contrast=contrast,detail=detail,origin=origin,destination=l,status='Both history groups not supported') for l in labels]
    # Suppress the complete observed allocation of an arm if any small cell is present.
    publish = np.all(tab >= 10, axis=1)
    x = independent_columns(x)
    cl, names = pd.factorize(d.family)
    method = 'No estimate'; result = None
    if np.all(tab >= 10):
        for lam in (0.,1.):
            try:
                # Known pure strata approach boundaries at ordinary ML.
                if lam==0 and d.assign(category=to).groupby('stratum').category.nunique().eq(1).any():
                    continue
                check = fit(x, to, cl, q, lam)
                p0,p1,o,lo,hi,_,emin,score = check
                if emin < 1e-5 or not np.all(np.isfinite(np.r_[p0,p1,o,lo,hi])):
                    continue
                assert abs(p0.sum()-1)<1e-10 and abs(p1.sum()-1)<1e-10
                method = 'Family-clustered multinomial MLE' if lam==0 else 'Nuisance L2 sensitivity (nominal CI)'
                result = check
                break
            except (ValueError, np.linalg.LinAlgError):
                continue
    rows=[]
    for j,label in enumerate(labels):
        r = dict(contrast=contrast,detail=detail,origin=origin,destination=label,reference_destination=labels[0],
                 status=method, windows=n, people=d.person.nunique(), families=len(names),
                 history_events=int(tab[1,j]) if publish[1] else None, history_windows=int(tab[1].sum()) if publish[1] else None,
                 reference_events=int(tab[0,j]) if publish[0] else None, reference_windows=int(tab[0].sum()) if publish[0] else None,
                 history_observed=tab[1,j]/tab[1].sum() if publish[1] else None,
                 reference_observed=tab[0,j]/tab[0].sum() if publish[0] else None)
        if result is not None:
            r.update(history_standardized=p1[j],reference_standardized=p0[j],difference_pp=100*(p1[j]-p0[j]),
                     OR=o[j-1] if j else 1., lo=lo[j-1] if j else None, hi=hi[j-1] if j else None,
                     model_design_rank=x.shape[1], min_information_eigenvalue=float(emin),max_score=float(score))
        rows.append(r)
    return rows


def main():
    rows=[]
    cells={1:'Physical only',2:'Psychological only',4:'Cognitive only'}
    for contrast in COMPS:
        d,y,x,_=load(contrast,'M_severity')
        assert d.code_fixed1.isin([0,1,2,4]).all() and d.code_fixed2.isin(range(8)).all()
        a,b=d.code_fixed1.to_numpy(dtype=int),d.code_fixed2.to_numpy(dtype=int)
        assert np.array_equal(group(b)==2,y==1)
        for fine in (False,True):
            origins={0:'No domain',**cells} if fine else {0:'No domain',1:'One domain'}
            for j,name in origins.items():
                use=(a==j) if fine or j==0 else a!=0
                end=b[use]
                if j==0 and fine:
                    to=np.select([end==0,end==1,end==2,end==4],[0,1,2,3],default=4)
                    levels=('No domain',*cells.values(),'PPC')
                elif j==0:
                    to=group(end); levels=('No domain','One domain','PPC')
                else:
                    # All non-PPC destinations are the explicit comparator for one-domain origins.
                    to=(group(end)==2).astype(int); levels=('No PPC','PPC')
                rows+=candidate(d.loc[use],x[use],name,to,levels,contrast,'Expanded' if fine else 'Main')
    x=pd.DataFrame(rows)
    x.to_csv(ROOT/'TABLES/TABLE2_OBSERVED_MULTISTATE_SOURCE.csv',index=False,encoding='utf-8-sig')
    display(x)
    print(x.groupby(['detail','contrast','status']).size().to_string())


def display(x):
    show=x[(x.detail.eq('Main') & ((x.origin.eq('No domain') & x.destination.ne('No domain')) | (x.origin.eq('One domain') & x.destination.eq('PPC'))))|
           (x.detail.eq('Expanded') & ((x.origin.eq('No domain') & x.destination.ne('No domain')) | (x.origin.ne('No domain') & x.destination.eq('PPC'))))].copy()
    show=show.drop_duplicates(['contrast','detail','origin','destination'])
    def ci(row):
        return f'{row.OR:.2f} ({row.lo:.2f}–{row.hi:.2f})' if pd.notna(row.OR) else 'Not estimated'
    show['Adjusted OR (95% CI)']=show.apply(ci,axis=1)
    for arm in ('history','reference'):
        show[arm.title()+' observed']=[f'{int(e):,}/{int(n):,} ({p:.1%})' if pd.notna(e) else 'Not displayed'
                                     for e,n,p in zip(show[arm+'_events'],show[arm+'_windows'],show[arm+'_observed'])]
    cols={'detail':'Panel','contrast':'Comparison','origin':'Index state','destination':'Next state','reference_destination':'Destination comparator',
          'windows':'Windows','history_standardized':'History probability','reference_standardized':'Reference probability','difference_pp':'Difference, pp','status':'Estimator / status'}
    front=['detail','contrast','origin','destination','reference_destination','windows','History observed','Reference observed','history_standardized','reference_standardized','difference_pp','Adjusted OR (95% CI)','status']
    show[front].rename(columns=cols).to_csv(ROOT/'TABLES/TABLE2_OBSERVED_TRANSITIONS_DISPLAY.csv',index=False,encoding='utf-8-sig')


if __name__=='__main__':
    if sys.argv[1:] == ['--display-only']:
        display(pd.read_csv(ROOT/'TABLES/TABLE2_OBSERVED_MULTISTATE_SOURCE.csv'))
    else:
        main()

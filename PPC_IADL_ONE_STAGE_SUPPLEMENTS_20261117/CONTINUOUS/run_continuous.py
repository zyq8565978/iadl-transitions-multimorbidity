"""Four-source continuous repeat-measurement supplement; no original outputs written."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import hashlib
import json
from pathlib import Path
import runpy
import sys
import time

import numpy as np
import pandas as pd
import patsy
import scipy
from scipy.linalg import block_diag
from scipy.optimize import minimize_scalar
from scipy.special import expit
from scipy.stats import t
import statsmodels

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
V3 = WORK / 'PPC_IADL_Dynamic_V3_20261116'
V4 = WORK / 'PPC_IADL_Dynamic_V4_20261116'
V5 = WORK / 'PPC_IADL_SHARE_V5_20261116'
OLD = runpy.run_path(str(V3 / 'CODE/long_v3.py'))
B = runpy.run_path(str(V4 / 'CODE/b_longitudinal.py'))
sys.path.insert(0, str(V5 / 'CODE'))
SH = runpy.run_path(str(V5 / 'CODE/share_longitudinal_v5.py'))
SOURCES = ('CHARLS', 'HRS', 'ELSA', 'SHARE')
CEILINGS = dict(CHARLS=30, HRS=8, ELSA=8, SHARE=12)
DOMAINS = OLD['DOMAINS']
UNITS = dict(physical='disease count (0–7)', psychological='percentage points of source symptom scale',
             cognitive='source fixed-reference mean subtest Z')
HASHES = {}


def path_hash(p):
    with p.open('rb') as f:
        value = hashlib.file_digest(f, 'sha256').hexdigest()
    HASHES[p.relative_to(WORK).as_posix()] = value
    return value


def inputs():
    z3 = json.loads((V3 / 'TABLES/LONG_RUN_V3.json').read_text(encoding='utf-8-sig'))
    z5 = json.loads((V5 / 'SHARE_ANALYSIS_LOCK.json').read_text(encoding='utf-8-sig'))
    for s in SOURCES:
        for k in ('panel', 'windows'):
            if s == 'SHARE':
                p = V5 / 'PRIVATE' / f'{s}_{k}.parquet'
                expected = z5['inputs'][p.relative_to(V5).as_posix()]
            else:
                p = (OLD['DIAG'] if s == 'CHARLS' else OLD['OLD']) / 'PRIVATE' / f'{s}_{k}.parquet'
                expected = z3['sources'][s][k]
            if path_hash(p) != expected:
                raise RuntimeError(f'{s} {k}: frozen input hash mismatch')
        if s == 'SHARE':
            p,e,z = SH['prepare']()
            score = pd.read_parquet(V5 / 'PRIVATE/SHARE_panel.parquet', columns=['person','wave','cesd'])
            p = p.merge(score,on=['person','wave'],validate='one_to_one',how='left',sort=False)
            yield s,(p,e,z)
        else:
            yield s,OLD['prepare'](s)


def initialize(s, raw, comp, domain, include):
    panel, entry, obs = raw
    field = DOMAINS[domain]
    formula = B['fit_formula'](s, comp, domain, 'index_included' if include else 'ri_linear')
    if s == 'SHARE':
        d = SH['sample'](entry, obs, comp, field)
        formula += ' + C(country)'
        if include:
            d = SH['add_initial'](d, obs, domain, comp)
    else:
        d = OLD['derived'](OLD['sample'](entry, obs, comp, domain, 6.)[0])
        check_formula = B['fit_formula'](s, comp, domain, 'ri_linear')
        check_x = patsy.dmatrix(check_formula, d, return_type='dataframe', NA_action='raise')
        p = V3 / 'PRIVATE' / f'LONG_{s}_{comp}_{domain}_06.npz'
        path_hash(p)
        with np.load(p, allow_pickle=False) as z:
            checks = dict(X=np.asarray(check_x), terms=check_x.columns.to_numpy(str),
                          y=d[field].to_numpy(float)/(7 if domain == 'physical' else 1),
                          person=d.person.to_numpy(str), cluster=d.cluster.to_numpy(str),
                          years=d.years.to_numpy(float))
            if not all(np.array_equal(z[k], value) for k, value in checks.items()):
                raise RuntimeError(f'{s} {comp} {domain}: V3 checkpoint identity failed')
        if include:
            start = obs.loc[obs.person.isin(d.person) & obs.wave.eq(obs.index_wave)].copy()
            assert start.person.is_unique and start.years.eq(0).all() and len(start) == d.person.nunique()
            assert np.array_equal(start[field], start[B['INITIAL'][domain]])
            start['hx'] = start.history.eq(OLD['COMPARE'][comp][0]).astype(int)
            d = OLD['derived'](pd.concat([start, d], ignore_index=True))
            d = d.assign(post=d.years.gt(0).astype(int), post_time2=d.years.gt(0)*d.time2,
                         other_domain_code=d.code_fixed1.astype(int) & (7-B['BITS'][domain]))
    if domain == 'psychological':
        d = d.merge(panel[['person', 'wave', 'cesd']], on=['person', 'wave'], how='left',
                    validate='one_to_one', sort=False)
        assert d.cesd.between(0, CEILINGS[s]).all()
        d['outcome'] = d.cesd * (100 / CEILINGS[s])
    else:
        d = d.assign(outcome=d[field]/(7 if domain == 'physical' else 1))
    assert np.isfinite(d.outcome).all() and d.years.between(0, 6).all()
    assert not d.duplicated(['person','wave']).any()
    assert d.loc[d.years.gt(0)].groupby('hx').wave.nunique().reindex([0,1]).ge(2).all()
    d = d.reset_index(drop=True)
    x = patsy.dmatrix(formula, d, return_type='dataframe', NA_action='raise')
    return d, x, formula


def safe_n(x):
    return '<10' if x < 10 else str(int((x+5)//10)*10)


def make_blocks(ds, shared):
    designs = [np.asarray(ds[s][1]) for s in SOURCES]
    parts = []
    for (s, x) in zip(SOURCES, designs):
        names = ds[s][1].columns
        j = [names.get_loc(key) for key in shared]
        k = [i for i in range(len(names)) if i not in j]
        parts.append((j, k, names))
    x = np.c_[np.concatenate([a[:, j] for a, (j,k,n) in zip(designs,parts)]),
              block_diag(*(a[:, k] for a, (j,k,n) in zip(designs,parts)))]
    terms = shared + [f'{s}:{names[k]}' for s, (_,ks,names) in zip(SOURCES,parts) for k in ks]
    assert np.linalg.matrix_rank(x) == x.shape[1]
    return x, parts, terms


def estimate(x, y, family, people, bounded, ri):
    # Fixed-effect computations are QR conditioned; transformation out of that
    # basis is applied before all analytic gradients and saved estimates.
    q, rt = np.linalg.qr(x, mode='reduced')
    n, p = x.shape
    scale = np.sqrt(n)
    z = q*scale
    back = np.linalg.solve(rt, np.eye(p))*scale
    assert np.max(np.abs(x @ back-z)) < 1e-7
    condition = np.linalg.cond(x)
    if ri:
        # V4 B1 uses REML with exchangeable within-person working errors; use
        # the exact restricted likelihood, profiled over common residual scale.
        gid = pd.factorize(people,sort=True)[0]
        counts = np.bincount(gid)
        zy = np.c_[z,y]
        sums = np.zeros((len(counts),p+1))
        np.add.at(sums, gid, zy)
        base = zy.T@zy
        def reml(lam, full=False):
            factor = lam/(1+lam*counts)
            vi = base-(sums.T*factor)@sums
            info = vi[:-1,:-1]
            rhs = vi[:-1,-1]
            coef = np.linalg.solve(info,rhs)
            rss = vi[-1,-1]-rhs@coef
            sign,logdet=np.linalg.slogdet(info)
            if sign<1 or rss<=0:
                raise ValueError('invalid REML matrix')
            obj = (n-p)*np.log(rss/(n-p))+np.log1p(lam*counts).sum()+logdet
            if not full:
                return obj
            r = y-z@coef
            vs = r-factor[gid]*np.bincount(gid,weights=r)[gid]
            # Independent implementation of the profiled GLS score.
            direct = B['working_inverse'](r[:,None],gid,1.,lam).ravel()
            assert np.allclose(vs,direct,atol=5e-10) and np.linalg.norm(z.T@vs, ord=np.inf)/n < 1e-8
            return coef, vs, np.linalg.inv(info), rss/(n-p)
        obj = lambda log1p: reml(np.expm1(log1p))
        fit = minimize_scalar(obj, bounds=(0.,np.log1p(999)),method='bounded',options={'xatol':1e-8})
        if not fit.success or fit.x > np.log1p(998):
            raise ValueError('REML optimizer not adequately interior at working upper bound')
        assert obj(fit.x) <= min(obj(0.),obj(np.log1p(1.)),obj(np.log1p(20.))) + 1e-6
        lam = np.expm1(fit.x) if obj(fit.x)<obj(0.)-1e-7 else 0.
        c,resid,bread,resvar = reml(lam,True)
        vz,g = B['robust'](z,z*resid[:,None],bread,family)
        qas = dict(residual_variance=resvar, intercept_variance=lam*resvar, working_icc=lam/(1+lam),
                   variance_boundary=lam==0., reml_evaluations=fit.nfev, restricted_objective=reml(lam),
                   independent_gls_score_error=np.max(np.abs(z.T@resid))/n)
    else:
        c,vz,g,qas = B['fit_index'](z,y,pd.DataFrame({'cluster':family}),bounded)
    b, cov=back@c,back@vz@back.T
    assert np.allclose(x@b,z@c,rtol=1e-8,atol=1e-8) and np.isfinite(cov).all()
    assert np.linalg.eigvalsh(cov).min() >= -1e-9
    return b,cov,g,dict(design_condition=condition,**qas)


def prediction_matrix(ds, refs, parts, shared, year, hx):
    parts_out=[]
    for s in SOURCES:
        _, x0, _ = ds[s]
        ref=refs[s]
        j,k,_=parts[SOURCES.index(s)]
        target=ref.assign(hx=hx,time2=year-2,post=int(year>0),post_time2=(year-2)*int(year>0))
        x = np.asarray(patsy.build_design_matrices([x0.design_info],target,NA_action='raise')[0])
        parts_out.append((x[:,j],x[:,k]))
    return np.c_[np.concatenate([x for x,k in parts_out]),block_diag(*(k for x,k in parts_out))]


def difference(a,b):
    return a[0]-b[0],a[1]-b[1]


def result(m, cov, critical, limits=None):
    value,grad=m
    v=grad@cov@grad
    assert np.isfinite(v) and v>=-1e-9
    se=float(np.sqrt(max(v,0)))
    lo, hi = value+np.array([-1,1])*critical*se
    if limits:
        if not limits[0]<value<limits[1]:
            raise ValueError('standardized fitted mean outside allowable bounds')
        if limits==(0,7):
            p=value/7
            lo,hi=7*expit(np.log(p/(1-p))+np.array([-1,1])*critical*se/(7*p*(1-p)))
    return dict(estimate=value,se=se,lower=lo,upper=hi)


def references(ds, raw, domain, lo, hi):
    out={}
    field=DOMAINS[domain]
    for s in SOURCES:
        df, _, _=ds[s]
        d = df.loc[df.years.gt(0)].copy()
        panel=raw[s][0]
        qualified=[]
        keys=['country','index_wave'] if s=='SHARE' else ['index_wave']
        for group,x in d.groupby(keys,observed=True):
            group=group if isinstance(group,tuple) else (group,)
            history_n=x.groupby('hx').person.nunique().reindex([0,1]).fillna(0)
            bounds=x.groupby('hx').years.agg(['min','max']).reindex([0,1])
            if not (history_n.ge(10).all() and bounds['min'].le(lo+1e-8).all()
                    and bounds['max'].ge(hi-1e-8).all()):
                continue
            if s=='SHARE':
                country,wave=group
                p=panel.loc[panel.country.eq(country)]
                valid_end = p.loc[p[field].notna()].groupby('wave').month.median().max()
            else:
                wave=group[0]
                p=panel
                last=OLD['LAST'][s][list(DOMAINS).index(domain)]
                valid_end=p.loc[p.wave.eq(last),'month'].median()
            horizon=(valid_end-p.loc[p.wave.eq(wave),'month'].median())/12
            if not np.isfinite(horizon) or np.floor(horizon+.5)<hi:
                continue
            qualified.append(group)
        target=d.drop_duplicates('person')
        ref=target.loc[pd.MultiIndex.from_frame(target[keys]).isin(qualified)]
        if not ref.groupby('hx').person.nunique().reindex([0,1]).fillna(0).ge(10).all():
            return None
        out[s]=ref
    return out


def run():
    raw=dict(inputs())
    results,curves,support,diags,coefs=[],[],[],[],[]
    for comp in OLD['COMPARE']:
        for domain in DOMAINS:
            for include in (False,True):
                started=time.perf_counter()
                bounded=domain=='physical'
                ri=not include and not bounded
                shared=['hx','hx:post','hx:post_time2'] if include else ['hx','hx:time2']
                base=dict(comparison=comp,domain=domain,scale=UNITS[domain],
                          model='index_included' if include else 'strictly_post_index',
                          working_model='fractional logit' if bounded else 'Gaussian RI REML' if ri else 'Gaussian independence')
                try:
                    ds={s:initialize(s,raw[s],comp,domain,include) for s in SOURCES}
                    for s,(d,xx,formula) in ds.items():
                        for h in (0,1):
                            side=d.loc[d.hx.eq(h)]
                            post=side.loc[side.years.gt(0)]
                            enough=side.person.nunique()>=10
                            support.append(dict(**base,source=s,hx=h,people=safe_n(side.person.nunique()),
                                                families=safe_n(side.cluster.nunique()),post_records=safe_n(len(post)),
                                                index_records='0' if not include else safe_n(int(side.years.eq(0).sum())),
                                                observed_earliest_year=post.years.min() if enough else None,
                                                observed_latest_year=post.years.max() if enough else None,formula=formula))
                    x,parts,terms=make_blocks(ds,shared)
                    d=pd.concat([ds[s][0].assign(source=s) for s in SOURCES],ignore_index=True)
                    families=d.source+'|'+d.cluster.astype(str)
                    people=d.source+'|'+d.person.astype(str)
                    y=d.outcome.to_numpy(float)
                    assert (bounded and np.all((y>=0)&(y<=1))) or not bounded
                    b,cov,g,qa=estimate(x,y,families,people,bounded,ri)
                    status=dict(**base,status='estimated',n_people=people.nunique(),n_families=g,
                                n_records=len(d),n_post_records=int(d.years.gt(0).sum()),n_predictors=len(b))
                    diag=dict(**status, **qa)
                    critical=t.ppf(.975,g-1)
                    for k in range(len(shared)):
                        v=result((b[k],np.eye(len(b))[k]),cov,critical)
                        coefs.append(dict(**status,term=shared[k],model_scale='logit disease fraction' if bounded else UNITS[domain],**v))
                    # Aggregate parameter/covariance only, no participant design data.
                    model_dir=HERE/'LOCAL_MODELS'
                    model_dir.mkdir(exist_ok=True)
                    np.savez_compressed(model_dir/f'{comp}_{domain}_{status["model"]}.npz', beta=b, covariance=cov, terms=np.array(terms))
                    refs=None
                    for lo,hi in [(2,4),(2.5,3.5),(3,4),(2,3)]:
                        refs=references(ds,raw,domain,lo,hi)
                        if refs:
                            break
                    if refs is None:
                        diag.update(curve_status='four_source_fixed_range_not_supported')
                    else:
                        diag.update(curve_status='available',curve_from=lo,curve_to=hi)
                        ref_data=dict(reference_people=sum(map(len,refs.values())),
                                      reference_sources=';'.join(SOURCES),range_start=lo,range_end=hi,
                                      **{f'reference_{s}_people':len(refs[s]) for s in SOURCES})
                        predictions={}
                        for year in ([0] if include else [])+list(np.arange(lo,hi+.01,.25)):
                            for h in (0,1):
                                new=prediction_matrix(ds,refs,parts,shared,year,h)
                                eta=new@b
                                if bounded:
                                    p=expit(eta)
                                    m=(float(7*p.mean()),7*(p*(1-p))@new/len(new))
                                else:
                                    m=(float(eta.mean()),new.mean(axis=0))
                                predictions[(year,h)]=m
                            a,b0=(predictions[(year,h)] for h in (1,0))
                            items=[('mean_h1',a),('mean_h0',b0),('mean_difference_h1_minus_h0',difference(a,b0))]
                            if include and year>0:
                                c1=difference(a,predictions[(0,1)])
                                c0=difference(b0,predictions[(0,0)])
                                items.extend([('change_from_index_h1',c1),('change_from_index_h0',c0),
                                              ('between_history_change_difference',difference(c1,c0))])
                            for name,m in items:
                                limits=(0,7) if bounded and name.startswith('mean_h') else (0,100) if domain=='psychological' and name.startswith('mean_h') else None
                                curves.append(dict(**status,**ref_data,functional=name,year=year,
                                                   **result(m,cov,critical,limits)))
                        for r in curves:
                            if all(r[k]==status[k] for k in ('comparison','domain','model')) and r['year'] in (0,lo,hi):
                                results.append(r.copy())
                        dlo=difference(predictions[(lo,1)],predictions[(lo,0)])
                        dhi=difference(predictions[(hi,1)],predictions[(hi,0)])
                        m=difference(dhi,dlo)
                        avg=(m[0]/(hi-lo),m[1]/(hi-lo))
                        results.append(dict(**status,**ref_data,
                                            functional='average_change_in_history_gap_per_year',
                                            **result(avg,cov,critical)))
                    diags.append(dict(**diag,elapsed_seconds=time.perf_counter()-started))
                except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                    diags.append(dict(**base,status='not_estimable',detail=str(exc).split('\n')[0][:230],
                                      elapsed_seconds=time.perf_counter()-started))
                print(comp,domain,include,diags[-1]['status'],diags[-1].get('curve_status',''),flush=True)
    for rows,name in ((results,'RESULTS'),(curves,'CURVE_SOURCE'),(support,'SOURCE_SUPPORT'),
                      (diags,'DIAGNOSTICS'),(coefs,'MODEL_HISTORY_TERMS')):
        pd.DataFrame(rows).to_csv(HERE/f'{name}.csv',index=False,encoding='utf-8-sig')
    record=dict(spec_sha256=path_hash(HERE/'SPEC_CN.md'), script_sha256=path_hash(HERE/'run_continuous.py'),
                input_hashes=HASHES, pandas=pd.__version__,numpy=np.__version__,scipy=scipy.__version__,
                statsmodels=statsmodels.__version__,patsy=patsy.__version__)
    (HERE/'INPUT_HASHES.json').write_text(json.dumps(record,indent=2),encoding='utf-8')


if __name__=='__main__':
    run()

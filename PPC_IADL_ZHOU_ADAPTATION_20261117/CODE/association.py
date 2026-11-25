"""V7 IADL history associations with already locked common three-domain profiles.

Read only the new project inputs. The former V6 PPC outcome is never read. Classes and
labels are treated as fixed for inference; posterior draws are a class-allocation
sensitivity, not a refit of the longitudinal measurement model.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from scipy.stats import chi2, f, t
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationError, PerfectSeparationWarning


SOURCES = ('CHARLS', 'HRS', 'ELSA', 'SHARE')
CONTRASTS = (('H11', 'H01'), ('H10', 'H00'))
MIN_CELL = 10
DRAWS = 40
ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('person', 'source', 'cluster', 'country', 'index_wave', 'history', 'age', 'sex',
          'education', 'Kgroup', 'code_fixed', 'chronic_index', 'symptom_z_index', 'cognition_index')


class Gate(ValueError):
    """Prespecified support or numerical non-estimability; no placeholder numbers."""


def digest(path):
    with path.open('rb') as inp:
        return hashlib.file_digest(inp, 'sha256').hexdigest()


def save(rows, path, required):
    x = pd.DataFrame(rows)
    x.reindex(columns=list(dict.fromkeys([*required, *x.columns]))).to_csv(path, index=False, encoding='utf-8-sig')


def read_locked(root):
    lock = json.loads((root/'QA'/'CLASS_LOCK.json').read_text(encoding='utf-8-sig'))
    k = int(lock['K'])
    if lock['admitted'] is not True or k < 2:
        return None, k
    pfile = root/'PRIVATE'/'posterior.csv'
    if 'posterior_sha256' in lock and digest(pfile) != lock['posterior_sha256']:
        raise RuntimeError('Locked posterior SHA-256 differs; no association was fit')
    posterior_fields = [f'p{i}' for i in range(1, k+1)]
    posterior = pd.read_csv(pfile, dtype={'person': 'string'}, usecols=['person', *posterior_fields])
    idx = pd.read_parquet(root/'PRIVATE'/'index.parquet', columns=list(FIELDS))
    idx['person'] = idx.person.astype('string')
    if (not idx.person.is_unique or not posterior.person.is_unique or idx.person.isna().any()
            or posterior.person.isna().any() or set(idx.person) != set(posterior.person)):
        raise RuntimeError('Locked profiles and one-index person set do not match')
    z = idx.merge(posterior, on='person', how='inner', validate='one_to_one')
    u = z[posterior_fields].to_numpy(dtype=float)
    if not np.isfinite(u).all() or not np.allclose(u.sum(1), 1, atol=1e-7) or (u < 0).any():
        raise RuntimeError('Invalid locked posterior probabilities')
    if set(z.source) != set(SOURCES):
        raise RuntimeError('Expected all four fitted-input cohorts')
    if not z.cluster.dropna().empty and z.groupby('cluster').source.nunique().gt(1).any():
        raise RuntimeError('Family clustering was not namespaced by source')
    z['profile'] = np.argmax(u, axis=1)+1
    return z, k


def cohort_overlap(z, contrast):
    """Pre-profile overlap. Source-wave-state, plus current difficulty for H11/H01."""
    index, reference = contrast
    z = z.loc[z.history.isin(contrast)].copy()
    n_hist = len(z)
    if (z.source == 'SHARE').any() and z.loc[z.source == 'SHARE', 'country'].isna().any():
        z = z.loc[~((z.source == 'SHARE') & z.country.isna())].copy()
    for key in ('age', 'education', 'Kgroup', 'chronic_index', 'symptom_z_index', 'cognition_index'):
        z[key] = pd.to_numeric(z[key], errors='coerce').replace([np.inf, -np.inf], np.nan)
    z = z.dropna(subset=['source', 'cluster', 'index_wave', 'code_fixed', 'age', 'sex', 'education',
                          'Kgroup', 'chronic_index', 'symptom_z_index', 'cognition_index'])
    n_complete = len(z)
    if index == 'H11' and not z.Kgroup.isin([1, 2, 3]).all():
        raise RuntimeError('H11/H01 requires frozen current IADL count categories 1, 2, or 3+')
    if index == 'H10' and not z.Kgroup.eq(0).all():
        raise RuntimeError('H10/H00 requires no current IADL difficulty')
    z['hx'] = z.history.eq(index).astype(int)
    key = ['source', 'index_wave', 'code_fixed'] + (['Kgroup'] if index == 'H11' else [])
    common = z.groupby(key, observed=True).hx.transform('nunique').eq(2)
    masked = lambda n: '<10' if 0 < n < MIN_CELL else 10*round(n/10)
    flow = dict(trajectory_history_people_rounded=masked(n_hist),
                complete_confounder_people_rounded=masked(n_complete),
                overlap_people_rounded=masked(int(common.sum())))
    return z.loc[common].copy().reset_index(drop=True), flow


def design(z, contrast, layer):
    """Common age/severity effects, source-specific waves, education and SHARE countries."""
    n = len(z)
    age = (z.age.to_numpy(dtype=float)-65)/10
    matrices = [np.ones((n, 1)), z[['hx']].to_numpy(dtype=float), age[:, None], (age*age)[:, None]]
    labels = ['intercept', 'history', 'age10', 'age10_squared']

    def indicators(cat, label):
        a = pd.get_dummies(cat, prefix=label, drop_first=True, dtype=float)
        matrices.append(a.to_numpy()); labels.extend(a.columns)

    def nested(by, variable, label):
        for src in SOURCES:
            idx = by.eq(src).to_numpy()
            if idx.any():
                a = pd.get_dummies(variable.loc[idx], prefix=f'{label}:{src}', drop_first=True, dtype=float)
                mat = np.zeros((n, a.shape[1]))
                mat[idx] = a.to_numpy(); matrices.append(mat); labels.extend(a.columns)

    indicators(z.source, 'source')
    indicators(z.sex, 'sex')
    indicators(z.code_fixed, 'index PPC state')
    nested(z.source, z.index_wave, 'index wave')
    nested(z.source.where(z.source.ne('HRS')), z.education, 'education')
    nested(z.source.where(z.source.eq('SHARE')), z.country, 'SHARE country')
    if z.source.eq('HRS').any():
        vals = np.where(z.source.eq('HRS'), (z.education.to_numpy()-12)/4, 0)
        matrices.append(vals[:,None]); labels.append('HRS education years / 4')
    if contrast[0] == 'H11':
        indicators(z.Kgroup, 'current IADL count')
    if layer == 'severity':
        matrices.append(z[['chronic_index', 'symptom_z_index', 'cognition_index']].to_numpy(dtype=float))
        labels += ['chronic_index', 'symptom_z_index', 'cognition_index']
    x = np.column_stack(matrices)
    # Do not manufacture effects after rank failure. The only allowed adaptation is
    # dropping columns constant in the sample, which have no meaning in this model.
    keep = np.r_[True, True, np.ptp(x[:, 2:], axis=0) > 1e-12]
    return x[:, keep], list(np.asarray(labels)[keep])


def support(z, labels, outcome='profile'):
    grid = pd.MultiIndex.from_product(([0, 1], labels))
    count = z.groupby(['hx', outcome], observed=True).size().reindex(grid, fill_value=0)
    families = z.groupby(['hx', outcome], observed=True).cluster.nunique().reindex(grid, fill_value=0)
    return int(count.min()), int(families.min())


def estimate(y, x, cluster, labels, inference=True):
    n, p = x.shape
    k = len(labels)
    q = p*(k-1)
    ids, levels = pd.factorize(cluster, sort=True)
    g = len(levels)
    if n <= q or g <= q or np.linalg.matrix_rank(x) != p:
        raise Gate('insufficient_independent_clusters_or_design_rank')
    if not np.array_equal(np.unique(y), np.arange(k)):
        raise Gate('one_or_more_required_profiles_absent')
    try:
        with warnings.catch_warnings():
            for category in (ConvergenceWarning, PerfectSeparationWarning, RuntimeWarning):
                warnings.simplefilter('error', category)
            model = sm.MNLogit(y, x, check_rank=False)
            with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
                fit = model.fit(method='newton', disp=False, maxiter=100, tol=1e-9)
    except (FloatingPointError, np.linalg.LinAlgError, ConvergenceWarning, PerfectSeparationError,
            PerfectSeparationWarning, RuntimeWarning) as e:
        raise Gate('multinomial_separation_or_nonconvergence') from e
    b = np.asarray(fit.params).ravel(order='F')
    hess = -model.hessian(b)
    score = model.score_obs(b)
    score_error = np.max(np.abs(score.sum(0)))
    if (not fit.mle_retvals.get('converged', False) or not np.isfinite(b).all() or
            np.max(np.abs(b)) > 18 or not np.isfinite(hess).all() or score_error > 1e-4):
        raise Gate('no_reliable_finite_maximum')
    eig = np.linalg.eigvalsh(hess)
    if eig.min() <= max(eig.max()*1e-10, 1e-8):
        raise Gate('insufficient_likelihood_curvature')
    ind = 1+p*np.arange(k-1)
    params = b.reshape((p, k-1), order='F')
    x0, x1 = x.copy(), x.copy()
    x0[:, 1], x1[:, 1] = 0, 1

    def prob(v, mat):
        return softmax(np.column_stack((np.zeros(n), v @ mat)), axis=1)

    p0, p1 = prob(x0, params), prob(x1, params)
    m0, m1 = p0.mean(0), p1.mean(0)
    pdiff = m1-m0
    if abs(pdiff.sum()) > 1e-10:
        raise RuntimeError('Probability differences do not sum to zero')
    if not inference:
        return b[ind], pdiff
    hinv = np.linalg.solve(hess, np.eye(q))
    totals = np.zeros((g, q))
    np.add.at(totals, ids, score)
    unc = hinv @ (totals.T @ totals) @ hinv
    independent = sm.stats.sandwich_covariance.cov_cluster(fit, ids, use_correction=False)
    if not np.allclose(unc, independent, rtol=1e-6, atol=1e-8):
        raise RuntimeError('Family-cluster sandwich disagrees with statsmodels')
    variance = ((unc+unc.T)/2) * g/(g-1) * (n-1)/(n-q)
    cov_history = variance[np.ix_(ind, ind)]
    if np.linalg.eigvalsh(cov_history).min() <= 1e-12:
        raise Gate('singular_robust_history_variance')
    wald = b[ind] @ np.linalg.solve(cov_history, b[ind])
    se = np.sqrt(np.diag(cov_history))
    crit = t.ppf(.975, g-1)
    pv = 2*t.sf(abs(b[ind]/se), g-1)
    holm = multipletests(pv, method='holm')[1]

    def gradient(a, pr):
        return np.array([np.concatenate([a.T @ (pr[:,j]*((j==ell)-pr[:,ell]))/n
                                          for ell in range(1,k)]) for j in range(k)])

    grad = gradient(x1,p1)-gradient(x0,p0)
    direction = np.linspace(-.5,.5,q)
    h = (1e-5*direction).reshape((p,k-1),order='F')
    approx = (prob(x1,params+h).mean(0)-prob(x0,params+h).mean(0)-
              prob(x1,params-h).mean(0)+prob(x0,params-h).mean(0))/2e-5
    err = max(abs(approx-grad@direction))
    if err > 1e-7 or max(abs(grad.sum(0))) > 1e-10:
        raise RuntimeError('Probability-difference gradient QA failed')
    v = np.einsum('ij,jk,ik->i', grad, variance, grad)
    if (v <= 0).any() or not np.isfinite(v).all():
        raise Gate('singular_robust_probability_variance')
    pdse = np.sqrt(v)
    rows = []
    for j, profile in enumerate(labels):
        row = dict(profile=profile, reference=labels[0], p_comparator=m0[j], p_history=m1[j],
                   risk_diff_pp=100*pdiff[j], rd_lo_pp=100*(pdiff[j]-crit*pdse[j]),
                   rd_hi_pp=100*(pdiff[j]+crit*pdse[j]))
        if j:
            h, e = b[ind[j-1]], se[j-1]
            row.update(log_or=h, log_or_se=e, or_est=np.exp(h), or_lo=np.exp(h-crit*e),
                       or_hi=np.exp(h+crit*e), p_value=pv[j-1], p_holm_within_fit=holm[j-1])
        rows.append(row)
    diag = dict(n_parameters=q, wald_global=wald, f_global=wald/(k-1),
                df1=k-1, df2=g-1, p_global=f.sf(wald/(k-1),k-1,g-1),
                max_abs_score=score_error, min_information_eigenvalue=eig.min(),
                max_sandwich_abs_difference=np.max(np.abs(unc-independent)),
                max_gradient_difference=err, max_abs_parameter=np.max(np.abs(b)))
    return rows, diag


def fit_assoc(z, source, contrast, layer, labels, model_kind):
    comparison = f'{contrast[0]}_vs_{contrast[1]}'
    blocks = z[z.profile.isin(labels)].copy().reset_index(drop=True)
    min_people, min_families = support(blocks, labels)
    used = set(blocks.source)
    sid = dict(source=source, comparison=comparison, layer=layer, model_kind=model_kind,
               profile_set='|'.join(map(str,labels)), sources_used='|'.join(s for s in SOURCES if s in used))
    # Do not release complementary small numbers. On supported full or pair tables,
    # every class-history cell is at least the published disclosure floor.
    diag = dict(**sid, people_exact_or_rounded=(len(blocks) if min_people >= MIN_CELL else 10*round(len(blocks)/10)),
                min_class_history_people=min_people if min_people >= MIN_CELL else '<10',
                min_class_history_families=min_families if min_families >= MIN_CELL else '<10')
    if min_people < MIN_CELL or min_families < MIN_CELL:
        why = 'one_or_more_profile_by_history_cells_below_people_or_family_floor'
        return [dict(**sid,profile=c,reference=labels[0],status=why) for c in labels], dict(**diag,status=why)
    try:
        x, terms = design(blocks, contrast, layer)
        cats = pd.Categorical(blocks.profile, categories=labels).codes
        rows, d = estimate(cats, x, blocks.cluster.to_numpy(), labels)
    except Gate as e:
        why = str(e)
        return [dict(**sid,profile=c,reference=labels[0],status=why) for c in labels], dict(**diag,status=why)
    mat = dict(n=len(blocks), n_comparator=int(blocks.hx.eq(0).sum()), n_history=int(blocks.hx.eq(1).sum()),
               families=blocks.cluster.nunique())
    return [dict(**sid,**r,**mat,status='estimated') for r in rows], dict(**diag,**d,**mat,status='estimated',design='|'.join(terms))


def posterior_sensitivity(z, k, contrast, point_ok):
    """Fixed-shape categorical allocations; complete stage-1 uncertainty is not estimated."""
    row = dict(source='POOLED', comparison=f'{contrast[0]}_vs_{contrast[1]}', layer='severity',
               draws_requested=DRAWS, what='fixed_common_trajectory_posterior_reallocation_point_estimates')
    labels = list(range(1,k+1))
    if not point_ok:
        return [dict(**row, profile=j, draws_valid=0, status='not_run_full_model_not_estimable') for j in labels]
    rng = np.random.default_rng(73329 + (contrast[0]=='H10'))
    prob = z[[f'p{i}' for i in labels]].to_numpy()
    cum = np.cumsum(prob, axis=1)
    cum[:,-1] = 1
    x, _ = design(z, contrast,'severity')
    temp = z[['hx','cluster']].copy()
    effects, margins = [], []
    for _ in range(DRAWS):
        y = np.sum(rng.random((len(z),1)) > cum, axis=1)
        temp['profile'] = y+1
        if min(support(temp,labels)) < MIN_CELL:
            continue
        try:
            h, pdiff = estimate(y, x, z.cluster.to_numpy(), labels, inference=False)
        except Gate:
            continue
        effects.append(np.r_[np.nan,h]); margins.append(pdiff)
    results = []
    for j in labels:
        r = dict(**row, profile=j, draws_valid=len(effects),status='insufficient_valid_draws')
        if len(effects) >= .8*DRAWS:
            r['status'] = 'estimated_empirical_quantiles_not_CI'
            for key, val in [('or',np.exp(np.asarray(effects)[:,j-1])), ('rd_pp',100*np.asarray(margins)[:,j-1])]:
                if np.isfinite(val).all():
                    for qname, value in zip(('p025','p50','p975'),np.quantile(val,[.025,.5,.975])):
                        r[f'{key}_{qname}'] = value
        results.append(r)
    return results


def reml_mkh(y, v):
    """REML profile objective, mKH variance floor, t(k-2) PI as primary V6."""
    k = len(y)
    if k < 3 or not np.isfinite(y).all() or not np.isfinite(v).all() or (v<=0).any():
        raise Gate('fewer_than_three_comparable_finite_sources')
    def obj(u):
        w = 1/(v+u)
        m = w@y/w.sum()
        return .5*(np.log(v+u).sum()+np.log(w.sum())+w@((y-m)**2))
    upper = 100*max(np.ptp(y)**2, v.max(), .01)
    fit = minimize_scalar(obj, bounds=(0,upper),method='bounded',options={'xatol':1e-13})
    if not fit.success or fit.x>=.999*upper:
        raise Gate('REML_optimization_unreliable')
    tau = 0. if obj(0)<=fit.fun+1e-13 else fit.x
    w = 1/(v+tau)
    b = w@y/w.sum()
    raw = w@((y-b)**2)/(k-1)
    e = np.sqrt(max(1.,raw)/w.sum())
    wi = 1/v
    fixed = wi@y/wi.sum()
    Q = wi@((y-fixed)**2)
    vt = (k-1)*wi.sum()/(wi.sum()**2-(wi**2).sum())
    half = t.ppf(.975,k-1)*e
    hp = t.ppf(.975,k-2)*np.sqrt(tau+e*e)
    if max(abs(b-half),abs(b+half),abs(b-hp),abs(b+hp)) > 700:
        raise Gate('transformed_meta_interval_outside_finite_range')
    return dict(log_or=b, log_or_se_mkh=e, or_est=np.exp(b), or_lo=np.exp(b-half), or_hi=np.exp(b+half),
                p_value=2*t.sf(abs(b/e),k-1), tau2=tau, i2_pct=100*tau/(tau+vt), q=Q, q_p=chi2.sf(Q,k-1),
                q_raw_kh=raw, prediction_lo=np.exp(b-hp), prediction_hi=np.exp(b+hp), prediction_df=k-2)


def meta(rows, k):
    result = []
    for h in CONTRASTS:
        comparison = f'{h[0]}_vs_{h[1]}'
        for j in range(2,k+1):
            # A three-plus class pairwise estimate has a different model/sample.
            # Never mix it with full multinomial estimates or count any source twice.
            kinds = ['full_multinomial'] if k==2 else ['full_multinomial','pairwise_class_vs_1']
            for kind in kinds:
                kept = [r for r in rows if r['status']=='estimated' and r['source'] in SOURCES and
                        r['comparison']==comparison and r['profile']==j and r['layer']=='severity' and r['model_kind']==kind]
                sources = [r['source'] for r in kept]
                if len(sources) != len(set(sources)):
                    raise RuntimeError('Meta has more than one estimate per source')
                b = dict(comparison=comparison, profile=j, reference=1, model_kind=kind, k=len(sources),
                         sources='|'.join(sources), method='REML_modified_Knapp_Hartung')
                try:
                    vals = reml_mkh(np.array([r['log_or'] for r in kept]),np.array([r['log_or_se']**2 for r in kept]))
                    result.append(dict(**b, **vals,status='estimated'))
                except Gate as e:
                    result.append(dict(**b,status=str(e)))
    return result


def run(root):
    z, k = read_locked(root)
    rows, diagnostics, flow, draws, pooled = [], [], [], [], []
    if z is not None:
        labels = list(range(1,k+1))
        for contrast in CONTRASTS:
            cmp = f'{contrast[0]}_vs_{contrast[1]}'
            for s in ('POOLED',*SOURCES):
                q, details = cohort_overlap(z if s=='POOLED' else z.loc[z.source.eq(s)],contrast)
                flow.append(dict(source=s,comparison=cmp,**details))
                r, d = fit_assoc(q,s,contrast,'severity',labels,'full_multinomial')
                rows.extend(r); diagnostics.append(d)
                if s=='POOLED':
                    draws.extend(posterior_sensitivity(q,k,contrast,d['status']=='estimated'))
                    r, d = fit_assoc(q,s,contrast,'basic_same_records',labels,'full_multinomial')
                    rows.extend(r); diagnostics.append(d)
                if k>2:
                    for j in labels[1:]:
                        r,d = fit_assoc(q,s,contrast,'severity',[1,j],'pairwise_class_vs_1')
                        rows.extend(r); diagnostics.append(d)
        pooled = meta(rows,k)
        status = 'conditional_association_attempts_completed'
    else:
        reason = 'outcome_blind_class_lock_not_admitted_or_single_profile'
        for h in CONTRASTS:
            key = f'{h[0]}_vs_{h[1]}'
            rows.append(dict(source='POOLED',comparison=key,status=reason))
            pooled.append(dict(comparison=key,k=0,status=reason))
        status = reason
    table = root/'TABLES'
    table.mkdir(exist_ok=True)
    save(rows,table/'IADL_HISTORY_PROFILE_ASSOCIATIONS.csv',['source','comparison','layer','model_kind','profile','reference','status'])
    save(diagnostics,table/'IADL_HISTORY_ASSOCIATION_QA.csv',['source','comparison','layer','model_kind','status'])
    save(flow,table/'IADL_HISTORY_ASSOCIATION_FLOW.csv',['source','comparison'])
    save(draws,table/'IADL_HISTORY_POSTERIOR_SENSITIVITY.csv',['source','comparison','profile','status'])
    save(pooled,table/'IADL_HISTORY_PROFILE_META.csv',['comparison','profile','reference','model_kind','k','sources','status'])
    out = dict(status=status,K=k, computed_full_models=sum(d['status']=='estimated' and d['model_kind']=='full_multinomial' for d in diagnostics),
               computed_pairs=sum(d['status']=='estimated' and d['model_kind']=='pairwise_class_vs_1' for d in diagnostics),
               computed_meta=sum(d['status']=='estimated' for d in pooled),
               reports_only_aggregate=True)
    (root/'QA'/'ASSOCIATION_STATUS.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT)
    run(p.parse_args().root.resolve())

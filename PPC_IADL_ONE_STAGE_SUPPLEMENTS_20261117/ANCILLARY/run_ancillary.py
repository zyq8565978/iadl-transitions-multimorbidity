"""One-stage analysis of frozen person-window ancillary questions; no old writes."""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.special import expit

ROOT = Path(__file__).resolve().parent
WS = ROOT.parents[1]
OLD = WS/'PPC_IADL_ZHOU_ADAPTATION_20261117'
sys.path.insert(0, str(OLD))
import run_ipd as core
from firth_engine_v6 import fit, family_plan, flic_shift

SOURCES = core.SOURCES
SPECS = ('known_prior_reference', 'known_prior_adjusted', 'both_visits_below_PPC',
         'physical_only_psychological', 'physical_only_cognitive',
         'ADL_same_records_reference', 'ADL_adjusted',
         'prior_health_same_records_reference', 'prior_health_adjusted')
B = 199


def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def independent_columns(x):
    """Preserve intercept and H, and the exact nuisance column space."""
    u = np.linalg.qr(x[:, :2], mode='reduced')[0]
    rest = x[:, 2:]-u@(u.T@x[:, 2:])
    _, r, piv = qr(rest, mode='economic', pivoting=True)
    tol = max(rest.shape)*np.finfo(float).eps*max(1, abs(r).max())
    rank = int((abs(np.diag(r)) > tol).sum())
    cols = np.r_[0, 1, 2+np.sort(piv[:rank])]
    reduced = x[:, cols]
    assert np.linalg.matrix_rank(reduced) == reduced.shape[1]
    reconstruction = reduced@np.linalg.lstsq(reduced, x, rcond=None)[0]
    assert np.max(abs(x-reconstruction)) < 1e-7
    mapping = np.linalg.lstsq(reduced, x, rcond=None)[0]
    for h in (0, 1):
        target = x.copy()
        target[:, 1] = h
        if np.max(abs(target[:, cols]@mapping-target)) >= 1e-7:
            raise ValueError('contrast_not_identifiable_under_equivalent_design')
    return cols


def prepare(comp, spec):
    blocks, rows, terms, support, hashes = [], [], [], [], {}
    for source in SOURCES:
        a, d, stem = core.original(source, comp, 'M_severity')
        for f in (stem.with_suffix('.npz'), stem.with_name(stem.name+'_rows.parquet')):
            hashes[str(f.relative_to(WS))] = digest(f)
        d = d.copy()
        d['_original_row'] = np.arange(len(d))
        extra = []
        if spec.startswith('known_prior'):
            d = d.loc[d.d_fixed0.notna()].copy()
            d['prior_ppc'] = d.d_fixed0.ge(2).astype(float)
            if spec.endswith('adjusted'):
                extra = ['prior_ppc']
        elif spec == 'both_visits_below_PPC':
            d = d.loc[d.d_fixed0.lt(2)].copy()
        elif spec.startswith('physical_only'):
            d = d.loc[d.p0.eq(1)].copy()
            bit = 2 if spec.endswith('psychological') else 4
            d['y'] = (d.code_fixed2.astype(int)&bit != 0).astype(float)
        elif spec.startswith('ADL_'):
            d['adl_common'] = d['adl_any1' if source == 'SHARE' else 'adl_any']
            d = d.loc[d.adl_common.notna()].copy()
            if spec == 'ADL_adjusted':
                extra = ['adl_common']
        elif spec.startswith('prior_health'):
            d = d.dropna(subset=['cesd0', 'cog_z0']).copy()
            if spec.endswith('adjusted'):
                extra = ['cesd0', 'cog_z0']
        context = (['country'] if source == 'SHARE' else [])+['wave', 'p0']
        if comp == 'H11_vs_H01':
            context += ['kgrp']
        supported = d.groupby(context, observed=True).hx.nunique()
        take = pd.MultiIndex.from_frame(d[context]).isin(supported[supported.eq(2)].index)
        d = d.loc[take].copy()
        if d.empty:
            support.append(dict(source=source, specification=spec, comparison=comp,
                                windows=0, people=0, families=0, status='no_supported_rows'))
            continue
        names = list(a['terms'])
        nuisance = [i for i, n in enumerate(names) if n not in ('Intercept', 'hx')]
        block = a['X'][d._original_row.to_numpy()][:, nuisance]
        if extra:
            block = np.c_[block, d[extra].to_numpy(float)]
        blocks.append(block)
        terms += [source+':'+str(names[i]) for i in nuisance]+[source+':'+e for e in extra]
        d['source'] = source
        d['person'] = source+'|'+d.person.astype(str)
        d['family'] = source+'|'+d.cluster.astype(str)
        d['stratum'] = source+'|'+d.country.astype(str) if source == 'SHARE' else source
        rows.append(d)
        s = dict(source=source, specification=spec, comparison=comp, windows=len(d),
                 people=d.person.nunique(), families=d.family.nunique(), events=int(d.y.sum()), status='included')
        for h in (0, 1):
            z = d.loc[d.hx.eq(h)]
            s.update({f'H{h}_windows':len(z), f'H{h}_events':int(z.y.sum()), f'H{h}_nonevents':int((1-z.y).sum())})
        support.append(s)
    d = pd.concat(rows, ignore_index=True)
    src = pd.Categorical(d.source, categories=SOURCES).codes
    block = np.zeros((len(d), sum(b.shape[1] for b in blocks)))
    ri = ci = 0
    for b in blocks:
        n, k = b.shape
        block[ri:ri+n, ci:ci+k] = b
        ri += n; ci += k
    x = np.c_[np.ones(len(d)), d.hx.to_numpy(float), np.eye(len(SOURCES))[src, 1:], block]
    names = ['Intercept', 'hx']+['source:'+s for s in SOURCES[1:]]+terms
    cols = independent_columns(x)
    removed = [names[i] for i in range(x.shape[1]) if i not in cols]
    x = x[:, cols]
    names = [names[i] for i in cols]
    assert not d.duplicated(['person', 'window']).any()
    assert np.isfinite(x).all() and d.y.isin([0, 1]).all()
    assert (d.groupby('family').stratum.nunique() == 1).all()
    return d, x, src, support, removed, names, hashes


def corrected(d, y, x, src, stem):
    beta, diag = fit(x, y)

    def functional(b, w):
        eta = x@b
        shifts = np.zeros(len(SOURCES))
        for i in np.unique(src):
            take = (src == i)&(w > 0)
            shifts[i] = flic_shift(eta[take], y[take], w[take])
        neutral = eta-x[:, 1]*b[1]+shifts[src]
        p1, p0 = (np.average(expit(neutral+h*b[1]), weights=w) for h in (1, 0))
        return np.array([b[1], p1, p0, p1-p0])

    center = functional(beta, np.ones(len(d)))
    seed = 61117020+int.from_bytes(hashlib.sha256(stem.encode()).digest()[:3], 'big')
    plan, inv = family_plan(d.family.to_numpy(str), d.stratum.to_numpy(str), seed, B)
    out = np.full((B, 4), np.nan)
    errors = Counter()
    for i, draw in enumerate(plan):
        w = draw[inv].astype(float)
        try:
            try:
                b, _ = fit(x, y, w, warm=beta)
            except ValueError as exc:
                if str(exc) != 'design_rank_deficient':
                    raise
                cols = independent_columns(x[w > 0])
                bb, _ = fit(x[:, cols], y, w, warm=beta[cols])
                b = np.zeros(x.shape[1])
                b[cols] = bb
            out[i] = functional(b, w)
        except (ValueError, AssertionError, np.linalg.LinAlgError) as e:
            errors[str(e)] += 1
    good = out[np.isfinite(out).all(axis=1)]
    valid = len(good) >= 180
    lo, hi = np.quantile(good, [.025, .975], axis=0) if valid else (np.full(4, np.nan), np.full(4, np.nan))
    se = good.std(0, ddof=1) if valid else np.full(4, np.nan)
    np.savez_compressed(ROOT/'PRIVATE'/f'{stem}_model.npz', beta=beta, draws=out,
                        # No individual arrays in model summary; the local audit rows are separate.
                        seed=seed)
    return dict(**core.columns(center, lo, hi, se), status='estimated' if valid else 'point_only',
                estimator='one_stage_Firth_source_FLIC', inference='stratified family bootstrap percentile',
                bootstrap_attempted=B, bootstrap_valid=len(good), bootstrap_failures=json.dumps(errors), **diag)


def run(comp, spec):
    start = perf_counter()
    stem = comp+'_'+spec
    d, x, src, support, removed, terms, hashes = prepare(comp, spec)
    row = dict(comparison=comp, specification=spec, sources=';'.join(d.source.unique()),
               windows=len(d), people=d.person.nunique(), families=d.family.nunique(), events=int(d.y.sum()),
               rank=x.shape[1], estimator='not_fitted', status='not_estimable')
    y = d.y.to_numpy(float)
    cells = [int(((d.hx == h)&(d.y == v)).sum()) for h in (0, 1) for v in (0, 1)]
    if min(cells) >= 10:
        try:
            ans = core.ordinary(d, y, x, src, False)
            row.update(status='estimated', estimator='one_stage_MLE', **ans)
        except (ValueError, AssertionError, np.linalg.LinAlgError) as e:
            row['MLE_failure'] = str(e)[:200]
            try:
                row.update(corrected(d, y, x, src, stem))
            except (ValueError, np.linalg.LinAlgError) as ex:
                row['failure'] = str(ex)[:200]
    else:
        row['failure'] = 'pooled_arm_outcome_support_below10'
    row['elapsed_seconds'] = perf_counter()-start
    row['target'] = 'pooled supported observed-outcome person-window empirical distribution'
    audit = dict(input_hashes=hashes, rank=x.shape[1], terms=terms,
                 equivalent_redundant_columns=removed, same_family_not_split_between_sources=True)
    (ROOT/'QA'/f'{stem}.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    d[['source', 'person', 'family', 'window', 'hx', 'y']].to_parquet(ROOT/'PRIVATE'/f'{stem}_rows.parquet', index=False)
    for path, expected in hashes.items():
        assert digest(WS/path) == expected
    pd.DataFrame([row]).to_csv(ROOT/'TABLES'/f'{stem}.csv', index=False, encoding='utf-8-sig')
    # Coarse source support prevents recovering small source/arm/outcome cells.
    safe = pd.DataFrame(support)
    count_cols = ['windows', 'people', 'families', 'events']+[f'H{h}_{v}' for h in (0, 1) for v in ('windows', 'events', 'nonevents')]
    for c in count_cols:
        if c in safe:
            safe[c] = safe[c].map(lambda n: '' if pd.isna(n) else '0' if n == 0 else '<10' if n < 10 else str(int((n+5)//10)*10))
    safe.to_csv(ROOT/'TABLES'/f'{stem}_support.csv', index=False, encoding='utf-8-sig')
    print(stem, row['status'], row['estimator'], row.get('bootstrap_valid'), round(row['elapsed_seconds'], 2), flush=True)


def collect():
    paths = [p for p in (ROOT/'TABLES').glob('H*.csv') if not p.stem.endswith('_support')]
    pd.concat([pd.read_csv(p) for p in paths], ignore_index=True).to_csv(ROOT/'TABLES/ANCILLARY_RESULTS.csv', index=False, encoding='utf-8-sig')
    pd.concat([pd.read_csv(p) for p in (ROOT/'TABLES').glob('*_support.csv')], ignore_index=True).to_csv(ROOT/'TABLES/ANCILLARY_SOURCE_SUPPORT.csv', index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    for folder in ('PRIVATE', 'TABLES', 'QA'):
        (ROOT/folder).mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison', choices=core.COMPS)
    parser.add_argument('--specification', choices=SPECS)
    parser.add_argument('--collect', action='store_true')
    args = parser.parse_args()
    if args.collect:
        collect()
    else:
        for comp in ([args.comparison] if args.comparison else core.COMPS):
            for spec in ([args.specification] if args.specification else SPECS):
                run(comp, spec)
        collect()

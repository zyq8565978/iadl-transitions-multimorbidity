"""Bounded same-objective closure of failed full-data multinomial fits only.

Run after the scheduled suite. Successful fits and the recorded penalty are
immutable; the native score, information and sandwich checks remain in force.
"""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
import hashlib
import inspect
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def precision_minimize(fun, x0, *, context, trace, solver=minimize, **kwargs):
    result = solver(fun, x0, **kwargs)
    for stage in range(3):
        finite = bool(np.isfinite(result.x).all() and np.isfinite(result.fun)
                      and np.isfinite(result.jac).all())
        score = float(np.max(np.abs(result.jac))) if finite else None
        trace.append(dict(**context, stage=stage, objective=float(result.fun),
                          normalized_score=score, success=bool(result.success),
                          optimizer_status=int(result.status), message=str(result.message),
                          iterations=int(result.nit), finite=finite))
        if not finite or score < 2e-7 or stage == 2:
            return result
        continuation = dict(kwargs)
        continuation['options'] = dict(ftol=1e-15, gtol=1e-10, maxiter=1700,
                                       maxcor=20, maxls=60)
        candidate = solver(fun, result.x, **continuation)
        tolerance = 1e-12 * max(1., abs(float(result.fun)))
        if not np.isfinite(candidate.fun) or candidate.fun > result.fun + tolerance:
            trace.append(dict(**context, stage=stage+1,
                              disposition='OBJECTIVE_NONINCREASE_CHECK_FAILED',
                              objective_tolerance=tolerance,
                              candidate_objective=float(candidate.fun)))
            return result
        result = candidate


def self_test():
    calls = []
    def objective(x):
        return float(x @ x), 2*x
    def premature(fun, x0, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return OptimizeResult(x=np.array([.01]), fun=.0001, jac=np.array([.02]),
                                  success=True, status=0, message='test premature stop', nit=1)
        return minimize(fun, x0, **kwargs)
    trace = []
    result = precision_minimize(objective, np.zeros(1), context={'unit_test': True},
                                trace=trace, solver=premature, jac=True, method='L-BFGS-B',
                                options={'ftol': 1e-13})
    assert np.max(abs(result.jac)) < 2e-7 and 1 < len(calls) <= 3
    assert calls[1]['options']['ftol'] == 1e-15
    assert result.fun <= .0001 + 1e-12
    print('PASS: same-objective continuation and unchanged score gate')


def main():
    if (ROOT/'QA/MULTINOMIAL_PRECISION_CLOSURE.json').exists():
        print('Bounded numerical closure already recorded; no additional attempts.')
        return
    progress = json.loads((ROOT/'QA/ANALYSIS_PROGRESS.json').read_text())
    assert len(progress['completed']) == progress['total'] == 60
    assert not any(v['returncode'] for v in progress['completed'])
    targets = {}
    for receipt in sorted((ROOT/'QA').glob('*_mi???.json')):
        if not receipt.name.startswith(('TRAJECTORY_', 'FORWARD_')):
            continue
        record = json.loads(receipt.read_text())
        if record.get('reason') == 'OPTIMIZER_OR_GRADIENT_GATE':
            stem = receipt.name.rsplit('_mi', 1)[0]
            targets.setdefault(stem, []).append(int(receipt.stem[-3:]))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive = ROOT/'ATTEMPTS'/('multinomial_initial_precision_'+stamp)
    archive.mkdir(parents=True, exist_ok=False)
    unchanged = {}
    for stem, failed in targets.items():
        for path in (ROOT/f'TABLES/{stem}.csv', ROOT/f'QA/{stem}_estimator.json'):
            target = archive/path.parent.name/path.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(path, target)
            if path.suffix == '.json':
                unchanged[str(path.relative_to(ROOT))] = digest(path)
        for mi in range(1, 51):
            receipt = ROOT/f'QA/{stem}_mi{mi:03d}.json'
            if mi in failed:
                shutil.copy2(receipt, archive/'QA'/receipt.name)
            else:
                asset = ROOT/f'PRIVATE/{stem}_mi{mi:03d}.npz'
                for path in (receipt, asset):
                    if path.exists():
                        unchanged[str(path.relative_to(ROOT))] = digest(path)
    (archive/'unchanged_hashes.json').write_text(json.dumps(unchanged, indent=2))
    import multinomial_suite as model
    model.effect_gate()
    context = {}
    trace = []
    original_functional = model.functional
    # Restrict attempted imputations as well as stems; other failure types stay
    # untouched even when they occur in the same specification.
    run_source = inspect.getsource(model.run_one)
    needle = 'if p.exists():continue'
    assert run_source.count(needle) == 1
    run_source = run_source.replace(needle,
        'if p.exists() or mi not in closure_targets[stem]:continue')
    model.closure_targets = targets
    exec(compile(run_source, 'bounded_failed_imputation_adapter', 'exec'), model.__dict__)
    original_run = model.run_one
    def tracked_functional(*args, **kwargs):
        caller = inspect.currentframe().f_back
        context.clear()
        context.update(stem=caller.f_locals['stem'], imputation=int(caller.f_locals['mi']))
        return original_functional(*args, **kwargs)
    def tracked_minimize(fun, x0, **kwargs):
        return precision_minimize(fun, x0, context=context.copy(), trace=trace, **kwargs)
    def selected_run(d, y, x, k, stem, meta, lam):
        if stem in targets:
            return original_run(d, y, x, k, stem, meta, lam)
    model.functional = tracked_functional
    model.ns['minimize'] = tracked_minimize
    model.run_one = selected_run
    for comparison in model.COMPS:
        if 'TRAJECTORY_'+comparison in targets:
            model.trajectory(comparison)
        if any(stem.startswith('FORWARD_'+comparison+'_') for stem in targets):
            model.forward(comparison)
    for relative, original_hash in unchanged.items():
        assert digest(ROOT/relative) == original_hash, relative
    pd.DataFrame(trace).to_csv(ROOT/'TABLES/MULTINOMIAL_PRECISION_ATTEMPTS.csv', index=False)
    summary = []
    for stem, failed in targets.items():
        pooled = pd.read_csv(ROOT/f'TABLES/{stem}.csv').status.iloc[0]
        for mi in failed:
            record = json.loads((ROOT/f'QA/{stem}_mi{mi:03d}.json').read_text())
            summary.append(dict(stem=stem, imputation=mi,
                                initial_status='OPTIMIZER_OR_GRADIENT_GATE',
                                closure_status=record['status'], reason=record.get('reason'),
                                normalized_score=record.get('score'),
                                information_min=record.get('information_min'),
                                pooled_status=pooled))
    pd.DataFrame(summary).to_csv(ROOT/'TABLES/MULTINOMIAL_PRECISION_CLOSURE.csv', index=False)
    report = dict(completed_at=datetime.now(timezone.utc).isoformat(),
                  original_attempt_archive=str(archive.relative_to(ROOT)), targets=targets,
                  successful_assets_and_penalties_unchanged=True,
                  unchanged_native_score_gate=2e-7, maximum_continuations=2,
                  objective_tolerance='1e-12 * max(1, abs(previous normalized objective))',
                  bootstrap_dispositions_unchanged=True, summary=summary)
    (ROOT/'QA/MULTINOMIAL_PRECISION_CLOSURE.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
        # Separately recorded, same-objective closure of later observed
        # full-data Firth failures; bootstrap failures are not revisited.
        from close_full_data_firth import main as close_firth
        close_firth()

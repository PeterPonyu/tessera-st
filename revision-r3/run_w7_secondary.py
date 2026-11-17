"""W7: two descriptive secondary analyses of PROTOCOL-R3 section 8 on the new-cohort targets,
as specified in PROTOCOL-R3-DEVIATIONS item 15 before they were run.

1. Label-free scale rule: the frozen P1 rule applied to diagnostics of unsupervised expression
   clusters (K-means at the silhouette-selected K in 3..2K), scored against the reference target.
2. Magnitude model: least squares of g* on h_adj(6) and the neighbourhood information gain,
   fitted on the calibration targets, with two baselines (provenance class mean; K and class balance).

Outputs (results/revision-r3/W7/): labelfree.csv, magnitude.csv, summary.json, run-lock.json.
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'external'))
import common_r3 as c3  # noqa: E402
import diagnostics_r3 as dg  # noqa: E402
import prepare_freeze as pf  # noqa: E402

OUT = c3.OUT / 'W7'
SIL_SAMPLE = 3000
FIXED = ('no smoothing', 'k=6', 'k=30', 'k=200')


def unsupervised_labels(x: np.ndarray, k_ref: int) -> tuple[np.ndarray, int, dict]:
    """K-means (10 initializations, random state 0) at the silhouette-selected K in 3..2*k_ref."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    best, scores = None, {}
    for k in range(3, 2 * k_ref + 1):
        lab = KMeans(k, n_init=10, random_state=0).fit(x).labels_
        s = float(silhouette_score(x, lab, sample_size=min(SIL_SAMPLE, len(x)), random_state=0))
        scores[k] = s
        if best is None or s > best[0]:
            best = (s, k, lab)
    return best[2].astype(np.int64), best[1], scores


def one(u: dict, target: str) -> dict:
    k_ref = int(u['K'][target])
    lab, k_sel, sil = unsupervised_labels(u['x'], k_ref)
    prof = dg.homophily_profile(u['coords'], u['frames'], lab)
    ell = dg.correlation_length(u['coords'], u['frames'], lab, u['um_per_unit'])
    pred = dg.predictability(u['x'], u['coords'], u['frames'], lab)
    row = SimpleNamespace(h_adj6=float(prof.loc[prof.k == 6, 'h_adj'].iloc[0]), ell_k=ell['ell_k'],
                          info_gain_max=pred['info_gain_max'])
    return dict(dataset=u['dataset'], target=target, K_ref=k_ref, K_selected=k_sel, silhouette=sil[k_sel],
                h_adj6_free=row.h_adj6, ell_k_free=row.ell_k, ell_censored_free=ell['censored'],
                kappa_self_free=pred['kappa_self'], info_gain_max_free=row.info_gain_max,
                choice_free=pf.rule_choice(row))


def labelfree(w6: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    import run_w6_local as w6l
    us = w6l.units()
    jobs = [(u, t) for u in us if u['cohort'] in ('spinal', 'colorectal', 'wu_breast') for t in u['targets']]
    rows = Parallel(n_jobs=int(os.environ.get('R3_JOBS', '12')), verbose=5)(delayed(one)(u, t) for u, t in jobs)
    free = pd.DataFrame(rows)
    t = w6.merge(free, on=['dataset', 'target'], validate='one_to_one')
    regret = pf.regret_table(t)                                  # label rule and fixed choices
    t_free = t.assign(h_adj6=t.h_adj6_free, ell_k=t.ell_k_free, info_gain_max=t.info_gain_max_free)
    regret['label-free rule'] = pf.regret_table(t_free)['rule'].to_numpy()
    regret.columns = [f'regret_{c}' for c in regret.columns]
    t = pd.concat([t.reset_index(drop=True), regret.reset_index(drop=True)], axis=1)
    t['choice_rule'] = [pf.rule_choice(r) for r in t.itertuples()]
    unit = t.groupby(['cohort', 'dataset'])[[c for c in t.columns if c.startswith('regret_')]].mean()
    lf = unit['regret_label-free rule']
    summary = dict(
        n_units=int(len(unit)), n_targets=int(len(t)),
        mean_regret={c.removeprefix('regret_'): float(unit[c].mean()) for c in unit.columns},
        by_cohort={c: {k.removeprefix('regret_'): float(v) for k, v in g.mean().items()}
                   for c, g in unit.groupby(level=0)},
        free_smoothed_targets=int((t.choice_free > 0).sum()), rule_smoothed_targets=int((t.choice_rule > 0).sum()),
        smoothing_helped_targets=int((t.g_star > 0).sum()),
        free_agrees_with_rule_targets=int(((t.choice_free > 0) == (t.choice_rule > 0)).sum()),
        free_lower_than={c: int((lf < unit[f'regret_{c}']).sum()) for c in ('rule', *FIXED)},
        free_higher_than={c: int((lf > unit[f'regret_{c}']).sum()) for c in ('rule', *FIXED)},
        k_selected_over_k_ref=dict(zip(t.dataset + '|' + t.target, (t.K_selected / t.K_ref).round(3))))
    return t, summary


def class_balance(diag: pd.DataFrame) -> pd.Series:
    d6 = diag[diag.k == 6].set_index(['dataset', 'target'])
    return 1.0 - d6['chance']


def magnitude(w1: pd.DataFrame, w6: pd.DataFrame, d1: pd.DataFrame, d6: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cal = w1.set_index(['dataset', 'target']).assign(balance=class_balance(d1)).reset_index()
    new = w6.set_index(['dataset', 'target']).assign(balance=class_balance(d6)).reset_index()

    def ols(train, test, cols):
        tr = train.dropna(subset=[*cols, 'g_star'])
        X = np.column_stack([np.ones(len(tr)), tr[cols].to_numpy(float)])
        beta, *_ = np.linalg.lstsq(X, tr.g_star.to_numpy(float), rcond=None)
        Xt = np.column_stack([np.ones(len(test)), test[cols].to_numpy(float)])
        return Xt @ beta, beta, len(tr)

    pred_m, beta_m, n_m = ols(cal, new, ['h_adj6', 'info_gain_max'])
    pred_b2, beta_b2, n_b2 = ols(cal, new, ['K', 'balance'])
    class_mean = cal.groupby('provenance').g_star.mean()
    pred_b1 = new.provenance.map(class_mean).to_numpy(float)
    out = new[['dataset', 'cohort', 'target', 'provenance', 'K', 'balance', 'h_adj6', 'info_gain_max', 'g_star']].copy()
    out['pred_model'], out['pred_class'], out['pred_k_balance'] = pred_m, pred_b1, pred_b2
    summary = dict(n_train=int(n_m), n_test=int(len(out)),
                   coefficients=dict(intercept=float(beta_m[0]), h_adj6=float(beta_m[1]), info_gain=float(beta_m[2])),
                   coefficients_k_balance=dict(intercept=float(beta_b2[0]), K=float(beta_b2[1]), balance=float(beta_b2[2])),
                   class_means={k: float(v) for k, v in class_mean.items()})
    for name, col in (('model', 'pred_model'), ('provenance class', 'pred_class'), ('K and class balance', 'pred_k_balance')):
        err = out[col] - out.g_star
        summary[name] = dict(mae=float(err.abs().mean()), spearman=float(stats.spearmanr(out[col], out.g_star)[0]))
    return out, summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c3.write_lock(OUT, 'W7: label-free scale rule and magnitude model (secondary, descriptive)', frozen_protocol=True)
    w1 = pd.read_csv(c3.OUT / 'W1' / 'targets.csv')
    w6 = pd.read_csv(c3.OUT / 'W6' / 'targets.csv')
    w6 = w6[w6.cohort.isin(['spinal', 'colorectal', 'wu_breast'])].reset_index(drop=True)
    d1 = pd.read_csv(c3.OUT / 'W1' / 'diagnostics.csv')
    d6 = pd.read_csv(c3.OUT / 'W6' / 'diagnostics.csv')
    mag, mag_summary = magnitude(w1, w6, d1, d6)
    mag.to_csv(OUT / 'magnitude.csv', index=False)
    lf, lf_summary = labelfree(w6)
    lf.to_csv(OUT / 'labelfree.csv', index=False)
    summary = dict(labelfree=lf_summary, magnitude=mag_summary)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'k_selected_over_k_ref'} for k, v in summary.items()},
                     indent=1))


if __name__ == '__main__':
    main()

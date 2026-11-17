"""Evaluate the frozen predictions of PROTOCOL-R3 and summarise the round-3 analyses.

P1  diagnostic scale rule versus fixed choices on the new cohorts (regret; one-sided
    Wilcoxon signed-rank, Holm across four comparisons).
P2  construction-scale dose-response on the NSCLC cells (thresholds of section 4/7).
P3  each encoder at its patient-excluded tuned configuration minus its depth-matched
    propagation and minus its same-graph mean, per target class; cohort-balanced mean,
    95% t-interval and one-sided tests against the SESOI, Holm across six comparisons.
Calibration summaries (W1-W3, W1b) are reported alongside as exploratory.

Writes results/revision-r3/summary/summary.json. Every number in the manuscript is read
from this file through revision-r3-macros.tex (make_macros_r3.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_r3 as c3  # noqa: E402
import prepare_freeze as pf  # noqa: E402

R = c3.OUT
NEW_COHORTS = ('spinal', 'colorectal', 'wu_breast')
REGION = {'pathologist', 'histology-anatomical', 'geometry-derived', 'computed-niche', 'dominant-category'}
DEPTH = {'graphst': 2, 'stagate': 1, 'tessera': 3}


def holm(p: dict) -> dict:
    keys = sorted(p, key=p.get)
    out, running = {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (len(keys) - i) * p[k]))
        out[k] = running
    return out


def target_class(prov: str) -> str:
    return 'region' if prov in REGION else 'celltype'


# ----------------------------------------------------------------------------- P1
def p1(w6: pd.DataFrame) -> dict:
    t = w6[w6.cohort.isin(NEW_COHORTS)].reset_index(drop=True)
    reg = pf.regret_table(t)
    reg['unit'] = t.dataset.values
    reg['cohort'] = t.cohort.values
    unit = reg.groupby(['cohort', 'unit']).mean(numeric_only=True)
    res = {'n_units': int(len(unit)), 'n_targets': int(len(t)),
           'mean_regret': {c: float(unit[c].mean()) for c in reg.columns if c not in ('unit', 'cohort')},
           'median_regret': {c: float(unit[c].median()) for c in reg.columns if c not in ('unit', 'cohort')},
           'by_cohort': {c: unit.loc[c].mean(numeric_only=True).round(4).to_dict() for c in unit.index.levels[0]
                         if c in unit.index.get_level_values(0)}}
    pvals = {}
    for fixed in ('no smoothing', 'k=6', 'k=30', 'k=200'):
        d = unit['rule'] - unit[fixed]
        pvals[fixed] = float(stats.wilcoxon(d, alternative='less', zero_method='zsplit').pvalue) if (d != 0).any() else 1.0
        res.setdefault('rule_better_units', {})[fixed] = int((d < 0).sum())
    res['p'] = pvals
    res['p_holm'] = holm(pvals)
    res['holds'] = bool(all(v < 0.05 for v in res['p_holm'].values()))
    return res


# ----------------------------------------------------------------------------- P2
def p2(w1b: pd.DataFrame, sesoi: float) -> dict:
    rho = {}
    for unit, g in w1b.groupby('unit'):
        rho[unit] = dict(ell=float(stats.spearmanr(g.k_c, g.ell_k).statistic),
                         k_star=float(stats.spearmanr(g.k_c, g.k_star).statistic))
    margin = w1b.groupby('k_c').tessera_minus_sgm.mean()
    return dict(rho=rho, ell_pass=int(sum(v['ell'] >= 0.9 for v in rho.values())),
                kstar_pass=int(sum(v['k_star'] >= 0.8 for v in rho.values())),
                margin_by_kc={int(k): float(v) for k, v in margin.items()}, sesoi=sesoi,
                holds=bool(sum(v['ell'] >= 0.9 for v in rho.values()) >= 4
                           and sum(v['k_star'] >= 0.8 for v in rho.values()) >= 4 and margin.max() <= sesoi),
                gain_peak_k=w1b.groupby('k_c').info_gain_k.median().astype(int).to_dict())


# ----------------------------------------------------------------------------- P3
def encoder_units(w4: pd.DataFrame, sel: pd.DataFrame, cp: pd.DataFrame, w6: pd.DataFrame) -> pd.DataFrame:
    prov = w6.set_index(['dataset', 'target']).provenance
    km = w4[(w4.rule == 'kmeans') & (w4.input_mode == 'common') & (w4.arm == 'released')]
    enc = km.groupby(['encoder', 'dataset', 'target', 'config']).ari.mean()
    cpk = cp[cp.rule == 'kmeans'].groupby(['encoder', 'dataset', 'target', 'config', 'L']).ari.mean()
    rows = []
    for r in sel.itertuples():
        if r.cohort not in NEW_COHORTS:
            continue
        key = (r.encoder, r.dataset, r.target, r.selected)
        if key not in enc.index:
            continue
        L = DEPTH[r.encoder]
        prop = cpk.get((*key, L), np.nan)
        mean1 = cpk.get((*key, 1), np.nan)
        rows.append(dict(encoder=r.encoder, cohort=r.cohort, dataset=r.dataset, target=r.target,
                         target_class=target_class(prov.get((r.dataset, r.target), 'expression-assigned')),
                         encoder_ari=enc[key], propagation_ari=prop, same_graph_mean_ari=mean1,
                         minus_propagation=enc[key] - prop, minus_same_graph_mean=enc[key] - mean1))
    return pd.DataFrame(rows)


def balanced_test(d: pd.DataFrame, column: str, sesoi_by_cohort: dict) -> dict:
    unit = d.groupby(['cohort', 'dataset'])[column].mean().dropna()
    cohorts = unit.index.get_level_values(0).unique()
    means = {c: unit.loc[c].mean() for c in cohorts}
    var = {c: unit.loc[c].var(ddof=1) / len(unit.loc[c]) if len(unit.loc[c]) > 1 else np.nan for c in cohorts}
    est = float(np.mean(list(means.values())))
    se = float(np.sqrt(np.nansum(list(var.values()))) / len(cohorts))
    df = max(1, len(unit) - len(cohorts))
    tcrit = stats.t.ppf(0.975, df)
    sesoi = float(np.mean([sesoi_by_cohort[c] for c in cohorts]))
    p_sup = float(stats.t.sf((est - sesoi) / se, df)) if se > 0 else np.nan
    p_inf = float(stats.t.sf((-sesoi - est) / se, df)) if se > 0 else np.nan
    return dict(estimate=est, se=se, df=int(df), ci=(est - tcrit * se, est + tcrit * se), sesoi=sesoi,
                n_units=int(len(unit)), n_cohorts=int(len(cohorts)), cohort_means={c: float(v) for c, v in means.items()},
                units_positive=int((unit > 0).sum()), p_encoder_superior=p_sup, p_counterpart_superior=p_inf)


def p3(units: pd.DataFrame, sesoi_by_cohort: dict) -> dict:
    out = {}
    for cls, g in units.groupby('target_class'):
        tests = {}
        for encoder, ge in g.groupby('encoder'):
            for column, name in (('minus_propagation', 'propagation'), ('minus_same_graph_mean', 'same_graph_mean')):
                tests[f'{encoder}|{name}'] = balanced_test(ge, column, sesoi_by_cohort)
        adj_sup = holm({k: v['p_encoder_superior'] for k, v in tests.items()})
        adj_inf = holm({k: v['p_counterpart_superior'] for k, v in tests.items()})
        for k, v in tests.items():
            v['p_holm_encoder_superior'], v['p_holm_counterpart_superior'] = adj_sup[k], adj_inf[k]
            v['verdict'] = ('encoder superior' if adj_sup[k] < 0.05 else
                            'counterpart superior' if adj_inf[k] < 0.05 else 'no resolvable difference')
        out[cls] = tests
    return out


def sesoi_new(w6: pd.DataFrame) -> dict:
    t = w6[w6.cohort.isin(NEW_COHORTS)]
    return {c: float(max(0.01, 0.1 * g.groupby('dataset').headroom.mean().mean())) for c, g in t.groupby('cohort')}


def main():
    out = R / 'summary'
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((R / 'freeze' / 'freeze_numbers.json').read_text())
    summary = {'freeze': freeze}
    summary['P2'] = p2(pd.read_csv(R / 'W1b' / 'targets.csv'), freeze['sesoi']['NSCLC|primary']['sesoi'])
    w6_path = R / 'W6' / 'targets.csv'
    if w6_path.exists():
        w6 = pd.read_csv(w6_path)
        summary['P1'] = p1(w6)
        summary['sesoi_new'] = sesoi_new(w6)
        w4 = R / 'W4'
        if (w4 / 'scores.csv').exists() and (w4 / 'counterparts.csv').exists():
            units = encoder_units(pd.read_csv(w4 / 'scores.csv'), pd.read_csv(w4 / 'selection.csv'),
                                  pd.read_csv(w4 / 'counterparts.csv'), w6)
            units.to_csv(out / 'p3_units.csv', index=False)
            summary['P3'] = p3(units, summary['sesoi_new'])
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
    for key in ('P1', 'P2', 'P3'):
        if key in summary:
            print(key, json.dumps(summary[key], default=float)[:1500])


if __name__ == '__main__':
    main()

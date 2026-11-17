"""Evaluate the frozen predictions of PROTOCOL-R3 and summarise the round-3 analyses.

P1  diagnostic scale rule versus fixed choices on the new cohorts (regret; one-sided
    Wilcoxon signed-rank, Holm across four comparisons).
P2  construction-scale dose-response on the NSCLC cells (thresholds of section 4/7).
P3  each encoder at its patient-excluded tuned configuration minus its depth-matched
    propagation and minus its same-graph mean, per target class; cohort-balanced mean,
    mean over units, 95% t-interval and one-sided tests against the SESOI, Holm across six comparisons.
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
ALPHA_ONE_SIDED = 0.025


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
                         target_class=target_class(prov.loc[(r.dataset, r.target)]),
                         encoder_ari=enc[key], propagation_ari=prop, same_graph_mean_ari=mean1,
                         minus_propagation=enc[key] - prop, minus_same_graph_mean=enc[key] - mean1))
    return pd.DataFrame(rows)


def balanced_test(d: pd.DataFrame, column: str, sesoi_by_cohort: dict) -> dict:
    """95% t-interval over units (section 6.1 and the frozen power code; deviation 17): seeds and targets
    are averaged within each unit, then the mean over units, SE = SD / sqrt(n), n - 1 degrees of freedom.
    The SESOI of a class that pools cohorts is the unit-weighted mean of the cohort SESOIs."""
    unit = d.groupby(['cohort', 'dataset'])[column].mean().dropna()
    cohorts = unit.index.get_level_values(0).unique()
    means = {c: unit.loc[c].mean() for c in cohorts}
    est = float(unit.mean())
    se = float(unit.std(ddof=1) / np.sqrt(len(unit))) if len(unit) > 1 else np.nan
    df = max(1, len(unit) - 1)
    tcrit = stats.t.ppf(0.975, df)
    sesoi = float(np.mean([sesoi_by_cohort[c] for c in unit.index.get_level_values(0)]))
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
            # Section 6.1 decides from a two-sided 95% interval against +/-SESOI, which is a one-sided
            # test at 0.025 in each direction; Holm is applied within each direction (deviation 14).
            v['verdict'] = ('encoder superior' if adj_sup[k] < ALPHA_ONE_SIDED else
                            'counterpart superior' if adj_inf[k] < ALPHA_ONE_SIDED else 'no resolvable difference')
        out[cls] = tests
    return out


# ----------------------------------------------------------------------------- P3, descriptive
CALIBRATION_GROUPS = {('NSCLC', 'primary'): 'LungReg', ('NSCLC', 'cell_type'): 'LungCt', ('BCC', 'primary'): 'Bcc',
                      ('HER2ST', 'all'): 'HerAll', ('HER2ST', 'boundary_T1'): 'HerTumor'}


def _balanced(d: pd.DataFrame, column: str) -> dict:
    """Mean over targets within a unit, over units within a cohort, then over cohorts (descriptive)."""
    unit = d.groupby(['cohort', 'dataset'])[column].mean().dropna()
    if unit.empty:
        return dict(estimate=None, n_units=0, units_positive=0)
    means = unit.groupby(level=0).mean()
    return dict(estimate=float(means.mean()), n_units=int(len(unit)), units_positive=int((unit > 0).sum()),
                cohort_means={c: float(v) for c, v in means.items()})


def p3_descriptive(w4: pd.DataFrame, sel: pd.DataFrame, cp: pd.DataFrame, w6: pd.DataFrame,
                   sesoi_by_cohort: dict) -> dict:
    """Component removals, propagation depth and Leiden replication in the new cohorts, and the
    calibration cohorts; every encoder at the configuration selected for that unit and target."""
    prov = w6.set_index(['dataset', 'target']).provenance
    common = w4[w4.input_mode == 'common']
    enc = common.groupby(['rule', 'encoder', 'dataset', 'target', 'config', 'arm']).ari.mean()
    cpm = cp.groupby(['rule', 'encoder', 'dataset', 'target', 'config', 'L']).ari.mean()
    rows = []
    for r in sel.itertuples():
        if r.cohort == 'colorectal_pilot':
            continue
        key = (r.encoder, r.dataset, r.target, r.selected)
        if ('kmeans', *key, 'released') not in enc.index:
            continue
        base = enc[('kmeans', *key, 'released')]
        row = dict(encoder=r.encoder, cohort=r.cohort, dataset=r.dataset, target=r.target, released=base)
        if r.cohort in NEW_COHORTS:
            row['target_class'] = target_class(prov.loc[(r.dataset, r.target)])
        else:
            row['group'] = CALIBRATION_GROUPS.get((r.cohort, r.target))
        for arm in common[common.encoder == r.encoder].arm.unique():
            if arm != 'released' and ('kmeans', *key, arm) in enc.index:
                row[f'arm|{arm}'] = enc[('kmeans', *key, arm)] - base
        for L in sorted({1, 2, 4, 8, DEPTH[r.encoder]}):
            if ('kmeans', *key, L) in cpm.index:
                row[f'minus_L|{L}'] = base - cpm[('kmeans', *key, L)]
        if ('leiden', *key, 'released') in enc.index:
            lb = enc[('leiden', *key, 'released')]
            for name, L in (('propagation', DEPTH[r.encoder]), ('same_graph_mean', 1)):
                if ('leiden', *key, L) in cpm.index:
                    row[f'leiden|{name}'] = lb - cpm[('leiden', *key, L)]
        rows.append(row)
    d = pd.DataFrame(rows)
    out = {'arms': {}, 'depth': {}, 'leiden': {}, 'calibration': {}}
    new = d[d.cohort.isin(NEW_COHORTS)]
    for (cls, encoder), g in new.groupby(['target_class', 'encoder']):
        out['arms'].setdefault(cls, {})[encoder] = {c.split('|', 1)[1]: _balanced(g, c) for c in g.columns
                                                    if c.startswith('arm|') and g[c].notna().any()}
        out['depth'].setdefault(cls, {})[encoder] = {int(c.split('|')[1]): _balanced(g, c) for c in g.columns
                                                     if c.startswith('minus_L|') and g[c].notna().any()}
        out['leiden'].setdefault(cls, {})[encoder] = {c.split('|')[1]: balanced_test(g.dropna(subset=[c]), c, sesoi_by_cohort)
                                                      for c in ('leiden|propagation', 'leiden|same_graph_mean')
                                                      if c in g and g[c].notna().any()}
    cal = d[~d.cohort.isin(NEW_COHORTS) & d.group.notna()] if 'group' in d else d.iloc[:0]
    for (group, encoder), g in cal.groupby(['group', 'encoder']):
        unit = g.groupby('dataset')[sorted({f'minus_L|{DEPTH[encoder]}', 'minus_L|1'})].mean()   # one column for STAGATE
        out['calibration'].setdefault(group, {})[encoder] = {
            name: dict(estimate=float(unit[col].mean()), n_units=int(unit[col].notna().sum()),
                       units_positive=int((unit[col] > 0).sum()))
            for name, col in (('propagation', f'minus_L|{DEPTH[encoder]}'), ('same_graph_mean', 'minus_L|1'))}
    out['n_rows'] = int(len(d))
    return out, d


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
            desc, rows = p3_descriptive(pd.read_csv(w4 / 'scores.csv'), pd.read_csv(w4 / 'selection.csv'),
                                        pd.read_csv(w4 / 'counterparts.csv'), w6, summary['sesoi_new'])
            rows.to_csv(out / 'p3_descriptive_rows.csv', index=False)
            summary['P3_descriptive'] = desc
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
    for key in ('P1', 'P2', 'P3'):
        if key in summary:
            print(key, json.dumps(summary[key], default=float)[:1500])


if __name__ == '__main__':
    main()

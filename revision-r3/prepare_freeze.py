"""Numbers fixed at the protocol freeze, computed from calibration outputs only (W1-W3).

- SESOI per cohort: 10% of spatial headroom (mean over units of the best of the 30
  label-selected controls minus PCA50), floored at 0.01 ARI.
- Power: expected 95% half-width of a unit-level mean difference at the planned unit
  counts, from the calibration spread of the encoder-minus-counterpart differences.
- P1 rule: regret of the diagnostic scale rule and of fixed choices on the calibration
  targets (the rule's thresholds are chosen here and frozen).

Writes results/revision-r3/freeze/freeze_numbers.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_r3 as c3  # noqa: E402

GRID = (6, 10, 15, 30, 50, 100, 200)
PLANNED_UNITS = {'spinal cord (mice)': 10, 'colorectal, new patients': 4, 'colorectal, all patients': 7,
                 'breast Visium (Wu)': 6, 'pooled new cancer units': 10, 'pooled new units': 20}
GAIN_THRESHOLD, HADJ_FALLBACK = 0.05, 0.5


def rule_choice(row) -> int:
    """0 = no smoothing; otherwise the k-grid value nearest the label correlation length."""
    helps = row.info_gain_max > GAIN_THRESHOLD if np.isfinite(row.info_gain_max) else row.h_adj6 > HADJ_FALLBACK
    if not helps:
        return 0
    g = np.array(GRID)
    return int(g[np.abs(np.log2(g) - np.log2(max(row.ell_k, 6))).argmin()])


def regret_table(t: pd.DataFrame) -> pd.DataFrame:
    ari = {k: t[f'ari_mean{k}'].to_numpy() for k in GRID}
    pca = t.ari_pca50.to_numpy()
    best = np.maximum(pca, np.max(np.stack(list(ari.values())), 0))
    choices = {'rule': [rule_choice(r) for r in t.itertuples()], 'no smoothing': [0] * len(t)}
    for k in (6, 30, 200):
        choices[f'k={k}'] = [k] * len(t)
    out = {}
    for name, ch in choices.items():
        got = np.array([pca[i] if c == 0 else ari[c][i] for i, c in enumerate(ch)])
        out[name] = best - got
    return pd.DataFrame(out)


def main():
    out_dir = c3.OUT / 'freeze'
    out_dir.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(c3.OUT / 'W1' / 'targets.csv')
    grid = pd.read_csv(c3.OUT / 'W2' / 'grid.csv')
    res = {}

    # SESOI from headroom (calibration cohorts with the full 30-configuration grid)
    g = grid[~grid.config.isin(['SameGraphMean', 'Tessera'])].groupby(['cohort', 'target', 'unit', 'config']).kmeans.mean()
    head = {}
    for (cohort, target), gg in g.groupby(level=[0, 1]):
        per_unit = gg.groupby(level=2).max() - gg[gg.index.get_level_values(3).str.startswith('PCA50')].droplevel([0, 1, 3])
        head[f'{cohort}|{target}'] = dict(headroom=float(per_unit.mean()), sesoi=float(max(0.01, 0.1 * per_unit.mean())))
    res['sesoi'] = head

    # Power: spread of unit-level encoder-minus-counterpart differences in calibration
    w3 = pd.read_csv(c3.OUT / 'W3' / 'scores.csv')
    km = w3[w3.rule == 'kmeans']
    prop3 = km[(km.arm == 'Propagate') & (km.L == 3)].groupby(['dataset', 'target']).ari.mean()
    tess = t.set_index(['dataset', 'target']).ari_tessera
    sgm = t.set_index(['dataset', 'target']).ari_same_graph_mean
    diffs = {'Tessera-ST minus same-graph mean': (tess - sgm).dropna(),
             'Tessera-ST minus depth-matched propagation (L=3)': (tess - prop3.reindex(tess.index)).dropna()}
    power = {}
    for name, d in diffs.items():
        sd = float(d.std(ddof=1))
        power[name] = dict(sd_units=sd, half_width={k: float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
                                                     for k, n in PLANNED_UNITS.items()})
    res['power'] = power

    # P1 rule on the calibration targets
    reg = regret_table(t)
    res['p1_calibration'] = {name: dict(median=float(reg[name].median()), mean=float(reg[name].mean()),
                                        p90=float(reg[name].quantile(0.9)),
                                        n_le_002=int((reg[name] <= 0.02).sum()), n=int(len(reg)))
                             for name in reg.columns}
    res['p1_rule'] = dict(gain_threshold=GAIN_THRESHOLD, h_adj_fallback=HADJ_FALLBACK, grid=list(GRID))
    c3.dump(out_dir / 'freeze_numbers.json', res)
    print(pd.Series({k: v['sesoi'] for k, v in head.items()}).round(3).to_string())
    print(pd.DataFrame({k: v['half_width'] for k, v in power.items()}).round(3).to_string())
    print(pd.DataFrame(res['p1_calibration']).T.round(3).to_string())


if __name__ == '__main__':
    main()

"""Patient-excluded selection of tuned GraphST and STAGATE configurations (PROTOCOL-R3, section 5).

Reads the grid scores (released arm, common input, K-means, seeds 1-3) and, for every
encoder, cohort and target, gives each unit the configuration with the highest mean
ARI over the other units of its cohort, the rule used for the neighbourhood controls.
The colorectal cohort also selects over the three pilot patients. Writes
selection.csv (one row per encoder, unit and target) and selected_configs.json
(dataset -> encoder -> configurations to refit with every arm at seeds 1-5).

Usage: python select_external_r3.py --scores <W4>/scores.csv --out <W4>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import functools  # noqa: E402
read_csv = functools.partial(pd.read_csv, float_precision="round_trip")  # pandas' default float parser segfaults on some 17-digit values

POOL = {'colorectal_pilot': 'colorectal'}      # cohorts selected together


def select(scores: pd.DataFrame) -> pd.DataFrame:
    g = scores[(scores.arm == 'released') & (scores.input_mode == 'common') & (scores.rule == 'kmeans')
               & scores.seed.isin([1, 2, 3])].copy()
    g['pool'] = g.cohort.map(lambda c: POOL.get(c, c))
    rows = []
    for (encoder, pool, target), h in g.groupby(['encoder', 'pool', 'target']):
        means = h.groupby(['dataset', 'config']).ari.mean().unstack()
        n_seeds = h.groupby(['dataset', 'config']).seed.nunique().unstack()
        complete = (n_seeds == 3).all(axis=1)
        for dataset in means.index:
            others = means.drop(index=dataset)
            if not len(others):
                continue
            chosen = others.mean().idxmax()
            rows.append(dict(encoder=encoder, pool=pool, cohort=h[h.dataset == dataset].cohort.iloc[0],
                             dataset=dataset, target=target, selected=chosen, n_other_units=len(others),
                             grid_complete=bool(complete.get(dataset, False)),
                             heldout_ari_at_selected=float(means.loc[dataset, chosen])))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    sel = select(read_csv(args.scores))
    sel.to_csv(args.out / 'selection.csv', index=False)
    configs = {}
    for r in sel[sel.cohort != 'colorectal_pilot'].itertuples():
        configs.setdefault(r.dataset, {}).setdefault(r.encoder, set()).add(r.selected)
    (args.out / 'selected_configs.json').write_text(
        json.dumps({d: {e: sorted(c) for e, c in v.items()} for d, v in configs.items()}, indent=1) + '\n')
    print(sel.groupby(['encoder', 'pool', 'target']).selected.agg(lambda s: s.value_counts().to_dict()).to_string())


if __name__ == '__main__':
    main()

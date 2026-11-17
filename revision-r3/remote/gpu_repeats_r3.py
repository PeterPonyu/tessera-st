"""Same-seed GPU repeats of Tessera-ST (PROTOCOL-R3, section 6): computational noise on lung and HER2ST.

Released configuration, seeds 1-3, three repeats each, on the GPU; K-means ARI at the
reference class count for every target. Writes gpu_repeats.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

EXT = Path(__file__).resolve().parent.parent / 'external'
sys.path.insert(0, str(EXT))


def main() -> None:
    import common_external as ce
    import run_tessera_r3 as rt
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    rows = []
    for dataset in [*ce.c.LUNG, *ce.HER2ST]:
        ds = ce.load_dataset(dataset)
        for seed in (1, 2, 3):
            for repeat in (1, 2, 3):
                z = rt.fit(ds['x_common'], ds['coords'], ds['frames'], 'released', rt.DEFAULT_CONFIG, seed,
                           'common', 'cuda')['embedding']
                for target, y in ds['targets'].items():
                    ok = y >= 0
                    pred = KMeans(ds['K'][target], n_init=10, random_state=seed).fit(z).labels_
                    rows.append(dict(dataset=dataset, target=target, seed=seed, repeat=repeat,
                                     ari=float(adjusted_rand_score(y[ok], pred[ok]))))
        print('repeats', dataset, flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(args.out) / 'gpu_repeats.csv', index=False)


if __name__ == '__main__':
    main()

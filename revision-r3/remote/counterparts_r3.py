"""Parameter-free counterparts on each encoder's own graph (PROTOCOL-R3, section 5).

For every dataset, encoder and configuration that stage 2 refits, the graph saved with
the released fit at seed 1 gives the same-graph mean (one hop) and the propagation P^L x
at the encoder's depth L, plus L in {1, 2, 4, 8}. Each is scored with K-means at seeds 1-5
and Leiden at seed 1. Writes counterparts.csv in the W4 output folder.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

EXT = Path(__file__).resolve().parent.parent / 'external'
sys.path.insert(0, str(EXT))
DEPTH = {'graphst': 2, 'stagate': 1, 'tessera': 3}


DATASETS: dict = {}   # loaded once in the parent and handed to fresh (spawned) workers (see main)


def init_worker(datasets: dict) -> None:
    DATASETS.update(datasets)


def one(task):
    out, dataset, encoder, config = task
    import common_external as ce
    import graph_ops as go
    from score_external_r3 import leiden_partition
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    tag = dataset.replace('/', '__')
    hits = sorted((Path(out) / 'embeddings' / tag).glob(f'{encoder}-common-released-{config}-seed1.npz'))
    if not hits:
        return []
    with np.load(hits[0]) as f:
        src, dst = f['src'].astype(np.int64), f['dst'].astype(np.int64)
    ds = DATASETS.get(dataset) or ce.load_dataset(dataset)
    go.assert_within_frames(src, dst, ds['frames'])
    rows = []
    for L in sorted({1, 2, 4, 8, DEPTH[encoder]}):
        rep = go.propagate(ds['x_common'], src, dst, L).astype(np.float32)
        for target, y in ds['targets'].items():
            k, ok = ds['K'][target], y >= 0
            for s in (1, 2, 3, 4, 5):
                pred = KMeans(k, n_init=10, random_state=s).fit(rep).labels_
                rows.append(dict(dataset=dataset, encoder=encoder, config=config, L=L, depth=DEPTH[encoder],
                                 target=target, seed=s, rule='kmeans', ari=float(adjusted_rand_score(y[ok], pred[ok]))))
            pred = leiden_partition(rep, k, 1)[0]
            rows.append(dict(dataset=dataset, encoder=encoder, config=config, L=L, depth=DEPTH[encoder],
                             target=target, seed=1, rule='leiden', ari=float(adjusted_rand_score(y[ok], pred[ok]))))
    return rows


def tasks_for(out: str) -> list[tuple]:
    selected = json.loads((Path(out) / 'selected_configs.json').read_text())
    default = {'graphst': 'k3-e600', 'stagate': 'knn6-e1000', 'tessera': 'k6-w0.5-t0.2-e120'}
    return sorted({(out, d, e, c) for d, by in selected.items() for e, cs in by.items() for c in [*cs, default[e]]})


def main() -> None:
    """Shards replace the worker pool: run N independent processes with --shard i --nshards N, then --merge.
    Each shard loads only its datasets, checks them against the fits' fingerprint, and writes one file whole."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--processes', type=int, default=1, help='unused; kept so old calls still parse')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--merge', action='store_true', help='concatenate the shard files into counterparts.csv')
    args = ap.parse_args()
    out, tasks = Path(args.out), tasks_for(args.out)
    parts = out / 'counterparts-shards'
    if args.merge:
        files = sorted(parts.glob('shard-*.csv'))
        rows = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        covered = set(zip(rows.dataset, rows.encoder, rows.config))
        missing = [t for t in tasks if (t[1], t[2], t[3]) not in covered]
        rows.to_csv(out / 'counterparts.csv', index=False)
        print(f'{len(files)} shard files, {len(tasks)} graphs planned, {len(missing)} without rows, {len(rows)} rows')
        for t in missing[:20]:
            print('  missing', t[1:])
        if missing:
            raise SystemExit(1)
        return
    mine = tasks[args.shard::args.nshards]
    import common_external as ce
    for d in sorted({t[1] for t in mine}):
        rec = next((json.loads(p.read_text()) for p in sorted((out / 'fits' / d.replace('/', '__')).glob('*.json'))), None)
        for _ in range(3):
            ds = ce.load_dataset(d)
            if rec is None or ds['fingerprint'] == rec['dataset_fingerprint']:
                break
        else:
            raise ValueError(f'{d}: dataset differs from the one the encoders were fitted on')
        DATASETS[d] = {key: ds[key] for key in ('x_common', 'frames', 'targets', 'K', 'fingerprint')}
    rows = [r for task in mine for r in one(task)]
    parts.mkdir(exist_ok=True)
    final = parts / f'shard-{args.shard:02d}-of-{args.nshards:02d}.csv'
    tmp = final.with_suffix('.csv.tmp')
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(final)
    print(f'shard {args.shard}/{args.nshards}: {len(mine)} graphs, {len(rows)} rows')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Score saved WP4 embeddings; decoupled from fitting, runs in either environment.

For every fit record with status ok and every target of its dataset, at K = the reference class count:
  kmeans  KMeans(K, n_init=10, random_state=seed): the round-2 matched K-means
  leiden  revision-r1 run_k_sweep.leiden_partition: undirected 15-nearest-neighbour graph of the embedding,
          RBConfiguration Leiden, resolution bisected towards K (seed = the fit seed)
Both cluster every cell; ARI and NMI use labelled cells (label >= 0). The dataset is reloaded and its
fingerprint must equal the one recorded at fit time.
    python -B score_external_r3.py --processes 8
Writes scores/<dataset>/<tag>.csv per fit (reused while the record's embedding hash is unchanged) and scores.csv
with columns cohort, unit, dataset, target, encoder, arm, config, seed, rule, K, ari, nmi (the plan's score
schema), then input_mode, K_achieved, n_scored, embedding_sha256.
"""
from __future__ import annotations

import argparse
import functools
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import common_external as ce  # noqa: E402
from run_external_r3 import OUT  # noqa: E402
from run_k_sweep import leiden_partition  # noqa: E402  (revision-r1, on the path via common_external)

COLUMNS = ["cohort", "unit", "dataset", "target", "encoder", "arm", "config", "seed", "rule", "K", "ari", "nmi",
           "input_mode", "K_achieved", "n_scored", "embedding_sha256"]


@functools.lru_cache(maxsize=2)
def reference(dataset_id: str) -> dict:
    ds = ce.load_dataset(dataset_id)
    return {"targets": ds["targets"], "K": ds["K"], "fingerprint": ds["fingerprint"]}


def score_record(record: dict, out: Path, rules=("kmeans", "leiden")) -> list[dict]:
    ref = reference(record["dataset"])
    if ref["fingerprint"] != record["dataset_fingerprint"]:
        raise ValueError(f"{record['dataset']}: dataset changed since the fit")
    with np.load(out / record["embedding_file"]) as saved:
        embedding = saved["embedding"].astype(np.float32)
    if len(embedding) != record["n_cells"]:
        raise ValueError(f"{record['embedding_file']}: {len(embedding)} rows, record says {record['n_cells']}")
    seed, rows = int(record["seed"]), []
    for target, y in ref["targets"].items():
        k, ok = ref["K"][target], y >= 0
        partitions = {}
        if "kmeans" in rules:
            partitions["kmeans"] = KMeans(k, n_init=10, random_state=seed).fit(embedding).labels_
        if "leiden" in rules:
            partitions["leiden"] = leiden_partition(embedding, k, seed)[0]
        for rule, pred in partitions.items():
            rows.append(dict(cohort=record["cohort"], unit=record["unit"], dataset=record["dataset"], target=target,
                             encoder=record["encoder"], arm=record["arm"], config=record["config"], seed=seed,
                             rule=rule, K=k, ari=float(adjusted_rand_score(y[ok], pred[ok])),
                             nmi=float(normalized_mutual_info_score(y[ok], pred[ok])),
                             input_mode=record["input_mode"], K_achieved=int(len(np.unique(pred))),
                             n_scored=int(ok.sum()), embedding_sha256=record["embedding_sha256"]))
    return rows


def score_path(record_path: Path, out: Path) -> Path:
    return out / "scores" / record_path.parent.name / f"{record_path.stem}.csv"


def current(path: Path, record: dict, rules=("kmeans", "leiden")) -> bool:
    if not path.exists():
        return False
    saved = pd.read_csv(path)
    return set(saved.embedding_sha256) == {record["embedding_sha256"]} and set(rules) <= set(saved.rule)


def score_group(task) -> tuple[int, int]:
    out, record_paths, rules = task
    done = 0
    for record_path in record_paths:
        record = json.loads(record_path.read_text())
        target = score_path(record_path, out)
        if current(target, record, rules):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(score_record(record, out, rules), columns=COLUMNS).to_csv(target, index=False)
        done += 1
    return done, len(record_paths)


def limit_threads() -> None:
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    threadpool_limits(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--processes", type=int, default=1)
    ap.add_argument("--rules", nargs="+", default=["kmeans", "leiden"], choices=["kmeans", "leiden"],
                    help="clustering rules to score; a cached score counts as current only if it holds them all")
    args = ap.parse_args()
    out = Path(args.out)
    groups, failed = {}, 0
    for record_path in sorted((out / "fits").glob("*/*.json")):
        if json.loads(record_path.read_text())["status"] != "ok":
            failed += 1
            continue
        groups.setdefault(record_path.parent.name, []).append(record_path)
    tasks = [(out, paths, tuple(args.rules)) for paths in groups.values()]
    if args.processes > 1:
        with mp.Pool(args.processes, initializer=limit_threads) as pool:
            results = list(pool.imap_unordered(score_group, tasks))
    else:
        results = [score_group(task) for task in tasks]
    frames = [pd.read_csv(score_path(p, out)) for paths in groups.values() for p in paths]
    scores = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    scores = scores.sort_values(["encoder", "dataset", "input_mode", "arm", "config", "seed", "target", "rule"])
    scores.to_csv(out / "scores.csv", index=False)
    print(f"scored {sum(r[0] for r in results)} new of {sum(r[1] for r in results)} ok fits "
          f"({failed} failed fits skipped); {len(scores)} rows -> {out / 'scores.csv'}")


if __name__ == "__main__":
    main()

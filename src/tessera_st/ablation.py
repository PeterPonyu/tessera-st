"""Benchmark + ablation runner.

One table holds three things side by side:
  * reference baselines  — a non-spatial floor and a naive over-smoothing baseline;
  * the curated component grid — full + four leave-one-out + backbone;
  * a data-driven pick — the *best Tessera config on THIS data*, not an assumed "full".

Design stance: the four components are **hypotheses**, not commitments. "full" is never assumed
to be the final model. The ablation exists to falsify components and keep only what earns its
place; if a component does not help under honest evaluation, the selected config drops it.

Note on SOTA: the `backbone` row (no components) is a standard graph-autoencoder + clustering —
a brand-neutral stand-in for the graph spatial-domain encoder family. Parity against the *named*
methods (run from their own repos) is a real-DLPFC gate, not something synthetic data can settle.
"""

from __future__ import annotations

import time
from dataclasses import asdict, replace

import numpy as np

from tessera_st.config import TrainConfig, curated_ablation_grid
from tessera_st.eval import (
    ari,
    asw,
    boundary_f1,
    calinski_harabasz,
    chaos_score,
    davies_bouldin,
    expected_calibration_error,
    nmi,
    percentage_abnormal_spots,
    small_domain_iou,
)
from tessera_st.eval.markers import marker_purity
from tessera_st.train import fit_predict

# Metrics grouped by the *dimension* they probe. ARI/NMI are one family (label agreement);
# the value of the table is that the other rows are orthogonal axes.
METRIC_DIMENSIONS = {
    "label-agreement": ["ARI", "NMI"],
    "spatial-coherence": ["CHAOS", "PAS"],
    "geometric (internal)": ["ASW", "DBI", "CAL"],
    "boundary": ["boundary_F1"],
    "small-domain": ["small_IoU"],
    "calibration": ["ECE"],
    "marker-prior": ["marker_purity"],  # lower circularity but conditioned; not ground truth
}
HIGHER_IS_BETTER = {
    "ARI": 1, "NMI": 1, "ASW": 1, "CAL": 1, "boundary_F1": 1, "small_IoU": 1, "marker_purity": 1,
    "CHAOS": -1, "PAS": -1, "DBI": -1, "ECE": -1,
}
TABLE_COLS = ["config", "ARI", "NMI", "CHAOS", "PAS", "boundary_F1", "small_IoU", "ECE",
              "marker_purity"]


def _aligned(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Majority-vote map of predicted clusters onto true labels (per-spot correctness for ECE
    only; never used to score ARI/NMI)."""
    out = np.full_like(pred, -1)
    for c in np.unique(pred):
        mask = pred == c
        valid = true[mask][true[mask] >= 0]
        if len(valid):
            out[mask] = np.bincount(valid).argmax()
    return out


def _hard_confidence(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Post-hoc per-spot confidence for a hard clusterer, so its ECE is comparable.

    Soft-assigns each spot to cluster centroids by distance-softmax and takes the max
    probability. This is an uncalibrated, descriptive confidence proxy; it
    just lets the calibration axis include the baselines instead of leaving a blank cell.
    """
    clusters = np.unique(labels)
    cent = np.stack([features[labels == c].mean(axis=0) for c in clusters])
    d2 = ((features[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
    logits = -d2 / (np.median(d2) + 1e-8)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    q = e / e.sum(axis=1, keepdims=True)
    return q.max(axis=1)


def _metric_row(
    name: str,
    coords: np.ndarray,
    true: np.ndarray,
    pred: np.ndarray,
    embed: np.ndarray,
    confidence: np.ndarray | None,
    kind: str,
    flags: dict | None = None,
    note: str = "",
    layer_scores: dict | None = None,
) -> dict:
    """Score one labelling on every dimension. ECE is None for hard clusterers (no confidence);
    marker_purity is None when no external marker scores are available (e.g. synthetic data)."""
    row: dict = {"config": name, "kind": kind}
    if flags:
        row.update(flags)
    row.update(
        {
            "ARI": round(ari(true, pred), 4),
            "NMI": round(nmi(true, pred), 4),
            "CHAOS": round(chaos_score(coords, pred), 4),
            "PAS": round(percentage_abnormal_spots(coords, pred), 4),
            "ASW": round(asw(embed, pred), 4),
            "DBI": round(davies_bouldin(embed, pred), 4),
            "CAL": round(calinski_harabasz(embed, pred), 2),
            "boundary_F1": round(boundary_f1(coords, true, pred), 4),
            "small_IoU": round(small_domain_iou(true, pred), 4),
        }
    )
    if confidence is not None:
        valid = np.asarray(true) >= 0
        if np.asarray(confidence).shape != np.asarray(pred).shape:
            raise ValueError("confidence must match predicted labels")
        # Cluster numbers are arbitrary. Compare mapped predictions to true
        # labels, never to the original cluster numbers themselves.
        correct = (_aligned(pred, true)[valid] == np.asarray(true)[valid]).astype(float)
        row["ECE"] = (round(expected_calibration_error(np.asarray(confidence)[valid], correct), 4)
                      if valid.any() else None)
    else:
        row["ECE"] = None
    mp = marker_purity(pred, layer_scores) if layer_scores else None
    row["marker_purity"] = round(mp, 4) if mp is not None else None
    if note:
        row["note"] = note
    return row


def reference_baseline_rows(
    expr: np.ndarray,
    coords: np.ndarray,
    true_labels: np.ndarray,
    n_clusters: int,
    seed: int = 0,
    knn_k: int = 6,
    layer_scores: dict | None = None,
    field_ids=None,
) -> list[dict]:
    """Brand-neutral reference clusterers, scored on the same metrics as Tessera.

    - nonspatial-kmeans: clusters raw expression. The floor any spatial method must clear.
    - smoothed-kmeans:   averages features over neighbours then clusters. The transparent
                         over-smoothing baseline whose failure mode the edge gate targets.
    """
    from tessera_st.eval.baselines import expression_kmeans, mean_smoothed_kmeans
    from tessera_st.model.graph import build_knn_edges

    edge_index, _ = build_knn_edges(coords, k=knn_k, field_ids=field_ids)
    builders = [
        ("ref:nonspatial-kmeans", "non-spatial floor",
         lambda: expression_kmeans(expr, n_clusters, seed)),
        ("ref:smoothed-kmeans", "naive over-smoothing",
         lambda: mean_smoothed_kmeans(expr, edge_index, n_clusters, rounds=2, seed=seed)),
    ]
    rows = []
    for name, note, build in builders:
        t0 = time.perf_counter()
        labels = build()
        dt = time.perf_counter() - t0
        conf = _hard_confidence(expr, labels)  # post-hoc, so ECE is no longer blank
        row = _metric_row(name, coords, true_labels, labels, expr, conf,
                          kind="reference", note=note, layer_scores=layer_scores)
        row["runtime_s"] = round(dt, 3)
        rows.append(row)
    return rows


def run_ablation(
    expr: np.ndarray,
    coords: np.ndarray,
    true_labels: np.ndarray,
    n_clusters: int | None = None,
    train_cfg: TrainConfig | None = None,
    layer_scores: dict | None = None,
    field_ids=None,
) -> list[dict]:
    """Train every config in the curated grid and score it."""
    train_cfg = train_cfg or TrainConfig()
    n_clusters = n_clusters or int(len(np.unique(true_labels[true_labels >= 0])))
    rows: list[dict] = []
    for cfg in curated_ablation_grid():
        t0 = time.perf_counter()
        res = fit_predict(expr, coords, n_clusters, cfg, train_cfg=train_cfg,
                          field_ids=field_ids)
        dt = time.perf_counter() - t0
        row = _metric_row(
            cfg.name, coords, true_labels, res.labels, res.embed, res.confidence,
            kind="tessera", flags=cfg.as_dict(), layer_scores=layer_scores,
        )
        row["runtime_s"] = round(dt, 2)
        rows.append(row)
    return rows


def run_benchmark(
    expr: np.ndarray,
    coords: np.ndarray,
    true_labels: np.ndarray,
    n_clusters: int | None = None,
    train_cfg: TrainConfig | None = None,
    layer_scores: dict | None = None,
    field_ids=None,
) -> list[dict]:
    """Reference baselines + the Tessera ablation grid, in one table."""
    train_cfg = train_cfg or TrainConfig()
    n_clusters = n_clusters or int(len(np.unique(true_labels[true_labels >= 0])))
    rows = reference_baseline_rows(expr, coords, true_labels, n_clusters, seed=train_cfg.seed,
                                   layer_scores=layer_scores, field_ids=field_ids)
    rows += run_ablation(expr, coords, true_labels, n_clusters, train_cfg,
                         layer_scores=layer_scores, field_ids=field_ids)
    return rows


_METRIC_KEYS = ["ARI", "NMI", "CHAOS", "PAS", "ASW", "DBI", "CAL", "boundary_F1", "small_IoU",
                "ECE", "marker_purity", "runtime_s"]
_FLAG_KEYS = ["edge_gating", "multi_scale", "boundary_contrastive", "calibrated_uncertainty"]


def aggregate_seeds(per_seed_rows: list[list[dict]]) -> list[dict]:
    """Collapse repeated runs into mean + std per config per metric (stability axis)."""
    by_cfg: dict[str, list[dict]] = {}
    order: list[str] = []
    for seed_rows in per_seed_rows:
        for r in seed_rows:
            if r["config"] not in by_cfg:
                by_cfg[r["config"]] = []
                order.append(r["config"])
            by_cfg[r["config"]].append(r)
    out = []
    for name in order:
        rs = by_cfg[name]
        agg: dict = {"config": name, "kind": rs[0].get("kind"), "n_seeds": len(rs)}
        for fk in _FLAG_KEYS:
            if fk in rs[0]:
                agg[fk] = rs[0][fk]
        for k in _METRIC_KEYS:
            vals = [r[k] for r in rs if r.get(k) is not None]
            if vals:
                agg[k] = round(float(np.mean(vals)), 4)
                agg[k + "_std"] = round(float(np.std(vals)), 4)
            else:
                agg[k] = None
        out.append(agg)
    return out


def run_benchmark_multiseed(
    expr: np.ndarray,
    coords: np.ndarray,
    true_labels: np.ndarray,
    seeds: list[int],
    n_clusters: int | None = None,
    train_cfg: TrainConfig | None = None,
    layer_scores: dict | None = None,
    field_ids=None,
) -> list[dict]:
    """Run the full benchmark under several seeds and aggregate to mean±std."""
    train_cfg = train_cfg or TrainConfig()
    per_seed = [
        run_benchmark(expr, coords, true_labels, n_clusters, replace(train_cfg, seed=sd),
                      layer_scores=layer_scores, field_ids=field_ids)
        for sd in seeds
    ]
    return aggregate_seeds(per_seed)


SELECTION_VIEWS = ("ARI", "marker_purity", "CHAOS", "small_IoU", "ECE")


def select_config(rows: list[dict], key: str = "ARI") -> dict | None:
    """Best Tessera config under ONE metric — only ever a single partial view. Never treat its
    output as 'the' selected model; `selection_summary` shows several such views and their
    disagreement. None values are excluded (not treated as worst)."""
    cands = [r for r in rows if r.get("kind") == "tessera" and r.get(key) is not None]
    if not cands:
        return None
    sign = HIGHER_IS_BETTER.get(key, 1)
    return max(cands, key=lambda r: sign * r[key])


def selection_summary(rows: list[dict], views: tuple[str, ...] = SELECTION_VIEWS) -> str:
    """Multi-metric view: which config wins under several DIFFERENT metrics. Deliberately does
    NOT crown one 'data-selected' model or a single metric to steer by — every metric is partial,
    semi-circular and conditioned, so we expose the disagreement and triangulate instead."""
    tess = [r for r in rows if r.get("kind") == "tessera"]
    if not tess:
        return "no Tessera config to rank"
    lines, winners = [], []
    for v in views:
        b = select_config(rows, v)
        if b is not None and b.get(v) is not None:
            lines.append(f"  best by {v:<13}: {b['config']:<26} ({v}={b[v]})")
            winners.append(b["config"])
    n_distinct = len(set(winners))
    head = (
        f"multi-metric view — {n_distinct} different config(s) win under different metrics; "
        "there is NO single 'best'.\nEvery metric is partial / semi-circular / conditioned — do "
        "NOT steer training or selection by any one of them. Corroborate across views, then decide "
        "by which trade-offs matter:"
    )
    return head + "\n" + "\n".join(lines)


def format_table(rows: list[dict], cols: list[str] | None = None) -> str:
    """Render the table. '↑/↓' mark metric direction; '—' is an inapplicable metric."""
    cols = cols or TABLE_COLS
    has_std = any(any(str(k).endswith("_std") for k in r) for r in rows)
    metric_w = 16 if has_std else 11
    config_w = max(len("config"), max((len(str(r["config"])) for r in rows), default=6))

    def width(c: str) -> int:
        return config_w if c == "config" else metric_w

    def cell(r: dict, c: str) -> str:
        v = r.get(c)
        if v is None:
            return "—"
        sd = r.get(f"{c}_std")
        return f"{v}±{sd}" if sd is not None else str(v)

    def header(c: str) -> str:
        if c == "config":
            return c
        return f"{c}{'↑' if HIGHER_IS_BETTER.get(c, 1) > 0 else '↓'}"

    head = " | ".join(f"{header(c):>{width(c)}}" for c in cols)
    sep = "-|-".join("-" * width(c) for c in cols)
    lines = []
    prev_kind = None
    for r in rows:
        if prev_kind == "reference" and r.get("kind") != "reference":
            lines.append("-|-".join("-" * width(c) for c in cols))  # divider before Tessera rows
        lines.append(" | ".join(f"{cell(r, c):>{width(c)}}" for c in cols))
        prev_kind = r.get("kind")
    legend = (
        "\nref:* = baselines (non-spatial floor / naive over-smoothing) · backbone = standard "
        "graph-autoencoder (SOTA-class stand-in)\n"
        "baseline ECE uses post-hoc distance-softmax confidence (Tessera confidence is withheld; legacy "
        "cluster head); '±' columns are std over seeds when present\n"
        "ALL metrics are semi-circular AND conditioned — none is ground truth (ARI/NMI: GT is "
        "expression-derived; ASW/DBI: scored in the method's own embedding; CHAOS/PAS: smoothing is "
        "bias+judge, gameable by one big domain; marker_purity: LOWER circularity but conditioned on "
        "a marker prior that drifts across platform/sample). Each is one partial view — triangulate "
        "across all, enthrone none, and do not steer training by any single one.\n"
        "dimensions: ARI/NMI=label-agreement · CHAOS/PAS=spatial-coherence · ASW/DBI=geometric · "
        "boundary_F1=seam sharpness · small_IoU=small-domain recovery · ECE=calibration · "
        "marker_purity=marker-prior view (less circular, still conditioned)"
    )
    return f"{head}\n{sep}\n" + "\n".join(lines) + legend


def to_records(rows: list[dict], train_cfg: TrainConfig) -> dict:
    # store best-by-each-metric (not a single 'selected'), to keep the record multi-angle
    best_by = {}
    for v in SELECTION_VIEWS:
        b = select_config(rows, v)
        best_by[v] = b["config"] if b else None
    return {"train_config": asdict(train_cfg), "best_by_metric": best_by, "rows": rows}

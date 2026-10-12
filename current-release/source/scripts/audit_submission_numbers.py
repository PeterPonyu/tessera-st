#!/usr/bin/env python3
"""Re-derive saved cohort statistics without importing their producer functions.

This is a numerical audit of existing records, not a new experiment or replication.
"""
from __future__ import annotations
import itertools
import math
import numpy as np
import pandas as pd
from common import ROOT, read, dump, sha, OBJECTS


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def sign_flip(values):
    values = list(map(float, values))
    observed = abs(math.fsum(values) / len(values))
    statistics = [abs(math.fsum(s * d for s, d in zip(signs, values)) / len(values))
                  for signs in itertools.product((-1, 1), repeat=len(values))]
    return sum(value >= observed - 1e-12 for value in statistics) / len(statistics)


def interval(values):
    values = np.asarray(values, float)
    indices = np.random.RandomState(0).randint(len(values), size=(10000, len(values)))
    return np.quantile(values[indices].mean(axis=1), [.025, .975])


def check_contrast(record, values):
    values = np.asarray(values, float)
    close(values, record["deltas"])
    close(values.mean(), record["mean"])
    assert int((values > 0).sum()) == record["n_positive"]
    assert int((values < 0).sum()) == record["n_negative"]
    assert int((values == 0).sum()) == record["n_zero"]
    close(sign_flip(values), record["exact_sign_flip_p"])
    if "bootstrap95" in record:
        close(interval(values), record["bootstrap95"])


def main():
    hard_dir = ROOT / "validation-extension/results/hardening"
    receipt = ROOT / "validation-extension/results/run-hardening.json"
    result = read(receipt)["result"]
    arms = pd.read_csv(hard_dir / "arms.csv")
    grid = pd.read_csv(hard_dir / "grid.csv")
    seams = pd.read_csv(hard_dir / "seams.csv")
    assert not arms.duplicated(["unit", "target", "arm", "seed"]).any()
    assert not grid.duplicated(["unit", "target", "config", "seed"]).any()
    assert arms.groupby(["unit", "target", "arm"]).size().eq(3).all()
    assert grid.groupby(["unit", "target", "config"]).size().eq(3).all()
    assert arms.seed.isin([1, 2, 3]).all() and grid.seed.isin([1, 2, 3]).all()
    contrast_count, selection_count, cells = 0, 0, 0
    for target, record in result["targets"].items():
        scores = arms[arms.target == target].groupby(["unit", "arm"]).ari.mean().unstack()
        means = grid[grid.target == target].groupby(["unit", "config"]).ari.mean().unstack()
        assert means.shape == (8, 30)
        spatial = [name for name in means if not name.startswith("PCA50")]
        for unit in record["units"]:
            training = means.loc[[u for u in record["units"] if u != unit], spatial].mean()
            chosen = record["selected_control"][unit]
            close(training[chosen], training.max())
            scores.loc[unit, "NestedControl"] = means.loc[unit, chosen]
            scores.loc[unit, "PCA50"] = means.loc[unit, [c for c in means if c.startswith("PCA50")][0]]
            selection_count += 1
            for arm, value in record["per_unit"][unit].items():
                close(scores.loc[unit, arm], value)
                cells += 1
        for record in record["contrasts"].values():
            delta = scores.loc[record["units"], record["a"]] - scores.loc[record["units"], record["b"]]
            check_contrast(record, delta)
            contrast_count += 1
    seam_means = 0
    for target, record in result["seams"].items():
        scores = seams[seams.target == target].groupby("unit").mean(numeric_only=True)
        for unit, values in record["per_unit"].items():
            for name, expected in values.items():
                close(scores.loc[unit, name], expected)
        for name, expected in record["mean"].items():
            values = scores.loc[list(record["per_unit"]), name].to_numpy()
            close(values.mean(), expected["mean"])
            close(interval(values), expected["bootstrap95"])
            seam_means += 1
        for record in record["differences"].values():
            check_contrast(record, scores.loc[record["units"], record["a"]] - scores.loc[record["units"], record["b"]])
            contrast_count += 1
    land_path = ROOT / "source/manuscript/ledgers/landscape.csv"
    land = pd.read_csv(land_path).set_index("object")
    assert len(land) == 24 and land.index.is_unique
    primary_path = ROOT / "source/manuscript/ledgers/all_primary_scores.csv"
    primary = pd.read_csv(primary_path).groupby(["object", "arm"]).ari.mean().unstack()
    scale_path = ROOT / "results/followup-20261010/scale_controls.csv"
    scale = pd.read_csv(scale_path)
    for name in OBJECTS:
        sub = scale[scale.object == name]
        assert sub.groupby(["kind", "k", "lam"], dropna=False).ngroups == 30
        best = sub.groupby(["kind", "k", "lam"], dropna=False).ari.mean().max()
        for key, actual in [("tessera", primary.loc[name, "Tessera"]),
                            ("same_graph_mean", primary.loc[name, "NeighborMean"]),
                            ("pca50", primary.loc[name, "PCA50"]), ("oracle_control", best)]:
            close(land.loc[name, key], actual)
    for unit, values in result["targets"]["all"]["per_unit"].items():
        name = "HER2ST_" + unit
        best = grid[(grid.unit == unit) & (grid.target == "all")].groupby("config").ari.mean().max()
        for key, actual in [("tessera", values["Tessera"]), ("same_graph_mean", values["UniformSameGraph"]),
                            ("pca50", values["PCA50"]), ("oracle_control", best)]:
            close(land.loc[name, key], actual)
    breadth_path = ROOT / "validation-extension/results/breadth/grid.csv"
    breadth = pd.read_csv(breadth_path)
    b_receipt = ROOT / "validation-extension/results/run-breadth.json"
    for unit, values in read(b_receipt)["result"]["per_unit"].items():
        best = breadth[breadth.unit == unit].groupby("config").ari.mean().max()
        for key, actual in [("tessera", values["Tessera"]), ("same_graph_mean", values["UniformSameGraph"]),
                            ("pca50", values["PCA50"]), ("oracle_control", best)]:
            close(land.loc[unit, key], actual)
    for name, prefix in [("same_graph_mean", "same_graph"), ("pca50", "pca50"), ("oracle_control", "oracle")]:
        close(land.tessera - land[name], land["tessera_minus_" + prefix])
    sources = [receipt, b_receipt, land_path, primary_path, scale_path, breadth_path,
               hard_dir / "arms.csv", hard_dir / "grid.csv", hard_dir / "seams.csv"]
    dump(ROOT / "qa/submission_number_audit.json", {
        "status": "PASS", "cohort_targets": len(result["targets"]),
        "patient_excluded_selections": selection_count, "saved_patient_arm_cells": cells,
        "exact_sign_flip_and_bootstrap_contrasts": contrast_count,
        "seam_mean_and_interval_pairs": seam_means, "landscape_rows": len(land),
        "landscape_arm_cells": len(land) * 4,
        "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
        "audit_source_sha256": sha(__file__),
        "scope": "Independent numerical implementation over saved score tables; no refitting, new specimens or biological replication.",
        "bootstrap_scope": "Conditional on selected controls; no nested scale-selection resampling."})
    print(f"NUMBERS PASS: {selection_count} selections, {contrast_count} contrasts, {len(land)} landscape objects")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Label-free choice of CASTLE's expression-similarity fallback graph (E4).

CASTLE's README names a fallback "to expression-based similarity when labels
are uncertain or unavailable" without naming the mode or threshold. Its code
offers three expression-similarity modes for single-cell-resolution input,
with thresholds given only in code comments: 'pca_cosine_knn' 0.4-0.6,
'snn_jaccard' 0.01, 'adaptive_rbf' 0.2-0.3. This script builds each graph with
CASTLE's own functions on the exact features of the E4 runs (same
preprocessing as run_native.graph_adata, then CASTLE preprocess/get_feature)
and records node degrees. No labels are read.

Rule (fixed before any CASTLE fit was scored): use the first mode, in the
code's order, whose documented threshold (the most permissive one when a range
is given) leaves the median cell with at least one edge in every patient.
Output: results/revision-r1/E4/castle_fallback_graph_density.csv and .json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_common as nc  # noqa: E402
import run_native as rn  # noqa: E402

MODES = [("pca_cosine_knn", [0.4, 0.5, 0.6]), ("snn_jaccard", [0.01]), ("adaptive_rbf", [0.2, 0.3])]


def main() -> None:
    warnings.simplefilter("ignore")
    sys.path.insert(0, str(nc.CASTLE_PARENT))
    from CASTLE.preprocess import construct_interaction_celltype, get_feature, preprocess
    rows = []
    for cohort, patients in rn.PATIENTS.items():
        for patient in patients:
            inp = nc.load_input(cohort, patient)
            adata, _, _ = rn.graph_adata(inp)
            if "highly_variable" not in adata.var:
                preprocess(adata)
            get_feature(adata)  # feat and permuted feat_a, as in the CASTLE constructor
            for mode, thresholds in MODES:
                for thr in thresholds:
                    construct_interaction_celltype(adata, mode=mode, k_spatial=5, k_celltype=10, sim_threshold=thr,
                                                   weight1=False, weight2=False)
                    deg = np.asarray((adata.obsm["adj"] != 0).sum(1)).ravel()
                    rows.append({"cohort": cohort, "patient": patient, "mode": mode, "threshold": thr,
                                 "mean_degree": float(deg.mean()), "median_degree": float(np.median(deg)),
                                 "isolated_fraction": float((deg == 0).mean())})
            print(cohort, patient, flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(nc.E4 / "castle_fallback_graph_density.csv", index=False)
    chosen = None
    for mode, thresholds in MODES:
        permissive = min(thresholds)  # lower threshold keeps more edges in all three modes
        sub = table[(table["mode"] == mode) & (table.threshold == permissive)]
        if (sub.median_degree >= 1).all():
            chosen = {"mode": mode, "threshold": permissive}
            break
    record = {"rule": "first mode in code order whose most permissive documented threshold gives median degree >= 1 "
                      "in every patient; labels never read",
              "modes_in_code_order": [m for m, _ in MODES], "chosen": chosen}
    (nc.E4 / "castle_fallback_choice.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

"""Benchmark-table assembly — reference baselines + selection, all torch-free (fast CI)."""

import numpy as np

from tessera_st.ablation import format_table, reference_baseline_rows, select_config
from tessera_st.data.synthetic import make_tessellation


def _rows():
    s = make_tessellation(n_side=12, n_genes=16, seed=0)
    n = int(len(np.unique(s.labels)))
    return reference_baseline_rows(s.expr, s.coords, s.labels, n, seed=0), s


def test_reference_baselines_score_on_all_dimensions():
    rows, _ = _rows()
    assert len(rows) == 2
    assert {r["config"] for r in rows} == {"ref:nonspatial-kmeans", "ref:smoothed-kmeans"}
    for r in rows:
        assert r["kind"] == "reference"
        # baselines now carry a post-hoc distance-softmax ECE so the calibration axis is comparable
        assert isinstance(r["ECE"], float)
        assert isinstance(r["runtime_s"], float)
        for k in ["ARI", "NMI", "CHAOS", "PAS", "boundary_F1", "small_IoU", "ASW", "DBI"]:
            assert isinstance(r[k], float)


def test_format_renders_none_metric_as_dash():
    row = {"config": "x", "kind": "tessera", "ARI": 0.5, "NMI": 0.5, "CHAOS": 1.0,
           "PAS": 0.1, "boundary_F1": 0.5, "small_IoU": 0.2, "ECE": None}
    assert "—" in format_table([row])


def test_select_config_ignores_reference_rows():
    rows, _ = _rows()
    # only reference rows present -> no Tessera config to select
    assert select_config(rows) is None
    # a synthetic tessera row should win selection by ARI
    rows = rows + [{"config": "full", "kind": "tessera", "ARI": 0.99, "edge_gating": True,
                    "multi_scale": True, "boundary_contrastive": True,
                    "calibrated_uncertainty": True}]
    assert select_config(rows)["config"] == "full"

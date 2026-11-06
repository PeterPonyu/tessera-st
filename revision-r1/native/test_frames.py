"""Frame-separation tests for E4 (PROTOCOL-R1.md: test_no_cross_frame_edges).

Run in the tessera-native environment:
    python -B -m pytest -q test_frames.py      (or: python -B test_frames.py)

After the per-frame offset, every edge of the graphs that GraphST, BANKSY and
CASTLE build from the coordinates must join two cells of the same frame; the
generic 30-nearest-neighbour graph is checked as well. run_native.py repeats
the check on the graphs of every actual fit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import pdist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_common as nc  # noqa: E402

UNITS = [("NSCLC", p) for p in ["NSCLC_Lung5_Rep2", "NSCLC_Lung6", "NSCLC_Lung9_Rep1", "NSCLC_Lung12",
                                "NSCLC_Lung13"]] + \
        [("HER2ST", p) for p in ["A1", "B1", "C1", "D1", "E1", "F1", "G2", "H1"]]


def test_offset_separates_overlapping_frames():
    rng = np.random.RandomState(0)
    xy = rng.rand(300, 2) * 100
    coords = np.vstack([xy, xy + 0.5])
    frames = np.array(["a"] * 300 + ["b"] * 300)
    assert nc.knn_frame_check(coords, frames)["cross_frame_edges"] > 0  # the checker sees crossings
    shifted, info = nc.offset_frames(coords, frames)
    assert nc.knn_frame_check(shifted, frames)["cross_frame_edges"] == 0
    assert info["gap"] >= 50 * info["median_nn"]
    for f in ("a", "b"):
        ix = frames == f
        assert np.allclose(pdist(coords[ix]), pdist(shifted[ix]))  # translation only


@pytest.mark.parametrize("cohort,patient", UNITS)
def test_no_cross_frame_edges(cohort, patient):
    import anndata as ad
    from GraphST.preprocess import construct_interaction
    from banksy.initialize_banksy import initialize_banksy
    if str(nc.CASTLE_PARENT) not in sys.path:
        sys.path.insert(0, str(nc.CASTLE_PARENT))
    from CASTLE.preprocess import construct_interaction_celltype

    inp = nc.load_input(cohort, patient)
    frames = inp["frame"].astype(str)
    coords, info = nc.offset_frames(inp["coords"], frames)
    n = len(coords)
    for f in np.unique(frames):  # within-frame geometry unchanged
        ix = np.flatnonzero(frames == f)[:400]
        assert np.allclose(pdist(inp["coords"][ix]), pdist(coords[ix]))
    checks = {"knn30": nc.knn_frame_check(coords, frames, 30)}

    g = ad.AnnData(np.zeros((n, 1), np.float32))
    g.obsm["spatial"] = coords
    construct_interaction(g)  # GraphST, 3 nearest neighbours
    checks["GraphST_adj"] = nc.edge_frame_check(g.obsm["adj"], frames)
    checks["GraphST_graph_neigh"] = nc.edge_frame_check(g.obsm["graph_neigh"], frames)

    b = ad.AnnData(np.zeros((n, 1), np.float32))
    b.obs["x"], b.obs["y"] = coords[:, 0], coords[:, 1]
    b.obsm["spatial"] = coords
    bd = initialize_banksy(b, ("x", "y", "spatial"), 15, nbr_weight_decay="scaled_gaussian", max_m=1,
                           plt_edge_hist=False, plt_nbr_weights=False, plt_agf_angles=False, plt_theta=False)
    for m in (0, 1):  # m = 1 uses 2 x k_geom = 30 neighbours
        checks[f"BANKSY_weights_m{m}"] = nc.edge_frame_check(bd["scaled_gaussian"]["weights"][m], frames)

    # CASTLE: same features as the runs (run_native.graph_adata, then CASTLE preprocess/get_feature).
    import run_native as rn
    from CASTLE.preprocess import get_feature, preprocess
    c, _, _ = rn.graph_adata(inp)
    assert np.allclose(c.obsm["spatial"], coords)
    if "highly_variable" not in c.var:
        preprocess(c)
    if cohort == "NSCLC":
        c.obs["cell_type"] = inp["celltype_name"].astype(str)
    get_feature(c, cell_type="cell_type" if cohort == "NSCLC" else None)
    choice = json.loads((nc.E4 / "castle_fallback_choice.json").read_text())["chosen"]
    graphs = [("pca_cosine_knn", 0.4), ("snn_jaccard", 0.01), ("adaptive_rbf", 0.2)]
    assert (choice["mode"], choice["threshold"]) in graphs
    graphs += [("celltype", None)] if cohort == "NSCLC" else []
    for mode, thr in graphs:
        construct_interaction_celltype(c, mode=mode, cell_type="cell_type" if mode == "celltype" else None,
                                       k_spatial=5, k_celltype=10, sim_threshold=thr, weight1=False, weight2=False)
        for key in ("adj", "graph_neigh", "adj_a", "graph_neigh_a"):
            if key in c.obsm:
                checks[f"CASTLE_{mode}_{key}"] = nc.edge_frame_check(c.obsm[key], frames)
        for key in ("adj_a", "graph_neigh_a"):
            c.obsm.pop(key, None)

    print(cohort, patient, "frames", len(info["frames"]), "gap", round(info["gap"], 1),
          {k: v["n_edges"] for k, v in checks.items()})
    bad = {k: v for k, v in checks.items() if v["cross_frame_edges"]}
    assert not bad, bad
    # The graphs used by the runs must be non-empty, so their checks cannot pass vacuously.
    used = ["knn30", "GraphST_adj", "BANKSY_weights_m0", "BANKSY_weights_m1", f"CASTLE_{choice['mode']}_adj"]
    used += ["CASTLE_celltype_adj"] if cohort == "NSCLC" else []
    assert all(checks[k]["n_edges"] > 0 for k in used), {k: checks[k] for k in used}


if __name__ == "__main__":
    sys.exit(pytest.main(["-q", "-s", str(Path(__file__).resolve())]))

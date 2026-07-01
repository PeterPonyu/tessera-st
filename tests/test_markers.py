"""External marker anchor — torch-free, fast."""

import numpy as np

from tessera_st.eval.markers import compute_layer_scores, marker_purity


def _two_layer_fixture():
    n = 20
    expr = np.zeros((n, 4))
    expr[:10, 0:2] = 5.0  # first 10 spots express layer-A markers
    expr[10:, 2:4] = 5.0  # last 10 express layer-B markers
    names = ["a1", "a2", "b1", "b2"]
    md = {"LA": ["a1", "a2"], "LB": ["b1", "b2"]}
    return compute_layer_scores(expr, names, md)


def test_marker_purity_rewards_layer_aligned_domains():
    scores = _two_layer_fixture()
    aligned = np.array([0] * 10 + [1] * 10)       # domains match the marker blocks
    shuffled = np.array([0, 1] * 10)              # domains cut across markers
    assert marker_purity(aligned, scores) > marker_purity(shuffled, scores)


def test_marker_purity_none_without_scores():
    assert marker_purity(np.array([0, 1]), {}) is None


def test_compute_skips_absent_genes():
    expr = np.random.RandomState(0).randn(10, 2)
    scores = compute_layer_scores(expr, ["x", "y"], {"L": ["x", "missing"], "Z": ["nope"]})
    assert "L" in scores and "Z" not in scores  # Z has no present markers -> dropped


def test_marker_purity_not_gamed_by_one_big_domain():
    scores = _two_layer_fixture()
    one_domain = np.zeros(20, dtype=int)
    aligned = np.array([0] * 10 + [1] * 10)
    # degenerate single domain should not beat the genuinely layer-aligned split
    assert marker_purity(one_domain, scores) < marker_purity(aligned, scores)

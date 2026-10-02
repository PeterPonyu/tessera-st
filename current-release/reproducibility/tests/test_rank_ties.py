"""Regression for row-order invariant partial Spearman rank treatment."""
import ast
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

CAP = Path(__file__).resolve().parents[2]


def functions():
    tree = ast.parse((CAP / "experiments/stat_rigor.py").read_text())
    names = {"rankz", "partial_spearman", "partial_spearman_multi"}
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"np": np, "rankdata": rankdata}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "<stat_rigor functions>", "exec"), namespace)
    return namespace


def test_rankz_midrank_and_row_permutation_invariance():
    funcs = functions()
    raw = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1], float)
    z = funcs["rankz"](raw)
    assert len(set(z[raw == 0])) == len(set(z[raw == 1])) == 1
    rng = np.random.default_rng(7)
    for _ in range(100):
        order = rng.permutation(len(raw))
        np.testing.assert_allclose(funcs["rankz"](raw[order]), z[order])


def test_joint_partial_rho_is_order_invariant_with_tied_confounders():
    funcs = functions()
    x = np.array([.2, .1, .75, .4, .55, .8, .05, .3, .65, .95, .9])
    y = np.array([.1, .8, .65, .3, .35, .55, .15, .4, .9, .85, .7])
    k = np.array([2, 2, 7, 5, 5, 7, 3, 3, 3, 7, 2])
    dim = np.array([3000, 351, 155, 1020, 33, 36, 3000, 34, 3000, 3000, 32])
    mod = np.array([0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1])
    joint = funcs["partial_spearman_multi"](x, y, [k, dim, mod])
    one = funcs["partial_spearman"](x, y, mod)
    for order in (np.arange(11)[::-1], *[np.random.default_rng(s).permutation(11) for s in range(30)]):
        assert funcs["partial_spearman_multi"](x[order], y[order],
                                                  [k[order], dim[order], mod[order]]) == joint
        assert funcs["partial_spearman"](x[order], y[order], mod[order]) == one

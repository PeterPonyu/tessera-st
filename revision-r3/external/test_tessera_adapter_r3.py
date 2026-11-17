"""Tests of the Tessera-ST adapter (run with the dl environment, CPU)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import run_tessera_r3 as rt  # noqa: E402

torch.set_num_threads(1)


def toy(n=120, seed=3):
    rng = np.random.default_rng(seed)
    xy = rng.random((2 * n, 2)) * 10
    frames = np.repeat(["a", "b"], n)
    x = rng.normal(size=(2 * n, 10)).astype(np.float32)
    return x, xy, frames


def test_config_grid_and_parsing():
    assert len(rt.CONFIGS) == 24 and rt.DEFAULT_CONFIG in rt.CONFIGS
    assert rt.parse_full("k15-w1.0-t0.5-e200") == (15, 1.0, 0.5, 200)
    assert rt.parse_config(rt.DEFAULT_CONFIG) == (6, 120)
    assert rt.CONFIG_FORMAT.format(6, 120) == rt.DEFAULT_CONFIG


@pytest.mark.parametrize("arm", rt.ARMS)
def test_every_arm_fits_within_frames(arm):
    x, xy, frames = toy()
    out = rt.fit(x, xy, frames, arm, "k6-w0.5-t0.2-e120", 1)
    assert out["embedding"].shape == (len(x), 32) and np.isfinite(out["embedding"]).all()
    assert (frames[out["src"]] == frames[out["dst"]]).all()
    if rt.depth(arm) == 0:
        assert (out["src"] == out["dst"]).all()
    else:
        assert (out["src"] != out["dst"]).all()


def test_released_equals_round1_training_loop():
    import train_core as tc
    x, xy, frames = toy()
    a = rt.fit(x, xy, frames, "released", "k15-w1.0-t0.5-e120", 2)["embedding"]
    b = tc.train_core(x, xy, frames, rt.HEAD_K, "Tessera", 2, "cpu", epochs=120, graph_k=15,
                      contrastive_weight=1.0, contrastive_temp=0.5)["checkpoints"][120]["z"]
    assert np.array_equal(a, b)


def test_encoder_edges_match_fit():
    x, xy, frames = toy()
    src, dst = rt.encoder_edges(xy, frames, "released", "k15-w0.5-t0.2-e120", 1)
    out = rt.fit(x, xy, frames, "released", "k15-w0.5-t0.2-e120", 1)
    assert np.array_equal(src, out["src"]) and np.array_equal(dst, out["dst"])

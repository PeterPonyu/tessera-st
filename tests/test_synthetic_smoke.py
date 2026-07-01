"""End-to-end smoke: the deep model trains on a synthetic tessellation and produces
schema-valid outputs. Proves the machine runs; proves nothing biological."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tessera_st.config import AblationConfig, TrainConfig  # noqa: E402
from tessera_st.data.synthetic import make_tessellation  # noqa: E402
from tessera_st.train import fit_predict  # noqa: E402


def test_full_model_fit_predict_runs():
    slide = make_tessellation(n_side=16, n_genes=24, seed=0)
    n_clusters = int(len(np.unique(slide.labels)))
    cfg = TrainConfig(epochs=20, device="cpu", seed=0)
    res = fit_predict(slide.expr, slide.coords, n_clusters, AblationConfig(), train_cfg=cfg)

    n = slide.coords.shape[0]
    assert res.labels.shape == (n,)
    assert res.embed.shape[0] == n
    assert res.boundary.shape == (n,)
    assert np.all((res.boundary >= 0) & (res.boundary <= 1))
    assert np.all((res.confidence >= 0) & (res.confidence <= 1))
    assert len(np.unique(res.labels)) >= 2
    assert res.history  # loss was recorded


def test_loss_decreases():
    slide = make_tessellation(n_side=16, n_genes=24, seed=1)
    cfg = TrainConfig(epochs=80, device="cpu", seed=1)
    res = fit_predict(slide.expr, slide.coords, 5, AblationConfig(), train_cfg=cfg)
    losses = [h["loss"] for h in res.history]
    assert losses[-1] < losses[0]  # the model actually learns

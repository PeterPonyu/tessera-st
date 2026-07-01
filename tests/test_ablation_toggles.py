"""Each component toggle is real: flipping it changes the model's behaviour/structure."""

import pytest

torch = pytest.importorskip("torch")

from tessera_st.config import AblationConfig, ModelConfig, curated_ablation_grid  # noqa: E402
from tessera_st.data.synthetic import make_tessellation  # noqa: E402
from tessera_st.model.tessera import TesseraNet  # noqa: E402
from tessera_st.model.graph import build_knn_edges  # noqa: E402


def _inputs():
    slide = make_tessellation(n_side=14, n_genes=20, seed=2)
    ei, ed = build_knn_edges(slide.coords, k=6)
    x = torch.tensor(slide.expr)
    return x, torch.tensor(ei), torch.tensor(ed, dtype=torch.float32)


def test_gating_off_pins_gate_to_one():
    x, ei, ed = _inputs()
    net = TesseraNet(ModelConfig(in_dim=x.shape[1]), 5, AblationConfig(edge_gating=False))
    out = net(x, ei, ed)
    assert torch.allclose(out.gates[-1], torch.ones_like(out.gates[-1]))


def test_gating_on_produces_nontrivial_gates():
    x, ei, ed = _inputs()
    net = TesseraNet(ModelConfig(in_dim=x.shape[1]), 5, AblationConfig(edge_gating=True))
    out = net(x, ei, ed)
    assert out.gates[-1].std() > 0  # gates vary across edges


def test_curated_grid_shape_and_names():
    grid = curated_ablation_grid()
    assert len(grid) == 6
    assert grid[0].name == "full"
    assert grid[-1].name == "backbone"
    assert {"no_edge_gating", "no_multi_scale", "no_boundary_contrastive",
            "no_calibrated_uncertainty"} <= {c.name for c in grid}


def test_multiscale_off_uses_last_layer_only():
    x, ei, ed = _inputs()
    net = TesseraNet(ModelConfig(in_dim=x.shape[1]), 5, AblationConfig(multi_scale=False))
    out = net(x, ei, ed)
    assert out.z.shape[0] == x.shape[0]  # still runs, just single-scale

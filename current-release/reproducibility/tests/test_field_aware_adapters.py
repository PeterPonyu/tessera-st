"""Behaviour tests for the retained-panel adapters, not tests of favourable ARI."""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"src"),str(ROOT/"experiments")]
from field_aware_adapters import directed_graph, neighbor_mean, sedr_graph, stagate_input, external_embedding
from field_aware_panel_20260923 import expression_only, contiguity, fields_from_data


def toy():
    xy = np.tile(np.c_[np.arange(10), np.zeros(10)],(2,1))
    return xy, np.array(["a"]*10+["b"]*10)


def test_stagate_runtime_graph_keeps_frames_and_zero_distance_neighbors():
    import torch
    from _roots import external_root
    sys.path.insert(0,str(external_root()/"STAGATE"))
    from STAGATE_pyG.utils import Transfer_pytorch_Data
    xy, f = toy(); xy[1] = xy[0]
    a = stagate_input(np.zeros((20,3)), xy, f)
    edge = Transfer_pytorch_Data(a).edge_index.numpy()
    assert (f[edge[0]] == f[edge[1]]).all()
    assert any(i == 0 and j == 1 for i,j in edge.T)


def test_sedr_positive_negative_and_normalized_graph_stay_within_frame():
    xy,f = toy()
    graph = sedr_graph(xy,f,3)
    for key in ["adj_norm","adj_label","mask"]:
        t = graph[key].coalesce(); src,dst = t.indices().numpy()
        assert (f[src] == f[dst]).all()
    mask = graph["mask"].coalesce()
    src,dst = mask.indices().numpy()
    neg = mask.values().numpy() == 0
    positive = set(map(tuple,graph["adj_label"].indices().numpy().T))
    assert neg.any() and all((i,j) not in positive for i,j in zip(src[neg],dst[neg]))


def test_singleton_policy_and_frequency_adjusted_metric():
    xy=np.zeros((5,2)); f=np.array(["a","a","a","a","b"])
    Z=np.arange(10).reshape(5,2)
    np.testing.assert_array_equal(neighbor_mean(Z,xy,f)[4],Z[4])
    metric=contiguity(xy,f,np.array([0,0,1,1,2]))
    assert metric["labelled_singletons_excluded"] == 1
    assert metric["contiguity_cell_weighted"] == pytest.approx(1/3)
    assert metric["within_field_permutation_expectation"] == pytest.approx(1/3)
    assert metric["chance_adjusted"] == pytest.approx(0)


def test_expression_only_has_no_spatial_inputs_and_is_repeatable():
    import inspect
    assert list(inspect.signature(expression_only).parameters) == ["Z","k","seed"]
    Z=np.random.default_rng(1).normal(size=(24,5))
    np.testing.assert_array_equal(expression_only(Z,3,1),expression_only(Z,3,1))


def test_missing_or_multiple_implicit_library_frames_fail():
    import anndata as ad
    a=ad.AnnData(np.zeros((2,2)))
    with pytest.raises(ValueError): fields_from_data(a,"uns:spatial")
    a.uns["spatial"]={"a":{},"b":{}}
    with pytest.raises(ValueError): fields_from_data(a,"uns:spatial")


@pytest.mark.parametrize("method",["SpaceFlow","GraphST","SpaGCN"])
def test_unadapted_external_methods_cannot_silently_pool_fields(method):
    xy,f=toy()
    with pytest.raises(RuntimeError, match="Not run"):
        external_embedding(method,np.zeros((20,3)),xy,f,1)

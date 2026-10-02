"""Spatial identity, negative-control and result-recording regression checks."""
from pathlib import Path
import sys
import json
import ast
import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

CAP = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(CAP/'src'), str(CAP/'experiments')]
from tessera_st.spatial import neighbor_rows
from tessera_st.model.graph import build_knn_edges
from tessera_st.eval.refine import refine_labels
from coordinate_guard import require_single_coordinate_frame
import protocol_guard as guard


def test_single_frame_graph_and_refinement_preserve_original_geometry():
    xy = np.random.RandomState(3).normal(size=(40,2))
    dist, idx = NearestNeighbors(n_neighbors=7).fit(xy).kneighbors(xy)
    src, dst = np.repeat(np.arange(40),6), idx[:,1:].ravel()
    expected = np.stack([np.r_[dst,src],np.r_[src,dst]])
    edge, length = build_knn_edges(xy)
    np.testing.assert_array_equal(edge, expected)
    np.testing.assert_array_equal(length, np.tile(dist[:,1:].ravel(),2).astype(np.float32))
    labels = np.arange(40) % 3
    old = labels.copy()
    for _ in range(2):
        old = np.array([np.bincount(old[row]).argmax() for row in idx])
    np.testing.assert_array_equal(refine_labels(labels,xy),old)


def test_overlapping_local_coordinates_do_not_make_cross_field_edges():
    one = np.random.RandomState(4).normal(size=(12,2))
    xy = np.concatenate([one,one])
    fields = np.array(['a']*12+['b']*12)
    edge, dist = build_knn_edges(xy,field_ids=fields)
    assert len(dist) == 2*24*6 and np.all(fields[edge[0]]==fields[edge[1]])
    labels = np.array([0]*12+[1]*12)
    np.testing.assert_array_equal(refine_labels(labels,xy,field_ids=fields),labels)


def test_duplicate_coordinate_is_a_neighbor_but_self_is_not():
    xy = np.array([[0,0],[0,0],[0,0],[1,0]],float)
    idx, _ = neighbor_rows(xy,k=2)
    assert all(i not in row and len(row)==2 for i,row in enumerate(idx))
    assert 1 in idx[0] or 2 in idx[0]


def test_small_fields_cap_degree_and_singletons_keep_their_labels():
    xy = np.zeros((4,2)); fields = ['a','a','b','c']
    idx, _ = neighbor_rows(xy,field_ids=fields)
    assert [len(r) for r in idx] == [1,1,0,0]
    np.testing.assert_array_equal(refine_labels(np.array([2,2,3,4]),xy,field_ids=fields),[2,2,3,4])


@pytest.mark.parametrize('fields', [['a'], ['a',None],['a',float('nan')]])
def test_malformed_coordinate_identity_rejected(fields):
    with pytest.raises(ValueError):
        neighbor_rows(np.zeros((2,2)),field_ids=fields)


def test_legacy_adapters_fail_closed_on_real_frame_keys():
    import anndata as ad
    import pandas as pd
    for key in ['library_id','section_id','sample_Xtile_Ytile']:
        obj = ad.AnnData(X=np.zeros((2,1)),obs=pd.DataFrame({key:['a','b']},index=['0','1']))
        with pytest.raises(RuntimeError,match='COORDINATE HOLD'):
            require_single_coordinate_frame(obj,'toy')


def test_h5ad_adapter_preserves_explicit_frame_identity():
    import pandas as pd
    from tessera_st.data.dlpfc import _infer_field_ids

    obs = pd.DataFrame({"library_id": ["lib-a", "lib-b", "lib-a"]})
    np.testing.assert_array_equal(
        _infer_field_ids(obs, 3), np.array(["lib-a", "lib-b", "lib-a"], dtype=object)
    )
    # No identity column means the caller must make the single-frame assumption explicit;
    # the loader does not manufacture one from coordinates.
    assert _infer_field_ids(pd.DataFrame({"batch": ["x", "x"]}), 2) is None


def test_expression_only_control_never_uses_coordinate_refinement():
    labels = np.array([0,1,1])
    for xy in [np.zeros((3,2)), np.array([[0,0],[100,0],[-100,0]])]:
        options = list(guard.refinement_candidates('floor:nonspatial',labels,xy))
        assert len(options)==1 and options[0][0]=='raw'
        np.testing.assert_array_equal(options[0][1],labels)


def test_seed_identity_prevents_shifted_partial_pairing():
    per = {'floor:nonspatial': {1:.2,2:.1,3:.3}, 'Tessera': {1:.3,3:.7}}
    means, _, missing = guard.complete_seed_summary(per,[1,2,3])
    assert 'Tessera' not in means and missing['Tessera']==[2]
    with pytest.raises(ValueError,match='duplicate'):
        guard.record_seed(per,'Tessera',1,.4)
    with pytest.raises(ValueError,match='control'):
        guard.complete_seed_summary({'floor:nonspatial':{1:.2}},[1,2,3])
    with pytest.raises(ValueError,match='unexpected'):
        guard.complete_seed_summary({'floor:nonspatial':{1:.2,4:.3}},[1,2,3])


def test_candidate_cache_requires_protocol_seed_and_content_identity(tmp_path):
    input_file = tmp_path/'input.dat'; input_file.write_text('original input')
    row = dict(platform='toy',protocol_id=guard.PROTOCOL_ID,input_fingerprint=guard.fingerprint([input_file]),
               seeds=[1,2,3],per_seed={'floor:nonspatial':{1:.1,2:.2,3:.3}},means={'floor:nonspatial':.2})
    restored = json.loads(json.dumps(row))
    guard.require_cached_input(restored,[input_file])
    with pytest.raises(ValueError,match='historical'):
        guard.require_candidate_rows([dict(row,protocol_id='old')])
    with pytest.raises(ValueError,match='means'):
        guard.require_candidate_rows([dict(row,means={'floor:nonspatial':.8})])
    with pytest.raises(ValueError,match='incomplete'):
        guard.require_complete_platforms([row],['toy','missing'])
    input_file.write_text('changed input')
    with pytest.raises(ValueError,match='changed'):
        guard.require_cached_input(row,[input_file])


def test_nonsignificant_and_negative_outcomes_have_no_acceptance_gate():
    per = {'floor:nonspatial':{1:-.2},'Tessera':{1:-.3}}
    means,_,_ = guard.complete_seed_summary(per,[1])
    assert means['Tessera'] < means['floor:nonspatial']
    for name in ['expanded_bench.py','expand_data.py','native_baselines.py','science_harden_20260918_r2_n12.py','stat_rigor.py','gtfree_proxy.py','mechanism_synth.py']:
        tree = ast.parse((CAP/'experiments'/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Assert):
                assert not any(word in ast.unparse(node.test) for word in ['pvalue','correlation','winners'])
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='best_config_ari':
                assert any(kw.arg=='method' for kw in node.keywords)


def test_ece_correctness_is_invariant_to_arbitrary_cluster_numbers():
    from tessera_st.ablation import _metric_row
    xy = np.random.RandomState(21).normal(size=(8,2))
    truth = np.array([0,0,0,1,1,1,-1,-1])
    predicted = np.array([0,0,1,1,1,1,0,1])
    def score(p):
        return _metric_row('toy',xy,truth,p,xy,np.full(8,.8),kind='test')['ECE']
    assert score(predicted) == score(np.where(predicted==0,7,3))
    # Five of six labelled points are correct after majority mapping.
    assert score(predicted) == round(abs(5/6-.8),4)


def test_model_head_has_no_calibration_and_public_output_withholds_confidence():
    import torch
    from tessera_st.config import AblationConfig,ModelConfig,TrainConfig
    from tessera_st.model.tessera import TesseraNet
    from tessera_st.losses import total_loss
    from tessera_st.train import fit_predict
    torch.set_num_threads(1)
    xy = np.random.RandomState(10).normal(size=(16,2))
    expr = np.random.RandomState(11).normal(size=(16,4)).astype(np.float32)
    edges, distances = build_knn_edges(xy,k=3)
    model = TesseraNet(ModelConfig(4,hidden_dim=8,embed_dim=4,n_layers=2),2,AblationConfig())
    x = torch.tensor(expr); edge = torch.tensor(edges); dist = torch.tensor(distances)
    out = model(x,edge,dist)
    loss,_ = total_loss(out,x,edge,TrainConfig(),AblationConfig())
    loss.backward()
    assert all(p.grad is None for p in model.cluster.parameters())
    result = fit_predict(expr,xy,2,AblationConfig(),TrainConfig(epochs=1,device='cpu'),
                         ModelConfig(4,hidden_dim=8,embed_dim=4,n_layers=2),
                         knn_k=3,refine=True,field_ids=['a']*8+['b']*8)
    assert result.confidence is None
    assert result.labels.shape == (16,) and np.isfinite(result.embed).all()

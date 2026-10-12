from pathlib import Path
import inspect
import sys
import numpy as np
import pytest
import torch
from sklearn.cluster import KMeans

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'source/scripts'),str(ROOT/'source/src')]
from prepare_data import selection
from run_experiments import graph,configuration,neighbor_embedding,train_net
from tessera_st.train import fit_predict
from tessera_st.config import TrainConfig
from tessera_st.model.encoder import scatter_mean_gated
from common import encode

def test_selection_cannot_accept_labels():
    assert list(inspect.signature(selection).parameters)==['fields','eligible','strata']
    f=np.repeat(['f1','f2','f3','f4'],2500)
    keep=selection(f,np.ones(len(f),bool))
    assert len(keep)==5000
    for field in np.unique(f[keep]): assert np.sum(f[keep]==field)==2500
    assert np.array_equal(keep,selection(f,np.ones(len(f),bool)))

def test_small_field_duplicate_and_singleton_graph():
    xy=np.array([[0,0],[0,0],[1,0],[0,0]],float);f=np.array(['a','a','a','b'])
    edge,dist=graph(xy,f,'Tessera',1)
    assert np.all(f[edge[0]]==f[edge[1]])
    assert not np.any(edge[0]==edge[1])
    assert 3 not in edge
    assert {0,1}.issubset(set(edge.ravel()))
    assert np.isfinite(dist).all()

def test_permuted_graph_preserves_frames_and_changes_neighbors():
    xy=np.c_[np.arange(40),np.zeros(40)];f=np.repeat(['a','b'],20)
    e,_=graph(xy,f,'Tessera',1);p,_=graph(xy,f,'PermutedXY',1)
    assert not np.array_equal(e,p)
    assert np.all(f[p[0]]==f[p[1]])
    assert np.array_equal(p,graph(xy,f,'PermutedXY',1)[0])

def test_component_switches_are_specific():
    assert not configuration('NoGate').edge_gating
    assert not configuration('NoScale').multi_scale
    assert not configuration('NoContrast').boundary_contrastive
    assert not configuration('SelfOnly').boundary_contrastive
    assert not configuration('SelfOnly').edge_gating
    assert all(not configuration(a).calibrated_uncertainty for a in ['Tessera','NoGate','NoScale','NoContrast','SelfOnly'])

def test_self_only_has_no_coordinate_dependence():
    rng=np.random.RandomState(0);xy=rng.normal(size=(20,2));f=np.repeat('a',20)
    e,d=graph(xy,f,'SelfOnly',1);p,q=graph(xy+999,f,'SelfOnly',1)
    assert np.array_equal(e,p) and np.array_equal(d,q)
    x=rng.normal(size=(20,4)).astype(np.float32)
    assert np.allclose(neighbor_embedding(x,e),x)

def test_unknowns_do_not_create_a_class():
    y,c=encode(['Tumor','nan','unknown','Stroma','Tumor'])
    assert c==['Stroma','Tumor'] and y.tolist()==[1,-1,-1,0,1]

def test_gate_normalization_uniform_scale_cancels():
    m=torch.tensor([[1.,2.],[4.,6.]])
    dst=torch.tensor([0,0]);g=torch.tensor([.2,.8])
    a=scatter_mean_gated(m,dst,g,1);b=scatter_mean_gated(m,dst,g*.1,1)
    assert torch.allclose(a,b,atol=1e-6)

def test_full_encoder_matches_existing_training_path_cpu():
    torch.set_num_threads(2)
    rng=np.random.RandomState(1);x=rng.normal(size=(32,5)).astype(np.float32)
    xy=rng.normal(size=(32,2));f=np.repeat(['a','b'],16)
    old=fit_predict(x,xy,3,configuration('Tessera'),TrainConfig(epochs=120,seed=1,device='cpu'),field_ids=f,refine=False)
    model,z,b,_,_,_,history,_,_,_=train_net(x,xy,f,3,'Tessera',1,'cpu')
    # Repeated threaded floating-point reductions can differ by a few ulps.
    # Require close embeddings plus exactly equal returned cluster labels.
    np.testing.assert_allclose(z,old.embed,rtol=1e-5,atol=1e-5)
    assert np.allclose(b,old.boundary,atol=1e-6)
    assert np.array_equal(KMeans(3,n_init=10,random_state=1).fit_predict(z),old.labels)
    assert history[0]['loss']==pytest.approx(old.history[0]['loss'])

def test_source_transform_cannot_fit_target():
    from run_transfer import transform_source
    rng=np.random.RandomState(1);src=rng.normal(size=(60,8)).astype(np.float32)
    tgt=rng.normal(size=(20,8)).astype(np.float32)
    a,_,m,s,p=transform_source(src,tgt)
    b,_,m2,s2,p2=transform_source(src,tgt*100+1000)
    assert np.array_equal(a,b) and np.array_equal(m,m2) and np.array_equal(s,s2)
    assert np.array_equal(p.components_,p2.components_)

def test_transfer_absent_categories_are_errors_not_dropped():
    from run_transfer import transfer_score
    r=transfer_score(['a','b','c','c'],np.array(['a','b','a','a']),np.array(['a','b']))
    assert r['absent_source_target_classes']==['c']
    assert r['unsupported_target_cell_fraction']==.5
    assert r['accuracy']==.5
    assert r['balanced_accuracy']==pytest.approx(2/3)

def test_prepared_inputs_contain_no_pickled_object_arrays():
    for path in (ROOT/'data').glob('*/input.npz'):
        d=np.load(path,allow_pickle=False)
        assert all(d[k].dtype.kind!='O' for k in d.files)

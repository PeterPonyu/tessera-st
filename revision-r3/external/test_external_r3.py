"""WP4 tests on CPU with toy inputs. An encoder's tests skip where it is not installed, so run in both environments:
    /home/zeyufu/miniconda3/envs/tessera-native/bin/python -B -m pytest -q -p no:cacheprovider test_external_r3.py
    /home/zeyufu/miniconda3/envs/dl/bin/python -B -m pytest -q -p no:cacheprovider test_external_r3.py
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only
os.environ.setdefault("TQDM_DISABLE", "1")

import importlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.sparse.csgraph import shortest_path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import graph_ops as go  # noqa: E402

HAS = {"graphst": importlib.util.find_spec("GraphST") is not None,
       "stagate": importlib.util.find_spec("torch_geometric") is not None}
ENCODERS = [pytest.param(e, marks=pytest.mark.skipif(not HAS[e], reason=f"{e} is not installed in this environment"))
            for e in ("graphst", "stagate")]
graphst_only = pytest.mark.skipif(not HAS["graphst"], reason="GraphST lives in env tessera-native")
stagate_only = pytest.mark.skipif(not HAS["stagate"], reason="STAGATE needs torch_geometric (env dl)")
N = 300


def adapter(encoder):
    return importlib.import_module(go.ADAPTERS[encoder])


@pytest.fixture(scope="module")
def toy():
    rng = np.random.RandomState(0)
    xy = rng.rand(N // 2, 2) * 100
    coords = np.vstack([xy, xy + 0.5])  # two frames over the same area: a pooled graph would join them
    frames = np.array(["a"] * (N // 2) + ["b"] * (N // 2))
    x = rng.randn(N, 20).astype(np.float32)
    counts = rng.poisson(2.0, size=(N, 400)).astype(np.float32)
    native = {"counts": counts, "genes": np.array([f"g{i}" for i in range(400)]),
              "obs_id": np.array([f"c{i}" for i in range(N)])}
    return dict(x=x, coords=coords, frames=frames, native=native)


def hops(src, dst, n=N):
    """hops[j, i]: shortest number of message steps from j to i."""
    return shortest_path(sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)), unweighted=True)


# graph_ops ---------------------------------------------------------------------------------------------------

def test_same_graph_mean_and_propagate_hand_values():
    pairs = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2), (3, 2), (2, 3), (3, 3), (4, 3), (3, 4), (4, 4)]
    src, dst = map(np.array, zip(*pairs))  # path 0-1-2-3-4, both directions, with self-loops
    x = np.array([[1, 0], [2, 3], [4, 0], [8, 3], [16, 0]], float)
    one = [[3 / 2, 3 / 2], [7 / 3, 1], [14 / 3, 2], [28 / 3, 1], [12, 3 / 2]]
    two = [[23 / 12, 5 / 4], [17 / 6, 3 / 2], [49 / 9, 4 / 3], [26 / 3, 3 / 2], [32 / 3, 5 / 4]]
    np.testing.assert_allclose(go.same_graph_mean(x, src, dst), one, rtol=1e-6)
    np.testing.assert_allclose(go.propagate(x, src, dst, 1), one, rtol=1e-6)
    np.testing.assert_allclose(go.propagate(x, src, dst, 2), two, rtol=1e-6)
    np.testing.assert_allclose(go.propagate(x, src, dst, 0), x)
    # direction: an edge 0 -> 1 feeds node 1 only
    np.testing.assert_allclose(go.same_graph_mean([[1.0], [3.0]], np.array([0, 0, 1]), np.array([0, 1, 1])),
                               [[1.0], [2.0]])
    with pytest.raises(ValueError):
        go.same_graph_mean([[1.0], [3.0]], np.array([0]), np.array([1]))  # node 0 receives nothing


def test_load_r3_reads_the_cohort_writer(tmp_path, monkeypatch):
    """Contract with the WP5 writer, revision-r3/cohorts/common_cohorts.write_dataset (run into tmp_path)."""
    import common_external as ce
    sys.path.insert(0, str(HERE.parent / "cohorts"))
    cohorts = pytest.importorskip("common_cohorts")
    rng = np.random.RandomState(0)
    frames = np.array(["s1"] * 60 + ["s2"] * 60)
    y = rng.randint(-1, 3, len(frames))
    cohorts.write_dataset(tmp_path, "toy", "mouse1", rng.randn(len(frames), 50), rng.rand(len(frames), 2) * 100,
                          frames, {"laminae": (y, ["I", "II", "III"], "geometry-derived")}, {"unit": "Mouse 1"})
    monkeypatch.setattr(ce, "DATA_R3", tmp_path)
    assert ce.r3_ids() == ["toy/mouse1"] and not ce.has_native("toy/mouse1")
    ds = ce.load_dataset("toy/mouse1")
    assert ds["unit"] == "Mouse 1" and ds["K"] == {"laminae": 3} and np.array_equal(ds["targets"]["laminae"], y)
    assert ds["native"] is None and ds["x_common"].shape == (len(frames), 50)


def test_assert_within_frames():
    frames = np.array(["a", "a", "b"])
    assert go.assert_within_frames(np.array([0, 1]), np.array([1, 0]), frames) == 2
    with pytest.raises(AssertionError):
        go.assert_within_frames(np.array([0]), np.array([2]), frames)


# both encoders -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("encoder", ENCODERS)
def test_no_cross_frame_edges(encoder, toy):
    from sklearn.neighbors import NearestNeighbors
    idx = NearestNeighbors(n_neighbors=7).fit(toy["coords"]).kneighbors(toy["coords"], return_distance=False)
    assert (toy["frames"][idx] != toy["frames"][:, None]).any()  # the raw coordinates would cross frames
    a = adapter(encoder)
    for arm in a.ARMS:
        for config in a.CONFIGS:
            src, dst = go.edges_from_encoder(encoder, toy["coords"], toy["frames"], config, arm, seed=1)
            assert go.assert_within_frames(src, dst, toy["frames"]) >= N
            assert (src != dst).any() == (a.depth(arm) > 0)  # spatial arms carry neighbour edges


@pytest.mark.parametrize("encoder", ENCODERS)
def test_permuted_xy_changes_graph_keeps_frames(encoder, toy):
    coords, frames = toy["coords"], toy["frames"]
    base = go.edges_from_encoder(encoder, coords, frames, "default", "released", 1)
    perm = go.edges_from_encoder(encoder, coords, frames, "default", "permuted_xy", 1)
    again = go.edges_from_encoder(encoder, coords, frames, "default", "permuted_xy", 1)
    other = go.edges_from_encoder(encoder, coords, frames, "default", "permuted_xy", 2)
    assert set(zip(*base)) != set(zip(*perm)) and set(zip(*perm)) != set(zip(*other))
    assert all(np.array_equal(p, q) for p, q in zip(perm, again))
    go.assert_within_frames(*perm, frames)
    moved = go.permute_within_frames(coords, frames, 1)
    for f in ("a", "b"):
        ix = frames == f
        assert sorted(map(tuple, moved[ix])) == sorted(map(tuple, coords[ix]))  # positions stay in their frame
    assert not np.array_equal(moved, coords)


@pytest.mark.parametrize("encoder", ENCODERS)
def test_smoke_fit_every_arm(encoder, toy):
    a = adapter(encoder)
    for arm in a.ARMS:
        r = a.fit(toy["x"], toy["coords"], toy["frames"], arm, "default", 1, "common", "cpu")
        assert r["embedding"].shape == (N, 20 if encoder == "graphst" else 30) and np.isfinite(r["embedding"]).all()
        edges = go.edges_from_encoder(encoder, toy["coords"], toy["frames"], "default", arm, 1)
        assert np.array_equal(edges[0], r["src"]) and np.array_equal(edges[1], r["dst"])
        go.assert_within_frames(r["src"], r["dst"], toy["frames"])
        assert r["depth"] == a.depth(arm) and r["seconds"] > 0 and r["peak_gpu_bytes"] is None
        if encoder == "graphst":
            assert r["settings"]["beta"] == (0.0 if arm == "no_contrast" else 1)
    r = a.fit(toy["native"], toy["coords"], toy["frames"], "released", "default", 1, "native", "cpu")
    assert len(r["embedding"]) == N and np.isfinite(r["embedding"]).all()


@pytest.mark.parametrize("encoder", ENCODERS)
def test_runner_record_lock_resume_and_score(encoder, tmp_path):
    """One real fit through the runner (HER2ST C1, 176 spots), then a resume, the locks and the scorer."""
    import common_external as ce
    import run_external_r3 as rr
    import score_external_r3 as se
    a = adapter(encoder)
    ds = ce.load_dataset("HER2ST-C1")
    config = a.CONFIGS[0]
    job = dict(dataset="HER2ST-C1", encoder=encoder, arm="released", config=config, seed=1, input_mode="common")
    env = rr.versions(encoder, "cpu")
    assert rr.run_job(a, job, ds, "cpu", tmp_path, env).startswith("ok")
    p = rr.paths(job, tmp_path)
    record = json.loads(p["record"].read_text())
    assert record["status"] == "ok" and not p["lock"].exists() and record["dataset_fingerprint"] == ds["fingerprint"]
    with np.load(p["embedding"]) as saved:
        assert saved["embedding"].dtype == np.float16 and len(saved["embedding"]) == len(ds["coords"])
        go.assert_within_frames(saved["src"], saved["dst"], ds["frames"])
    assert rr.run_job(a, job, ds, "cpu", tmp_path, env) == "skip (ok)"
    other = dict(job, seed=2)
    lock = rr.paths(other, tmp_path)["lock"]
    lock.write_text(json.dumps({"host": rr.socket.gethostname(), "pid": os.getpid()}))
    assert rr.run_job(a, other, ds, "cpu", tmp_path, env) == "skip (locked)"  # a live process holds it
    lock.write_text(json.dumps({"host": rr.socket.gethostname(), "pid": 2 ** 22 + 12345}))
    assert rr.take_lock(lock)  # a dead process's lock is stale and taken over
    rows = se.score_record(record, tmp_path)
    expected = {(t, rule) for t in ds["targets"] for rule in ("kmeans", "leiden")}
    assert {(r["target"], r["rule"]) for r in rows} == expected
    assert all(np.isfinite(r["ari"]) and r["K"] == ds["K"][r["target"]] for r in rows)


# GraphST -----------------------------------------------------------------------------------------------------

@graphst_only
def test_identity_encoder_keeps_spatial_readout(toy):
    import torch
    from GraphST.model import Encoder
    import run_graphst_r3 as g

    def seen_in_training(arm):
        model, _ = g.prepare(toy["x"], toy["coords"], toy["frames"], arm, "k3-e2", 1, "common", "cpu")
        seen = []

        def hook(module, inputs):
            if isinstance(module, Encoder):
                seen.append((inputs[2].detach().numpy().copy(), module.graph_neigh.detach().numpy().copy()))
        handle = torch.nn.modules.module.register_module_forward_pre_hook(hook)
        try:
            model.train()
        finally:
            handle.remove()
        return seen

    adata, *_ = g.build_adata(toy["x"], toy["coords"], toy["frames"], "released", "k3-e2", 1, "common")
    spatial = adata.obsm["graph_neigh"] + np.eye(N)  # GraphST's own k-NN readout mask, plus I
    assert (adata.obsm["graph_neigh"].sum(1) == 3).all()
    seen = seen_in_training("identity_encoder")
    assert len(seen) == 3  # two epochs and the final embedding pass
    for adj, mask in seen:
        assert np.array_equal(adj, np.eye(N))  # no message passing
        assert np.array_equal(mask, spatial)  # spatial positives kept
    for adj, mask in seen_in_training("released"):  # the hook does see a spatial operator when there is one
        assert not np.array_equal(adj, np.eye(N)) and np.array_equal(mask, spatial)


@graphst_only
def test_graphst_depth(toy):
    """Each output row of 'emb' depends on cells at most DEPTH = 2 steps away on the operator's graph."""
    import torch
    from GraphST.model import Encoder
    import run_graphst_r3 as g
    model, info = g.prepare(toy["x"], toy["coords"], toy["frames"], "released", "default", 1, "common", "cpu")
    torch.manual_seed(0)
    encoder = Encoder(model.dim_input, model.dim_output, model.graph_neigh).eval()
    distance = hops(info["src"], info["dst"])
    i = 0

    def row(x):
        t = torch.tensor(x)
        return encoder(t, t, model.adj)[1].detach().numpy()[i]
    base = row(toy["x"])
    for steps, changes in ((g.DEPTH, True), (g.DEPTH + 1, False)):
        j = int(np.flatnonzero(distance[:, i] == steps)[0])
        x = toy["x"].copy()
        x[j] += 5.0
        assert (np.abs(row(x) - base).max() > 1e-4) == changes


# STAGATE -----------------------------------------------------------------------------------------------------

@stagate_only
def test_uniform_attention_equal_within_neighbourhood(toy):
    import torch
    from torch_geometric.nn.conv import MessagePassing
    import run_stagate_r3 as s

    def attention(arm):
        captured, hooked = [], set()

        def message_hook(module, inputs, output):
            captured.append((inputs[0]["index"].numpy().copy(), module._alpha.detach().numpy().ravel().copy()))

        def pre(module, args):
            if isinstance(module, MessagePassing) and id(module) not in hooked:
                hooked.add(id(module))
                module.register_message_forward_hook(message_hook)
        handle = torch.nn.modules.module.register_module_forward_pre_hook(pre)
        try:
            s.fit(toy["x"], toy["coords"], toy["frames"], arm, "knn6-e3", 1, "common", "cpu")
        finally:
            handle.remove()
        return captured

    def spread(index, alpha):
        lo = np.full(N, np.inf)
        hi = np.full(N, -np.inf)
        np.minimum.at(lo, index, alpha)
        np.maximum.at(hi, index, alpha)
        return hi - lo

    uniform = attention("uniform_attention")
    assert len(uniform) == 8  # conv1 and the tied conv3, in three epochs and the final pass
    for index, alpha in uniform:
        indegree = np.bincount(index, minlength=N)
        assert spread(index, alpha).max() < 1e-6
        np.testing.assert_allclose(alpha, 1.0 / indegree[index], rtol=1e-5)
    released = attention("released")
    assert max(spread(index, alpha).max() for index, alpha in released) > 1e-4  # released attention is not uniform


@stagate_only
def test_stagate_depth(toy):
    """The latent of a cell depends on its in-neighbours (DEPTH = 1 step) and on nothing further."""
    import torch
    import STAGATE_pyG as ST
    import run_stagate_r3 as s
    src, dst = go.edges_from_encoder("stagate", toy["coords"], toy["frames"], "default", "released", 1)
    torch.manual_seed(0)
    model = ST.STAGATE(hidden_dims=[toy["x"].shape[1], 512, 30]).eval()
    edge_index = torch.tensor(np.stack([src, dst]))
    distance = hops(src, dst)
    i = next(i for i in range(N) if (distance[:, i] == 1).any() and (distance[:, i] == 2).any())

    def row(x):
        return model(torch.tensor(x), edge_index)[0].detach().numpy()[i]
    base = row(toy["x"])
    for steps, changes in ((s.DEPTH, True), (s.DEPTH + 1, False)):
        j = int(np.flatnonzero(distance[:, i] == steps)[0])
        x = toy["x"].copy()
        x[j] += 5.0
        assert (np.abs(row(x) - base).max() > 1e-4) == changes

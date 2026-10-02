#!/usr/bin/env python3
"""Independent rescore and raw-source identity checks of every retained result.

No calling the producer's graph or contiguity functions. This checks numerical
outputs and source/protocol binding, not submission readiness or external validity.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"experiments/generated/field-identified-fixed-backend-20260923-v1"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def main():
    protocol=json.loads((OUT/"protocol.json").read_text())
    for path,digest in protocol["code_sha256"].items():
        assert sha(path)==digest, f"Code identity changed: {path}"
    check={}
    total=0
    failures=[]
    for name in protocol["retained_inputs"]:
        folder=OUT/name
        meta=json.loads((folder/"input.json").read_text())
        # These are locally generated records, not untrusted downloads. Pandas
        # string columns were serialized as object arrays by the frozen runner.
        rows=np.load(folder/"rows.npz", allow_pickle=True)
        assert sha(meta["path"])==meta["source_sha256"]
        a=ad.read_h5ad(meta["path"],backed="r")
        try:
            keep=np.arange(a.n_obs)
            if len(keep)>16000:
                keep=np.sort(np.random.RandomState(0).choice(len(keep),16000,replace=False))
            np.testing.assert_array_equal(keep,rows["original_index"])
            np.testing.assert_array_equal(np.asarray(a.obs_names,str)[keep],rows["obs_id"])
            np.testing.assert_array_equal(np.asarray(a.obsm["spatial"])[keep],rows["coordinates"])
            if meta["field_key"]=="uns:spatial":
                assert len(a.uns["spatial"])==1
                raw_field=np.repeat(next(iter(a.uns["spatial"])),len(keep))
            else:
                raw_field=a.obs[meta["field_key"]].astype(str).to_numpy()[keep]
            np.testing.assert_array_equal(raw_field,rows["field"])
            raw_label=a.obs[meta["label_key"]].astype(str).to_numpy()[keep]
            labels=np.array([meta["classes"].get(v,-1) for v in raw_label])
            np.testing.assert_array_equal(labels,rows["reference_label"])
        finally:
            a.file.close()
        xy,f,y=rows["coordinates"],rows["field"],rows["reference_label"]
        s,t=rows["spatial_src"],rows["spatial_dst"]
        assert (f[s]==f[t]).all() and (s!=t).all()
        good=y>=0
        xx,ff,yy=xy[good],f[good],y[good]
        agreements=[]; chance=[]; edges=0; singletons=0
        for field in np.unique(ff):
            ix=np.flatnonzero(ff==field)
            if len(ix)==1:
                singletons+=1
                continue
            count=np.bincount(yy[ix])
            expected=np.sum(count*(count-1))/(len(ix)*(len(ix)-1))
            idx=NearestNeighbors(n_neighbors=min(7,len(ix))).fit(xx[ix]).kneighbors(xx[ix],return_distance=False)
            for i,near in enumerate(idx):
                near=near[near!=i][:6]
                agreements.append(np.mean(yy[ix[near]]==yy[ix[i]]))
                chance.append(expected)
                edges+=len(near)
        assert abs(np.mean(agreements)-meta["contiguity_cell_weighted"])<1e-12
        assert abs(np.mean(chance)-meta["within_field_permutation_expectation"])<1e-12
        assert edges==meta["labelled_directed_edges"] and singletons==meta["labelled_singletons_excluded"]
        scores=0
        for method in protocol["methods"]:
            for seed in protocol["seeds"]:
                path=folder/f"{method}-seed{seed}.json"
                r=json.loads(path.read_text())
                assert r["method"]==method and r["seed"]==seed and r["dataset"]==name
                assert r["protocol_sha256"]==sha(OUT/"protocol.json")
                assert r["source_sha256"]==meta["source_sha256"]
                if r["status"]!="complete":
                    failures.append({"dataset":name,"method":method,"seed":seed,"error":r["error"]})
                    continue
                predfile=folder/f"{method}-seed{seed}.npz"
                assert sha(predfile)==r["predictions_sha256"]
                pred=np.load(predfile)
                for key,val in [("raw","raw_ari"),("refined","refined_ari"),
                    ("diagnostic_only_refined_control","diagnostic_only_refined_control_ari"),
                    ("diagnostic_only_pooled_coordinates","diagnostic_only_pooled_coordinate_ari")]:
                    if val in r:
                        assert pred[key].shape==y.shape
                        assert abs(adjusted_rand_score(y[good],pred[key][good])-r[val])<1e-12
                if method=="expression-only":
                    assert not r["spatial_refinement"] and "refined_ari" not in r
                    assert "refined" not in pred
                scores+=1
        check[name]={"n_cells":len(y),"n_fields":len(set(f)),"zero_cross_field_edges":True,
            "completed_method_seeds_rescored":scores,"contiguity_raw_source_reproduced":True}
        total+=scores
    result=dict(status="raw-source identity, spatial metrics and prediction rescore passed",
        scientific_ready=False,protocol_sha256=sha(OUT/"protocol.json"),datasets=check,
        completed_method_seeds=total,expected_method_seeds=len(check)*len(protocol["methods"])*len(protocol["seeds"]),
        failed_runs=failures,unavailable_methods=protocol["unavailable_methods"])
    (OUT/"QA.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()

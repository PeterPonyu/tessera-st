#!/usr/bin/env python3
"""Replay generator diagnostics and tie-rank audit without model training."""
import ast
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from tessera_st.data.synthetic import make_tessellation
OUT=ROOT/"manuscript/ledgers/revision_20260923"


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    file=ROOT/"experiments/mechanism_decoupled_v2.py"
    tree=ast.parse(file.read_text())
    fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="scramble_labels_regenerate")
    env={"np":np};exec(compile(ast.Module(body=[fun],type_ignores=[]),str(file),"exec"),env)
    records=[]
    for s,params in [("S1",dict(n_side=34,n_genes=40,n_domains=5,domain_signal=.3,iid_noise=1.,small_domain=True)),
                     ("S2",dict(n_side=40,n_genes=60,n_domains=6,domain_signal=.35,iid_noise=1.,small_domain=True))]:
        for seed in range(1,6):
            tissue=make_tessellation(seed=seed,**params)
            k=int(tissue.labels.max())+1
            means=np.stack([tissue.expr[tissue.labels==d].mean(0) for d in range(k)])
            idx=NearestNeighbors(n_neighbors=7).fit(tissue.coords).kneighbors(tissue.coords)[1][:,1:]
            for level in [0.,1.]:
                y,expr=env["scramble_labels_regenerate"](tissue.labels,means,level,
                    np.random.default_rng(seed*10000+int(level*1000)),k,params["n_genes"],params["iid_noise"])
                centered=expr-expr.mean(0); den=(centered**2).sum(0)
                moran=float((np.sum(centered*centered[idx].mean(1),axis=0)/den).mean())
                counts=np.bincount(y,minlength=k)
                records.append(dict(substrate=s,seed=seed,level=level,expression_moran=moran,
                    small_class_fraction=float(counts[-1]/len(y)),residual_sd=float((expr-means[y]).std()),
                    random_same_label=float(np.sum(counts*(counts-1))/(len(y)*(len(y)-1)))))
    rows=json.loads((ROOT/"experiments/expanded_bench.json").read_text())["rows"]
    stat=ast.parse((ROOT/"experiments/stat_rigor.py").read_text())
    const={}
    for n in stat.body:
        if isinstance(n,ast.Assign):
            for target in n.targets:
                if isinstance(target,ast.Name) and target.id in ["RAW_VARS","MODALITY"]:const[target.id]=ast.literal_eval(n.value)
    x=np.array([r["GT_contiguity"] for r in rows]); y=np.array([r["spatial_advantage"] for r in rows])
    columns=[np.array([r["k"] for r in rows]),np.array([min(const["RAW_VARS"][r["platform"]],3000) for r in rows]),
             np.array([const["MODALITY"][r["platform"]] for r in rows])]
    def part(x,y,c,ranking):
        Z=np.column_stack([np.ones(len(x))]+[ranking(v) for v in c])
        a,b=ranking(x),ranking(y)
        return float(np.corrcoef(a-Z@np.linalg.lstsq(Z,a,rcond=None)[0],b-Z@np.linalg.lstsq(Z,b,rcond=None)[0])[0,1])
    old=lambda v:np.argsort(np.argsort(v)).astype(float)
    old_r=[];new_r=[];rng=np.random.default_rng(223)
    for _ in range(200):
        p=rng.permutation(len(x))
        old_r.append(part(x[p],y[p],[c[p] for c in columns],old))
        new_r.append(part(x[p],y[p],[c[p] for c in columns],rankdata))
    result=dict(scope="Generator diagnostic and historical-panel rank counterexample; not repaired tissue correlation",
        generator_rows=records,legacy_joint_partial_range=[min(old_r),max(old_r)],
        midrank_joint_partial_range=[min(new_r),max(new_r)],
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [file,ROOT/"src/tessera_st/data/synthetic.py",
            ROOT/"experiments/expanded_bench.json",ROOT/"experiments/stat_rigor.py"]})
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"estimand-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print('Generator endpoints:',len(records),'rank range:',result['midrank_joint_partial_range'])


if __name__=="__main__":main()

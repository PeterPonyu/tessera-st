#!/usr/bin/env python3
"""Summaries from complete seed identities; no significance acceptance rule."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, rankdata

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"experiments/generated/field-identified-fixed-backend-20260923-v1"


def partial(x,y,columns):
    design=np.column_stack([np.ones(len(x))]+[rankdata(c,method="average") for c in columns])
    a,b=rankdata(x),rankdata(y)
    ra=a-design@np.linalg.lstsq(design,a,rcond=None)[0]
    rb=b-design@np.linalg.lstsq(design,b,rcond=None)[0]
    if np.linalg.norm(ra)<1e-10 or np.linalg.norm(rb)<1e-10:return None
    return float(np.corrcoef(ra,rb)[0,1])


def main():
    protocol=json.loads((OUT/"protocol.json").read_text())
    rows=[]
    for name in protocol["retained_inputs"]:
        folder=OUT/name
        meta=json.loads((folder/"input.json").read_text())
        methods={}
        for m in protocol["methods"]:
            records=[json.loads((folder/f"{m}-seed{s}.json").read_text()) for s in protocol["seeds"]]
            if any(r["status"]!="complete" for r in records):
                methods[m]={"status":"incomplete","records":records}
                continue
            raw=[r["raw_ari"] for r in records]
            methods[m]={"status":"complete","seeds":protocol["seeds"],"raw_per_seed":raw,
                        "raw_mean":float(np.mean(raw)),"raw_sd":float(np.std(raw,ddof=1))}
            for key in ["refined_ari","diagnostic_only_refined_control_ari","diagnostic_only_pooled_coordinate_ari"]:
                if all(key in r for r in records):
                    methods[m][key+"_mean"]=float(np.mean([r[key] for r in records]))
        if methods["expression-only"]["status"]!="complete":raise ValueError("Missing expression-only comparator")
        floor=methods["expression-only"]["raw_mean"]
        for m,r in methods.items():
            if r["status"]=="complete":r["delta_ari_vs_expression"]=r["raw_mean"]-floor
        rows.append({**meta,"methods":methods})
    correlations={}
    for m in protocol["methods"][1:]:
        if any(r["methods"][m]["status"]!="complete" for r in rows):
            correlations[m]={"status":"not_estimated_incomplete_panel"};continue
        x=np.array([r["contiguity_cell_weighted"] for r in rows])
        adjusted=np.array([r["chance_adjusted"] for r in rows])
        y=np.array([r["methods"][m]["delta_ari_vs_expression"] for r in rows])
        ordinary=spearmanr(x,y); adj=spearmanr(adjusted,y)
        cols=[np.array([r[k] for r in rows]) for k in ["n_classes","n_features"]]
        joint=partial(x,y,cols)
        diffs=[]
        for seed in range(100):
            perm=np.random.default_rng(seed).permutation(len(x))
            diffs.append(abs(partial(x[perm],y[perm],[c[perm] for c in cols])-joint))
        correlations[m]={"n":len(rows),"rho":float(ordinary.statistic),
            "p_uncorrected_descriptive":float(ordinary.pvalue),"chance_adjusted_rho":float(adj.statistic),
            "partial_class_count_feature_dim":joint,"permutation_order_max_abs_difference":max(diffs),
            "scope":"descriptive seven-object convenience panel; seeds not independent tissues; no confounding exclusion"}
    summary=dict(protocol_id=protocol["protocol_id"],rows=rows,correlations=correlations,
        status="new field-identified audit panel; historical n11/n12/n14 and unadapted methods remain HOLD",
        note="Fixed named contrasts are primary. No oracle maximum used to select a method. Three seeds give SD, not tissue-level uncertainty.")
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"datasets":len(rows),"correlations":correlations},indent=2))


if __name__=="__main__":main()

"""One-platform BayesSpace worker for parallelizing the tail of the panel run.

Reuses bayesspace_bench.py's PLATFORMS/load_any/bayesspace_labels/best_config_ari/NONSPATIAL
unchanged (imported, not reimplemented) so numbers are bit-identical to what the sequential
script would have produced for the same platform/seed. Writes one small JSON per platform;
a separate merge step folds these into bayesspace_bench_partial.json before the final aggregate
pass. Infra-only addition to enable running independent platforms concurrently; does not change
any published claim or the staged script's own logic.
"""
import json
import re
import sys

sys.path.insert(0, "experiments")
import numpy as np  # noqa: E402

import bayesspace_bench as bb  # noqa: E402

name = sys.argv[1]
path, gt = bb.PLATFORMS[name]
base = {r["platform"]: r for r in json.load(open("experiments/expanded_bench.json"))["rows"]}
lognorm, coords, true = bb.load_any(path, gt)
k = int(len(np.unique(true[true >= 0])))
cont = base[name]["GT_contiguity"]
platform_arg = "Visium" if "Visium" in name else "ST"
lab = bb.bayesspace_labels(lognorm, coords, k, platform_arg, seed=1)
if lab is None:
    sys.exit(f"BayesSpace failed for {name}")
bs = round(bb.best_config_ari(lab, true, coords, k), 4)
means = dict(base[name]["means"])
means["BayesSpace"] = bs
winner = max(means, key=lambda m: means[m])
nonsp = means.get(bb.NONSPATIAL, 0.0)
sadv = round(max(v for m, v in means.items() if m != bb.NONSPATIAL) - nonsp, 3)
row = {
    "platform": name, "GT_contiguity": cont, "k": k,
    "BayesSpace_ari": bs, "winner_with_bayesspace": winner,
    "spatial_advantage_with_bayesspace": sadv,
    "spatial_advantage_published": base[name]["spatial_advantage"],
    "means_with_bayesspace": {m: round(v, 4) for m, v in means.items()},
}
slug = re.sub(r"[^a-zA-Z0-9]+", "_", name)
json.dump(row, open(f"experiments/bayesspace_parallel_{slug}.json", "w"), indent=2)
print(f"DONE {name}: ARI={bs} winner={winner} adv={sadv:+.3f}", flush=True)

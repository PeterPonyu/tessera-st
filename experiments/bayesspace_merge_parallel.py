"""Fold experiments/bayesspace_parallel_*.json worker outputs into the canonical
bayesspace_bench_partial.json checkpoint, then delete the temp files. Run once, after all
parallel workers have finished, before re-invoking bayesspace_bench.py for the final aggregate.
"""
import glob
import json
import os

partial = "experiments/bayesspace_bench_partial.json"
rows = json.load(open(partial))["rows"] if os.path.exists(partial) else []
done = {r["platform"] for r in rows}

for f in sorted(glob.glob("experiments/bayesspace_parallel_*.json")):
    row = json.load(open(f))
    if row["platform"] not in done:
        rows.append(row)
        done.add(row["platform"])
        print(f"merged {row['platform']}")
    os.remove(f)

json.dump({"rows": rows}, open(partial, "w"), indent=2)
print(f"{len(rows)} platforms now in {partial}")

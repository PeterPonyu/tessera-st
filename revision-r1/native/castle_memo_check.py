#!/usr/bin/env python3
"""Check that run_native.CastleMemo leaves CASTLE's output unchanged (E4).

Plain CASTLE (per-epoch rebuilds) and memoized CASTLE are fitted with the same
seed; a second plain fit measures the run-to-run difference on this GPU.
Writes results/revision-r1/E4/castle_memo_check.json.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_common as nc  # noqa: E402
import run_native as rn  # noqa: E402

CASES = [("HER2ST", "C1", "expression", 600), ("NSCLC", "NSCLC_Lung13", "celltype", 20),
         ("NSCLC", "NSCLC_Lung13", "expression", 20)]


def main() -> None:
    warnings.simplefilter("ignore")
    out = []
    for cohort, patient, variant, epochs in CASES:
        inp = nc.load_input(cohort, patient)
        emb, secs = {}, {}
        for name, memo in (("plain_1", False), ("plain_2", False), ("memo", True)):
            t0 = time.perf_counter()
            emb[name] = rn.fit_castle(inp, 1, variant, [], "cuda", epochs=epochs, memo=memo)[0]
            secs[name] = time.perf_counter() - t0
        diff = lambda a, b: float(np.max(np.abs(emb[a].astype(np.float64) - emb[b])))
        rec = {"cohort": cohort, "patient": patient, "variant": variant, "epochs": epochs, "seed": 1,
               "plain_vs_plain_max_abs": diff("plain_1", "plain_2"), "plain_vs_memo_max_abs": diff("plain_1", "memo"),
               "plain_equals_memo_bitwise": bool(np.array_equal(emb["plain_1"], emb["memo"])),
               "seconds": secs}
        out.append(rec)
        print(json.dumps(rec), flush=True)
    (nc.E4 / "castle_memo_check.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()

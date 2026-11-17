"""Tessera-ST adapter for the WP4 runner (same interface as run_graphst_r3 and run_stagate_r3).

Arms: released; no_gate; no_contrast; self_only; permuted_xy_no_contrast (coordinates
permuted within frames, contrastive term removed); contrast_no_mp (spatial positives
without message passing, revision-r3/arms_r3.py). Configurations: the round-2 tuning
grid, graph neighbours {6, 15, 30} x contrastive weight {0.5, 1.0} x temperature
{0.2, 0.5} x epochs {120, 200}. The untrained head has two clusters for every dataset,
as in the round-2 added analyses, so no label or class count enters fitting.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
import common_external  # noqa: E402,F401  (round-1/2 import paths)
import train_core as tc  # noqa: E402

ENCODER = "tessera"
ARMS = ("released", "no_gate", "no_contrast", "self_only", "permuted_xy_no_contrast", "contrast_no_mp")
NEIGHBORS, WEIGHTS, TEMPS, EPOCHS = (6, 15, 30), (0.5, 1.0), (0.2, 0.5), (120, 200)
CONFIGS = tuple(f"k{k}-w{w}-t{t}-e{e}" for k in NEIGHBORS for w in WEIGHTS for t in TEMPS for e in EPOCHS)
DEFAULT_CONFIG = "k6-w0.5-t0.2-e120"
CONFIG_FORMAT = "k{}-w0.5-t0.2-e{}"   # used by the runner's timing sample (released weight and temperature)
DEPTH = 3
SCALING_POWER = 1
HEAD_K = 2
R1_ARM = {"released": "Tessera", "no_gate": "NoGate", "no_contrast": "NoContrast", "self_only": "SelfOnly",
          "permuted_xy_no_contrast": "PermutedXY-NoContrast"}


def parse_full(config) -> tuple[int, float, float, int]:
    config = DEFAULT_CONFIG if config in (None, "default") else config
    k, w, t, e = config.split("-")
    if not (k[0], w[0], t[0], e[0]) == ("k", "w", "t", "e"):
        raise ValueError(f"Tessera-ST config {config!r} is not k<n>-w<weight>-t<temp>-e<epochs>")
    return int(k[1:]), float(w[1:]), float(t[1:]), int(e[1:])


def parse_config(config) -> tuple[int, int]:
    k, _, _, e = parse_full(config)
    return k, e


def depth(arm: str) -> int:
    return 0 if arm in ("self_only", "contrast_no_mp") else DEPTH


def encoder_edges(coords, frames, arm="released", config="default", seed=1):
    """The graph Tessera-ST passes messages on for this arm and configuration (self edges when it passes none)."""
    k, _, _, _ = parse_full(config)
    edge, _ = tc.build_graph(coords, np.asarray(frames), "SelfOnly" if depth(arm) == 0 else R1_ARM.get(arm, "Tessera"),
                             seed, k)
    return edge[0], edge[1]


def fit(x, coords, frames, arm, config, seed, input_mode="common", device="cpu") -> dict:
    if input_mode != "common":
        raise ValueError("Tessera-ST runs on the common input only")
    k, w, t, e = parse_full(config)
    x = np.asarray(x, np.float32)
    frames = np.asarray(frames)
    if arm == "contrast_no_mp":
        import arms_r3
        out = arms_r3.train_contrast_no_mp(x, coords, frames, HEAD_K, seed, device, epochs=e, graph_k=k,
                                           contrastive_weight=w, contrastive_temp=t)
        z, edge, memory = out["z"], out["forward_edges"], None
    else:
        out = tc.train_core(x, coords, frames, HEAD_K, R1_ARM[arm], seed, device, epochs=e, graph_k=k,
                            contrastive_weight=w, contrastive_temp=t)
        z, edge, memory = out["checkpoints"][e]["z"], out["edge"], out["memory"]
    z = np.asarray(z, np.float32)
    if not np.isfinite(z).all():
        raise ValueError("non-finite Tessera-ST embedding")
    return dict(embedding=z, src=np.asarray(edge[0]), dst=np.asarray(edge[1]), seconds=float(out["seconds"]),
                peak_gpu_bytes=memory, depth=depth(arm), epochs=e, head_k=HEAD_K, graph_k=k,
                contrastive_weight=w, contrastive_temp=t)

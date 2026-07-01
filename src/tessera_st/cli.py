"""Command-line entry points.

  tessera smoke-synth     # offline synthetic tessellation: train + ablation grid
  tessera ablate-real     # real DLPFC .h5ad: full ablation grid (needs [real-data])
"""

from __future__ import annotations

import argparse
import json
import sys

from tessera_st.ablation import (
    format_table,
    run_benchmark,
    run_benchmark_multiseed,
    selection_summary,
    to_records,
)
from tessera_st.config import TrainConfig


def _parse_seeds(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    return [int(x) for x in spec.split(",") if x.strip()]


def _bench(expr, coords, labels, args, layer_scores=None):
    cfg = TrainConfig(epochs=args.epochs, seed=args.seed, device=args.device)
    seeds = _parse_seeds(getattr(args, "seeds", None))
    if seeds:
        rows = run_benchmark_multiseed(expr, coords, labels, seeds, train_cfg=cfg,
                                       layer_scores=layer_scores)
    else:
        rows = run_benchmark(expr, coords, labels, train_cfg=cfg, layer_scores=layer_scores)
    return rows, cfg


def _emit(rows, cfg, out) -> int:
    print(format_table(rows))
    print("\n" + selection_summary(rows))
    if out:
        with open(out, "w") as fh:
            json.dump(to_records(rows, cfg), fh, indent=2)
        print(f"\nwrote {out}")
    return 0


def _smoke_synth(args: argparse.Namespace) -> int:
    from tessera_st.data.synthetic import make_tessellation

    slide = make_tessellation(n_side=args.n_side, n_genes=args.n_genes, seed=args.seed)
    rows, cfg = _bench(slide.expr, slide.coords, slide.labels, args)
    return _emit(rows, cfg, args.out)


def _ablate_real(args: argparse.Namespace) -> int:
    from tessera_st.data.dlpfc import load_h5ad
    from tessera_st.eval.markers import DLPFC_LAYER_MARKERS

    slide = load_h5ad(args.h5ad, label_key=args.label_key, marker_dict=DLPFC_LAYER_MARKERS)
    rows, cfg = _bench(slide.expr, slide.coords, slide.labels, args,
                       layer_scores=slide.layer_marker_scores)
    return _emit(rows, cfg, args.out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tessera", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("smoke-synth", help="offline synthetic ablation grid")
    s.add_argument("--n-side", type=int, default=28)
    s.add_argument("--n-genes", type=int, default=50)
    s.add_argument("--epochs", type=int, default=150)
    s.add_argument("--seed", type=int, default=17)
    s.add_argument("--seeds", default=None, help="comma list, e.g. 1,2,3 -> mean±std over seeds")
    s.add_argument("--device", default="auto")
    s.add_argument("--out", default=None)
    s.set_defaults(func=_smoke_synth)

    r = sub.add_parser("ablate-real", help="real DLPFC .h5ad ablation grid")
    r.add_argument("h5ad")
    r.add_argument("--label-key", default="layer_guess")
    r.add_argument("--epochs", type=int, default=400)
    r.add_argument("--seed", type=int, default=17)
    r.add_argument("--seeds", default=None, help="comma list, e.g. 1,2,3 -> mean±std over seeds")
    r.add_argument("--device", default="auto")
    r.add_argument("--out", default=None)
    r.set_defaults(func=_ablate_real)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

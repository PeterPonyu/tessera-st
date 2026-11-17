#!/usr/bin/env python3
"""WP4 resumable batch runner: GraphST and STAGATE fits over (dataset, encoder, arm, config, seed, input_mode).

Each encoder runs in its own environment. An encoder that cannot be imported by this interpreter is handed to
its own (TESSERA_R3_PY_GRAPHST, default env tessera-native; TESSERA_R3_PY_STAGATE, default env dl):
    python -B run_external_r3.py --device cuda                      # all jobs, resumable
    python -B run_external_r3.py --dry-run --timing-epochs 60       # job counts and a wall-time estimate
Default plan: datasets 'core' (lung 5, BCC 4, HER2ST 8, prepared data-r3 cohorts); every arm x every config of
each encoder x seeds 1-5 on the common input; the native input (where round 2 defined one) for the released arm.
Outputs under $TESSERA_R3_OUT (default results/revision-r3/W4), tag = <encoder>-<input_mode>-<arm>-<config>-seed<seed>:
    fits/<dataset>/<tag>.json        one record per fit: settings, seconds, peak GPU memory, versions, graph checks
    embeddings/<dataset>/<tag>.npz   embedding (float16) and the encoder's own graph src, dst (int32)
    fits/<dataset>/<tag>.lock        held while the fit runs; a lock left by a dead process on this host is stale
A finished record (status ok, or failed unless --retry-failed) is never refitted.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "8")
os.environ.setdefault("TQDM_DISABLE", "1")

import argparse  # noqa: E402
import importlib  # noqa: E402
import importlib.metadata  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import common_external as ce  # noqa: E402

OUT = Path(os.environ.get("TESSERA_R3_OUT") or ce.c.RESULTS / "revision-r3" / "W4")
PYTHON = {"graphst": os.environ.get("TESSERA_R3_PY_GRAPHST", "/home/zeyufu/miniconda3/envs/tessera-native/bin/python"),
          "stagate": os.environ.get("TESSERA_R3_PY_STAGATE", "/home/zeyufu/miniconda3/envs/dl/bin/python"),
          "tessera": os.environ.get("TESSERA_R3_PY_TESSERA", "/home/zeyufu/miniconda3/envs/dl/bin/python")}
NEEDS = {"graphst": "GraphST", "stagate": "torch_geometric"}
MODULES = {"graphst": "run_graphst_r3", "stagate": "run_stagate_r3", "tessera": "run_tessera_r3"}
PACKAGES = ["numpy", "scipy", "pandas", "scikit-learn", "scanpy", "anndata", "torch", "POT", "GraphST",
            "torch_geometric", "leidenalg", "igraph"]
SEEDS = [1, 2, 3, 4, 5]


def importable(encoder: str) -> bool:
    return importlib.util.find_spec(NEEDS[encoder]) is not None


def tag(encoder, mode, arm, config, seed) -> str:
    return f"{encoder}-{mode}-{arm}-{config}-seed{seed}"


def folder(dataset_id: str) -> str:
    return dataset_id.replace("/", "__")


def paths(job: dict, out: Path) -> dict:
    name = tag(job["encoder"], job["input_mode"], job["arm"], job["config"], job["seed"])
    fits, emb = out / "fits" / folder(job["dataset"]), out / "embeddings" / folder(job["dataset"])
    return {"record": fits / f"{name}.json", "lock": fits / f"{name}.lock", "embedding": emb / f"{name}.npz"}


def plan(adapter, datasets, args) -> list[dict]:
    arms = args.arms or adapter.ARMS
    named = {"all": adapter.CONFIGS, "default": (adapter.DEFAULT_CONFIG,)}
    configs = named.get(args.configs[0], args.configs) if len(args.configs) == 1 else args.configs
    if set(arms) - set(adapter.ARMS) or set(configs) - set(adapter.CONFIGS):
        raise SystemExit(f"{adapter.ENCODER}: unknown arm or config in {arms} {configs}")
    jobs = []
    for ds in datasets:
        for mode in args.input_modes:
            if mode == "native" and not ce.has_native(ds):
                continue
            for arm in arms:
                if mode == "native" and arm not in args.native_arms:
                    continue
                for config in configs:
                    for seed in args.seeds:
                        jobs.append(dict(dataset=ds, encoder=adapter.ENCODER, arm=arm, config=config, seed=seed,
                                         input_mode=mode))
    return jobs


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def take_lock(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                held = json.loads(path.read_text())
            except (OSError, ValueError):
                return False
            if held.get("host") == socket.gethostname() and not pid_alive(int(held.get("pid", -1))):
                path.unlink(missing_ok=True)  # stale: its process is gone
                continue
            return False
        with os.fdopen(fd, "w") as handle:
            json.dump({"host": socket.gethostname(), "pid": os.getpid(), "started": time.time()}, handle)
        return True
    return False


def status(job: dict, out: Path) -> str:
    p = paths(job, out)
    if p["record"].exists():
        return json.loads(p["record"].read_text())["status"]
    return "locked" if p["lock"].exists() else "pending"


def git(root: Path, *args) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def versions(encoder: str, device: str) -> dict:
    import torch
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    info = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "host": socket.gethostname(), "packages": packages, "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if str(device).startswith("cuda") else None,
            "torch_threads": torch.get_num_threads(),
            "scripts_sha256": {p.name: ce.sha(p) for p in sorted(HERE.glob("*.py"))}}
    if encoder == "graphst":
        import GraphST
        root = Path(GraphST.__file__).parent
        info["GraphST_files_sha256"] = {p.name: ce.sha(p) for p in sorted(root.glob("*.py"))}
    elif encoder == "tessera":
        import tessera_st
        root = Path(tessera_st.__file__).parent
        info["tessera_st_files_sha256"] = {p.relative_to(root).as_posix(): ce.sha(p) for p in sorted(root.rglob("*.py"))}
    else:
        import run_stagate_r3 as rs
        files = sorted((rs.STAGATE_ROOT / "STAGATE_pyG").glob("*.py"))
        info["STAGATE"] = {"root": str(rs.STAGATE_ROOT), "commit": git(rs.STAGATE_ROOT, "rev-parse", "HEAD"),
                           "status": git(rs.STAGATE_ROOT, "status", "--porcelain", "--untracked-files=no"),
                           "files_sha256": {p.name: ce.sha(p) for p in files}}
    return info


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float) + "\n")
    os.replace(tmp, path)


def run_job(adapter, job, ds, device, out, env, retry_failed=False) -> str:
    p = paths(job, out)
    if p["record"].exists():
        previous = json.loads(p["record"].read_text())["status"]
        if previous == "ok" or not retry_failed:
            return f"skip ({previous})"
    if not take_lock(p["lock"]):
        return "skip (locked)"
    record = dict(job, cohort=ds["cohort"], unit=ds["unit"], device=device, n_cells=int(len(ds["coords"])),
                  dataset_fingerprint=ds["fingerprint"], sources=ds["sources"], started=time.time(), versions=env)
    try:
        x = ds["x_common"] if job["input_mode"] == "common" else ds["native"]
        fit = adapter.fit(x, ds["coords"], ds["frames"], job["arm"], job["config"], job["seed"], job["input_mode"],
                          device)
        embedding = fit["embedding"].astype(np.float16)
        if not np.isfinite(embedding).all():
            raise ValueError("embedding overflows float16")
        p["embedding"].parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p["embedding"], embedding=embedding, src=fit["src"].astype(np.int32),
                            dst=fit["dst"].astype(np.int32))
        record.update(status="ok", embedding_file=str(p["embedding"].relative_to(out)),
                      embedding_sha256=ce.sha(p["embedding"]), embedding_shape=list(embedding.shape),
                      **{k: v for k, v in fit.items() if k not in ("embedding", "src", "dst")})
    except Exception as exc:  # recorded; never replaced by another setting
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        record["finished"] = time.time()
        if "status" in record:  # interrupted fits leave no record
            write_json(p["record"], record)
        p["lock"].unlink(missing_ok=True)
    return record["status"] if record["status"] == "failed" else f"ok {record['seconds']:.1f}s"


def timing(adapter, jobs, meta, args) -> dict:
    """One released fit per input mode on the largest planned dataset (or --timing-dataset)."""
    samples = {}
    for mode in sorted({j["input_mode"] for j in jobs}):
        ids = sorted({j["dataset"] for j in jobs if j["input_mode"] == mode}, key=lambda d: meta[d]["n"])
        ds_id = args.timing_dataset if args.timing_dataset in ids else ids[-1]
        ds = ce.load_dataset(ds_id)
        k, epochs = adapter.parse_config(adapter.DEFAULT_CONFIG)
        epochs = args.timing_epochs or epochs
        x = ds["x_common"] if mode == "common" else ds["native"]
        fit = adapter.fit(x, ds["coords"], ds["frames"], "released", adapter.CONFIG_FORMAT.format(k, epochs), 1,
                          mode, args.device)
        samples[mode] = dict(dataset=ds_id, n=meta[ds_id]["n"], epochs=epochs, setup_s=fit["setup_seconds"],
                             train_s=fit["train_seconds"], per_epoch_s=fit["train_seconds"] / epochs,
                             peak_gpu_bytes=fit["peak_gpu_bytes"])
    return samples


def estimate(adapter, job, meta, sample) -> float:
    scale = (meta[job["dataset"]]["n"] / sample["n"]) ** adapter.SCALING_POWER
    _, epochs = adapter.parse_config(job["config"])
    return (sample["setup_s"] + sample["per_epoch_s"] * epochs) * scale


def dry_run(adapter, jobs, meta, args, out) -> None:
    states = {id(j): status(j, out) for j in jobs}
    print(f"\n{adapter.ENCODER}: {len(jobs)} jobs over {len({j['dataset'] for j in jobs})} datasets, "
          f"{len(args.seeds)} seeds; output {out}")
    samples = {} if args.no_timing else timing(adapter, jobs, meta, args)
    for mode, s in samples.items():
        print(f"  timing sample ({args.device}, {mode}): {s['dataset']} n={s['n']}, released, {s['epochs']} epochs: "
              f"setup {s['setup_s']:.1f} s, train {s['train_s']:.1f} s ({s['per_epoch_s'] * 1e3:.1f} ms/epoch)")
    print(f"  {'arm':<18}{'mode':<8}{'jobs':>6}{'done':>6}{'failed':>7}{'locked':>7}{'pending':>8}{'est_h':>9}")
    groups = sorted({(j["arm"], j["input_mode"]) for j in jobs}, key=lambda g: (g[1], adapter.ARMS.index(g[0])))
    total = 0.0
    for arm, mode in groups:
        part = [j for j in jobs if (j["arm"], j["input_mode"]) == (arm, mode)]
        count = Counter(states[id(j)] for j in part)
        hours = sum(estimate(adapter, j, meta, samples[mode]) for j in part
                    if states[id(j)] == "pending") / 3600 if samples else float("nan")
        total += hours
        print(f"  {arm:<18}{mode:<8}{len(part):>6}{count['ok']:>6}{count['failed']:>7}{count['locked']:>7}"
              f"{count['pending']:>8}{hours:>9.2f}")
    print(f"  estimated wall time of pending jobs, one process on this device: {total:.2f} h")


def dataset_meta(datasets) -> dict:
    meta = {}
    for ds_id in datasets:
        ds = ce.load_dataset(ds_id)
        meta[ds_id] = {"n": int(len(ds["coords"])), "native": ds["native"] is not None}
    return meta


def handle(encoder: str, args) -> None:
    adapter = importlib.import_module(MODULES[encoder])
    datasets = ce.dataset_ids(args.datasets[0]) if args.datasets[0] in ("core", "all") else args.datasets
    jobs = plan(adapter, datasets, args)
    out = Path(args.out)
    if args.dry_run:
        dry_run(adapter, jobs, dataset_meta(sorted({j["dataset"] for j in jobs})), args, out)
        return
    env = versions(encoder, args.device)
    done = 0
    for ds_id in dict.fromkeys(j["dataset"] for j in jobs):
        todo = [j for j in jobs if j["dataset"] == ds_id and (status(j, out) == "pending" or
                                                              (args.retry_failed and status(j, out) == "failed"))]
        if not todo:
            continue
        ds = ce.load_dataset(ds_id)
        for job in todo:
            if args.limit is not None and done >= args.limit:
                return
            result = run_job(adapter, job, ds, args.device, out, env, args.retry_failed)
            done += not result.startswith("skip")
            print(ds_id, tag(job["encoder"], job["input_mode"], job["arm"], job["config"], job["seed"]), result,
                  flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoders", nargs="+", default=["graphst", "stagate"], choices=sorted(MODULES))
    ap.add_argument("--datasets", nargs="+", default=["core"], help="'core', 'all' or dataset ids")
    ap.add_argument("--arms", nargs="+", default=None, help="default: every arm of the encoder")
    ap.add_argument("--configs", nargs="+", default=["all"], help="'all', 'default' or config names")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--input-modes", nargs="+", default=["common", "native"], choices=["common", "native"])
    ap.add_argument("--native-arms", nargs="+", default=["released"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="stop after this many fits")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-timing", action="store_true", help="dry run without the timing sample")
    ap.add_argument("--timing-epochs", type=int, default=None, help="epochs of the timing fit (default config's)")
    ap.add_argument("--timing-dataset", default=None)
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # run finally blocks: release the job lock
    for encoder in args.encoders:
        if importable(encoder):
            handle(encoder, args)
            continue
        argv = sys.argv[1:]
        start = argv.index("--encoders") if "--encoders" in argv else len(argv)
        stop = start + 1
        while stop < len(argv) and not argv[stop].startswith("--"):
            stop += 1
        argv = argv[:start] + argv[stop:] + ["--encoders", encoder]
        print(f"{encoder}: not importable here; running {PYTHON[encoder]}", flush=True)
        code = subprocess.run([PYTHON[encoder], "-B", str(Path(__file__).resolve()), *argv]).returncode
        if code:
            raise SystemExit(f"{encoder} runner exited with {code}")


if __name__ == "__main__":
    main()

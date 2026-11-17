"""Datasets for the WP4 external-encoder runs (GraphST, STAGATE); imports in both environments.

load_dataset(dataset_id) returns a dict:
  id, cohort, unit        unit = patient (the unit of inference); dataset = id
  x_common   float32      shared input of every control: input.npz['x'] (PCA50; 36 markers for MIBI) for
                          the 13 bounded objects; for HER2ST the frozen round-2 path common_r1.her2st_unit
                          (run_her2st_validation.load_joined + preprocess), not a re-implementation
  native     dict | None  input for each method's own preprocessing, as in round 2 (E4 inputs):
                          NSCLC {'expr_log', 'genes', 'obs_id'} (frozen log-normalized panel, no counts);
                          HER2ST {'counts', 'genes', 'obs_id'}; None for the other bounded objects, whose
                          stored gene matrices were already gene-selected, so no method's own pipeline applies
  coords float64 (n, 2), frames str (n,), obs_id str (n,)
  targets    {name: int64 labels, -1 = missing};  K {name: reference class count}
  fingerprint             SHA-256 over x_common, coords, frames and targets (checked again at scoring)
Paths follow common_r1 (TESSERA_DATA, TESSERA_RESULTS, TESSERA_VALIDATION override them); future cohorts
live under TESSERA_DATA_R3 (default projects/tessera-st/data-r3) in the cohorts lane's format, see load_r3.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROJECT = REPO.parent
R1 = REPO / "revision-r1"
for _path in (R1, R1 / "native", HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import common_r1 as c  # noqa: E402  (frozen import paths, loaders, encode)

DATA_R3 = Path(os.environ.get("TESSERA_DATA_R3") or PROJECT / "data-r3")
E4_INPUTS = c.RESULTS / "revision-r1" / "E4" / "inputs"
HER2ST = [f"HER2ST-{stem}" for stem in c.HER2ST]


def sha(path) -> str:
    return c.sha(path)


def r3_ids() -> list[str]:
    """Prepared future-cohort datasets, as '<cohort>/<dataset>' (none exist before WP5)."""
    return sorted(f"{p.parts[-3]}/{p.parts[-2]}" for p in DATA_R3.glob("*/*/input.npz") if p.parts[-3] != "raw")


def dataset_ids(group: str = "core") -> list[str]:
    """core: lung 5, BCC 4, HER2ST 8 and every prepared data-r3 dataset (the WP4 cancer core and WP5);
    all: the 13 bounded objects, HER2ST 8 and data-r3."""
    if group == "core":
        return [*c.LUNG, *c.BCC, *HER2ST, *r3_ids()]
    if group == "all":
        return [*c.OBJECTS, *HER2ST, *r3_ids()]
    raise ValueError(f"unknown dataset group {group!r}")


def has_native(dataset_id: str) -> bool:
    if dataset_id in c.LUNG or dataset_id in HER2ST:
        return True
    if "/" in dataset_id:
        with np.load(DATA_R3 / dataset_id / "input.npz", allow_pickle=False) as d:
            return "counts" in d.files
    return False


def _e4_input(cohort: str, patient: str) -> dict:
    path = E4_INPUTS / f"{cohort}-{patient}.npz"
    with np.load(path, allow_pickle=False) as d:
        out = {k: d[k] for k in d.files}
    out["sha256"] = sha(path)
    return out


def _fingerprint(ds: dict) -> str:
    digest = hashlib.sha256()
    for array in (ds["x_common"], ds["coords"], ds["frames"].astype("U")):
        digest.update(np.ascontiguousarray(array).tobytes())
    for name in sorted(ds["targets"]):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(ds["targets"][name]).tobytes())
    return digest.hexdigest()


def _finish(ds: dict) -> dict:
    n = len(ds["x_common"])
    ds["x_common"] = np.asarray(ds["x_common"], np.float32)
    ds["coords"] = np.asarray(ds["coords"], np.float64)
    ds["frames"] = np.asarray(ds["frames"]).astype(str)
    ds["obs_id"] = np.asarray(ds["obs_id"]).astype(str)
    ds["targets"] = {t: np.asarray(y, np.int64) for t, y in ds["targets"].items()}
    for key in ("coords", "frames", "obs_id"):
        if len(ds[key]) != n:
            raise ValueError(f"{ds['id']}: {key} has {len(ds[key])} rows, expected {n}")
    for t, y in ds["targets"].items():
        if len(y) != n or y.max() >= ds["K"][t] or (y < -1).any():
            raise ValueError(f"{ds['id']}: target {t} does not fit K = {ds['K'][t]}")
    if ds["native"] is not None:
        matrix = ds["native"].get("counts", ds["native"].get("expr_log"))
        if matrix is None or len(matrix) != n or not np.array_equal(ds["native"]["obs_id"].astype(str), ds["obs_id"]):
            raise ValueError(f"{ds['id']}: native input is not row-aligned with x_common")
    ds["fingerprint"] = _fingerprint(ds)
    return ds


def load_bounded(name: str) -> dict:
    data, meta = c.load_primary(name)  # frozen loader; checks the input hash against input.json
    targets = {"primary": data["ref"]}
    K = {"primary": int(meta["n_classes"])}
    native, sources = None, {"input.npz": meta["matrix_sha256"]}
    if name in c.LUNG:
        y, classes = c.celltype_labels(data)
        targets["cell_type"], K["cell_type"] = y, len(classes)
        e4 = _e4_input("NSCLC", name)
        if not (np.array_equal(e4["coords"], data["coords"]) and np.array_equal(e4["obs_id"], data["obs_id"])
                and np.array_equal(e4["y_region"], data["ref"]) and np.array_equal(e4["y_celltype"], y)):
            raise ValueError(f"{name}: E4 input disagrees with the bounded object")
        native = {"expr_log": e4["expr_log"], "genes": e4["genes"], "obs_id": e4["obs_id"]}
        sources["e4_input.npz"] = e4["sha256"]
    return _finish(dict(id=name, cohort=str(meta["cohort"]), unit=name, x_common=data["x"], coords=data["coords"],
                        frames=data["field"], obs_id=data["obs_id"], targets=targets, K=K, native=native,
                        sources=sources))


def load_her2st(stem: str) -> dict:
    unit = c.her2st_unit(stem)  # frozen join, preprocessing (PCA50) and targets of round 2
    e4 = _e4_input("HER2ST", stem)
    if not (np.array_equal(e4["coords"], unit["coords"]) and np.array_equal(e4["y_all"], unit["targets"]["all"][0])
            and np.array_equal(e4["y_boundary_T1"], unit["targets"]["boundary_T1"][0])):
        raise ValueError(f"HER2ST {stem}: E4 input disagrees with the frozen unit")
    targets = {t: y for t, (y, _) in unit["targets"].items()}
    K = {t: len(classes) for t, (_, classes) in unit["targets"].items()}
    native = {"counts": e4["counts"], "genes": e4["genes"], "obs_id": e4["obs_id"]}
    return _finish(dict(id=f"HER2ST-{stem}", cohort="HER2ST", unit=f"HER2ST-{unit['base']['patient']}",
                        x_common=unit["x"], coords=unit["coords"], frames=unit["fields"], obs_id=e4["obs_id"],
                        targets=targets, K=K, native=native, sources={"e4_input.npz": e4["sha256"]}))


def load_r3(cohort: str, dataset: str) -> dict:
    """Loader stub for cohorts prepared in WP5 (none on disk yet).

    Contract of the writer revision-r3/cohorts/common_cohorts.write_dataset: data-r3/<cohort>/<dataset>/
    input.npz holds x (PCA50), coords and field, and for each target t the codes y_<t> (-1 = missing) and the
    class names classes_<t> (K = their number); input.json gives the unit. obs_id is optional. The writer
    stores no counts; a dataset gets a native input only if counts and genes are added.
    """
    folder = DATA_R3 / cohort / dataset
    with np.load(folder / "input.npz", allow_pickle=False) as d:
        arrays = {k: d[k] for k in d.files}
    info = json.loads((folder / "input.json").read_text())
    names = [key[2:] for key in arrays if key.startswith("y_")]
    n = len(arrays["x"])
    obs_id = arrays.get("obs_id", np.array([f"{dataset}:{i}" for i in range(n)]))
    native = None
    if "counts" in arrays:
        native = {"counts": arrays["counts"], "genes": arrays["genes"], "obs_id": obs_id}
    return _finish(dict(id=f"{cohort}/{dataset}", cohort=cohort, unit=str(info.get("unit", dataset)),
                        x_common=arrays["x"], coords=arrays["coords"], frames=arrays["field"], obs_id=obs_id,
                        targets={t: arrays[f"y_{t}"] for t in names},
                        K={t: len(arrays[f"classes_{t}"]) for t in names}, native=native,
                        sources={"input.npz": sha(folder / "input.npz")}))


def load_dataset(dataset_id: str) -> dict:
    if dataset_id in c.OBJECTS:
        return load_bounded(dataset_id)
    if dataset_id in HER2ST:
        return load_her2st(dataset_id.split("-", 1)[1])
    if "/" in dataset_id:
        return load_r3(*dataset_id.split("/", 1))
    raise ValueError(f"unknown dataset {dataset_id!r}")

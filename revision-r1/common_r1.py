"""Paths, frozen-module imports and shared helpers for the revision R1 analyses.

The frozen estimator, preprocessing, graph and control code are imported from
``current-release/source`` and the project validation folders; nothing there
is modified. Data are read from the project-level ``data/`` and ``results/``
directories, and new outputs go to ``results/revision-r1/`` only.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PROJECT = REPO.parent
FROZEN = REPO / 'current-release' / 'source'
PROTOCOL = HERE / 'PROTOCOL-R1.md'


def _first(env: str, candidates, probe: str) -> Path:
    """Environment override, else the first candidate containing ``probe``.

    The author's project keeps full data next to the repository; the complete software
    archive keeps the same files under ``current-release/``.
    """
    if os.environ.get(env):
        return Path(os.environ[env])
    for path in candidates:
        if (path / probe).exists():
            return path
    return candidates[0]


DATA = _first('TESSERA_DATA', [PROJECT / 'data', REPO / 'current-release' / 'data'], 'NSCLC_Lung5_Rep2/input.npz')
RESULTS = _first('TESSERA_RESULTS', [PROJECT / 'results', REPO / 'current-release' / 'results'],
                 'NSCLC_Lung5_Rep2/Tessera-seed1.npz')
VALIDATION = _first('TESSERA_VALIDATION', [PROJECT / 'validation', REPO / 'current-release'],
                    'validation-external/raw/A1.tsv.gz')
OUT = Path(os.environ.get('TESSERA_R1_OUT') or RESULTS / 'revision-r1')

for _path in [VALIDATION / 'validation-extension', VALIDATION / 'validation-external',
              FROZEN / 'src', FROZEN / 'scripts']:
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_experiments as rx  # noqa: E402
import run_followup as fu  # noqa: E402
from common import OBJECTS, SEEDS, encode, valid  # noqa: E402

fu.DATA = DATA
fu.RESULTS = RESULTS

LUNG = OBJECTS[:5]
BCC = OBJECTS[5:9]
OTHER = OBJECTS[9:]
HER2ST = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1']
TISSUES = ['DLPFC_151673', 'STARmap_visual_cortex', 'MERFISH_hypothalamus']
CHECKPOINTS = [40, 80, 120, 160, 200, 300, 400]


def sha(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def environment() -> dict:
    import torch
    packages = {}
    for name in ['numpy', 'scipy', 'scikit-learn', 'torch', 'pandas']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(hostname=socket.gethostname(), python=sys.version, platform=platform.platform(),
                packages=packages, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                torch_threads=torch.get_num_threads())


def write_lock(directory: Path, analysis: str, extra: dict | None = None, name: str = 'run-lock.json') -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    scripts = sorted(HERE.glob('*.py'))
    record = dict(analysis=analysis, protocol='PROTOCOL-R1.md', protocol_sha256=sha(PROTOCOL),
                  scripts_sha256={p.name: sha(p) for p in scripts},
                  frozen_sha256={p.relative_to(REPO).as_posix(): sha(p) for p in
                                 sorted((FROZEN / 'src').rglob('*.py'))},
                  environment=environment(), **(extra or {}))
    path = directory / name
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['protocol_sha256'] != record['protocol_sha256']:
            raise ValueError(f'{analysis}: protocol changed since this output root was locked')
    else:
        path.write_text(json.dumps(record, indent=2) + '\n')
    return record


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def load_primary(name: str):
    data, meta = fu.load(name)
    return data, meta


def celltype_labels(data):
    y, classes = encode(data['secondary_name'])
    return y, classes


@functools.lru_cache(maxsize=None)
def her2st_unit(stem: str) -> dict:
    """One HER2ST section with the frozen join, preprocessing and targets."""
    import run_extension as rex
    import run_enhanced_targets as et
    import run_her2st_validation as hv
    base, matrix, genes, coords, raw = hv.load_joined(stem)
    pcs, *_ = hv.preprocess(matrix, genes)
    fields = np.full(len(raw), stem)
    targets = {'all': encode(raw), 'compartments': encode(et.recode(raw, 'compartments')),
               'boundary_T1': encode(rex.boundary_labels(raw, 'T1'))}
    return dict(name=stem, cohort='HER2ST', x=pcs, coords=coords, fields=fields, targets=targets, base=base)


def her2st_units() -> dict:
    return {stem: her2st_unit(stem) for stem in HER2ST}


def tissue_units() -> dict:
    import run_extension as rex
    if os.environ.get('TESSERA_LAB_ROOT'):
        lab = Path(os.environ['TESSERA_LAB_ROOT'])
        rex.BREADTH = {name: (lab / path.relative_to(rex.LAB), column) for name, (path, column) in rex.BREADTH.items()}
    units = {}
    for name in TISSUES:
        info, pcs, coords, fields, raw = rex.load_breadth(name)
        units[name] = dict(name=name, cohort='tissue', x=pcs, coords=coords, fields=fields,
                           targets={'anatomy': encode(raw)}, info=info)
    return units


exact_sign_flip = fu.exact_sign_flip


def select_nested(means, held_out, exclude_prefix=None):
    """Configuration with the highest mean over the other patients.

    ``means`` is a patients x configurations table of seed-mean ARI. The held-out
    patient's row is dropped before anything is computed.
    """
    others = means.drop(index=held_out)
    columns = [c for c in others.columns if not (exclude_prefix and str(c).startswith(exclude_prefix))]
    return others[columns].mean().idxmax()


def bootstrap_interval(values, draws: int = 10000, seed: int = 0):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(deltas, margin: float = 0.02, bootstrap: bool = False) -> dict:
    deltas = np.asarray(deltas, float)
    n = len(deltas)
    pos, neg = int((deltas > 0).sum()), int((deltas < 0).sum())
    mean = float(deltas.mean())
    if mean <= -margin and neg > n / 2:
        outcome = 'retained'
    elif mean >= margin and pos > n / 2:
        outcome = 'reversed'
    else:
        outcome = 'closed'
    result = dict(n=n, mean=mean, n_pos=pos, n_neg=neg, p_sign_flip=exact_sign_flip(deltas),
                  min=float(deltas.min()), max=float(deltas.max()), outcome=outcome)
    if bootstrap:
        result['bootstrap_95'] = bootstrap_interval(deltas)
    return result

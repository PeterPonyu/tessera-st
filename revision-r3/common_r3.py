"""Paths, loaders and run records for the round-3 revision analyses.

The frozen estimator, preprocessing and control code are imported through the
round-1 helpers (``revision-r1/common_r1.py``); nothing there is modified. New
outputs go to ``results/revision-r3/`` only.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
R1 = REPO / 'revision-r1'
if str(R1) not in sys.path:
    sys.path.insert(0, str(R1))

import common_r1 as c1  # noqa: E402  (also puts the frozen modules on sys.path)
from common import encode  # noqa: E402

PROTOCOL = HERE / 'PROTOCOL-R3.md'
OUT = Path(os.environ.get('TESSERA_R3_OUT') or c1.RESULTS / 'revision-r3')
E1_EMB = c1.RESULTS / 'revision-r1' / 'E1' / 'embeddings'
SEEDS = (1, 2, 3)

# Provenance class of every existing target (PROTOCOL-R3, section 2).
PROVENANCE = {
    ('NSCLC', 'primary'): 'computed-niche',
    ('NSCLC', 'celltype'): 'expression-assigned',
    ('BCC', 'primary'): 'expression-assigned',
    ('Breast_Visium', 'primary'): 'dominant-category',
    ('Colorectal_TVA232', 'primary'): 'expression-assigned',
    ('MIBI', 'primary'): 'expression-assigned',
    ('OpenST_HNSCC26', 'primary'): 'expression-assigned',   # source molecular annotation
    ('HER2ST', 'all'): 'pathologist',
    ('HER2ST', 'compartments'): 'pathologist',
    ('HER2ST', 'boundary_T1'): 'pathologist',
    ('DLPFC_151673', 'anatomy'): 'histology-anatomical',
    ('STARmap_visual_cortex', 'anatomy'): 'histology-anatomical',
    ('MERFISH_hypothalamus', 'anatomy'): 'histology-anatomical',
}

# Development data (Supplementary Note S1): reported, never counted as independent.
DEVELOPMENT = {'MIBI', 'OpenST_HNSCC26', 'DLPFC_151673', 'STARmap_visual_cortex', 'MERFISH_hypothalamus'}


def _cohort(name: str) -> str:
    if name.startswith('NSCLC'):
        return 'NSCLC'
    if name.startswith('BCC'):
        return 'BCC'
    return name


def bounded_units() -> list[dict]:
    """The 13 bounded objects with their primary target and, for lung, cell types."""
    units = []
    for name in c1.OBJECTS:
        d, meta = c1.load_primary(name)
        cohort = _cohort(name)
        targets = {'primary': (np.asarray(d['ref'], np.int64), list(meta['classes']))}
        if cohort == 'NSCLC':
            y, classes = c1.celltype_labels(d)
            targets['celltype'] = (np.asarray(y, np.int64), list(classes))
        units.append(dict(name=name, cohort=cohort, unit=name, x=np.asarray(d['x'], np.float64),
                          coords=np.asarray(d['coords'], float), frames=np.asarray(d['field']).astype(str),
                          targets=targets, tessera=('primary', name)))
    return units


def her2st_units() -> list[dict]:
    units = []
    for stem in c1.HER2ST:
        u = c1.her2st_unit(stem)
        targets = {t: (np.asarray(y, np.int64), list(classes)) for t, (y, classes) in u['targets'].items()}
        units.append(dict(name=f'HER2ST_{stem}', cohort='HER2ST', unit=stem, x=np.asarray(u['x'], np.float64),
                          coords=np.asarray(u['coords'], float), frames=np.asarray(u['fields']).astype(str),
                          targets=targets, tessera=('e1', stem)))
    return units


def tissue_units() -> list[dict]:
    units = []
    for name, u in c1.tissue_units().items():
        y, classes = u['targets']['anatomy']
        units.append(dict(name=name, cohort=name, unit=name, x=np.asarray(u['x'], np.float64),
                          coords=np.asarray(u['coords'], float), frames=np.asarray(u['fields']).astype(str),
                          targets={'anatomy': (np.asarray(y, np.int64), list(classes))}, tessera=('e1', name)))
    return units


def existing_units() -> list[dict]:
    """All 24 datasets analyzed before round 3, in manuscript order."""
    return bounded_units() + her2st_units() + tissue_units()


def tessera_embeddings(unit: dict) -> list[np.ndarray]:
    """Saved 120-epoch Tessera-ST embeddings for seeds 1-3.

    Bounded objects use the original primary fits; HER2ST and the tissue panel use
    the round-1 refits (one CPU thread, fixed two-cluster head), the primary fits of
    the analyses added in review.
    """
    kind, key = unit['tessera']
    if kind == 'primary':
        return [np.load(c1.RESULTS / key / f'Tessera-seed{s}.npz')['embedding'] for s in SEEDS]
    return [np.load(E1_EMB / f'{key}-Tessera-seed{s}.npz')['z120'] for s in SEEDS]


def provenance(unit: dict, target: str) -> str:
    return PROVENANCE[(unit['cohort'], target)]


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def write_lock(directory: Path, analysis: str, frozen_protocol: bool = False, extra: dict | None = None) -> dict:
    """Record protocol and script hashes, the environment and the git head before an analysis writes.

    An output root whose earlier lock names a different protocol hash is refused when
    ``frozen_protocol`` is set (post-freeze analyses); calibration analyses run under
    the draft record the draft hash and are rerun after any definition change.
    """
    import subprocess
    directory.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain', '--', 'revision-r3'],
                           capture_output=True, text=True).stdout.strip()
    record = dict(analysis=analysis, protocol='PROTOCOL-R3.md', protocol_sha256=c1.sha(PROTOCOL),
                  protocol_frozen=frozen_protocol, git_head=head, revision_r3_uncommitted=bool(dirty),
                  scripts_sha256={p.name: c1.sha(p) for p in sorted(HERE.glob('*.py'))},
                  environment=c1.environment(), **(extra or {}))
    path = directory / 'run-lock.json'
    if path.exists() and frozen_protocol:
        previous = json.loads(path.read_text())
        if previous['protocol_sha256'] != record['protocol_sha256']:
            raise ValueError(f'{analysis}: protocol changed since this output root was locked')
    path.write_text(json.dumps(record, indent=2) + '\n')
    return record

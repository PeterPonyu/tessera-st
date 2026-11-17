"""Paths, the frozen file manifest and the bounded-input writer shared by the round-3 cohort scripts.

Raw files live in ``data-r3/raw/<cohort>/`` next to their download receipts; prepared inputs in
``data-r3/<cohort>/<dataset>/``. The frozen preprocessing (``run_her2st_validation.preprocess``) and
the frozen label encoder are imported, never copied, so the new inputs are normalized, reduced and
encoded like the existing ones. The bound follows the wording of PROTOCOL-R3 section 3.3, which
differs from the round-2 ``prepare_data.selection`` (salted hash; subsampling only when no whole
frame fits), so that function is not used here.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import common_r3 as c3  # noqa: E402  (puts the frozen modules on sys.path through common_r1)
import diagnostics_r3 as dg  # noqa: E402
import run_her2st_validation as hv  # noqa: E402  frozen preprocessing: preprocess(matrix, genes)
from common import encode  # noqa: E402  frozen label encoder

c1 = c3.c1
DATA_R3 = Path(os.environ.get('TESSERA_DATA_R3') or c1.PROJECT / 'data-r3')
CAP = 6000
VISIUM_PITCH_UM = 100.0
UNLABELLED = ('unlabelled', 'unlabeled')   # missing in addition to the frozen encoder's empty/NaN tokens

# Frozen file manifest, read from the Zenodo API on 17 Nov 2026: file -> (bytes, md5). fetch_r3.py
# fetches nothing else and refuses when the live listing disagrees. The colorectal files are the
# pilot rule's choice (fetch_r3.colorectal_selection), re-derived from the live listing before a fetch.
COHORTS = {
    'spinal': dict(
        record='18039571', doi='10.5281/zenodo.18039571',
        source='Wang et al. 2026, PNAS 123:e2623177123',
        files={'MERFISH_spinal_cord_resolved_0718.h5ad': (98970238, 'ce06f62c0ec4973581dae17bb76f0cd9'),
               'SNI_merged_0917.h5ad': (115901575, '7e90a801ee57b8fec06cd03c8630f01b')}),
    'colorectal': dict(
        record='7760264', doi='10.5281/zenodo.7760264',
        source='Valdeolivas et al. 2024, npj Precis Oncol 8:10, doi:10.1038/s41698-023-00488-4',
        files={'SN048_A121573_Rep1.zip': (149798389, '608a39f21da059024121407967a76b8a'),
               'SN048_A416371_Rep2.zip': (133062616, '9a49067a4a6d89521601b562011dc02e'),
               'SN123_A798015_Rep1.zip': (81914465, '0e589a2c96546fff5ca21cb18d597791'),
               'SN123_A938797_Rep1_X.zip': (108570769, '08e3c7fe308db62b658cacc6b4be4e1c')}),
    'wu_breast': dict(
        record='4739739', doi='10.5281/zenodo.4739739',
        source='Wu et al. 2021, Nat Genet; CC BY 4.0',
        files={'filtered_count_matrices.tar.gz': (155703418, 'ea1220f0606c4d4e9307468fb2d5427a'),
               'spatial.tar.gz': (54282871, '29b02ae433d459d19dc52b75a529e8e9'),
               'metadata.tar.gz': (217533, '1d0b34ead70635c094ac74ac58a88d68')}),
}

sha256 = c1.sha
dump = c3.dump


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def receipt_path(path) -> Path:
    return Path(path).with_name(Path(path).name + '.receipt.json')


def verified(path, receipt=None) -> str:
    """SHA-256 of a raw file, checked against its download receipt."""
    expected = json.loads(Path(receipt or receipt_path(path)).read_text())['sha256']
    digest = sha256(path)
    if digest != expected:
        raise RuntimeError(f'{Path(path).name}: SHA-256 differs from its receipt')
    return digest


def freeze_record() -> dict | None:
    """The commit titled 'Freeze PROTOCOL-R3', if it exists and the protocol has no uncommitted change."""
    git = ['git', '-C', str(c1.REPO)]
    path = 'revision-r3/PROTOCOL-R3.md'
    log = subprocess.run(git + ['log', '--format=%H%x09%cI%x09%s', '--', path], capture_output=True, text=True)
    dirty = subprocess.run(git + ['status', '--porcelain', '--', path], capture_output=True, text=True)
    if log.returncode or dirty.returncode or dirty.stdout.strip():
        return None
    for line in log.stdout.splitlines():
        commit, when, subject = line.split('\t', 2)
        if subject.startswith('Freeze PROTOCOL-R3'):
            return dict(freeze_commit=commit, freeze_committed=when, protocol_sha256=sha256(c3.PROTOCOL))
    return None


def bound(frames, eligible):
    """Label-blind bound of one unit (PROTOCOL-R3, section 3.3); returns kept row indices and a record.

    Whole frames are visited in SHA-256 order of the frame identifier (UTF-8) and each is admitted if
    the running total of eligible observations stays at or below 6,000. A first frame that alone
    exceeds 6,000 is subsampled to 6,000 with numpy RandomState(0). Labels are not an argument.
    """
    frames = np.asarray(frames, str)
    eligible = np.asarray(eligible, bool)
    order = sorted(set(frames[eligible]), key=lambda f: hashlib.sha256(f.encode()).hexdigest())
    members = {f: np.flatnonzero(eligible & (frames == f)) for f in order}
    if order and len(members[order[0]]) > CAP:
        chosen = {order[0]: np.random.RandomState(0).choice(members[order[0]], CAP, replace=False)}
    else:
        chosen, total = {}, 0
        for f in order:
            if total + len(members[f]) <= CAP:
                chosen[f], total = members[f], total + len(members[f])
    keep = np.sort(np.concatenate(list(chosen.values()))) if chosen else np.empty(0, np.int64)
    visits = [dict(frame=f, n_eligible=len(members[f]), n_selected=len(chosen.get(f, ()))) for f in order]
    record = dict(rule='PROTOCOL-R3 3.3: whole frames in SHA-256 order of the frame identifier, each admitted if the '
                       'total stays <= 6,000; an oversized first frame is subsampled with RandomState(0); labels not read',
                  cap=CAP, n_rows=len(frames), n_eligible=int(eligible.sum()), n_selected=len(keep), visits=visits)
    return keep, record


def visium_scale(coords, frames) -> dict:
    """Micrometres per pixel from the 100 um Visium spot pitch and the median nearest-neighbour distance."""
    _, dist, _ = dg.frame_knn(coords, frames, 1)
    return dict(coordinate_units='full-resolution image pixels',
                um_per_coordinate_unit=VISIUM_PITCH_UM / float(np.nanmedian(dist[:, 0])),
                um_per_coordinate_unit_rule='100 um Visium spot pitch / median within-frame nearest-neighbour distance')


def encode_released(raw):
    """Classes as released; empty, NaN or 'unlabelled' values become -1 (otherwise the frozen encoder)."""
    raw = pd.Series(np.asarray(raw, dtype=object)).fillna('').astype(str).to_numpy(str)
    raw = np.where(np.isin(np.char.lower(np.char.strip(raw)), UNLABELLED), '', raw)
    y, classes = encode(raw)
    return np.asarray(y, np.int64), [str(c) for c in classes]


def write_dataset(root, cohort, dataset, x, coords, frames, targets, meta) -> dict:
    """Write ``<root>/<cohort>/<dataset>/input.npz`` and ``input.json``, then refresh ``<root>/index.json``.

    ``targets`` maps a target name to (codes with -1 missing, class names, provenance class).
    """
    x, coords, frames = np.asarray(x, np.float32), np.asarray(coords, np.float64), np.asarray(frames, str)
    n = len(x)
    if coords.shape != (n, 2) or frames.shape != (n,) or not (np.isfinite(x).all() and np.isfinite(coords).all()):
        raise ValueError(f'{dataset}: inconsistent or nonfinite arrays')
    arrays, described = dict(x=x, coords=coords, field=frames), {}
    for name, (y, classes, provenance) in targets.items():
        y = np.asarray(y, np.int64)
        if y.shape != (n,) or y.min(initial=-1) < -1 or y.max(initial=-1) >= len(classes):
            raise ValueError(f'{dataset}: codes of target {name} out of range')
        arrays[f'y_{name}'], arrays[f'classes_{name}'] = y, np.asarray(classes, str)
        described[name] = dict(provenance=provenance, n_classes=len(classes), n_missing=int((y < 0).sum()),
                               class_counts={c: int((y == i).sum()) for i, c in enumerate(classes)})
    out = Path(root) / cohort / dataset
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'input.npz', **arrays)
    _, dist, _ = dg.frame_knn(coords, frames, 6)
    names, sizes = np.unique(frames, return_counts=True)
    info = dict(cohort=cohort, dataset=dataset, **meta, n_obs=n, n_pc=int(x.shape[1]),
                frames={str(f): int(s) for f, s in zip(names, sizes)},
                median_nn_distance=float(np.nanmedian(dist[:, 0])), median_6nn_distance=float(np.nanmedian(dist[:, -1])),
                targets=described, input_sha256=sha256(out / 'input.npz'), written_utc=utc(),
                protocol_sha256=sha256(c3.PROTOCOL), protocol_freeze=freeze_record(),
                code_sha256={'run_her2st_validation.py': sha256(hv.__file__),
                             **{p.name: sha256(p) for p in sorted(HERE.glob('*.py'))}})
    dump(out / 'input.json', info)
    write_index(root)
    return info


def write_index(root) -> None:
    """``<root>/index.json``: every prepared dataset, rebuilt from the input.json files on disk."""
    rows = []
    for path in sorted(Path(root).glob('*/*/input.json')):
        if path.parts[-3] == 'raw':
            continue
        info = json.loads(path.read_text())
        rows.append(dict(cohort=info['cohort'], dataset=info['dataset'], unit=info['unit'], n_obs=info['n_obs'],
                         n_frames=len(info['frames']), targets=sorted(info['targets']),
                         input=str(path.parent.relative_to(root) / 'input.npz'), input_sha256=info['input_sha256']))
    dump(Path(root) / 'index.json', dict(generated_utc=utc(), datasets=rows))


def md5_and_sha256(path) -> tuple[str, str]:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            md5.update(block)
            sha.update(block)
    return md5.hexdigest(), sha.hexdigest()

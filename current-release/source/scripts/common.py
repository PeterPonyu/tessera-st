"""Shared traceability helpers for the frozen tumor-associated expansion."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
SEEDS = [1, 2, 3]
ARMS = ['PCA50', 'PCA30', 'NeighborMean', 'Tessera', 'NoGate', 'NoScale',
        'NoContrast', 'SelfOnly', 'PermutedXY']
OBJECTS = ['NSCLC_Lung5_Rep2', 'NSCLC_Lung6', 'NSCLC_Lung9_Rep1',
           'NSCLC_Lung12', 'NSCLC_Lung13', 'BCC_PatientA', 'BCC_PatientB',
           'BCC_PatientC', 'BCC_PatientD', 'Breast_Visium', 'Colorectal_TVA232',
           'MIBI', 'OpenST_HNSCC26']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def read(path):
    return json.loads(Path(path).read_text())

def valid(raw):
    return ~np.isin(np.char.lower(np.char.strip(np.asarray(raw, str))),
                   ['', 'nan', 'none', '<na>', 'unknown', 'na'])

def encode(raw):
    raw = np.asarray(raw, str)
    ok = valid(raw)
    classes = sorted(set(raw[ok]))
    lookup = {c: i for i, c in enumerate(classes)}
    y = np.array([lookup[v] if keep else -1 for v, keep in zip(raw, ok)])
    return y, classes

def code_hashes():
    files = sorted((ROOT / 'source').rglob('*.py')) + [ROOT / 'PROTOCOL.md']
    return {str(p.relative_to(ROOT)): sha(p) for p in files}

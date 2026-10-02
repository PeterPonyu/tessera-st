"""Portable data roots for local experiment scripts.

Set one of:
  TESSERA_DATA_ROOT / SPATIAL_OMICS_ROOT  — path to the spatial-omics data checkout
  LABS_ROOT                              — parent of ``active/spatial-omics-reform``

If unset, falls back to a sibling ``spatial-omics-reform`` directory next to this
repo (no machine-specific absolute path in source).
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def spatial_omics_root() -> Path:
    for key in ("TESSERA_DATA_ROOT", "SPATIAL_OMICS_ROOT"):
        raw = os.environ.get(key)
        if raw:
            return Path(raw).expanduser().resolve()
    labs = os.environ.get("LABS_ROOT")
    if labs:
        return (Path(labs).expanduser() / "active" / "spatial-omics-reform").resolve()
    return (_REPO_ROOT.parent / "spatial-omics-reform").resolve()


def data_root() -> Path:
    return spatial_omics_root() / "data"


def external_root() -> Path:
    return spatial_omics_root() / "external"

"""Path contract for re-running the submitted figure producers unchanged.

The copied producers in this folder differ from their originals only in where
they read and write: inputs are the frozen project data, results, validation
records and legacy manuscript ledgers (read-only); every output goes below OUT,
a scratch folder outside the project (set INVENTORY_OUT to choose it).
"""
from pathlib import Path
import os

PROJECT = Path(__file__).resolve().parents[4]
assert (PROJECT / 'data').is_dir() and (PROJECT / 'results').is_dir(), PROJECT
LEGACY = PROJECT / 'manuscripts/scientific-reports'
LEGACY_MAN = LEGACY / 'source/manuscript'
SUBMITTED = PROJECT / 'submissions/frontiers-in-genetics/neighborhood-averaging-spatial-gains'
DENSE_R2 = PROJECT / 'submissions/frontiers-in-genetics/dense-20261014-r2'
VALIDATION_EXTENSION = PROJECT / 'validation/validation-extension/results'
OUT = Path(os.environ.get('INVENTORY_OUT', '/tmp/tessera-figure-inventory')).resolve()
assert PROJECT not in OUT.parents and OUT != PROJECT, 'inventory output must stay outside the project'

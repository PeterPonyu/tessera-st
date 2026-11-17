#!/usr/bin/env python3
"""Bounded per-mouse inputs from the spinal-cord MERFISH release (PROTOCOL-R3, section 3).

Unit = mouse, frame = section. In every branch the cell-type labels at each deposited level are
targets (``--celltype``, coarsest level first, named celltype_l1, celltype_l2, ...). Laminae come from
branch A (deposited per-cell lamina labels, ``--lamina``) or branch B (deposited per-cell relative
depth, ``--depth``, binned at the boundaries the paper states numerically, LAMINA_BINS); branch C
(neither) builds no lamina target. Column names come from the inspect_r3.py receipt and are passed
explicitly; the branch follows from which lamina source is given.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_cohorts as cc  # noqa: E402

COHORT = 'spinal'
FILES = {'healthy': 'MERFISH_spinal_cord_resolved_0718.h5ad', 'sni': 'SNI_merged_0917.h5ad'}
EXPECTED_MICE = {'healthy': 10}

# Branch B only: laminae as bins of the deposited relative depth, copied from the paper before any
# target is built, as (lamina name, lower edge inclusive, upper edge exclusive) in the file's depth
# units, in increasing depth. Cells outside every bin are missing (-1). While this table is empty,
# branch B refuses to run.
LAMINA_BINS: list[tuple[str, float, float]] = []

UNIT_WORDS = {'um': 1.0, '\u00b5m': 1.0, '\u03bcm': 1.0, 'micron': 1.0, 'microns': 1.0, 'micrometer': 1.0,
              'micrometre': 1.0, 'micrometers': 1.0, 'micrometres': 1.0, 'mm': 1000.0}


def check_bins(bins) -> None:
    if not bins:
        raise ValueError('branch B needs LAMINA_BINS copied from the paper; the table is empty')
    edges = [(float(lo), float(hi)) for _, lo, hi in bins]
    if any(lo >= hi for lo, hi in edges) or any(a[1] > b[0] for a, b in zip(edges, edges[1:])):
        raise ValueError('LAMINA_BINS must be increasing, non-overlapping intervals')


def lamina_from_depth(depth, bins=None):
    """Branch B target: the LAMINA_BINS interval holding each cell's deposited relative depth."""
    bins = LAMINA_BINS if bins is None else bins
    check_bins(bins)
    depth = np.asarray(depth, float)
    y = np.full(len(depth), -1, np.int64)
    for code, (_, lo, hi) in enumerate(bins):
        y[(depth >= lo) & (depth < hi)] = code
    return y, [str(name) for name, _, _ in bins]


def coordinates(a, spec: str):
    """``obsm:KEY`` (a two-column array) or ``obs:XCOL,YCOL``; returns the array and the field names."""
    kind, _, name = spec.partition(':')
    if kind == 'obsm':
        xy, names = np.asarray(a.obsm[name], float), [name]
    elif kind == 'obs' and len(name.split(',')) == 2:
        names = name.split(',')
        xy = a.obs[names].to_numpy(float)
    else:
        raise ValueError('--coords must be obsm:KEY or obs:XCOL,YCOL')
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f'coordinates {spec} are not two-dimensional')
    return xy, names


def coordinate_unit(a, names):
    """The coordinate unit stated in the file (a uns entry naming a unit, or a unit suffix on the
    coordinate field names) and where it was stated; (None, None) if the file does not say."""
    for key, value in a.uns.items():
        if 'unit' in str(key).lower() and isinstance(value, str) and value.strip().lower() in UNIT_WORDS:
            return value.strip().lower(), f'uns[{key!r}]'
    for name in names:
        match = re.search(r'[_\s(\[]([a-z\u00b5\u03bc]+)[)\]]?$', name.lower())
        if match and match.group(1) in UNIT_WORDS:
            return match.group(1), f'field name {name!r}'
    return None, None


def expression(a, spec: str):
    if spec == 'X':
        return a.X, a.var_names.to_numpy(str)
    if spec == 'raw':
        return a.raw.X, a.raw.var_names.to_numpy(str)
    if spec.startswith('layer:'):
        return a.layers[spec[6:]], a.var_names.to_numpy(str)
    raise ValueError("--matrix must be 'X', 'raw' or 'layer:<name>'")


def prepare(h5ad, tag, *, mouse, section, coords, celltypes, lamina=None, depth=None, group=None, matrix='X',
            confirm_branch_c=False, root=cc.DATA_R3) -> list[dict]:
    """Write one bounded input per mouse; returns their input.json records."""
    import anndata as ad
    if lamina and depth:
        raise ValueError('give --lamina (branch A) or --depth (branch B), not both')
    if not celltypes:
        raise ValueError('at least one deposited cell-type level is required')
    branch = 'A' if lamina else 'B' if depth else 'C'
    if branch == 'B':
        check_bins(LAMINA_BINS)                       # refuse before anything is read or written
    h5ad = Path(h5ad)
    digest = cc.verified(h5ad)
    a = ad.read_h5ad(h5ad)
    obs = a.obs
    hints = [str(c) for c in obs.columns if re.search('lamin|depth', str(c), re.I)] if branch == 'C' else []
    if hints and not confirm_branch_c:
        raise ValueError(f'obs columns {hints} may hold laminae or depth: pass --lamina or --depth, '
                         'or --confirm-branch-c')
    if obs[mouse].isna().any() or obs[section].isna().any():
        raise ValueError('missing mouse or section identifiers')
    mice = obs[mouse].astype(str).to_numpy()
    frames = np.array([f'{m}|{s}' for m, s in zip(mice, obs[section].astype(str).to_numpy())])
    units = sorted(set(mice))
    if tag in EXPECTED_MICE and len(units) != EXPECTED_MICE[tag]:
        raise ValueError(f'{len(units)} mice in column {mouse!r}; the release describes {EXPECTED_MICE[tag]}')
    values, genes = expression(a, matrix)
    xy, names = coordinates(a, coords)
    eligible = np.isfinite(xy).all(1) & (np.asarray(abs(values).sum(1)).ravel() > 0)
    unit_name, unit_source = coordinate_unit(a, names)
    spec = cc.COHORTS[COHORT]
    written = []
    for unit in units:
        rows = np.flatnonzero(mice == unit)
        groups = sorted(set(obs[group].astype(str).to_numpy()[rows])) if group else [None]
        if len(groups) != 1:
            raise ValueError(f'mouse {unit}: column {group!r} is not constant within the mouse')
        keep, selection = cc.bound(frames[rows], eligible[rows])     # label-blind: no target is passed
        rows = rows[keep]
        block = values[rows]
        dense = block.toarray() if sparse.issparse(block) else np.asarray(block)
        x, normalized, removed, n_pc, explained = cc.hv.preprocess(dense.astype(float), genes)
        targets = {f'celltype_l{level}': (*cc.encode_released(obs[column].to_numpy()[rows]), 'expression-assigned')
                   for level, column in enumerate(celltypes, 1)}
        if branch == 'A':
            targets['lamina'] = (*cc.encode_released(obs[lamina].to_numpy()[rows]), 'geometry-derived')
        elif branch == 'B':
            targets['lamina'] = (*lamina_from_depth(obs[depth].to_numpy()[rows]), 'geometry-derived')
        meta = dict(unit=str(unit), unit_kind='mouse', frame_kind='section', group=groups[0],
                    source=spec['source'], source_doi=spec['doi'], source_file=h5ad.name, source_sha256=digest,
                    protocol_branch=branch, lamina_bins=[list(b) for b in LAMINA_BINS] if branch == 'B' else None,
                    lamina_like_columns_not_used=hints,
                    columns=dict(mouse=mouse, section=section, coords=coords, lamina=lamina, depth=depth, group=group,
                                 **{f'celltype_l{level}': column for level, column in enumerate(celltypes, 1)}),
                    matrix=matrix, n_features=int(len(genes)), count_normalization=normalized,
                    control_features_removed=removed, pca_explained_variance=explained,
                    coordinate_units=unit_name, coordinate_unit_source=unit_source,
                    um_per_coordinate_unit=UNIT_WORDS.get(unit_name) if unit_name else None,
                    eligibility='finite coordinates and nonzero expression; labels not read', selection=selection)
        dataset = f'{tag}-' + re.sub(r'[^A-Za-z0-9._-]+', '_', str(unit))
        written.append(cc.write_dataset(root, COHORT, dataset, x, xy[rows], frames[rows], targets, meta))
    return written


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--file', choices=sorted(FILES), default='healthy')
    parser.add_argument('--mouse', required=True, help='obs column holding the mouse')
    parser.add_argument('--section', required=True, help='obs column holding the section (frame)')
    parser.add_argument('--coords', required=True, help='obsm:KEY or obs:XCOL,YCOL')
    parser.add_argument('--celltype', action='append', required=True,
                        help='obs column of one deposited cell-type level; repeat for each level, coarsest first')
    parser.add_argument('--lamina', help='obs column of deposited per-cell laminae (branch A)')
    parser.add_argument('--depth', help='obs column of deposited per-cell relative depth (branch B)')
    parser.add_argument('--group', help='obs column holding the condition (sham or SNI), recorded per mouse')
    parser.add_argument('--matrix', default='X', help="'X', 'raw' or 'layer:<name>' (the raw-count matrix)")
    parser.add_argument('--confirm-branch-c', action='store_true',
                        help='proceed without laminae although an obs column name mentions laminae or depth')
    parser.add_argument('--root', type=Path, default=cc.DATA_R3, help='data-r3 folder')
    args = parser.parse_args(argv)
    h5ad = args.root / 'raw' / COHORT / FILES[args.file]
    for info in prepare(h5ad, args.file, mouse=args.mouse, section=args.section, coords=args.coords,
                        celltypes=args.celltype, lamina=args.lamina, depth=args.depth, group=args.group,
                        matrix=args.matrix, confirm_branch_c=args.confirm_branch_c, root=args.root):
        print('PREPARED', info['dataset'], info['n_obs'], 'cells,', len(info['frames']), 'sections, branch',
              info['protocol_branch'], flush=True)


if __name__ == '__main__':
    main()

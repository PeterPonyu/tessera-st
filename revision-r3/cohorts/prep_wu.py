#!/usr/bin/env python3
"""Bounded inputs for the Wu et al. 2021 breast Visium cohort (PROTOCOL-R3, section 3).

Six patients, one section each, read straight from the three released archives (filtered counts,
spatial positions, spot metadata; images are skipped). Targets: every released pathologist class of
the column ``Classification`` as released (empty or 'unlabelled' values missing), and tumor versus
non-tumor, where a class is tumor if its name contains 'cancer', 'DCIS' or 'Invasive' (any case).
"""
from __future__ import annotations

import argparse
import gzip
import io
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_cohorts as cc  # noqa: E402

COHORT = 'wu_breast'
PATIENTS = ('1142243F', '1160920F', 'CID4290', 'CID4465', 'CID44971', 'CID4535')
LABEL_COLUMN = 'Classification'
TUMOR_WORDS = ('cancer', 'dcis', 'invasive')
MIN_MATCH = 0.95          # the pilot's annotation-join threshold
NEEDED = ('matrix.mtx.gz', 'matrix.mtx', 'features.tsv.gz', 'genes.tsv.gz', 'barcodes.tsv.gz',
          'tissue_positions_list.csv', 'tissue_positions.csv', '_metadata.csv')
POSITIONS = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']


def tumor_mapping(classes) -> dict:
    """Secondary target: a released class is 'tumor' if its name contains 'cancer', 'DCIS' or 'Invasive'."""
    return {c: 'tumor' if any(word in c.lower() for word in TUMOR_WORDS) else 'non-tumor' for c in classes}


def members(raw_dir: Path) -> dict:
    """The needed small members of the three archives, each archive read once (images are skipped)."""
    out = {}
    for name in cc.COHORTS[COHORT]['files']:
        with tarfile.open(raw_dir / name, 'r|gz') as tar:
            for item in tar:
                base = Path(item.name).name
                if item.isfile() and not base.startswith('._') and base.endswith(NEEDED):
                    out[item.name] = tar.extractfile(item).read()
    return out


def member(files: dict, patient: str, suffixes) -> bytes:
    """The single member of ``patient`` ending with one of ``suffixes`` (gzip removed); fails otherwise."""
    token = re.compile(rf'(?<![A-Za-z0-9]){re.escape(patient)}(?![0-9])')
    hits = [name for name in files if Path(name).name.endswith(suffixes) and token.search(name)]
    if len(hits) != 1:
        raise ValueError(f'{patient}: expected one member ending in {suffixes}, found {hits}')
    data = files[hits[0]]
    return gzip.decompress(data) if data[:2] == b'\x1f\x8b' else data   # some '.gz' members are stored uncompressed


def load(files: dict, patient: str):
    counts = mmread(io.BytesIO(member(files, patient, ('matrix.mtx.gz', 'matrix.mtx')))).tocsr()
    features = pd.read_csv(io.BytesIO(member(files, patient, ('features.tsv.gz', 'genes.tsv.gz'))), sep='\t',
                           header=None, dtype=str)
    barcodes = pd.read_csv(io.BytesIO(member(files, patient, ('barcodes.tsv.gz',))), sep='\t', header=None,
                           dtype=str)[0].to_numpy(str)
    if counts.shape == (len(features), len(barcodes)):
        counts = counts.T.tocsr()
    if counts.shape != (len(barcodes), len(features)):
        raise ValueError(f'{patient}: matrix shape {counts.shape} does not match barcodes and features')
    genes = features[1 if features.shape[1] > 1 else 0].to_numpy(str)
    if features.shape[1] > 2:
        expressed = features[2].to_numpy(str) == 'Gene Expression'
        counts, genes = counts[:, expressed], genes[expressed]
    if not np.isfinite(counts.data).all() or (counts.data < 0).any() or not np.equal(counts.data, np.round(counts.data)).all():
        raise ValueError(f'{patient}: counts are not finite nonnegative integers')
    text = member(files, patient, ('tissue_positions_list.csv', 'tissue_positions.csv')).decode()
    header = text.splitlines()[0].startswith('barcode,')
    positions = pd.read_csv(io.StringIO(text), header=0 if header else None, names=None if header else POSITIONS)
    positions = positions.set_index('barcode')
    meta = pd.read_csv(io.BytesIO(member(files, patient, ('_metadata.csv',))), dtype=str, index_col=0)
    if LABEL_COLUMN not in meta.columns:
        raise ValueError(f'{patient}: the spot metadata has no {LABEL_COLUMN!r} column')
    if len(set(barcodes)) != len(barcodes) or not positions.index.is_unique or not meta.index.is_unique:
        raise ValueError(f'{patient}: duplicate barcodes')
    if not pd.Index(barcodes).isin(positions.index).all():
        raise ValueError(f'{patient}: count barcodes absent from the positions')
    return counts, genes, barcodes, positions.reindex(barcodes), meta


def join_labels(meta: pd.DataFrame, barcodes, patient: str):
    """Exact barcode join; if under 95% match, one deterministic retry with a leading '<patient>_' removed."""
    index, rule = meta.index, 'exact'
    rate = float(index.isin(barcodes).mean())
    if rate < MIN_MATCH and index.str.startswith(f'{patient}_').all():
        index, rule = index.str.slice(len(patient) + 1), f"exact after removing the prefix '{patient}_'"
        rate = float(index.isin(barcodes).mean())
    if rate < MIN_MATCH or not index.is_unique:
        raise ValueError(f'{patient}: {rate:.3f} of annotated spots match the count barcodes (needs {MIN_MATCH})')
    labels = pd.Series(meta[LABEL_COLUMN].to_numpy(), index=index).reindex(barcodes).to_numpy(object)
    return labels, dict(rule=rule, match_rate=rate, n_annotated_rows=len(meta), n_count_barcodes=len(barcodes))


def prepare(root: Path = cc.DATA_R3, patients=PATIENTS) -> list[dict]:
    raw_dir = root / 'raw' / COHORT
    digests = {name: cc.verified(raw_dir / name) for name in cc.COHORTS[COHORT]['files']}
    files = members(raw_dir)
    spec, written = cc.COHORTS[COHORT], []
    for patient in patients:
        counts, genes, barcodes, positions, meta = load(files, patient)
        labels, join = join_labels(meta, barcodes, patient)
        xy = positions[['pxl_col_in_fullres', 'pxl_row_in_fullres']].to_numpy(float)
        eligible = ((positions['in_tissue'].to_numpy() == 1) & (np.asarray(counts.sum(1)).ravel() > 0)
                    & np.isfinite(xy).all(1))
        frames = np.full(len(barcodes), patient)
        keep, selection = cc.bound(frames, eligible)                   # label-blind
        x, normalized, removed, n_pc, explained = cc.hv.preprocess(counts[keep].toarray().astype(float), genes)
        y_all, classes = cc.encode_released(labels[keep])
        mapping = tumor_mapping(classes)
        is_tumor = np.array([mapping[c] == 'tumor' for c in classes], np.int64)
        y_tumor = np.full(len(y_all), -1, np.int64)
        y_tumor[y_all >= 0] = is_tumor[y_all[y_all >= 0]]
        info = dict(unit=patient, unit_kind='patient', frame_kind='section', source=spec['source'],
                    source_doi=spec['doi'], source_files_sha256=digests, label_column=LABEL_COLUMN, join=join,
                    tumor_rule="tumor if the class name contains 'cancer', 'DCIS' or 'Invasive' (case-insensitive)",
                    tumor_mapping=mapping, n_features=int(len(genes)), count_normalization=normalized,
                    control_features_removed=removed, pca_explained_variance=explained,
                    **cc.visium_scale(xy[keep], frames[keep]),
                    eligibility='in-tissue spots with nonzero counts and finite positions; labels not read',
                    selection=selection)
        targets = {'all': (y_all, classes, 'pathologist'), 'tumor': (y_tumor, ['non-tumor', 'tumor'], 'pathologist')}
        written.append(cc.write_dataset(root, COHORT, patient, x, xy[keep], frames[keep], targets, info))
    return written


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--root', type=Path, default=cc.DATA_R3, help='data-r3 folder')
    args = parser.parse_args(argv)
    for info in prepare(args.root):
        print('PREPARED', info['dataset'], info['n_obs'], 'spots,', info['targets']['all']['n_classes'], 'classes',
              flush=True)


if __name__ == '__main__':
    main()

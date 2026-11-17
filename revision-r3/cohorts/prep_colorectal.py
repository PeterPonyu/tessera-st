#!/usr/bin/env python3
"""Bounded inputs for the four colorectal Visium patients not in the pilot (PROTOCOL-R3, section 3).

One section per patient (the pilot's smallest-archive rule, fixed in common_cohorts.COHORTS). The
Space Ranger parsing, the exact barcode join with the pathologist labels and the eligibility checks
are the pilot's own ``run_pilot.load_raw`` (current-release/validation-independent-crc), imported
unchanged and pointed at a temporary folder that links the verified archives. A section failing a
pilot check is recorded in exclusions.json and never replaced by another section.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_cohorts as cc  # noqa: E402

sys.path.insert(0, str(cc.c1.REPO / 'current-release' / 'validation-independent-crc'))
import run_pilot  # noqa: E402

COHORT = 'colorectal'
ANNOTATIONS = (cc.c1.PROJECT / 'revisions' / 'tessera-20261008-expanded' / 'validation-independent-crc' / 'raw'
               / 'Pathology_SpotAnnotations.zip')
ANNOTATIONS_MD5 = '92882eb0db7c2cacc5077cb5fc9578fd'      # Zenodo 7760264 listing


@contextmanager
def pilot_folder(archive: Path, annotations: Path, archive_sha256: str):
    """A temporary folder laid out as run_pilot.load_raw expects, with run_pilot.HERE pointed at it."""
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / 'raw'
        raw.mkdir()
        (raw / archive.name).symlink_to(archive.resolve())
        (raw / 'Pathology_SpotAnnotations.zip').symlink_to(annotations.resolve())
        cc.dump(Path(tmp) / 'fetch-receipt.json', dict(files=[dict(filename=archive.name, sha256=archive_sha256)]))
        saved, run_pilot.HERE = run_pilot.HERE, Path(tmp)
        try:
            yield
        finally:
            run_pilot.HERE = saved


def prepare(section: str, root: Path, annotations: Path, annotations_sha256: str) -> dict:
    archive = root / 'raw' / COHORT / f'{section}.zip'
    digest = cc.verified(archive)
    with pilot_folder(archive, annotations, digest):
        counts, genes, barcodes, xy, raw_labels, _, join, _ = run_pilot.load_raw(section)
    frames = np.full(len(barcodes), section)
    keep, selection = cc.bound(frames, np.ones(len(barcodes), bool))     # label-blind
    x, normalized, removed, n_pc, explained = cc.hv.preprocess(counts[keep].toarray().astype(float), genes)
    y, classes = cc.encode_released(raw_labels[keep])
    spec = cc.COHORTS[COHORT]
    meta = dict(unit=join['patient'], unit_kind='patient', frame_kind='section', section=section,
                source=spec['source'], source_doi=spec['doi'], source_file=archive.name, source_sha256=digest,
                annotation_file=annotations.name, annotation_sha256=annotations_sha256,
                join={k: v for k, v in join.items() if k not in ('classes', 'class_counts')},
                n_features=int(len(genes)), count_normalization=normalized, control_features_removed=removed,
                pca_explained_variance=explained, **cc.visium_scale(xy[keep], frames[keep]),
                eligibility='in-tissue spots with nonzero counts (pilot load_raw); labels not read by the bound',
                selection=selection)
    return cc.write_dataset(root, COHORT, join['patient'], x, xy[keep], frames[keep],
                            {'all': (y, classes, 'pathologist')}, meta)


def prepare_all(sections, root: Path = cc.DATA_R3, annotations: Path = ANNOTATIONS, annotations_receipt=None):
    annotations_sha256 = cc.verified(annotations, annotations_receipt or
                                     annotations.with_name(annotations.name + '.sha256.json'))
    excluded = []
    for section in sections:
        try:
            info = prepare(section, root, annotations, annotations_sha256)
            print('PREPARED', info['dataset'], section, info['n_obs'], 'spots,', info['targets']['all']['n_classes'],
                  'classes', flush=True)
        except ValueError as error:        # a pilot eligibility or parsing check failed
            excluded.append(dict(section=section, reason=str(error)))
            print('EXCLUDED', section, error, flush=True)
    cc.dump(root / COHORT / 'exclusions.json', dict(written_utc=cc.utc(), excluded=excluded))
    return excluded


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--annotations', type=Path, default=ANNOTATIONS, help='Pathology_SpotAnnotations.zip')
    parser.add_argument('--root', type=Path, default=cc.DATA_R3, help='data-r3 folder')
    args = parser.parse_args(argv)
    if cc.md5_and_sha256(args.annotations)[0] != ANNOTATIONS_MD5:
        raise SystemExit(f'{args.annotations} differs from the Zenodo md5')
    if prepare_all([Path(name).stem for name in cc.COHORTS[COHORT]['files']], args.root, args.annotations):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

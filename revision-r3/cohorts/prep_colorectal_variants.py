"""Prepare the two new colorectal patients whose pathology tables use header variants.

The frozen pilot parser accepts the label column only as 'Pathologist Annotation(s)'.
The tables of SN123_A798015_Rep1 ('Pathologist annotation') and SN123_A938797_Rep1_X
('Pathologist_KH') each hold exactly one label column under another name. This script
renames that header in a temporary copy of the annotation archive and runs the
unchanged preparation; the frozen parser still checks that barcodes are unique. The
original header is recorded in input.json. Decided on 17 Nov 2026 after reading the
headers only, before any score (PROTOCOL-R3-DEVIATIONS.md, item 12).
"""
from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_cohorts as cc  # noqa: E402
import prep_colorectal as pc  # noqa: E402

VARIANTS = {'SN123_A798015_Rep1': 'Pathologist annotation', 'SN123_A938797_Rep1_X': 'Pathologist_KH'}
CANONICAL = 'Pathologist Annotation'


def renamed_archive(src: Path, dst: Path) -> None:
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            for section, header in VARIANTS.items():
                if item.filename.endswith(f'Pathologist_Annotations_{section}.csv'):
                    first, rest = data.split(b'\n', 1)
                    columns = first.decode().rstrip('\r').split(',')
                    assert columns == ['Barcode', header], (item.filename, columns)
                    data = f'Barcode,{CANONICAL}\n'.encode() + rest
            zout.writestr(item, data)


def main() -> None:
    original_sha = cc.verified(pc.ANNOTATIONS, pc.ANNOTATIONS.with_name(pc.ANNOTATIONS.name + '.sha256.json'))
    with tempfile.TemporaryDirectory() as tmp:
        variant = Path(tmp) / 'Pathology_SpotAnnotations.zip'
        renamed_archive(pc.ANNOTATIONS, variant)
        variant_sha = cc.sha256(variant)
        for section, header in VARIANTS.items():
            info = pc.prepare(section, cc.DATA_R3, variant, variant_sha)
            path = cc.DATA_R3 / pc.COHORT / info['dataset'] / 'input.json'
            meta = json.loads(path.read_text())
            meta['annotation_header_variant'] = dict(original_header=header, renamed_to=CANONICAL,
                                                     original_archive_sha256=original_sha,
                                                     renamed_archive_sha256=variant_sha)
            path.write_text(json.dumps(meta, indent=2) + '\n')
            print('PREPARED', info['dataset'], section, info['n_obs'], 'spots,',
                  info['targets']['all']['n_classes'], 'classes (header variant)', flush=True)
    excl = cc.DATA_R3 / pc.COHORT / 'exclusions.json'
    record = json.loads(excl.read_text())
    record['resolved_by_header_variant'] = sorted(VARIANTS)
    excl.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()

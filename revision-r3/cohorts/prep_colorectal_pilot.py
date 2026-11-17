"""Bounded inputs for the three colorectal pilot patients, in the round-3 format.

They are not new data: the pilot analysed them in October. PROTOCOL-R3 (section 5)
uses them only for the patient-excluded selection of tuned encoder configurations in
the colorectal cohort, and they are reported separately from the four new patients.
The archives are the pilot's own verified downloads; their October receipts are used.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_cohorts as cc  # noqa: E402
import prep_colorectal as pc  # noqa: E402

PILOT_RAW = cc.c1.PROJECT / 'revisions' / 'tessera-20261008-expanded' / 'validation-independent-crc' / 'raw'
SECTIONS = ('SN84_A120838_Rep1', 'SN124_A595688_Rep2', 'SN124_A551763_Rep2')
COHORT = 'colorectal_pilot'


def main() -> None:
    raw = cc.DATA_R3 / 'raw' / COHORT
    raw.mkdir(parents=True, exist_ok=True)
    for section in SECTIONS:
        for suffix in ('.zip', '.zip.sha256.json'):
            link = raw / f'{section}{suffix}'
            if not link.exists():
                link.symlink_to(PILOT_RAW / f'{section}{suffix}')
    original = pc.cc.receipt_path
    pc.cc.receipt_path = lambda path: Path(str(path) + '.sha256.json')     # the pilot's receipt format
    pc.COHORT = COHORT
    cc.COHORTS[COHORT] = cc.COHORTS['colorectal']
    try:
        if pc.prepare_all(SECTIONS, cc.DATA_R3, pc.ANNOTATIONS):
            raise SystemExit(1)
    finally:
        pc.cc.receipt_path = original


if __name__ == '__main__':
    main()

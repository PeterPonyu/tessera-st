#!/usr/bin/env python3
"""Plan, and after the protocol freeze perform, the verified downloads of the round-3 cohorts.

Default: a dry run that reads each Zenodo record's API metadata over a direct connection and prints
every planned file with its size and md5. ``--execute <cohort>`` downloads that cohort's files into
``data-r3/raw/<cohort>/`` only once the commit titled 'Freeze PROTOCOL-R3' exists: direct connection
(the local proxy measured slower), resumable ``.part`` files, the Zenodo md5 verified, SHA-256
computed, one receipt JSON per file. Files outside the frozen manifest (common_cohorts.COHORTS) are
never fetched.
"""
from __future__ import annotations

import argparse
import http.client
import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_cohorts as cc  # noqa: E402

API = 'https://zenodo.org/api/records/{}'
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # ignores HTTP(S)_PROXY
SAMPLE_ARCHIVE = re.compile(r'^SN\d+_(A\d+)_Rep\d+(?:_[A-Za-z0-9]+)?\.zip$')
PILOT_PLAN = cc.c1.REPO / 'current-release' / 'validation-independent-crc' / 'download-plan.json'


def record_metadata(record: str, opener=DIRECT) -> dict:
    request = urllib.request.Request(API.format(record), headers={'Accept': 'application/json'})
    with opener.open(request, timeout=60) as response:
        return json.load(response)


def listing(metadata: dict) -> dict:
    """File name -> bytes, md5 and download URL, from a record's API metadata."""
    files = {}
    for f in metadata['files']:
        if not f['checksum'].startswith('md5:'):
            raise ValueError(f'{f["key"]}: checksum is not an md5')
        files[f['key']] = dict(bytes=int(f['size']), md5=f['checksum'][4:], url=f['links']['self'])
    return files


def pilot_files() -> list[str]:
    return [job['filename'] for job in json.loads(PILOT_PLAN.read_text()) if SAMPLE_ARCHIVE.match(job['filename'])]


def colorectal_selection(sizes: dict, pilot: list[str]) -> dict:
    """The pilot rule on record 7760264: one section per patient, the smallest archive (ties by name).

    ``sizes`` maps archive names to bytes. The rule must reproduce the pilot, whose sections were the
    three smallest per-patient archives; it returns {patient: (archive, bytes)} for the other patients.
    """
    best = {}
    for name, size in sorted(sizes.items(), key=lambda item: (item[1], item[0])):
        match = SAMPLE_ARCHIVE.match(name)
        if match and match.group(1) not in best:
            best[match.group(1)] = (name, size)
    ranked = sorted(best.values(), key=lambda item: (item[1], item[0]))
    if {name for name, _ in ranked[:3]} != set(pilot):
        raise ValueError('the smallest-archive rule does not reproduce the pilot sections')
    pilot_patients = {SAMPLE_ARCHIVE.match(name).group(1) for name in pilot}
    return {patient: best[patient] for patient in sorted(best) if patient not in pilot_patients}


def plan(cohort: str, metadata: dict) -> tuple[list[dict], dict | None]:
    """Planned jobs of one cohort, each flagged by whether the live listing matches the frozen manifest."""
    live, spec = listing(metadata), cc.COHORTS[cohort]
    selection = None
    if cohort == 'colorectal':
        pilot = pilot_files()
        chosen = colorectal_selection({name: f['bytes'] for name, f in live.items()}, pilot)
        if sorted(name for name, _ in chosen.values()) != sorted(spec['files']):
            raise ValueError('colorectal: the live smallest-archive selection differs from the frozen manifest')
        selection = dict(rule='one section per patient by smallest archive; the pilot took the three smallest '
                              'per-patient archives; the remaining patients are prepared',
                         pilot=pilot, selected={patient: dict(archive=name, bytes=size)
                                                for patient, (name, size) in chosen.items()})
    jobs = []
    for name, (size, md5) in spec['files'].items():
        got = live.get(name, {})
        jobs.append(dict(cohort=cohort, file=name, bytes=size, md5=md5, url=got.get('url'),
                         matches_manifest=got.get('bytes') == size and got.get('md5') == md5))
    return jobs, selection


def download(url: str, dest, size: int, md5: str, opener=DIRECT, attempts: int = 5, extra=None) -> dict:
    """Resumable download to ``dest``: size and Zenodo md5 verified, SHA-256 computed, receipt written."""
    dest = Path(dest)
    part = dest.with_name(dest.name + '.part')
    dest.parent.mkdir(parents=True, exist_ok=True)
    resumed = part.stat().st_size if part.exists() else 0
    started = cc.utc()
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        if have == size:
            break
        request = urllib.request.Request(url, headers={'Range': f'bytes={have}-'} if have else {})
        try:
            with opener.open(request, timeout=60) as response, \
                    part.open('ab' if have and response.status == 206 else 'wb') as out:
                shutil.copyfileobj(response, out, 1 << 20)
        except urllib.error.HTTPError as error:
            if error.code < 500 or attempt == attempts:
                raise
            time.sleep(min(30, 2 ** attempt))
        except (OSError, http.client.HTTPException):
            if attempt == attempts:
                raise
            time.sleep(min(30, 2 ** attempt))
    got = part.stat().st_size if part.exists() else 0
    if got != size:
        raise ValueError(f'{dest.name}: {got} bytes received, {size} expected')
    digest_md5, digest_sha256 = cc.md5_and_sha256(part)
    if digest_md5 != md5:
        part.unlink()
        raise ValueError(f'{dest.name}: md5 {digest_md5} differs from the Zenodo md5 {md5}')
    part.replace(dest)
    receipt = dict(url=url, file=dest.name, bytes=size, md5=digest_md5, sha256=digest_sha256,
                   utc_start=started, utc_end=cc.utc(), resumed_from_bytes=resumed,
                   connection='direct, no proxy', **(extra or {}))
    cc.dump(cc.receipt_path(dest), receipt)
    return receipt


def fetch(job: dict, root: Path, extra: dict) -> None:
    dest = root / 'raw' / job['cohort'] / job['file']
    if dest.exists():
        digest_md5, digest_sha256 = cc.md5_and_sha256(dest)
        receipt = cc.receipt_path(dest)
        if (dest.stat().st_size == job['bytes'] and digest_md5 == job['md5'] and receipt.exists()
                and json.loads(receipt.read_text())['sha256'] == digest_sha256):
            print('PRESENT', dest.name, flush=True)
            return
        raise SystemExit(f'{dest} exists without a matching receipt; move it aside first')
    rec = download(job['url'], dest, job['bytes'], job['md5'], extra=extra)
    print('FETCHED', dest.name, rec['bytes'], 'bytes; sha256', rec['sha256'], flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--execute', choices=sorted(cc.COHORTS), help='download this cohort (needs the freeze commit)')
    parser.add_argument('--root', type=Path, default=cc.DATA_R3, help='data-r3 folder')
    args = parser.parse_args(argv)
    freeze = None
    if args.execute:
        freeze = cc.freeze_record()          # checked before any network request
        if freeze is None:
            raise SystemExit('PROTOCOL-R3 is not frozen (no "Freeze PROTOCOL-R3" commit, or uncommitted protocol '
                             'changes); nothing fetched')
    total = 0
    for cohort in [args.execute] if args.execute else list(cc.COHORTS):
        spec = cc.COHORTS[cohort]
        metadata = record_metadata(spec['record'])
        jobs, selection = plan(cohort, metadata)
        license_id = (metadata['metadata'].get('license') or {}).get('id') or 'none stated on the record'
        print(f'\n{cohort}: Zenodo {spec["record"]}, doi:{spec["doi"]}, license {license_id}')
        versions = metadata['metadata'].get('relations', {}).get('version', [{}])[0]
        if versions.get('is_last') is False:
            latest = record_metadata(f'{spec["record"]}/versions/latest')
            files = ', '.join(f'{f["key"]} ({f["size"]:,} B)' for f in latest['files'])
            print(f'  NOTE: not the latest version; latest is {latest["id"]} (doi:{latest["doi"]}): {files}')
        if selection:
            for patient, chosen in selection['selected'].items():
                print(f'  patient {patient}: smallest archive {chosen["archive"]}')
            print(f'  pilot (excluded): {", ".join(selection["pilot"])}')
        for job in jobs:
            state = 'matches frozen manifest' if job['matches_manifest'] else 'DIFFERS FROM FROZEN MANIFEST'
            print(f'  {job["file"]:<40} {job["bytes"]:>12,} B  md5:{job["md5"]}  {state}')
        total += sum(job['bytes'] for job in jobs)
        if args.execute:
            if not all(job['matches_manifest'] for job in jobs):
                raise SystemExit('the live listing differs from the frozen manifest; nothing fetched')
            if selection:
                cc.dump(args.root / 'raw' / cohort / 'selection.json', dict(selection, written_utc=cc.utc()))
            for job in jobs:
                fetch(job, args.root, dict(record=spec['record'], doi=spec['doi'], **freeze))
    print(f'\nTotal {total:,} bytes ({total / 2**20:.1f} MiB).' + ('' if args.execute else ' Dry run: nothing fetched.'))


if __name__ == '__main__':
    main()

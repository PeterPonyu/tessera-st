"""Adapter for original HER2ST/extension download calls, restricted to archived hashes.

This package-only helper replaces the workspace fetch_gate path in the portable
runner. It accepts any explicitly configured proxy and never changes environment
variables. Historical fetch_gate bytes and receipts remain preserved separately.
"""
import json
import os
from pathlib import Path
from acquire import ROOT, acquire, check, sha

MAX_AUTO = 10 * 1024 * 1024


def fetch(url, dest, yes_large=False, timeout=60):
    manifest = json.loads((ROOT / 'provenance/upstream-inputs.json').read_text())
    records = [row for rows in manifest['acquisition'].values() for row in rows if row.get('url') == url]
    if len(records) != 1:
        return {'ok': False, 'blocked': True, 'url': url, 'reason': 'URL is absent from the checksum-bound source manifest'}
    record = records[0]
    try:
        result = acquire(record, Path(dest), download=os.environ.get('TESSERA_ENABLE_DOWNLOAD') == '1',
                         proxy=os.environ.get('TESSERA_FETCH_PROXY'),
                         source_dir=os.environ.get('TESSERA_SOURCE_DIR'))
        return {'ok': True, 'url': url, 'path': str(dest), 'bytes': record['bytes'],
                'sha256': record['sha256'], 'portable_acquisition': result['status']}
    except (OSError, ValueError) as exc:
        return {'ok': False, 'blocked': True, 'reason': str(exc)}

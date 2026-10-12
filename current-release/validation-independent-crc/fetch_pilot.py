#!/usr/bin/env python3
"""Fetch only the prelocked CRC pilot through the workspace's download gate."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
GATE = REPO / 'research/pipelines/fetch_gate.py'


def sha(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    lock = json.loads((HERE / 'protocol-lock.json').read_text())
    if sha(HERE / 'PROTOCOL.md') != lock['protocol_sha256']:
        raise ValueError('protocol changed after lock')
    if os.environ.get('SC_FETCH_PROXY') != 'http://127.0.0.1:7890':
        raise ValueError('SC_FETCH_PROXY must explicitly select ClashParty 7890')
    spec = importlib.util.spec_from_file_location('crc_fetch_gate', GATE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    plan = json.loads((HERE / 'download-plan.json').read_text())
    raw = HERE / 'raw'
    raw.mkdir(exist_ok=True)
    records = []
    for job in plan:
        dest = raw / job['filename']
        started = time.perf_counter()
        if dest.exists():
            rec = dict(ok=True, cached=True, bytes=dest.stat().st_size, sha256=sha(dest))
        else:
            head = gate.head_info(job['url'], proxy=os.environ['SC_FETCH_PROXY'], timeout=20)
            if not head['ok'] or head['content_length'] != job['bytes'] or job['bytes'] > 1024**3:
                raise ValueError(f'failed exact HEAD: {job["filename"]}')
            rec = gate.fetch(job['url'], dest, yes_large=job['bytes'] > gate.MAX_AUTO, timeout=90)
        if not rec.get('ok'):
            save(HERE / 'fetch-failure.json', rec)
            raise RuntimeError('download failed; see fetch-failure.json')
        if dest.stat().st_size != job['bytes'] or 'md5:' + sha(dest, 'md5') != job['checksum']:
            raise ValueError(f'publisher size/checksum mismatch: {dest.name}')
        rec.update(filename=dest.name, url=job['url'], publisher_md5=job['checksum'],
                   sha256=sha(dest), seconds=time.perf_counter() - started)
        records.append(rec)
        save(HERE / 'fetch-receipt.json', dict(protocol_sha256=lock['protocol_sha256'],
             fetch_gate_sha256=sha(GATE), script_sha256=sha(Path(__file__)), files=records))
        with zipfile.ZipFile(dest) as z:
            entries = [i for i in z.infolist() if not i.is_dir() and not i.filename.startswith('__MACOSX/')]
            save(raw / (dest.stem + '-contents.json'),
                 [{'name': i.filename, 'bytes': i.file_size} for i in entries])
            print('LANDED', dest.name, dest.stat().st_size, 'bytes', len(entries), 'members', flush=True)
        # Numerical preparation can start as soon as the first sample lands.
        print('READY', dest.name, flush=True)


if __name__ == '__main__':
    main()

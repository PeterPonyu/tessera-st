#!/usr/bin/env python3
"""Checksum-bound upstream acquisition; default mode verifies manually staged files."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import ssl
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path, algorithm='sha256'):
    digest = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def check(path, record):
    if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
        raise ValueError('size/SHA256 mismatch: ' + str(path))
    if record.get('md5') and sha(path, 'md5') != record['md5']:
        raise ValueError('publisher MD5 mismatch: ' + str(path))


def acquire(record, destination, *, download=False, proxy=None, source_dir=None):
    destination = Path(destination)
    if destination.exists():
        check(destination, record)
        return {'file': record['filename'], 'status': 'verified-existing', 'sha256': record['sha256']}
    if source_dir:
        source = Path(source_dir) / record['filename']
        if source.is_file():
            check(source, record)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            return {'file': record['filename'], 'status': 'copied-verified-manual-source', 'sha256': record['sha256']}
    if not download:
        raise FileNotFoundError('Stage the checksum-bound file manually or use --download: ' + str(destination))
    if not record.get('url'):
        raise ValueError('No exact upstream download URL is recorded: ' + record['filename'])
    handlers = [urllib.request.ProxyHandler({'http': proxy, 'https': proxy} if proxy else {}),
                urllib.request.HTTPSHandler(context=ssl.create_default_context())]
    opener = urllib.request.build_opener(*handlers)
    request = urllib.request.Request(record['url'], headers={'User-Agent': 'tessera-controls-audit/2026.10.11'})
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.partial')
    try:
        with opener.open(request, timeout=120) as response, temporary.open('xb') as stream:
            length = 0
            for block in iter(lambda: response.read(1 << 20), b''):
                length += len(block)
                if length > record['bytes']:
                    raise ValueError('Upstream stream exceeds recorded byte count')
                stream.write(block)
        check(temporary, record)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {'file': record['filename'], 'status': 'downloaded-and-verified', 'sha256': record['sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort', choices=['her2st', 'crc'])
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, help='Manually acquired files, checked before copying')
    parser.add_argument('--download', action='store_true', help='Explicitly enable checksum-bound network acquisition')
    parser.add_argument('--proxy', help='Optional HTTP(S) proxy, used only by this request; direct by default')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'provenance/upstream-inputs.json').read_text())
    records = [acquire(record, args.destination / record['filename'], download=args.download,
                       proxy=args.proxy, source_dir=args.source_dir) for record in manifest['acquisition'][args.cohort]]
    print(json.dumps({'status': 'passed', 'cohort': args.cohort, 'records': records}, indent=2))


if __name__ == '__main__':
    main()

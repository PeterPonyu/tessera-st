#!/usr/bin/env python3
"""Seal the current extraction after any final document/citation overlay."""
from __future__ import annotations
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True, help='Destination outside package root')
    args = parser.parse_args()
    target = args.archive.resolve()
    if ROOT in target.parents:
        parser.error('Archive must be outside package root')
    files = sorted(path for path in ROOT.rglob('*') if path.is_file() and path.name != 'MANIFEST.sha256')
    if any(path.is_symlink() for path in ROOT.rglob('*')):
        raise ValueError('Cannot seal package with symlinks')
    if any('__pycache__' in path.parts or path.suffix == '.pyc' for path in files):
        raise ValueError('Cannot seal Python cache files; use PYTHONDONTWRITEBYTECODE=1')
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in files}
    manifest = ROOT / 'MANIFEST.sha256'
    manifest.write_text(''.join(digest + '  ' + name + '\n' for name, digest in hashes.items()))
    files.append(manifest)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT))
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(files)
        for name, digest in hashes.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
    print(json.dumps({'status': 'passed', 'archive': str(target), 'archive_bytes': target.stat().st_size,
                     'archive_sha256': sha(target), 'files': len(files),
                     'uncompressed_bytes': sum(path.stat().st_size for path in files),
                     'manifest_sha256': sha(manifest), 'zip_crc': 'passed'}, indent=2))


if __name__ == '__main__':
    main()

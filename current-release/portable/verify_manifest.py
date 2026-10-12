#!/usr/bin/env python3
"""Validate sealed package bytes and disallow symbolic links."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = ROOT / 'MANIFEST.sha256'
    count = 0
    total = 0
    for line in manifest.read_text().splitlines():
        digest, name = line.split('  ', 1)
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Nonportable manifest path')
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing file or symlink: ' + name)
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                h.update(block)
        if h.hexdigest() != digest:
            raise ValueError('SHA256 mismatch: ' + name)
        count += 1
        total += path.stat().st_size
    links = [str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_symlink()]
    if links:
        raise ValueError('Package includes symlinks: ' + repr(links))
    print(json.dumps({'status': 'passed', 'manifest_files': count, 'manifest_bytes': total,
                     'symlinks': 0, 'network_requests': 0}, indent=2))


if __name__ == '__main__':
    main()

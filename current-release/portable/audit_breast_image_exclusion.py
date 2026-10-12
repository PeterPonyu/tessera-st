#!/usr/bin/env python3
"""Scan final PDF/raster payloads for exact historical 10x breast image content.

Read-only supplemental document QA. Requires pypdf and Pillow; the four default
saved-result commands do not require these supplemental PDF-inspection tools.
"""
from pathlib import Path
import hashlib
import json
from pypdf import PdfReader
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    forbidden = json.loads((ROOT / 'qa/release-closure/old-breast-image-fingerprints.json').read_text())
    old_streams = {row['decoded_rgb_sha256'] for row in forbidden['embedded_images']}
    old_file = forbidden['old_figure07_pdf_sha256']
    raw_hash = json.loads((ROOT / 'provenance/breast-image-source-and-rights.json').read_text())['image_sha256']
    matches = []
    pdfs = []
    rasters = []
    images = 0
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():
            continue
        name = str(path.relative_to(ROOT))
        suffix = path.suffix.lower()
        if suffix not in {'.pdf', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp', '.gif', '.bmp'}:
            continue
        raw = path.read_bytes()
        if digest(raw) in {old_file, raw_hash}:
            matches.append({'path': name, 'reason': 'historical figure07 PDF or original 10x breast image hash'})
        if suffix == '.pdf':
            reader = PdfReader(path)
            count = 0
            for page_index, page in enumerate(reader.pages):
                for img in page.images:
                    rgb = img.image.convert('RGB')
                    count += 1
                    if digest(rgb.tobytes()) in old_streams:
                        matches.append({'path': name, 'page': page_index + 1, 'image': img.name,
                                        'reason': 'historical figure07 decoded image pixels'})
            pdfs.append({'path': name, 'sha256': digest(raw), 'pages': len(reader.pages), 'embedded_images_checked': count})
            images += count
        else:
            with Image.open(path) as image:
                rgb = image.convert('RGB')
                if digest(rgb.tobytes()) in old_streams:
                    matches.append({'path': name, 'reason': 'historical figure07 decoded image pixels'})
                rasters.append({'path': name, 'sha256': digest(raw), 'dimensions': list(rgb.size)})
    report = {'status': 'passed' if not matches else 'failed', 'pdf_files_checked': len(pdfs),
              'embedded_pdf_images_checked': images, 'raster_files_checked': len(rasters),
              'historical_breast_content_matches': matches, 'pdfs': pdfs, 'rasters': rasters,
              'scope': 'Exact byte/pixel fingerprints of the excluded historical breast content plus current source/visual evidence; CRC images are distinct dataset-licensed assets retained intentionally.'}
    print(json.dumps(report, indent=2))
    if matches:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Render the Scientific Reports PDFs and bind overview sheets to their hashes.

Mechanical part only: page images, overview sheets, embedded fonts, caption pages and declaration pages.
The manual visual attestation (qa/resumed-document-review/visual-review.json) is written by
write_srep_visual_review.py after the overview sheets of the current build have been opened.
Documents whose hash is already in the manifest keep their record, so an unchanged supplement is not re-rendered.
"""
from __future__ import annotations
import re
import subprocess
from PIL import Image, ImageDraw
from common import ROOT, dump, read, sha

OUT = ROOT / 'qa/resumed-document-review'
SHEET = (4, 2)          # columns, rows of pages per overview sheet


def pdf_text(pdf):
    return subprocess.run(['pdftotext', '-layout', str(pdf), '-'], text=True, capture_output=True,
                          check=True).stdout.split('\f')


def render(key, pdf, digest):
    folder = OUT / f'{key}-{digest[:12]}'
    folder.mkdir(parents=True, exist_ok=True)
    subprocess.run(['pdftoppm', '-r', '110', '-png', str(pdf), str(folder / 'page')], check=True)
    pages = sorted(folder.glob('page-*.png'))
    records = [{'page': i + 1, 'path': str(p.relative_to(ROOT)), 'sha256': sha(p)} for i, p in enumerate(pages)]
    sheets, per = [], SHEET[0] * SHEET[1]
    for start in range(0, len(pages), per):
        chunk = pages[start:start + per]
        w, h = 560, 724
        sheet = Image.new('RGB', (SHEET[0] * w, SHEET[1] * h), '#e7eaed')
        draw = ImageDraw.Draw(sheet)
        for i, path in enumerate(chunk):
            image = Image.open(path).convert('RGB')
            image.thumbnail((w - 10, h - 24))
            x, y = (i % SHEET[0]) * w, (i // SHEET[0]) * h
            draw.text((x + 8, y + 6), f'{key} | page {start + i + 1}', fill='#263744')
            sheet.paste(image, (x + (w - image.width) // 2, y + 22))
        target = folder / f'overview-{start // per + 1:02d}.jpg'
        sheet.save(target, quality=80)
        sheets.append({'path': str(target.relative_to(ROOT)), 'sha256': sha(target),
                       'pages': list(range(start + 1, start + len(chunk) + 1))})
    text = pdf_text(pdf)
    fonts = subprocess.run(['pdffonts', str(pdf)], text=True, capture_output=True, check=True).stdout
    (folder / 'fonts.txt').write_text(fonts)
    emb = [line.split()[-5] for line in fonts.splitlines()[2:] if line.strip()]
    return {'pdf': str(pdf.relative_to(ROOT)), 'sha256': digest, 'pages': len(pages), 'page_images': records,
            'overview_sheets': sheets,
            'figure_caption_pages': [i + 1 for i, t in enumerate(text) if re.search(r'(?m)^\s*Figure\s+\d+\s*[:.]', t)],
            'declaration_pages': [i + 1 for i, t in enumerate(text)
                                  if re.search(r'Data Availability|Additional Information|Author contributions', t)],
            'embedded_fonts': all(e == 'yes' for e in emb)}


def main():
    manifest_path = OUT / 'render-manifest.json'
    manifest = read(manifest_path)
    check = read(ROOT / 'qa/scientific-reports/format-check.json')
    for key in check['documents']:
        pdf = ROOT / 'output/scientific-reports' / f'scientific-reports-{key}.pdf'
        digest = sha(pdf)
        if manifest['documents'].get(key, {}).get('sha256') == digest:
            print(key, 'unchanged; record kept')
            continue
        manifest['documents'][key] = render(key, pdf, digest)
        print(key, manifest['documents'][key]['pages'], 'pages rendered')
    for key in list(manifest['documents']):
        if key not in check['documents']:
            del manifest['documents'][key]
    manifest.update({'format_report_sha256': sha(ROOT / 'qa/scientific-reports/format-check.json'),
                     'verifier_sha256': sha(ROOT / 'source/scripts/render_srep_qa.py'),
                     'review_status': 'rendered; manual review pending',
                     'scope': 'Document hashes, declaration order, embedded fonts, page images and overview sheets; '
                              'no model fitting or scientific peer review.'})
    dump(manifest_path, manifest)


if __name__ == '__main__':
    main()

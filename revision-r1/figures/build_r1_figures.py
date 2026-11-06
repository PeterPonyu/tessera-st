#!/usr/bin/env python3
"""Build the revision-1 figure set: vector PDFs, 300-dpi RGB JPGs and QA renders.

Usage (figure drawing needs only matplotlib; PDF post-processing needs PyMuPDF):
  /home/zeyufu/miniconda3/envs/dl/bin/python build_r1_figures.py draw [names...]
  /home/zeyufu/miniconda3/bin/python3 build_r1_figures.py finish
  python build_r1_figures.py all     # both steps, in an environment with matplotlib and PyMuPDF
`draw` writes the PDFs/SVGs to source/figures-r1/; `finish` applies the 6-pt white border
kept from the submitted figures, exports JPGs to figures-r1/, renders 150-dpi QA PNGs and
records fonts, text and page geometry. Environment overrides: R1_PDF_OUT, R1_JPG_OUT, R1_QA_OUT.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MAIN = ['figure-01', 'figure-02', 'figure-03', 'figure-04', 'figure-05', 'figure-06', 'figure-07', 'figure-08',
        'figure-09']
SUPPLEMENT = [f'figure-S{i}' for i in range(1, 17)]
ORDER = MAIN + SUPPLEMENT
# Submitted figures that carried the 14 Oct 6-pt white border, mapped to their revision-1 descendants.
BORDERED = {'figure-03': 'old 3', 'figure-04': 'old 4', 'figure-07': 'old 9', 'figure-S1': 'old S1',
            'figure-S2': 'old S2', 'figure-S3': 'old S3', 'figure-S14': 'old 4d'}
PAD = 6.0
UPLOAD_WIDTH = 2067  # 17.5 cm at 300 dpi


def draw(names):
    import r1_style as S
    import r1_new_figures
    import r1_schematics
    import r1_statistical
    names = names or ORDER
    for name in names:
        if name in r1_schematics.SCHEMATICS:
            r1_schematics.build([name])
        elif name in r1_new_figures.NEW:
            r1_new_figures.NEW[name]()
        else:
            r1_statistical.STATISTICAL[name]()
        print('drew', name, flush=True)
    S.QA_OUT.mkdir(parents=True, exist_ok=True)
    record = S.QA_OUT / 'draw-audit.json'
    previous = json.loads(record.read_text()) if record.exists() else {}
    previous.setdefault('audit', {}).update(S.finite(S.AUDIT))
    previous.setdefault('plot_data', {}).update(S.finite(S.PLOT_DATA))
    previous.setdefault('inputs_sha256', {}).update(S.INPUTS)
    record.write_text(json.dumps(previous, indent=1, allow_nan=False) + '\n')


def pad_in_place(path):
    import fitz
    temp = path.with_suffix('.padded.pdf')
    with fitz.open(path) as doc:
        page = doc[0]
        text_before = page.get_text()
        new = fitz.open()
        out = new.new_page(width=page.rect.width + 2 * PAD, height=page.rect.height + 2 * PAD)
        out.show_pdf_page(fitz.Rect(PAD, PAD, PAD + page.rect.width, PAD + page.rect.height), doc, 0)
        new.save(temp, garbage=4, deflate=True)
        new.close()
    with fitz.open(temp) as check:
        assert check[0].get_text() == text_before, path
    temp.replace(path)


def fonts(path):
    out = subprocess.run(['pdffonts', str(path)], check=True, text=True, capture_output=True).stdout
    rows = [line.split() for line in out.splitlines()[2:] if line.strip()]
    return [{'name': r[0], 'type': ' '.join(r[1:-5]), 'embedded': r[-5], 'subset': r[-4]} for r in rows]


def finish():
    import fitz
    from PIL import Image
    sys.path.insert(0, str(HERE))
    import r1_style as S
    S.JPG_OUT.mkdir(parents=True, exist_ok=True)
    (S.QA_OUT / 'png150').mkdir(parents=True, exist_ok=True)
    report = {}
    for name in ORDER:
        pdf = S.PDF_OUT / (name + '.pdf')
        if not pdf.exists():
            continue
        with fitz.open(pdf) as doc:
            width = doc[0].rect.width
        assert abs(width - S.W) < .01 or (name in BORDERED and abs(width - S.W - 2 * PAD) < .01), (name, width)
        if name in BORDERED and abs(width - S.W) < .01:  # freshly drawn canvas: add the border once
            pad_in_place(pdf)
        with fitz.open(pdf) as doc:
            page = doc[0]
            zoom = UPLOAD_WIDTH / page.rect.width
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            Image.frombytes('RGB', (pix.width, pix.height), pix.samples).save(
                S.JPG_OUT / (name + '.jpg'), quality=95, dpi=(300, 300), optimize=True)
            page.get_pixmap(dpi=150, alpha=False).save(S.QA_OUT / 'png150' / (name + '.png'))
            text = page.get_text()
            report[name] = {'page_points': [round(page.rect.width, 2), round(page.rect.height, 2)],
                            'border_points_per_side': PAD if name in BORDERED else 0,
                            'jpg_pixels': [pix.width, pix.height], 'fonts': fonts(pdf),
                            'unrenamed_Tessera': len(re.findall(r'Tessera(?!-ST)', text)),
                            'Tessera-ST_occurrences': text.count('Tessera-ST'),
                            'Same-graph_mean_occurrences': len(re.findall(r'[Ss]ame-graph mean', text)),
                            'images': len(page.get_images())}
    (S.QA_OUT / 'finish-report.json').write_text(json.dumps(report, indent=1) + '\n')
    for name, r in report.items():
        names = sorted({f['name'].split('+')[-1] for f in r['fonts']})
        print(name, r['page_points'], r['jpg_pixels'], names, 'Tessera-ST', r['Tessera-ST_occurrences'],
              'unrenamed', r['unrenamed_Tessera'], flush=True)


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else 'draw'
    if command == 'draw':
        draw(sys.argv[2:])
    elif command == 'finish':
        finish()
    elif command == 'all':  # draw every figure, then border, export and audit them
        draw(sys.argv[2:])
        finish()
    else:
        raise SystemExit(__doc__)

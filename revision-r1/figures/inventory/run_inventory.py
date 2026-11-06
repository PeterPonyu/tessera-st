#!/usr/bin/env python3
"""Regenerate the 20 submitted Frontiers figures unchanged and compare them with the packet.

Run with the base interpreter (PyMuPDF): python3 run_inventory.py
Set INVENTORY_OUT to a scratch folder; nothing inside the project is written.

Chain reproduced (see FIGURE-GENERATORS.md):
  render_figures.py / render_external_figures.py / artwork.py   -> native PDFs (pgf, serif)
  style_journal_figures.py                                      -> journal PDFs (Arial substitution)
  300-dpi JPG at 17.5 cm (13 Oct) -> cropped JPG-in-PDF wrapper (14 Oct, dense r2)
  frontiers_redraw_figures.py (+ _current_)                     -> vector figures 2, 3, 4, 9, 12 (14 Oct, r3)
  6-pt white border (14 Oct evidence pass)                      -> figures 3, 4, 9, S1, S2, S3
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

import fitz
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inventory_paths import LEGACY_MAN, SUBMITTED, DENSE_R2, OUT  # noqa: E402

DL = '/home/zeyufu/miniconda3/envs/dl/bin/python'
BASE = sys.executable
PAD = 6.0          # points per side, as in the 14 Oct evidence pass
UPLOAD_WIDTH = 2067  # 17.5 cm at 300 dpi, as in the 13 Oct portal JPGs

RASTER = {  # submitted stem -> journal figure produced by the native generators
    'figure-01': 'figure01-workflow', 'figure-05': 'figure05-nsclc-maps', 'figure-06': 'figure06-bcc-maps',
    'figure-07': 'figure07-breast-histology', 'figure-08': 'figure08-breast-mixtures',
    'figure-10': 'figure11-stability', 'figure-11': 'figure15-external-cohort',
    'figure-S1': 'figure02-cohort', 'figure-S2': 'figure14-target-lineage', 'figure-S3': 'figure09-colorectal-proteins',
    'figure-S4': 'figure12-efficiency-mechanism', 'figure-S5': 'supplement01-hnscc', 'figure-S6': 'supplement02-mibi',
    'figure-S7': 'supplement03-rna-markers', 'figure-S8': 'figure16-external-maps'}
VECTOR = ['figure-02', 'figure-03', 'figure-04', 'figure-09', 'figure-12']
PADDED = {'figure-03', 'figure-04', 'figure-09', 'figure-S1', 'figure-S2', 'figure-S3'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(python, script):
    env = dict(os.environ, INVENTORY_OUT=str(OUT), PYTHONPATH=str(HERE))
    result = subprocess.run([python, str(HERE / script)], cwd=str(OUT), env=env, text=True, capture_output=True)
    (OUT / 'logs').mkdir(exist_ok=True)
    (OUT / 'logs' / (script + '.log')).write_text(result.stdout + '\n' + result.stderr)
    if result.returncode:
        raise RuntimeError(f'{script} failed; see {OUT}/logs/{script}.log\n{result.stderr[-3000:]}')


def pad(source, target):
    """Same operation as the 14 Oct pass: place the intact page on a page 2*PAD larger."""
    with fitz.open(source) as doc:
        page = doc[0]
        new = fitz.open()
        out = new.new_page(width=page.rect.width + 2 * PAD, height=page.rect.height + 2 * PAD)
        out.show_pdf_page(fitz.Rect(PAD, PAD, PAD + page.rect.width, PAD + page.rect.height), doc, 0)
        new.save(target, garbage=4, deflate=True)
        new.close()


def wrap_jpeg(journal_pdf, jpeg, target, box):
    """Rasterize at 17.5 cm/300 dpi, then wrap the JPEG with the recorded 14 Oct crop box."""
    with fitz.open(journal_pdf) as doc:
        zoom = UPLOAD_WIDTH / doc[0].rect.width
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    Image.frombytes('RGB', (pix.width, pix.height), pix.samples).save(jpeg, quality=95, dpi=(300, 300))
    x0, y0, x1, y1 = box
    s = 72 / 300
    new = fitz.open()
    page = new.new_page(width=(x1 - x0) * s, height=(y1 - y0) * s)
    page.insert_image(fitz.Rect(-x0 * s, -y0 * s, (pix.width - x0) * s, (pix.height - y0) * s), filename=str(jpeg))
    new.save(target, garbage=4, deflate=True)
    new.close()
    return [pix.width, pix.height]


def render(path, dpi=150):
    with fitz.open(path) as doc:
        pix = doc[0].get_pixmap(dpi=dpi, alpha=False)
    return np.asarray(Image.frombytes('RGB', (pix.width, pix.height), pix.samples)).astype(np.int16)


def compare(submitted, regenerated, sheet):
    a, b = render(submitted), render(regenerated)
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    size_delta = [abs(a.shape[1] - b.shape[1]), abs(a.shape[0] - b.shape[0])]
    diff = np.abs(a[:h, :w] - b[:h, :w]).max(axis=2)
    canvas = np.full((h, 3 * w + 40, 3), 255, np.uint8)
    canvas[:, :w] = a[:h, :w]
    canvas[:, w + 20:2 * w + 20] = b[:h, :w]
    canvas[:, 2 * w + 40:] = 255 - np.clip(diff * 3, 0, 255)[..., None].astype(np.uint8)
    Image.fromarray(canvas).save(sheet)
    return {'pixel_size_delta_150dpi': size_delta, 'mean_abs_difference_0_255': round(float(diff.mean()), 3),
            'fraction_pixels_diff_gt_64': round(float((diff > 64).mean()), 5),
            'page_points_submitted': list(fitz.open(submitted)[0].rect)[2:],
            'page_points_regenerated': list(fitz.open(regenerated)[0].rect)[2:]}


def main():
    for sub in ['native-figures', 'journal-figures', 'qa', 'ledgers', 'assets', 'frontiers-redraw/source',
                'frontiers-redraw/qa', 'submitted-equivalent', 'upload-jpg', 'compare']:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    for name in ['replicates.csv', 'landscape.csv']:  # produced by make_external_tables.py; read-only inputs
        shutil.copy2(LEGACY_MAN / 'ledgers' / name, OUT / 'ledgers' / name)
    run(DL, 'render_figures.py')
    run(DL, 'render_external_figures.py')
    run(DL, 'artwork.py')
    run(BASE, 'style_journal_figures.py')
    run(BASE, 'frontiers_redraw_figures.py')
    crops = {Path(e['input']).stem: e for e in json.loads((DENSE_R2 / 'qa/figure-border-crops.json').read_text())}
    record = {'output_root': str(OUT), 'figures': {}}
    for stem, journal in RASTER.items():
        target = OUT / 'submitted-equivalent' / (stem + '.pdf')
        wrapped = OUT / 'submitted-equivalent' / (stem + '.unpadded.pdf') if stem in PADDED else target
        pixels = wrap_jpeg(OUT / 'journal-figures' / (journal + '.pdf'), OUT / 'upload-jpg' / (stem + '.jpg'),
                           wrapped, crops[stem]['retained_pixel_box'])
        if stem in PADDED:
            pad(wrapped, target)
        record['figures'][stem] = {'chain': f'{journal}: native -> journal -> JPG {pixels} -> crop '
                                            f'{crops[stem]["retained_pixel_box"]}' + (' -> 6-pt border' if stem in PADDED else '')}
    for stem in VECTOR:
        source = OUT / 'frontiers-redraw/source' / (stem + '.pdf')
        target = OUT / 'submitted-equivalent' / (stem + '.pdf')
        pad(source, target) if stem in PADDED else shutil.copy2(source, target)
        record['figures'][stem] = {'chain': 'frontiers redraw (14 Oct r3)' + (' -> 6-pt border' if stem in PADDED else '')}
    for stem in sorted(record['figures']):
        result = compare(SUBMITTED / 'source' / (stem + '.pdf'), OUT / 'submitted-equivalent' / (stem + '.pdf'),
                         OUT / 'compare' / (stem + '.png'))
        record['figures'][stem].update(result)
        print(stem, json.dumps(result), flush=True)
    # Intermediate checks: regenerated ledgers and native/journal figures against the legacy manuscript copies.
    record['ledgers_byte_identical'] = {name: sha(OUT / 'ledgers' / name) == sha(LEGACY_MAN / 'ledgers' / name)
                                        for name in ['all_primary_scores.csv', 'primary_summary.csv', 'patient_contrasts.json',
                                                     'all_transfer_scores.csv', 'seed_stability.csv', 'histology_roi.json']}
    record['journal_vs_legacy'] = {}
    for journal in sorted(set(RASTER.values()) | {'figure13-architecture', 'figure03-comparison', 'figure04-ablations',
                                                  'figure10-patient-transfer', 'figure17-tissue-landscape'}):
        result = compare(LEGACY_MAN / 'journal-figures' / (journal + '.pdf'), OUT / 'journal-figures' / (journal + '.pdf'),
                         OUT / 'compare' / ('journal-' + journal + '.png'))
        record['journal_vs_legacy'][journal] = result
    (OUT / 'inventory-comparison.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record['ledgers_byte_identical']))


if __name__ == '__main__':
    main()

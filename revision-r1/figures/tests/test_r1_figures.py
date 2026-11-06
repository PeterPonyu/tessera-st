"""Checks for the revision-1 figure set (run with the base interpreter, which has PyMuPDF):

  /home/zeyufu/miniconda3/bin/python3 -m pytest -q tests/test_r1_figures.py

Data values: the plotted arrays recorded at draw time must equal the plot-data record
saved with the submitted 14 Oct figures, and annotation numbers printed in the old
journal figures (ARI badges, sign counts, Spearman rho) must reappear unchanged.
Presentation: complete numbering, Arial-only embedded fonts, no unrenamed method
label, no superseded arm name, 300-dpi RGB JPGs at 17.5 cm.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import fitz
import numpy as np
import pytest
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parents[2]
PACKET = PROJECT / 'submissions/frontiers-in-genetics/revision-1-2028706'
PDF = Path(os.environ.get('R1_PDF_OUT', PACKET / 'source/figures-r1'))
JPG = Path(os.environ.get('R1_JPG_OUT', PACKET / 'figures-r1'))
QA = Path(os.environ.get('R1_QA_OUT', HERE / 'qa'))
SUBMITTED_PLOT = PROJECT / 'submissions/frontiers-in-genetics/final-typeset-20261014-r3/qa/plot-data.json'
JOURNAL = PROJECT / 'manuscripts/scientific-reports/source/manuscript/journal-figures'
EXPECTED = [f'figure-0{i}' for i in range(1, 10)] + [f'figure-S{i}' for i in range(1, 17)]
SUMMARY = PROJECT / 'results/revision-r1/summary'
TARGETS = [('NSCLC', 'primary', 'NSCLC regions'), ('BCC', 'primary', 'BCC cell types'),
           ('HER2ST', 'all', 'HER2ST, all labels'), ('HER2ST', 'boundary_T1', 'HER2ST, tumor boundary')]


def text(path):
    with fitz.open(path) as doc:
        return doc[0].get_text()


def numbers(value):
    return value.replace('−', '-').replace('−', '-')


@pytest.fixture(scope='module')
def plot():
    return json.loads((QA / 'draw-audit.json').read_text())['plot_data']


@pytest.fixture(scope='module')
def submitted():
    return json.loads(SUBMITTED_PLOT.read_text())


def close(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.allclose(a, b, rtol=0, atol=1e-12, equal_nan=True), np.nanmax(np.abs(a - b))


def test_numbering_complete():
    pdfs = sorted(p.stem for p in PDF.glob('figure-*.pdf'))
    jpgs = sorted(p.stem for p in JPG.glob('figure-*.jpg'))
    assert pdfs == sorted(EXPECTED) and jpgs == sorted(EXPECTED), (pdfs, jpgs)


@pytest.mark.parametrize('name', EXPECTED)
def test_fonts_are_arial_only(name):
    out = subprocess.run(['pdffonts', str(PDF / (name + '.pdf'))], check=True, text=True, capture_output=True).stdout
    rows = [line.split() for line in out.splitlines()[2:] if line.strip()]
    assert rows, name
    for row in rows:
        assert row[0].split('+')[-1] in {'ArialMT', 'Arial-BoldMT'}, (name, row)
        assert row[-5] == 'yes', (name, row)


@pytest.mark.parametrize('name', EXPECTED)
def test_names(name):
    content = text(PDF / (name + '.pdf'))
    assert not re.search(r'Tessera(?!-ST)', content), name
    assert not re.search(r'Neighbou?r mean|[Ss]ix-neighbou?r|\bMean 6\b|\bMean 200\b', content), name


@pytest.mark.parametrize('name', EXPECTED)
def test_jpg_export(name):
    with Image.open(JPG / (name + '.jpg')) as im:
        assert im.mode == 'RGB' and im.width == 2067, (name, im.mode, im.size)
        assert tuple(round(v) for v in im.info['dpi']) == (300, 300), (name, im.info.get('dpi'))
    with fitz.open(PDF / (name + '.pdf')) as doc:
        assert len(doc) == 1
        width = doc[0].rect.width
        assert abs(width - 496.8) < .01 or abs(width - 508.8) < .01, (name, width)


def test_figure3_values(plot, submitted):
    for key in ['bounded_vs_pca', 'bounded_vs_tuned', 'whole_vs_pca', 'whole_vs_tuned']:
        close(plot['figure-03'][key], submitted['figure3'][key])


def test_figure4_and_S14_values(plot, submitted):
    close(plot['figure-04']['full_minus_control'], submitted['figure4']['full_minus_control'])
    close(plot['figure-S14']['bins'], submitted['figure4']['bins'])
    for label in ['Same region', 'Different region']:
        close(plot['figure-S14']['histogram_densities'][label], submitted['figure4']['histogram_densities'][label])
    assert plot['figure-S14']['anchor'] == submitted['figure4']['anchor'] and plot['figure-S14']['seed'] == 1


def test_figure7_values(plot, submitted):
    keys = ['region_balanced_accuracy', 'cell_type_balanced_accuracy', 'region_accuracy', 'cell_type_accuracy']
    for i, key in enumerate(keys):
        close(plot['figure-07'][key], submitted[f'figure9_{i}'])


def test_landscape_values(plot, submitted):
    rows = submitted['figure12']
    for name, field in [('figure-09', 'tessera_minus_same_graph'), ('figure-09', 'tessera_minus_oracle'),
                        ('figure-S16', 'tessera_minus_pca50'), ('figure-S16', 'gate_auroc'),
                        ('figure-S16', 'raw_difference_auroc')]:
        close(plot[name][field], [r[field] for r in rows])
    assert plot['figure-09']['best_uniform_scale_multiplicities'] == submitted['figure12_scale_multiplicities']


@pytest.mark.parametrize('old,new', [('figure05-nsclc-maps', 'figure-S9'), ('figure06-bcc-maps', 'figure-S10'),
                                     ('figure07-breast-histology', 'figure-S11'), ('supplement01-hnscc', 'figure-S5')])
def test_ari_badges_unchanged(old, new):
    before = Counter(re.findall(r'ARI \d\.\d\d', text(JOURNAL / (old + '.pdf'))))
    after = Counter(re.findall(r'ARI \d\.\d\d', text(PDF / (new + '.pdf'))))
    assert before and before == after, (old, before, after)


def test_external_badges_unchanged():
    def badges(path):
        return Counter(numbers(t) for t in re.findall(r'(?m)^[−−-]?\d\.\d\d$', text(path)))
    before, after = badges(JOURNAL / 'figure16-external-maps.pdf'), badges(PDF / 'figure-S8.pdf')
    assert len(before) and before == after, (before, after)


def test_sign_counts_unchanged():
    def counts(path):
        return Counter(numbers(t) for t in re.findall(r'\+\d+/[−−-]\d+', text(path)))
    before, after = counts(JOURNAL / 'figure15-external-cohort.pdf'), counts(PDF / 'figure-08.pdf')
    assert sum(before.values()) == 9 and before == after, (before, after)


def test_spearman_unchanged():
    old = re.findall(r'=\s*([−−-]?\d\.\d\d)', text(JOURNAL / 'figure08-breast-mixtures.pdf'))
    new = re.findall(r'ρ\s*=\s*([−−-]?\d\.\d\d)', text(PDF / 'figure-S12.pdf'))
    assert [numbers(v) for v in old] == [numbers(v) for v in new] == ['-0.09'], (old, new)


def test_heatmap_annotations_match_submitted():
    """Positive cells print with an explicit '+', which no tick label carries (full matrix: test above)."""
    def positives(path):
        return Counter(re.findall(r'(?m)^\+\d\.\d\d$', text(path)))
    submitted = PROJECT / 'submissions/frontiers-in-genetics/neighborhood-averaging-spatial-gains/source/figure-04.pdf'
    old, new = positives(submitted), positives(PDF / 'figure-04.pdf')
    assert sum(old.values()) > 20 and old == new, (old, new)


def test_figure5_values(plot):
    """Budget curves and E2 rows equal the ledger columns named in the figure specification."""
    import pandas as pd
    budget = pd.read_csv(SUMMARY / 'fig_budget.csv')
    e2 = pd.read_csv(SUMMARY / 'table_E2.csv')
    record = plot['figure-05']
    for cohort, target, title in TARGETS[:3]:
        sub = budget[(budget.cohort == cohort) & (budget.target == target)]
        pivot = sub.pivot(index='patient', columns='epoch', values='minus_control')
        assert record[title]['patients'] == list(pivot.index) and record[title]['epochs'] == list(pivot.columns)
        close(record[title]['minus_control'], pivot.to_numpy())
        close(record[title]['cohort_mean'], pivot.mean(axis=0).to_numpy())
        close(pivot.to_numpy(), sub.pivot(index='patient', columns='epoch', values='tessera').to_numpy()
              - sub.pivot(index='patient', columns='epoch', values='tuned_control').to_numpy())
    rows = {'Released': ('tessera_default', 'nested_control'), 'Tuned': ('tessera_nested', 'nested_control'),
            'No gate': ('nogate_selected', 'tessera_nested'), 'No contrast': ('nocontrast_selected', 'tessera_nested')}
    for cohort, target, title in TARGETS:
        sub = e2[(e2.cohort == cohort) & (e2.target == target)]
        for label, (a, b) in rows.items():
            entry = record['d' if label in ('Released', 'Tuned') else 'e'][f'{title}: {label}']
            values = (sub[a] - sub[b]).to_numpy()
            close(entry['patients'], values)
            assert entry['n_positive'] == int((values > 0).sum()) and entry['n_negative'] == int((values < 0).sum())


def test_figure6_values(plot):
    """Cluster-number curves and label-free rows equal the ledger column Tessera120-TunedControl."""
    import pandas as pd
    k = pd.read_csv(SUMMARY / 'fig_k.csv')
    free = pd.read_csv(SUMMARY / 'fig_k_label_free.csv')
    record = plot['figure-06']
    column = 'Tessera120-TunedControl'
    for cohort, target, title in TARGETS:
        sub = k[(k.cohort == cohort) & (k.target == target)]
        full = list(range(int(sub.offset.min()), int(sub.offset.max()) + 1))
        assert record[title]['offsets'] == full  # every swept cluster number is drawn
        for geometry, key in [('euclidean', 'euclidean'), ('cosine', 'unit_length')]:
            pivot = sub[sub.geometry == geometry].pivot(index='unit', columns='offset', values=column).reindex(
                columns=full)
            close(record[title][key], pivot.to_numpy())
            mean = pivot.mean(axis=0).where(pivot.notna().all(axis=0))
            close(record[title]['mean_' + key], mean.to_numpy())
        rows = free[(free.cohort == cohort) & (free.target == target)]
        for geometry, label in [('euclidean', 'Euclidean'), ('cosine', 'Unit length')]:
            part = rows[rows.geometry == geometry]
            assert (part.K == part.K_label_free).all()
            close(record['e'][f'{title}: {label}']['patients'], part.set_index('unit').loc[record[title]['units'], column])


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))

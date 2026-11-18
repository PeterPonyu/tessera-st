"""Main figures of the prospective analyses, drawn natively in Arial from saved results.

Figures (vector PDF plus 300-dpi PNG, 180 mm wide):
  fig-concept       schematic: one synthetic tissue under three references, the label diagnostics,
                    matched attribution, and the prospective protocol
  fig-diagnostics   label diagnostics of the calibration targets and the new cohorts
  fig-construction  lung niches rebuilt at known construction scales
  fig-prospective   regret of the frozen scale rule in the new cohorts
  fig-mechanism     component and propagation contrasts of Tessera-ST; clustering-rule pairings
  fig-encoders      three encoders against propagation on their own graphs, propagation depth,
                    component removals and the calibration cohorts (skipped until summary.json has P3)
  fig-noise         seed spread and CPU/GPU agreement (a third panel appears once the same-seed
                    GPU repeats exist)

Inputs are read-only: results/revision-r3/{W1, W1b, W2, W3, W6, summary} and, when present,
W4/{scores, selection, counterparts}.csv with summary/p3_units.csv, and W4/gpu_repeats/gpu_repeats.csv.
Regret is computed with the frozen prepare_freeze.regret_table and
rule_choice. Every number printed on a figure is compared with summary/revision-r3-macros.tex before
the figure is saved; a disagreement stops the build. Layout reuses the revision-1 style module
(r1_style): Arial only, explicit point offsets from the canvas' upper-left corner, no automatic layout,
and checks that reject overlapping or off-canvas text and panel letters crowding an axes frame.

Usage:  /home/zeyufu/miniconda3/envs/dl/bin/python figures_r3.py [fig-name ...]
Environment overrides: TESSERA_R3_OUT (results root), R3_FIG_OUT (PDF/PNG folder),
R3_FIG_QA (plot-data record), R3_GPU_REPEATS (same-seed GPU repeat table).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import functools  # noqa: E402
read_csv = functools.partial(pd.read_csv, float_precision="round_trip")  # pandas' default float parser segfaults on some 17-digit values
from scipy import stats

HERE = Path(__file__).resolve().parent
R3_CODE = HERE.parent
REPO = R3_CODE.parent
PROJECT = REPO.parent
for _path in (REPO / 'revision-r1' / 'figures', R3_CODE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import r1_style as S  # noqa: E402  Arial set-up, type sizes, point canvas, overlap and clearance checks
from r1_style import (BASELINE, HEADER, INK, LEGEND, LIGHT, MINUS, NOTE, TICK, TITLE, X_AXIS, Y_AXIS,  # noqa: E402
                      Canvas, plain_log_ticks)
import prepare_freeze as pf  # noqa: E402  frozen regret_table and rule_choice

from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.ft2font import FT2Font  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

plt = S.plt

# ------------------------------------------------------------------ paths
R3 = Path(os.environ.get('TESSERA_R3_OUT') or PROJECT / 'results' / 'revision-r3')
OUT = Path(os.environ.get('R3_FIG_OUT') or PROJECT / 'manuscript' / 'figures-r3')   # set R3_FIG_OUT to the manuscript's figure folder
QA = Path(os.environ.get('R3_FIG_QA') or HERE / 'qa')
GPU_REPEATS = Path(os.environ.get('R3_GPU_REPEATS') or R3 / 'W4' / 'gpu_repeats' / 'gpu_repeats.csv')
MACROS = R3 / 'summary' / 'revision-r3-macros.tex'
for _protected in (R3, PROJECT / 'data', PROJECT / 'data-r3'):
    assert _protected != OUT and _protected not in OUT.parents and _protected not in QA.parents, _protected

W = 180 / 25.4 * 72  # 180 mm canvas, in points
NAMES = ['fig-concept', 'fig-diagnostics', 'fig-construction', 'fig-prospective', 'fig-mechanism', 'fig-encoders',
         'fig-noise']
NOISE_PANEL_C = 'fig-noise-panel-c.tex'


# ------------------------------------------------------------ typography
def _arial_files():
    """Resolve Arial regular and bold explicitly; refuse to draw with any substitute."""
    files = {}
    for weight, postscript in (('normal', 'ArialMT'), ('bold', 'Arial-BoldMT')):
        path = font_manager.findfont(FontProperties(family=S.FAMILY, weight=weight), fallback_to_default=False)
        found = FT2Font(path).postscript_name
        if found != postscript:
            raise RuntimeError(f'Arial ({weight}) is not available: matplotlib resolved {path} ({found})')
        files[weight] = path
    return files


ARIAL = _arial_files()
GLYPHS = {weight: set(FT2Font(path).get_charmap()) for weight, path in ARIAL.items()}

# --------------------------------------------------------------- palette
# One colour per provenance class, used in every figure; new cohorts add an open marker shape.
PROVENANCE = ['computed-niche', 'histology-anatomical', 'pathologist', 'dominant-category', 'expression-assigned']
PROV_COLOR = {'computed-niche': '#7b4fa0', 'histology-anatomical': '#c8962e', 'pathologist': '#4d8f3a',
              'dominant-category': '#c45c8a', 'expression-assigned': '#2c7bb6'}
PROV_NAME = {'computed-niche': 'Computed niche', 'histology-anatomical': 'Histology-anatomical', 'pathologist': 'Pathologist',
             'dominant-category': 'Dominant category', 'expression-assigned': 'Expression-assigned'}
NEW = ('spinal', 'colorectal', 'wu_breast')          # prospective cohorts; the colorectal pilot is not new
COHORT_NAME = {'spinal': 'Spinal cord', 'colorectal': 'Colorectal', 'wu_breast': 'Breast'}
COHORT_PROV = {'spinal': 'expression-assigned', 'colorectal': 'pathologist', 'wu_breast': 'pathologist'}
COHORT_UNIT = {'spinal': 'mice', 'colorectal': 'patients', 'wu_breast': 'patients'}
MARKER = {'calibration': 'o', 'spinal': '^', 'colorectal': 's', 'wu_breast': 'D'}
AREA = {'o': 1.0, '^': 1.2, 's': .8, 'D': .62}       # scatter areas that look alike across shapes
TESSERA = '#bd5b30'
SAME_GRAPH = '#657780'
PROPAGATION = '#1f8a8a'
GUIDE = '#8c8c8c'
DOT = '#a3abb0'
DASH = (0, (4.0, 2.2))
NICHE = PROV_COLOR['computed-niche']
LUNG = ['NSCLC_Lung5_Rep2', 'NSCLC_Lung6', 'NSCLC_Lung9_Rep1', 'NSCLC_Lung12', 'NSCLC_Lung13']
SHORT = {'NSCLC_Lung5_Rep2': 'Lung 5', 'NSCLC_Lung6': 'Lung 6', 'NSCLC_Lung9_Rep1': 'Lung 9',
         'NSCLC_Lung12': 'Lung 12', 'NSCLC_Lung13': 'Lung 13', 'BCC_PatientA': 'BCC A', 'BCC_PatientB': 'BCC B',
         'BCC_PatientC': 'BCC C', 'BCC_PatientD': 'BCC D', 'Breast_Visium': 'Breast Visium',
         'Colorectal_TVA232': 'Colorectal TVA', 'MIBI': 'MIBI', 'OpenST_HNSCC26': 'Open-ST HNSCC'}
BOUNDED = list(SHORT)                               # the 13 bounded calibration objects, manuscript order
PATIENT_MARKERS = ['o', 's', '^', 'D', 'v']
CHOICES = ['rule', 'no smoothing', 'k=6', 'k=30', 'k=200']
CHOICE_LABEL = {'rule': 'Rule', 'no smoothing': 'No\nsmoothing', 'k=6': 'k = 6', 'k=30': 'k = 30',
                'k=200': 'k = 200'}
CHOICE_MACRO = {'rule': 'Rule', 'no smoothing': 'None', 'k=6': 'Six', 'k=30': 'Thirty', 'k=200': 'TwoHundred'}
PAIRINGS = [('kmeans', 'kmeans'), ('kmeans', 'leiden'), ('leiden', 'kmeans'), ('leiden', 'leiden')]
PAIRING_LABEL = ['K/K', 'K/L', 'L/K', 'L/L']
SELECTION_TARGETS = [('NSCLC', 'primary', 'NSCLC regions', 'Lung'), ('BCC', 'primary', 'BCC cell types', 'Bcc'),
                     ('HER2ST', 'all', 'HER2ST all labels', 'Her'),
                     ('HER2ST', 'boundary_T1', 'HER2ST tumour/non-tumour', 'Tumor')]
DEPTHS = [1, 2, 3, 4, 8]

INPUTS: dict[str, str] = {}
PLOT_DATA: dict[str, object] = {}
AUDIT: dict[str, object] = {}


# ---------------------------------------------------------------- inputs
def track(path):
    path = Path(path)
    key = str(path.relative_to(PROJECT)) if PROJECT in path.parents else str(path)
    INPUTS[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


@lru_cache(maxsize=None)
def table(relative):
    return read_csv(track(R3 / relative))


@lru_cache(maxsize=None)
def record(relative):
    return json.loads(track(R3 / relative).read_text())


@lru_cache(maxsize=None)
def macros():
    """The manuscript's round-3 numbers, as printed (LaTeX minus replaced by the Unicode minus)."""
    text = track(MACROS).read_text()
    return {name: value.replace('\\ensuremath{-}', MINUS)
            for name, value in re.findall(r'^\\newcommand\{\\(\w+)\}\{(.*)\}$', text, re.M)}


def num(value, digits=3, sign=False):
    """Formatted as the macros are (make_macros_r3.f3/f2): no negative zero and no sign on zero."""
    value = 0.0 if round(value, digits) == 0 else value
    text = f'{value:+.{digits}f}' if sign and value != 0 else f'{value:.{digits}f}'
    return text.replace('-', MINUS)


def agree(text, name):
    """A number printed on a figure must equal the manuscript macro for the same quantity."""
    expected = macros()[name]
    if text != expected:
        raise AssertionError(f'{name}: the figure would print {text!r} but the macro is {expected!r}')
    return text


# --------------------------------------------------------------- drawing
def points(ax, x, y, provenance, cohort='calibration', size=15.0, zorder=3, lw=.85):
    """Calibration targets are filled circles; prospective-cohort targets are open shapes in the same colour."""
    marker = MARKER[cohort]
    colour = PROV_COLOR[provenance]
    if cohort == 'calibration':
        return ax.scatter(x, y, s=size * AREA[marker], marker=marker, color=colour, linewidths=0, zorder=zorder)
    return ax.scatter(x, y, s=size * AREA[marker], marker=marker, facecolors='white', edgecolors=colour,
                      linewidths=lw, zorder=zorder + 1)


def provenance_handles(classes=PROVENANCE, counts=None):
    return [Line2D([0], [0], marker='o', ls='', color=PROV_COLOR[c], markeredgewidth=0, markersize=5.2,
                   label=PROV_NAME[c] + (f' ({counts[c]})' if counts else '')) for c in classes]


def cohort_handles(calibration=True, coloured=False, counts=None):
    handles = []
    if calibration:
        handles.append(Line2D([0], [0], marker='o', ls='', color='#6f6f6f', markeredgewidth=0, markersize=5.2,
                              label='Calibration'))
    for c in NEW:
        marker = MARKER[c]
        label = f'Prospective: {COHORT_NAME[c].lower()}' if calibration else COHORT_NAME[c]
        if counts:
            label += f' ({counts[c]} {COHORT_UNIT[c]})'
        handles.append(Line2D([0], [0], marker=marker, ls='', markerfacecolor='white',
                              markeredgecolor=PROV_COLOR[COHORT_PROV[c]] if coloured else '#4d4d4d',
                              markeredgewidth=.9, markersize=5.2 * AREA[marker] ** .5, label=label))
    return handles


def swarm(values, diameter, points_per_unit):
    """Deterministic beeswarm: horizontal offsets (points) keeping markers of `diameter` points apart."""
    y = np.asarray(values, float) * points_per_unit
    offsets = np.zeros(len(y))
    placed = []
    for i in np.argsort(y, kind='stable'):
        for step in range(400):
            candidates = (0.0,) if step == 0 else (step * diameter / 2, -step * diameter / 2)
            fits = [dx for dx in candidates
                    if all((dx - px) ** 2 + (y[i] - py) ** 2 >= (.95 * diameter) ** 2 for px, py in placed)]
            if fits:
                offsets[i] = fits[0]
                break
        placed.append((offsets[i], y[i]))
    return offsets


def axes_points(cv, ax):
    """(width, height) of an axes in points."""
    box = ax.get_position()
    return box.width * cv.W, box.height * cv.H


def guide_line(ax, value, axis='x', **kwargs):
    style = dict(color=GUIDE, lw=.8, ls=DASH, zorder=1)
    style.update(kwargs)
    return (ax.axvline if axis == 'x' else ax.axhline)(value, **style)


def corner_note(ax, x, y, text, ha='right', va='bottom'):
    return ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=NOTE, color=INK, linespacing=1.25)


def sub_title(cv, ax, y, text):
    """A short title on the header baseline, left-aligned with its own axes."""
    return cv.text(ax.get_position().x0 * cv.W, y + BASELINE, text, size=TITLE, gid='panel-title')


def save(cv, name, plot_data):
    """Audit typography and layout, then write the vector PDF and a 300-dpi PNG."""
    fig = cv.fig
    letters, sizes = [], []
    for artist in S.visible_texts(fig):
        text = artist.get_text()
        for pattern in S.BANNED:
            assert not re.search(pattern, text), (name, text, pattern)
        assert artist.get_fontfamily() == [S.FAMILY], (name, text, artist.get_fontfamily())
        bold = artist.get_gid() == 'panel-label'
        weight = artist.get_fontweight()
        if bold:
            assert weight == 'bold' and re.fullmatch('[a-z]', text), (name, text)
            letters.append(text)
        else:
            assert weight in ('normal', 400), (name, text, weight)
            sizes.append(float(artist.get_fontsize()))
        missing = sorted({c for c in text if c != '\n' and ord(c) not in GLYPHS['bold' if bold else 'normal']})
        assert not missing, f'{name}: {text!r} needs glyphs Arial lacks: {missing}'
        font = font_manager.findfont(artist.get_fontproperties(), fallback_to_default=False)
        assert font == ARIAL['bold' if bold else 'normal'], (name, text, font)
    assert sorted(letters) == [chr(97 + i) for i in range(len(letters))], (name, letters)
    assert min(sizes) >= NOTE - 1e-9, (name, min(sizes))
    layout = S.check_text_layout(fig, name)
    clearance = S.check_letter_clearance(fig, name)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.pdf', metadata={'CreationDate': None, 'ModDate': None, 'Creator': 'matplotlib',
                                               'Producer': 'matplotlib pdf backend'})
    fig.savefig(OUT / f'{name}.png', dpi=300)
    plt.close(fig)
    AUDIT[name] = {'letters': sorted(letters), 'min_text_points': min(sizes), 'max_text_points': max(sizes),
                   **layout, **clearance}
    PLOT_DATA[name] = plot_data
    return OUT / f'{name}.pdf'


def write_record():
    QA.mkdir(parents=True, exist_ok=True)
    path = QA / 'plot-data.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    for key, value in (('audit', AUDIT), ('plot_data', PLOT_DATA), ('inputs_sha256', INPUTS)):
        previous.setdefault(key, {}).update(S.finite(value))
    path.write_text(json.dumps(previous, indent=1, allow_nan=False) + '\n')
    return path


# ======================================================= fig-diagnostics
def diagnostics_data():
    cal = table('W1/targets.csv')
    new = table('W6/targets.csv')
    new = new[new.cohort.isin(NEW)]
    rows = []
    for kind, frame, diagnostics in (('calibration', cal, table('W1/diagnostics.csv')),
                                     ('new', new, table('W6/diagnostics.csv'))):
        by_target = diagnostics.set_index(['dataset', 'target']).sort_index()
        for r in frame.itertuples():
            curve = by_target.loc[(r.dataset, r.target)].sort_values('k')
            curve = curve[np.isfinite(curve.h_adj)]
            rows.append(dict(dataset=r.dataset, target=r.target, provenance=r.provenance,
                             cohort='calibration' if kind == 'calibration' else r.cohort,
                             k=curve.k.astype(int).tolist(), h_adj=curve.h_adj.tolist(), h_adj6=float(r.h_adj6),
                             kappa=float(r.kappa_self), g_star=float(r.g_star)))
    finite = cal[['h_adj6', 'g_star']].dropna()
    rho = num(stats.spearmanr(finite.h_adj6, finite.g_star).statistic, 2)
    lung = cal[cal.cohort == 'NSCLC'].set_index(['dataset', 'target'])
    pairs = {m: [[float(lung.loc[(p, 'primary'), m]), float(lung.loc[(p, 'celltype'), m])] for p in LUNG]
             for m in ('h_adj6', 'kappa_self', 'g_star')}
    threshold = record('summary/summary.json')['freeze']['p1_rule']['h_adj_fallback']
    return dict(rows=rows, rho=rho, pairs=pairs, threshold=threshold)


def fig_diagnostics():
    d = diagnostics_data()
    rows = d['rows']
    rho = agree(d['rho'], 'rCalRhoHadj')
    order = {c: i for i, c in enumerate(['pathologist', 'expression-assigned', 'histology-anatomical',
                                          'dominant-category', 'computed-niche'])}
    cv = Canvas(600, width=W)
    cv.legend(provenance_handles(), W / 2, 1.0, ncol=5, columnspacing=1.5)
    cv.legend(cohort_handles(), W / 2, 13.5, ncol=4, columnspacing=1.5)
    top, gap, h = 30.0, 18.0, 166.0
    half = (W - gap) / 2

    # a: homophily across scale, calibration and new cohorts on matched axes
    sub_gap = 9.0
    pw = (half - Y_AXIS - sub_gap - 4) / 2
    ph = h - HEADER - X_AXIS
    cv.letter(0, top + BASELINE, 'a')
    a1 = cv.axes(Y_AXIS, top + HEADER, pw, ph)
    a2 = cv.axes(Y_AXIS + pw + sub_gap, top + HEADER, pw, ph)
    record_a = {}
    for ax, group, title in ((a1, 'calibration', 'Calibration targets'), (a2, 'new', 'Prospective cohorts')):
        sub_title(cv, ax, top, title)
        chosen = [r for r in rows if (r['cohort'] == 'calibration') == (group == 'calibration')]
        for r in sorted(chosen, key=lambda r: order[r['provenance']]):
            colour = PROV_COLOR[r['provenance']]
            ax.plot(r['k'], r['h_adj'], color=colour, lw=.7, alpha=.85, zorder=2, solid_capstyle='round')
            if r['cohort'] != 'calibration':
                marker = MARKER[r['cohort']]
                ax.scatter(r['k'], r['h_adj'], s=6.5 * AREA[marker], marker=marker, facecolors='white',
                           edgecolors=colour, linewidths=.55, zorder=3)
        ax.set_xscale('log')
        plain_log_ticks(ax.xaxis, [6, 30, 200])
        ax.set_xlim(5.3, 230)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, .2, .4, .6, .8, 1])
        ax.set_xlabel('Neighbours k')
        record_a[group] = {'n_lines': len(chosen), 'targets': [f"{r['dataset']}|{r['target']}" for r in chosen]}
    a1.set_ylabel('Adjusted homophily')
    a2.set_yticklabels([])

    # b: homophily against own-expression predictability
    xb = half + gap
    b = cv.panel(xb, top, half, h, 'b', 'Own-expression predictability', left=Y_AXIS, bottom=X_AXIS, right=6)
    shown_b = [r for r in rows if np.isfinite(r['kappa'])]
    for r in shown_b:
        points(b, r['h_adj6'], r['kappa'], r['provenance'], r['cohort'])
    b.set_xlim(0, 1)
    b.set_ylim(.2, .9)
    b.set_xticks([0, .2, .4, .6, .8, 1])
    b.set_xlabel('Adjusted homophily, k = 6')
    b.set_ylabel('Own-expression κ')

    # c: homophily against the label-selected gain
    y2 = top + h + 12
    c = cv.panel(0, y2, half, h, 'c', 'Gain from smoothing', left=Y_AXIS, bottom=X_AXIS, right=6)
    c.axhline(0, color=INK, lw=.6, zorder=1)
    guide_line(c, d['threshold'])
    for r in rows:
        points(c, r['h_adj6'], r['g_star'], r['provenance'], r['cohort'])
    c.set_xlim(0, 1)
    c.set_ylim(-.4, .5)
    c.set_xticks([0, .2, .4, .6, .8, 1])
    c.set_yticks([-.4, -.2, 0, .2, .4])
    c.set_xlabel('Adjusted homophily, k = 6')
    c.set_ylabel('Label-selected gain (ARI)')
    corner_note(c, .03, .97, f'Calibration ρ = {rho}', ha='left', va='top')

    # d: the same lung cells under two references
    xd = half + gap
    cv.header(xd, y2, 'd', 'Same lung cells')
    key = [Line2D([0], [0], marker='o', ls='', color=PROV_COLOR['computed-niche'], markeredgewidth=0, markersize=5,
                  label='Regions'),
           Line2D([0], [0], marker='o', ls='', color=PROV_COLOR['expression-assigned'], markeredgewidth=0,
                  markersize=5, label='Cell types')]
    cv.fig.legend(handles=key, loc='upper right', ncol=2, frameon=False, fontsize=LEGEND, columnspacing=1.0,
                  handletextpad=.3, borderaxespad=0, borderpad=0,
                  bbox_to_anchor=(cv.fx(xd + half - 2), cv.fy(y2 + 3.5)), bbox_transform=cv.fig.transFigure)
    names = [SHORT[p] for p in LUNG]
    lane = cv.text_width(names, TICK) + 6
    mini_gap = 14.0
    mw = (half - lane - 2 * mini_gap - 4) / 3
    specs = [('h_adj6', 'Homophily, k = 6', (0, 1), [0, .5, 1], ['0', '0.5', '1']),
             ('kappa_self', 'Own-expression κ', (0, 1), [0, .5, 1], ['0', '0.5', '1']),
             ('g_star', 'Selected gain', (-.3, .5), [-.2, 0, .2, .4], [f'{MINUS}0.2', '0', '', '0.4'])]
    for j, (metric, label, lim, ticks, tick_labels) in enumerate(specs):
        ax = cv.axes(xd + lane + j * (mw + mini_gap), y2 + HEADER, mw, h - HEADER - X_AXIS)
        values = np.array(d['pairs'][metric])
        for i, (region, celltype) in enumerate(values):
            ax.plot([region, celltype], [i, i], color=LIGHT, lw=1.6, zorder=1, solid_capstyle='round')
        ax.scatter(values[:, 0], np.arange(5), s=18, color=PROV_COLOR['computed-niche'], linewidths=0, zorder=3)
        ax.scatter(values[:, 1], np.arange(5), s=18, color=PROV_COLOR['expression-assigned'], linewidths=0,
                   zorder=3)
        if metric == 'g_star':
            ax.axvline(0, color=INK, lw=.6, zorder=0)
        ax.set_ylim(4.6, -.6)
        ax.set_yticks(np.arange(5), names if j == 0 else [])
        ax.tick_params(axis='y', length=0)
        ax.set_xlim(*lim)
        ax.set_xticks(ticks, tick_labels)
        ax.set_xlabel(label)
    cv.resize(y2 + h)
    return save(cv, 'fig-diagnostics', {
        'a': record_a,
        'b': {'n_points': len(shown_b), 'n_calibration': sum(r['cohort'] == 'calibration' for r in shown_b),
              'h_adj6': [r['h_adj6'] for r in shown_b], 'kappa': [r['kappa'] for r in shown_b]},
        'c': {'n_points': len(rows), 'n_calibration': sum(r['cohort'] == 'calibration' for r in rows),
              'h_adj6': [r['h_adj6'] for r in rows], 'g_star': [r['g_star'] for r in rows],
              'cohort': [r['cohort'] for r in rows], 'rho_printed': rho, 'threshold': d['threshold']},
        'd': {'patients': LUNG, **d['pairs']},
        'curves': {f"{r['dataset']}|{r['target']}": {'k': r['k'], 'h_adj': r['h_adj']} for r in rows}})


# ====================================================== fig-construction
def construction_data():
    t = table('W1b/targets.csv')
    kcs = sorted(int(k) for k in t.k_c.unique())
    per = {}
    for p in LUNG:
        g = t[t.unit == p].sort_values('k_c')
        assert g.k_c.tolist() == kcs, p
        per[p] = dict(ell=g.ell_k.tolist(), censored=g.ell_censored.astype(bool).tolist(), k_star=g.k_star.tolist(),
                      peak=g.info_gain_k.tolist(), margin=g.tessera_minus_sgm.tolist())
    grid = list(pf.GRID)
    curves = t.groupby('k_c')[[f'ari_mean{k}' for k in grid]].mean().loc[kcs].to_numpy()
    margin = t.groupby('k_c').tessera_minus_sgm.mean().loc[kcs]
    p2 = record('summary/summary.json')['P2']
    for k, v in margin.items():
        assert abs(p2['margin_by_kc'][str(k)] - v) < 1e-12, (k, v)
    return dict(kcs=kcs, per=per, grid=grid, curves=curves.tolist(), peaks=[grid[i] for i in curves.argmax(1)],
                margin=margin.tolist(), sesoi=float(p2['sesoi']))


def fig_construction():
    d = construction_data()
    kcs, grid = d['kcs'], d['grid']
    agree(num(min(d['margin']), sign=True), 'rPTwoMarginMin')
    agree(num(max(d['margin']), sign=True), 'rPTwoMarginMax')
    sesoi = num(d['sesoi'])
    freeze = record('summary/summary.json')['freeze']['sesoi']['NSCLC|primary']['sesoi']
    assert abs(freeze - d['sesoi']) < 1e-15 and sesoi == '0.029', (freeze, sesoi)
    cv = Canvas(600, width=W)
    handles = [Line2D([0], [0], marker=PATIENT_MARKERS[i], color=NICHE, lw=.7, markersize=4.3, markeredgewidth=0,
                      label=SHORT[p]) for i, p in enumerate(LUNG)]
    handles.append(Line2D([0], [0], marker='o', ls='', markersize=4.3, markerfacecolor='white', markeredgecolor=NICHE,
                          markeredgewidth=.9, label='Censored length'))
    cv.legend(handles, W / 2, 1.0, ncol=6, columnspacing=1.5)
    top, gap, h1 = 17.0, 12.0, 168.0
    w3 = (W - 2 * gap) / 3
    dodge = 1.075
    lim = (7.0, 570.0)
    ticks = [10, 25, 50, 100, 200, 400]
    record_abc = {}
    for i, (letter, title, metric, ylabel) in enumerate([
            ('a', 'Label correlation length', 'ell', 'Correlation length (neighbours)'),
            ('b', 'Best uniform-mean scale', 'k_star', 'Best mean scale (neighbours)'),
            ('c', 'Most informative scale', 'peak', 'Peak-gain scale (neighbours)')]):
        ax = cv.panel(i * (w3 + gap), top, w3, h1, letter, title, left=Y_AXIS, bottom=X_AXIS, right=6)
        ax.plot(lim, lim, color=GUIDE, lw=.7, ls=DASH, zorder=1)
        for j, p in enumerate(LUNG):
            r = d['per'][p]
            x = np.array(kcs, float) * dodge ** (j - 2)
            y = np.array(r[metric], float)
            hollow = np.array(r['censored']) if metric == 'ell' else np.zeros(len(x), bool)
            ax.plot(x, y, color=NICHE, lw=.7, alpha=.75, zorder=2)
            ax.scatter(x[~hollow], y[~hollow], marker=PATIENT_MARKERS[j], s=14 * AREA.get(PATIENT_MARKERS[j], .9),
                       color=NICHE, linewidths=0, zorder=3)
            ax.scatter(x[hollow], y[hollow], marker=PATIENT_MARKERS[j], s=14 * AREA.get(PATIENT_MARKERS[j], .9),
                       facecolors='white', edgecolors=NICHE, linewidths=.8, zorder=3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        plain_log_ticks(ax.xaxis, ticks)
        plain_log_ticks(ax.yaxis, ticks)
        ax.set_xlabel('Construction scale (neighbours)')
        ax.set_ylabel(ylabel)
        record_abc[letter] = {p: d['per'][p][metric] for p in LUNG}
    record_abc['a_censored'] = {p: d['per'][p]['censored'] for p in LUNG}

    # d: mean agreement of uniform means across k, one line per construction scale
    y2 = top + h1 + 12
    h2 = 166.0
    wd = 292.0
    legend_w = 64.0
    dd = cv.panel(0, y2, wd, h2, 'd', 'Uniform means across scale', left=Y_AXIS, bottom=X_AXIS, right=legend_w)
    ramp = LinearSegmentedColormap.from_list('niche-ramp', ['#cdb8e3', '#2e1747'])
    curve_handles = []
    for i, kc in enumerate(kcs):
        colour = ramp(i / (len(kcs) - 1))
        values = np.array(d['curves'][i])
        dd.plot(grid, values, color=colour, lw=1.2, zorder=2)
        dd.scatter(grid, values, s=7, color=colour, linewidths=0, zorder=3)
        best = int(np.argmax(values))
        dd.scatter([grid[best]], [values[best]], s=34, marker='o', facecolors=colour, edgecolors='white',
                   linewidths=.8, zorder=4)
        curve_handles.append(Line2D([0], [0], color=colour, lw=1.4, marker='o', markersize=4,
                                    markeredgecolor='white', markeredgewidth=.6, label=f'{kc}'))
    dd.set_xscale('log')
    plain_log_ticks(dd.xaxis, [6, 10, 15, 30, 50, 100, 200])
    dd.set_xlim(5.3, 230)
    assert np.max(d['curves']) < .53, np.max(d['curves'])
    dd.set_ylim(.1, .55)
    dd.set_yticks([.1, .2, .3, .4, .5])
    dd.set_xlabel('Neighbours k')
    dd.set_ylabel('Mean ARI, five patients')
    box = dd.get_position()
    lx = box.x1 * W + 14
    cv.text(lx, y2 + HEADER + 6, 'Construction', size=NOTE)
    cv.text(lx, y2 + HEADER + 14.5, 'scale', size=NOTE)
    cv.fig.legend(handles=curve_handles, loc='upper left', ncol=1, frameon=False, fontsize=LEGEND,
                  handlelength=1.6, handletextpad=.4, labelspacing=.45, borderaxespad=0, borderpad=0,
                  bbox_to_anchor=(cv.fx(lx), cv.fy(y2 + HEADER + 20)), bbox_transform=cv.fig.transFigure)

    # e: Tessera-ST minus its same-graph mean, with the lung smallest effect of interest
    xe = wd + 12
    e = cv.panel(xe, y2, W - xe, h2, 'e', f'Tessera-ST {MINUS} same-graph mean', left=Y_AXIS, bottom=X_AXIS,
                 right=34)
    e.axhspan(-d['sesoi'], d['sesoi'], color='#e3e3e3', lw=0, zorder=0)
    e.axhline(0, color=INK, lw=.6, zorder=1)
    for p in LUNG:
        e.plot(kcs, d['per'][p]['margin'], color=DOT, lw=.8, zorder=2, solid_capstyle='round')
    e.plot(kcs, d['margin'], color=TESSERA, lw=2.0, marker='o', markersize=3.2, zorder=4)
    e.set_xscale('log')
    plain_log_ticks(e.xaxis, ticks)
    e.set_xlim(*lim)
    e.set_ylim(-.24, .12)
    e.set_yticks([-.2, -.1, 0, .1])
    e.set_xlabel('Construction scale (neighbours)')
    e.set_ylabel('ARI difference')
    ebox = e.get_position()
    e_top = (1 - ebox.y1) * cv.H
    e_h = ebox.height * cv.H
    y_mid = e_top + e_h * (.12 - 0) / .36
    cv.text(ebox.x1 * W + 3, y_mid, f'±{sesoi}', size=NOTE, va='center')
    cv.resize(y2 + h2)
    return save(cv, 'fig-construction', {
        'k_c': kcs, **record_abc, 'd': {'k': grid, 'mean_ari': d['curves'], 'peak_k': d['peaks']},
        'e': {'patients': {p: d['per'][p]['margin'] for p in LUNG}, 'mean': d['margin'], 'band': d['sesoi'],
              'band_printed': f'±{sesoi}'}})


# ====================================================== fig-prospective
def prospective_data():
    w6 = table('W6/targets.csv')
    t = w6[w6.cohort.isin(NEW)].reset_index(drop=True)
    regret = pf.regret_table(t)
    regret['unit'] = t.dataset.values
    regret['cohort'] = t.cohort.values
    unit = regret.groupby(['cohort', 'unit']).mean(numeric_only=True)
    p1 = record('summary/summary.json')['P1']
    assert len(unit) == p1['n_units'] and len(t) == p1['n_targets'], (len(unit), len(t))
    for c in CHOICES:
        assert abs(unit[c].mean() - p1['mean_regret'][c]) < 1e-12, c
    cohort_means = unit.groupby(level=0).mean()
    return dict(cohort=[c for c, _ in unit.index], unit=[u for _, u in unit.index],
                regret={c: unit[c].tolist() for c in CHOICES}, mean={c: float(unit[c].mean()) for c in CHOICES},
                cohort_mean={k: {c: float(cohort_means.loc[k, c]) for c in CHOICES} for k in NEW},
                targets=dict(dataset=t.dataset.tolist(), target=t.target.tolist(), cohort=t.cohort.tolist(),
                             provenance=t.provenance.tolist(), h_adj6=t.h_adj6.tolist(), g_star=t.g_star.tolist(),
                             rule_choice=[pf.rule_choice(r) for r in t.itertuples()]),
                units_per_cohort={k: int((unit.index.get_level_values(0) == k).sum()) for k in NEW},
                threshold=record('summary/summary.json')['freeze']['p1_rule']['h_adj_fallback'])


def fig_prospective():
    d = prospective_data()
    printed = {c: agree(num(d['mean'][c]), f'rPOneRegret{CHOICE_MACRO[c]}') for c in CHOICES}
    for k, short in (('spinal', 'Spinal'), ('colorectal', 'Crc'), ('wu_breast', 'Wu')):
        agree(str(d['units_per_cohort'][k]), f'rNew{short}Units')
    cv = Canvas(600, width=W)
    cv.legend(cohort_handles(calibration=False, coloured=True, counts=d['units_per_cohort']), W / 2, 1.0, ncol=3,
              columnspacing=2.0)
    top = 17.0
    gap = 18.0
    half = (W - gap) / 2
    ha, h2 = 170.0, 166.0
    offsets = {'spinal': -.22, 'colorectal': 0.0, 'wu_breast': .22}
    cohorts = np.array(d['cohort'])

    # a: unit-level regret for the rule and the four fixed choices (full width, so tied zeros stay in their column)
    a = cv.panel(0, top, W, ha, 'a', 'Regret per unit', left=Y_AXIS, bottom=X_AXIS, right=4)
    ylim = (-.012, .41)
    a.set_ylim(*ylim)
    a.set_xlim(-.55, len(CHOICES) - .45)
    pw, ph = axes_points(cv, a)
    per_x = pw / (len(CHOICES) + .1)
    diameter = 4.6
    for i, c in enumerate(CHOICES):
        values = np.array(d['regret'][c])
        dx = swarm(values, diameter, ph / (ylim[1] - ylim[0])) / per_x
        assert np.abs(dx).max() < .45, (c, np.abs(dx).max())
        for k in NEW:
            sel = cohorts == k
            points(a, i + dx[sel], values[sel], COHORT_PROV[k], k, size=17)
        a.plot([i - .3, i + .3], [d['mean'][c]] * 2, color=INK, lw=1.5, zorder=6, solid_capstyle='butt')
        a.text(i, .392, printed[c], ha='center', va='top', fontsize=NOTE)
    a.set_xticks(np.arange(len(CHOICES)), [CHOICE_LABEL[c] for c in CHOICES], linespacing=1.0)
    a.tick_params(axis='x', length=0)
    a.set_yticks([0, .1, .2, .3])
    a.set_ylabel('Regret (ARI)')

    # c: cohort-mean regret by choice
    yc = top + ha + 12
    cc = cv.panel(half + gap, yc, half, h2, 'c', 'Cohort means', left=Y_AXIS, bottom=X_AXIS, right=4)
    for k in NEW:
        x = np.arange(len(CHOICES)) + offsets[k]
        y = [d['cohort_mean'][k][c] for c in CHOICES]
        cc.plot(x, y, color=PROV_COLOR[COHORT_PROV[k]], lw=.7, alpha=.6, zorder=2)
        points(cc, x, y, COHORT_PROV[k], k, size=20)
    cc.set_xlim(-.55, len(CHOICES) - .45)
    cc.set_ylim(-.012, .3)
    cc.set_yticks([0, .1, .2, .3])
    cc.set_xticks(np.arange(len(CHOICES)), [CHOICE_LABEL[c] for c in CHOICES], linespacing=1.0)
    cc.tick_params(axis='x', length=0)
    cc.set_ylabel('Mean regret (ARI)')

    # b: homophily against label-selected gain for the new targets
    b = cv.panel(0, yc, half, h2, 'b', 'Homophily and gain', left=Y_AXIS, bottom=X_AXIS, right=6)
    b.axhline(0, color=INK, lw=.6, zorder=1)
    guide_line(b, d['threshold'])
    tg = d['targets']
    for k in NEW:
        sel = np.array(tg['cohort']) == k
        points(b, np.array(tg['h_adj6'])[sel], np.array(tg['g_star'])[sel], COHORT_PROV[k], k, size=18)
    b.set_xlim(0, 1)
    b.set_ylim(-.4, .3)
    b.set_xticks([0, .2, .4, .6, .8, 1])
    b.set_yticks([-.4, -.2, 0, .2])
    b.set_xlabel('Adjusted homophily, k = 6')
    b.set_ylabel('Label-selected gain (ARI)')
    cv.resize(yc + h2)
    return save(cv, 'fig-prospective', {
        'a': {'units': d['unit'], 'cohort': d['cohort'], 'regret': d['regret'], 'mean': d['mean'],
              'printed': printed},
        'b': {'n_points': len(tg['h_adj6']), **tg, 'threshold': d['threshold']},
        'c': d['cohort_mean'], 'units_per_cohort': d['units_per_cohort']})


# ======================================================== fig-mechanism
def mechanism_data():
    t = table('W1/targets.csv').set_index(['dataset', 'target'])
    w3 = table('W3/scores.csv')
    km = w3[w3.rule == 'kmeans']
    cn = km[km.arm == 'ContrastNoMP'].groupby(['dataset', 'target']).ari.mean()
    prop = km[km.arm == 'Propagate'].groupby(['dataset', 'target', 'L']).ari.mean().unstack('L')
    assert set(cn.index) == set(t.index) == set(prop.index), 'W3 and W1 targets differ'
    index = list(t.index)
    tess = t.ari_tessera.loc[index]
    minus_prop = {L: (tess - prop.loc[index, float(L)]).tolist() for L in DEPTHS}
    provenance = t.provenance.loc[index].tolist()
    groups = {}
    for p in PROVENANCE:
        rows = [i for i, q in enumerate(provenance) if q == p]
        groups[p] = {'n': len(rows), 'median': [], 'q1': [], 'q3': []}
        for L in DEPTHS:
            v = pd.Series(np.array(minus_prop[L])[rows])
            groups[p]['median'].append(float(v.median()))
            groups[p]['q1'].append(float(v.quantile(.25)))
            groups[p]['q3'].append(float(v.quantile(.75)))
    s = table('W2/selection.csv')
    selection = {}
    for cohort, target, title, short in SELECTION_TARGETS:
        part = s[(s.cohort == cohort) & (s.target == target)]
        units = list(dict.fromkeys(part.unit))
        selection[title] = {'short': short, 'units': units, 'provenance': t.provenance[
            [i for i in t.index if i[0].startswith(cohort) and i[1] == target][0]], 'values': {}}
        for (sel, sco), label in zip(PAIRINGS, PAIRING_LABEL):
            g = part[(part.select_rule == sel) & (part.score_rule == sco)].set_index('unit').loc[units]
            selection[title]['values'][label] = g.tessera_minus_control.tolist()
    return dict(index=[f'{a}|{b}' for a, b in index], provenance=provenance, tessera=tess.tolist(),
                no_mp=cn.loc[index].tolist(), minus_prop=minus_prop, groups=groups, selection=selection)


def fig_mechanism():
    d = mechanism_data()
    tess, no_mp = np.array(d['tessera']), np.array(d['no_mp'])
    diff = tess - no_mp
    within = int((np.abs(diff) <= .02).sum())
    agree(str(within), 'rCnWithin')
    agree(str(len(diff)), 'rCnN')
    agree(num(np.median(diff), sign=True), 'rCnMedian')
    d3 = np.array(d['minus_prop'][3])
    agree(num(np.median(d3), sign=True), 'rPThreeMedian')
    agree(str(int((d3 < 0).sum())), 'rPThreeBelow')
    counts = {p: d['groups'][p]['n'] for p in PROVENANCE}
    cv = Canvas(600, width=W)
    cv.legend(provenance_handles(counts=counts), W / 2, 1.0, ncol=5, columnspacing=1.3)
    top, h1 = 17.0, 176.0

    # a: Tessera-ST against the arm with spatial positives and no message passing
    wa = 196.0
    a = cv.panel(0, top, wa, h1, 'a', 'Spatial positives alone', left=Y_AXIS, bottom=X_AXIS, right=6)
    lim = (0, .6)
    a.fill_between(lim, [lim[0] - .02, lim[1] - .02], [lim[0] + .02, lim[1] + .02], color='#e3e3e3', lw=0,
                   zorder=0)
    a.plot(lim, lim, color=GUIDE, lw=.7, ls=DASH, zorder=1)
    for i, p in enumerate(d['provenance']):
        points(a, no_mp[i], tess[i], p, size=15)
    a.set_xlim(*lim)
    a.set_ylim(*lim)
    a.set_xticks([0, .2, .4, .6])
    a.set_yticks([0, .2, .4, .6])
    a.set_xlabel('No message passing (ARI)')
    a.set_ylabel('Tessera-ST (ARI)')
    corner_note(a, .97, .04, f'Within ±0.02: {within} of {len(diff)}')

    # b: propagation depth profile by provenance class
    xb = wa + 16
    b = cv.panel(xb, top, W - xb, h1, 'b', 'Propagation on the encoder graph', left=Y_AXIS + 4, bottom=X_AXIS,
                 right=6)
    b.axhline(0, color=INK, lw=.6, zorder=1)
    guide_line(b, np.log2(3))
    shift = {p: (i - 2) * .075 for i, p in enumerate(PROVENANCE)}
    for p in PROVENANCE:
        g = d['groups'][p]
        x = np.log2(DEPTHS) + shift[p]
        b.vlines(x, g['q1'], g['q3'], color=PROV_COLOR[p], lw=1.0, zorder=2)
        b.plot(x, g['median'], color=PROV_COLOR[p], lw=1.1, marker='o', markersize=3.6, markeredgewidth=0, zorder=3)
    b.set_xlim(-.35, 3.35)
    b.set_xticks(np.log2(DEPTHS), [str(L) for L in DEPTHS])
    b.set_ylim(-.25, .25)
    b.set_yticks([-.2, -.1, 0, .1, .2])
    b.set_xlabel('Propagation steps L')
    b.set_ylabel(f'Tessera-ST {MINUS} propagation (ARI)')
    b.text(np.log2(3) + .05, .245, 'Encoder depth', ha='left', va='top', fontsize=NOTE)

    # c: the selected control under each pairing of selection and scoring rules
    y2 = top + h1 + 12
    h2 = 182.0
    left = Y_AXIS + 9
    sub_gap = 8.0
    sw = (W - left - 3 * sub_gap - 4) / 4
    ph = h2 - HEADER - X_AXIS - 12
    ylim = (-.42, .46)
    cv.letter(0, y2 + BASELINE, 'c')
    record_c = {}
    for j, (cohort, target, title, short) in enumerate(SELECTION_TARGETS):
        ax = cv.axes(left + j * (sw + sub_gap), y2 + HEADER, sw, ph)
        sub_title(cv, ax, y2, title)
        sel = d['selection'][title]
        colour = sel['provenance']
        ax.axhline(0, color=INK, lw=.6, zorder=1)
        ax.set_xlim(-.6, 3.6)
        ax.set_ylim(*ylim)
        pw_, ph_ = axes_points(cv, ax)
        per_x = pw_ / 4.2
        record_c[title] = {'units': sel['units'], 'values': sel['values'], 'mean': {}, 'printed': {},
                           'clipped': []}
        for i, label in enumerate(PAIRING_LABEL):
            values = np.array(sel['values'][label])
            shown = np.minimum(values, ylim[1] - .03)
            dx = swarm(shown, 4.4, ph_ / (ylim[1] - ylim[0])) / per_x
            inside = values <= ylim[1] - .03
            points(ax, i + dx[inside], values[inside], colour, size=15)
            for x_, v in zip(i + dx[~inside], values[~inside]):
                ax.scatter([x_], [ylim[1] - .03], s=22, marker='^', color=PROV_COLOR[colour], linewidths=0, zorder=4)
                ax.text(x_ + .16, ylim[1] - .03, num(v, 2), ha='left', va='center', fontsize=NOTE)
                record_c[title]['clipped'].append({'pairing': label, 'value': float(v), 'printed': num(v, 2)})
            mean = float(values.mean())
            text = agree(num(mean, sign=True), f'rLeiden{short}{"".join(r[0].upper() for r in PAIRINGS[i])}')
            ax.plot([i - .32, i + .32], [mean, mean], color=INK, lw=1.5, zorder=6, solid_capstyle='butt')
            ax.text(i, ylim[0] + .015, text, ha='center', va='bottom', fontsize=NOTE)
            record_c[title]['mean'][label] = mean
            record_c[title]['printed'][label] = text
        ax.set_xticks(np.arange(4), PAIRING_LABEL)
        ax.set_yticks([-.4, -.2, 0, .2, .4])
        if j == 0:
            ax.set_ylabel(f'Tessera-ST {MINUS} selected\ncontrol (ARI)', linespacing=1.15)
        else:
            ax.set_yticklabels([])
    xlabel_y = y2 + HEADER + ph + 22
    cv.text(left + (W - left - 4) / 2, xlabel_y, 'Control selected / scored with K-means (K) or Leiden (L)',
            size=S.LABEL, ha='center')
    cv.resize(xlabel_y + 6)
    return save(cv, 'fig-mechanism', {
        'a': {'targets': d['index'], 'provenance': d['provenance'], 'no_message_passing': d['no_mp'],
              'tessera': d['tessera'], 'within_002': within},
        'b': {'L': DEPTHS, 'groups': d['groups'], 'minus_propagation': {str(k): v for k, v in d['minus_prop'].items()}},
        'c': record_c})


# ========================================================= fig-encoders
ENCODERS = ['tessera', 'graphst', 'stagate']
ENCODER_NAME = {'tessera': 'Tessera-ST', 'graphst': 'GraphST', 'stagate': 'STAGATE'}
ENCODER_DEPTH = {'tessera': 3, 'graphst': 2, 'stagate': 1}     # message-passing steps of each embedding
CAL_ENCODERS = ['graphst', 'stagate']  # refitted on the calibration cohorts; Tessera-ST was not (its calibration
                                       # contrasts with propagation are the W3 record drawn in fig-mechanism)
TARGET_CLASSES = ['region', 'celltype']
CLASS_PROV = {'region': 'pathologist', 'celltype': 'expression-assigned'}   # the provenance of every new target
CLASS_TITLE = {'region': 'Regions: colorectal and breast patients', 'celltype': 'Cell types: spinal cord mice'}
# Rows of panel a, top to bottom. For STAGATE (depth one) propagation and the same-graph mean are one operator.
P3_ROWS = [('tessera', 'propagation', f'Tessera-ST {MINUS} three-step propagation'),
           ('tessera', 'same_graph_mean', f'Tessera-ST {MINUS} same-graph mean'),
           ('graphst', 'propagation', f'GraphST {MINUS} two-step propagation'),
           ('graphst', 'same_graph_mean', f'GraphST {MINUS} same-graph mean'),
           ('stagate', 'same_graph_mean', f'STAGATE {MINUS} same-graph mean\n= one-step propagation')]
P3_MACRO = {'region': 'Reg', 'celltype': 'Ct', 'tessera': 'Tess', 'graphst': 'Graph', 'stagate': 'Stag',
            'propagation': 'Prop', 'same_graph_mean': 'Mean'}
REMOVALS = {'tessera': [('no_gate', 'No gate'), ('no_contrast', 'No contrastive term'),
                        ('self_only', 'Self edges only'),
                        ('permuted_xy_no_contrast', 'Permuted coordinates,\nno contrastive term'),
                        ('contrast_no_mp', 'Spatial positives,\nno message passing')],
            'graphst': [('no_contrast', 'No contrastive term'), ('identity_encoder', 'Identity encoder\nadjacency'),
                        ('permuted_xy', 'Permuted coordinates')],
            'stagate': [('uniform_attention', 'Uniform attention'), ('identity_graph', 'Identity graph'),
                        ('permuted_xy', 'Permuted coordinates')]}
# Calibration targets of panel d; W4 names the lung cell types cell_type, W1 names them celltype.
CALIBRATION_TARGETS = [('NSCLC', 'primary', 'Lung regions'), ('NSCLC', 'cell_type', 'Lung cell types'),
                       ('BCC', 'primary', 'BCC cell types'), ('HER2ST', 'all', 'HER2ST all labels'),
                       ('HER2ST', 'boundary_T1', 'HER2ST tumour/non-tumour')]
BAND = '#e3e3e3'


def encoders_data():
    """Unit contrasts with seeds averaged first: K-means, common input, each unit's selected configuration.

    Rows follow analyze_r3.encoder_units; the panel-a units must equal summary/p3_units.csv and their
    means the P3 estimates in summary.json (mean over units), or the build stops."""
    p3 = record('summary/summary.json')['P3']
    s = table('W4/scores.csv')
    fits = s[(s.rule == 'kmeans') & (s.input_mode == 'common')].groupby(
        ['encoder', 'dataset', 'target', 'config', 'arm']).ari.mean()
    c = table('W4/counterparts.csv')
    steps = c[c.rule == 'kmeans'].groupby(['encoder', 'dataset', 'target', 'config', 'L']).ari.mean()
    provenance = table('W6/targets.csv').set_index(['dataset', 'target']).provenance
    calibration = {(k, t): title for k, t, title in CALIBRATION_TARGETS}
    rows = []
    for r in table('W4/selection.csv').itertuples():
        if r.cohort not in NEW and (r.cohort, r.target) not in calibration:
            continue
        key = (r.encoder, r.dataset, r.target, r.selected)
        if (*key, 'released') not in fits.index:
            continue  # as in the summary, a selected configuration without its refits is not scored
        released = fits[(*key, 'released')]
        if r.cohort in NEW:
            group = 'celltype' if provenance[(r.dataset, r.target)] == 'expression-assigned' else 'region'
        else:
            group = calibration[(r.cohort, r.target)]
        row = dict(encoder=r.encoder, cohort=r.cohort, dataset=r.dataset, target=r.target, group=group,
                   released=released)
        for L in sorted({1, 2, 4, 8, ENCODER_DEPTH[r.encoder]}):
            row[f'L{L}'] = released - steps.get((*key, L), np.nan)
        row['propagation'], row['same_graph_mean'] = row[f'L{ENCODER_DEPTH[r.encoder]}'], row['L1']
        for arm, _ in REMOVALS[r.encoder]:
            row[arm] = fits.get((*key, arm), np.nan) - released
        rows.append(row)
    t = pd.DataFrame(rows)
    new = t[t.cohort.isin(NEW)]
    saved = table('summary/p3_units.csv').merge(new, on=['encoder', 'cohort', 'dataset', 'target'], how='outer',
                                                 validate='one_to_one')
    for mine, theirs in (('released', 'encoder_ari'), ('propagation', 'minus_propagation'),
                         ('same_graph_mean', 'minus_same_graph_mean')):
        assert np.allclose(saved[mine], saved[theirs], rtol=0, atol=1e-12, equal_nan=True), \
            f'fig-encoders: {mine} differs from p3_units.csv'
    assert (saved.group == saved.target_class).all(), 'fig-encoders: target classes differ from p3_units.csv'

    a = {}
    for cls in TARGET_CLASSES:
        for enc, comp, _ in P3_ROWS:
            unit = new[(new.encoder == enc) & (new.group == cls)].groupby(['cohort', 'dataset'])[comp].mean().dropna()
            test = p3[cls][f'{enc}|{comp}']
            assert abs(unit.mean() - test['estimate']) < 1e-12 and len(unit) == test['n_units'], (cls, enc, comp)
            a.setdefault(cls, {})[f'{enc}|{comp}'] = dict(
                cohort=unit.index.get_level_values(0).tolist(), unit=unit.index.get_level_values(1).tolist(),
                values=unit.tolist(), estimate=float(test['estimate']), ci=[float(v) for v in test['ci']],
                sesoi=float(test['sesoi']))
        prop, mean = p3[cls]['stagate|propagation'], p3[cls]['stagate|same_graph_mean']
        assert abs(prop['estimate'] - mean['estimate']) < 1e-12 and prop['ci'] == mean['ci'], cls

    b = {}
    for enc in ENCODERS:
        Ls = sorted({1, 2, 4, 8, ENCODER_DEPTH[enc]})
        b[enc] = {'L': Ls}
        for cls in TARGET_CLASSES:
            unit = new[(new.encoder == enc) & (new.group == cls)].groupby(['cohort', 'dataset'])[
                [f'L{L}' for L in Ls]].mean()
            b[enc][cls] = dict(n=len(unit), median=unit.median().tolist(), q1=unit.quantile(.25).tolist(),
                               q3=unit.quantile(.75).tolist())

    removal = {}
    for enc in ENCODERS:
        for arm, label in REMOVALS[enc]:
            unit = new[new.encoder == enc].groupby(['group', 'cohort', 'dataset'])[arm].mean().dropna()
            groups = unit.index.get_level_values(0)
            removal.setdefault(enc, {})[arm] = dict(  # class means are over units, like the panel-a estimates
                label=label.replace('\n', ' '), group=groups.tolist(), cohort=unit.index.get_level_values(1).tolist(),
                unit=unit.index.get_level_values(2).tolist(), values=unit.tolist(),
                mean={cls: float(unit.loc[cls].mean()) for cls in TARGET_CLASSES if cls in groups})

    w1 = table('W1/targets.csv')
    cal = t[~t.cohort.isin(NEW)]
    assert set(cal.encoder) == set(CAL_ENCODERS), sorted(set(cal.encoder))
    d = {'provenance': {}}
    for cohort, target, title in CALIBRATION_TARGETS:
        found = w1[(w1.cohort == cohort) & (w1.target == {'cell_type': 'celltype'}.get(target, target))]
        assert found.provenance.nunique() == 1, (cohort, target)
        d['provenance'][title] = found.provenance.iloc[0]
        for enc in CAL_ENCODERS:
            v = cal[(cal.encoder == enc) & (cal.group == title)].set_index('dataset').propagation
            assert v.index.is_unique, (enc, title)
            d.setdefault(enc, {})[title] = dict(unit=v.index.tolist(), values=v.tolist(), mean=float(v.mean()))
    return dict(a=a, b=b, c=removal, d=d)


def ari_axis(ax, values, n, axis='x'):
    """Limits padded around the values and zero, at most n + 1 round ticks with one shared precision; the
    outermost ticks stay 9 points (about half a label) inside the limits, so labels stay over their axes."""
    v = np.array([x for x in np.ravel(values) if np.isfinite(x)] + [0.0])
    pad = .08 * max(float(v.max() - v.min()), .02)
    lo, hi = float(v.min()) - pad, float(v.max()) + pad
    step = next(s for s in (.005, .01, .02, .05, .1, .2, .5, 1.0) if (hi - lo) / s <= n)
    ticks = np.arange(np.ceil(lo / step), np.floor(hi / step) + 1) * step
    digits = 3 if step < .01 else 2 if step < .1 else 1
    labels = [num(t, digits) for t in ticks]
    box = ax.get_position()
    length = 72 * (box.width * ax.figure.get_figwidth() if axis == 'x' else box.height * ax.figure.get_figheight())
    room = 9.0 * (hi - lo) / length
    lo, hi = min(lo, ticks[0] - room), max(hi, ticks[-1] + room)
    if axis == 'x':
        ax.set_xlim(lo, hi)
        ax.set_xticks(ticks, labels)
    else:
        ax.set_ylim(lo, hi)
        ax.set_yticks(ticks, labels)
    return lo, hi


def fig_encoders():
    if 'P3' not in record('summary/summary.json'):
        return None  # the three-encoder results are not in yet; every other figure builds as before
    d = encoders_data()
    cv = Canvas(700, width=W)
    cv.legend(provenance_handles(['computed-niche', 'pathologist', 'expression-assigned']), W / 2, 1.0, ncol=3,
              columnspacing=1.5)
    cv.legend(cohort_handles(), W / 2, 13.5, ncol=4, columnspacing=1.5)
    head = HEADER + 14  # letter and title, then a line of sub-panel titles
    sub = 14.0          # the second header line sits this far below the first

    # a: each encoder against its parameter-free counterparts in the new cohorts
    top = 30.0
    labels = [label for *_, label in P3_ROWS]
    lane = cv.text_width([s for label in labels for s in label.split('\n')], TICK) + 7
    est_w = cv.text_width([f'{MINUS}0.000'], NOTE) + 9
    gap = 10.0
    pw = (W - lane - 2 * est_w - gap) / 2
    pitch = 15.0
    ph = pitch * len(P3_ROWS)
    cv.header(0, top, 'a', 'Primary test')
    every = [v for cls in TARGET_CLASSES for r in d['a'][cls].values()
             for v in [*r['values'], *r['ci'], r['sesoi'], -r['sesoi']]]
    record_a = {}
    for j, cls in enumerate(TARGET_CLASSES):
        ax = cv.axes(lane + j * (pw + est_w + gap), top + head, pw, ph)
        sub_title(cv, ax, top + sub, CLASS_TITLE[cls])
        lo, hi = ari_axis(ax, every, 5)
        ax.axvline(0, color=INK, lw=.6, zorder=1)
        record_a[cls] = {}
        for i, (enc, comp, label) in enumerate(P3_ROWS):
            r = d['a'][cls][f'{enc}|{comp}']
            ax.fill_betweenx([i - .51, i + .51], -r['sesoi'], r['sesoi'], color=BAND, lw=0, zorder=0)
            v, cohorts = np.array(r['values']), np.array(r['cohort'])
            dy = np.clip(swarm(v, 3.6, pw / (hi - lo)) / pitch, -.2, .2)
            for k in NEW:
                sel = cohorts == k
                if sel.any():
                    ax.scatter(v[sel], i - .16 + dy[sel], s=10 * AREA[MARKER[k]], marker=MARKER[k], color=DOT,
                               linewidths=0, zorder=3)
            y = i + .27
            ax.plot(r['ci'], [y, y], color=INK, lw=1.1, zorder=4, solid_capstyle='butt')
            ax.scatter([r['estimate']], [y], s=15, color=INK, linewidths=0, zorder=5)
            macro = f'rPThree{P3_MACRO[cls]}{P3_MACRO[enc]}{P3_MACRO[comp]}Est'
            text = agree(num(r['estimate'], sign=True), macro)
            ax.annotate(text, xy=(1, y), xycoords=('axes fraction', 'data'), xytext=(4, 0), textcoords='offset points',
                        ha='left', va='center', fontsize=NOTE)
            record_a[cls][f'{enc}|{comp}'] = {**r, 'label': label.replace('\n', ' '), 'printed': text, 'macro': macro}
        ax.set_ylim(len(P3_ROWS) - .5, -.5)
        ax.set_yticks(np.arange(len(P3_ROWS)), labels if j == 0 else [], linespacing=1.1)
        ax.tick_params(axis='y', length=0)
        ax.set_xlabel('ARI difference')

    # b: encoder minus propagation across steps, median and interquartile range over units
    yb = top + head + ph + X_AXIS + 6
    cv.header(0, yb, 'b', 'Propagation depth')
    sub_gap = 12.0
    left_b = Y_AXIS + 9  # a two-line axis title
    bw = (W - left_b - 2 * sub_gap - 4) / 3
    bh = 66.0
    every_b = [v for enc in ENCODERS for cls in TARGET_CLASSES for q in ('q1', 'median', 'q3')
               for v in d['b'][enc][cls][q]]
    for j, enc in enumerate(ENCODERS):
        ax = cv.axes(left_b + j * (bw + sub_gap), yb + head, bw, bh)
        sub_title(cv, ax, yb + sub, ENCODER_NAME[enc])
        Ls = d['b'][enc]['L']
        x = np.log2(Ls)
        ax.axhline(0, color=INK, lw=.6, zorder=1)
        guide_line(ax, np.log2(ENCODER_DEPTH[enc]))
        for cls in TARGET_CLASSES:
            g, colour = d['b'][enc][cls], PROV_COLOR[CLASS_PROV[cls]]
            ax.fill_between(x, g['q1'], g['q3'], color=colour, alpha=.18, lw=0, zorder=2)
            ax.plot(x, g['median'], color=colour, lw=1.1, marker='o', markersize=3.2, markeredgewidth=0, zorder=3)
        ax.set_xlim(-.3, 3.3)
        ax.set_xticks(x, [str(L) for L in Ls])
        ari_axis(ax, every_b, 5, axis='y')
        if j == 0:
            ax.set_ylabel(f'Encoder {MINUS}\npropagation (ARI)', linespacing=1.15)
        else:
            ax.set_yticklabels([])
    cv.text(left_b + (W - left_b - 4) / 2, yb + head + bh + 22, 'Propagation steps', size=S.LABEL, ha='center')

    # c: component removals minus the released encoder at the same configuration
    yc = yb + head + bh + X_AXIS + 6
    cv.header(0, yc, 'c', 'Component removals')
    lanes = {enc: cv.text_width([s for _, label in REMOVALS[enc] for s in label.split('\n')], TICK) + 6
             for enc in ENCODERS}
    sub_gap_c = 10.0
    cw = (W - sum(lanes.values()) - 2 * sub_gap_c - 4) / 3
    ch = 84.0
    every_c = [v for enc in ENCODERS for r in d['c'][enc].values() for v in [*r['values'], *r['mean'].values()]]
    x0 = 0.0
    for enc in ENCODERS:
        x0 += lanes[enc]
        ax = cv.axes(x0, yc + head, cw, ch)
        x0 += cw + sub_gap_c
        sub_title(cv, ax, yc + sub, ENCODER_NAME[enc])
        lo, hi = ari_axis(ax, every_c, 4)  # 0.2 steps; with 3 only the zero tick survived
        ax.axvline(0, color=INK, lw=.6, zorder=1)
        arms = REMOVALS[enc]
        row_pt = ch / len(arms)
        for i, (arm, _) in enumerate(arms):
            r = d['c'][enc][arm]
            for cls, offset in (('region', -4.2), ('celltype', 4.2)):  # points above and below the row centre
                sel = np.array(r['group']) == cls
                v, cohorts = np.array(r['values'])[sel], np.array(r['cohort'])[sel]
                y = i + offset / row_pt
                dy = np.clip(swarm(v, 3.4, cw / (hi - lo)) / row_pt, -3.0 / row_pt, 3.0 / row_pt)
                for k in NEW:
                    pick = cohorts == k
                    if pick.any():
                        points(ax, v[pick], y + dy[pick], CLASS_PROV[cls], k, size=9, lw=.6)
                if cls in r['mean']:
                    half = 3.6 / row_pt
                    ax.plot([r['mean'][cls]] * 2, [y - half, y + half], color=INK, lw=1.4, zorder=6,
                            solid_capstyle='butt')
        ax.set_ylim(len(arms) - .5, -.5)
        ax.set_yticks(np.arange(len(arms)), [label for _, label in arms], linespacing=1.05)
        ax.tick_params(axis='y', length=0)
    cv.text(W / 2, yc + head + ch + 22, f'Removal {MINUS} full encoder, same settings (ARI)', size=S.LABEL,
            ha='center')

    # d: calibration cohorts, descriptive
    yd = yc + head + ch + X_AXIS + 6
    cv.header(0, yd, 'd', 'Calibration cohorts')
    titles = [title for *_, title in CALIBRATION_TARGETS]
    lane_d = cv.text_width(titles, TICK) + 7
    dw = (W - lane_d - sub_gap - 4) / 2
    pitch_d = 12.5
    dh = pitch_d * len(titles)
    every_d = [v for enc in CAL_ENCODERS for r in d['d'][enc].values() for v in r['values']]
    for j, enc in enumerate(CAL_ENCODERS):
        ax = cv.axes(lane_d + j * (dw + sub_gap), yd + head, dw, dh)
        sub_title(cv, ax, yd + sub, ENCODER_NAME[enc])
        lo, hi = ari_axis(ax, every_d, 4)
        ax.axvline(0, color=INK, lw=.6, zorder=1)
        for i, title in enumerate(titles):
            r = d['d'][enc][title]
            v = np.array(r['values'])
            dy = np.clip(swarm(v, 3.6, dw / (hi - lo)) / pitch_d, -.3, .3)
            points(ax, v, i + dy, d['d']['provenance'][title], size=11)
            ax.plot([r['mean']] * 2, [i - .34, i + .34], color=INK, lw=1.4, zorder=6, solid_capstyle='butt')
        ax.set_ylim(len(titles) - .5, -.5)
        ax.set_yticks(np.arange(len(titles)), titles if j == 0 else [])
        ax.tick_params(axis='y', length=0)
    bottom = yd + head + dh + 22
    cv.text(lane_d + (W - lane_d - 4) / 2, bottom, f'Encoder {MINUS} depth-matched propagation (ARI)', size=S.LABEL,
            ha='center')
    cv.resize(bottom + 6)
    return save(cv, 'fig-encoders', {'a': record_a, 'b': d['b'], 'c': d['c'],
                                     'd': {enc: d['d'][enc] for enc in CAL_ENCODERS},
                                     'd_provenance': d['d']['provenance']})


# ============================================================ fig-noise
def noise_data():
    s = table('W2/seeds.csv')
    saved = record('W1/per_seed.json')
    assert sorted(s.dataset.unique()) == sorted(BOUNDED), 'seed table covers other datasets'
    seeds = {n: s[s.dataset == n].sort_values('seed').ari.tolist() for n in BOUNDED}
    spread = s.groupby('dataset').ari.agg(['std', 'min', 'max'])
    cpu = s[s.seed <= 3].sort_values(['dataset', 'seed'])
    gpu = [saved[f'{r.dataset}|primary|tessera'][r.seed - 1] for r in cpu.itertuples()]
    out = dict(seeds=seeds, sd_median=float(spread['std'].median()), range_max=float((spread['max'] - spread['min']).max()),
               cpu=cpu.ari.tolist(), gpu=gpu, pair_dataset=cpu.dataset.tolist(), pair_seed=cpu.seed.tolist())
    out['abs_diff_median'] = float(np.median(np.abs(np.array(out['cpu']) - np.array(gpu))))
    if GPU_REPEATS.exists():
        rep = read_csv(track(GPU_REPEATS))
        rng = rep.groupby(['dataset', 'target', 'seed']).ari.agg(lambda v: float(v.max() - v.min())).reset_index()
        rng['group'] = [repeat_group(a, b) for a, b in zip(rng.dataset, rng.target)]
        known = ['NSCLC regions', 'NSCLC cell types', 'HER2ST all labels', 'HER2ST compartments',
                 'HER2ST tumour/non-tumour']
        order = [g for g in known if g in set(rng.group)] + sorted(set(rng.group) - set(known))
        out['repeats'] = {g: rng[rng.group == g].ari.tolist() for g in order}
    return out


def repeat_group(dataset, target):
    target = str(target)
    if str(dataset).startswith('NSCLC'):
        return 'NSCLC regions' if target == 'primary' else 'NSCLC cell types' if 'cell' in target else f'NSCLC {target}'
    if str(dataset).startswith('HER2ST'):
        return {'all': 'HER2ST all labels', 'compartments': 'HER2ST compartments',
                'boundary_T1': 'HER2ST tumour/non-tumour'}.get(target, f'HER2ST {target}')
    return f'{dataset} {target}'


def fig_noise():
    d = noise_data()
    sd = agree(num(d['sd_median']), 'rSeedSdMedian')
    largest = agree(num(d['range_max']), 'rSeedRangeMax')
    median = agree(f"{d['abs_diff_median']:.4f}", 'rCpuGpuMedian')
    third = 'repeats' in d
    cv = Canvas(600, width=W)
    top, h = 2.0, 196.0
    names = [SHORT[n] for n in BOUNDED]
    lane = cv.text_width(names, TICK) + 7
    wa = 278.0
    gap = 14.0
    a = cv.panel(0, top, wa, h, 'a', 'Ten CPU seeds', left=lane, bottom=X_AXIS, right=6)
    for i, n in enumerate(BOUNDED):
        v = np.array(d['seeds'][n])
        a.plot([v.min(), v.max()], [i, i], color='#e8c3b0', lw=3.2, zorder=1, solid_capstyle='round')
        a.scatter(v, np.full(len(v), i), s=9, color=TESSERA, linewidths=0, alpha=.9, zorder=3)
    a.set_yticks(np.arange(len(BOUNDED)), names)
    a.tick_params(axis='y', length=0)
    a.set_ylim(len(BOUNDED) - .4, -.6)
    a.set_xlim(.04, .40)
    a.set_xticks([.1, .2, .3, .4])
    a.set_xlabel('Tessera-ST ARI')
    corner_note(a, .97, .03, f'Median SD {sd}\nLargest range {largest}')
    xb = wa + gap
    b = cv.panel(xb, top, W - xb, h, 'b', 'CPU refit and saved GPU fit', left=Y_AXIS, bottom=X_AXIS, right=6)
    lim = (.04, .40)
    b.plot(lim, lim, color=GUIDE, lw=.7, ls=DASH, zorder=1)
    b.scatter(d['gpu'], d['cpu'], s=13, color=TESSERA, linewidths=0, alpha=.9, zorder=3)
    b.set_xlim(*lim)
    b.set_ylim(*lim)
    b.set_xticks([.1, .2, .3, .4])
    b.set_yticks([.1, .2, .3, .4])
    b.set_xlabel('Saved GPU fit (ARI)')
    b.set_ylabel('CPU refit (ARI)')
    corner_note(b, .97, .03, f'Median |difference| {median}')
    record_c = None
    bottom = top + h
    if third:  # a full-width row under a and b, which keep the layout of the two-panel version
        groups = list(d['repeats'])
        y3 = top + h + 10
        h3 = HEADER + 15.0 * len(groups) + X_AXIS
        glane = cv.text_width(groups, TICK) + 7
        c = cv.panel(0, y3, W, h3, 'c', 'Same-seed GPU repeats', left=glane, bottom=X_AXIS, right=18)
        widest = max(max(v) for v in d['repeats'].values())
        c.set_xlim(0, max(.005, 1.15 * widest))
        c.set_ylim(len(groups) - .5, -.5)
        pw_, ph_ = axes_points(cv, c)
        for i, g in enumerate(groups):
            v = np.array(d['repeats'][g])
            dy = swarm(v, 3.6, pw_ / c.get_xlim()[1]) / (ph_ / len(groups))
            c.scatter(v, i + np.clip(dy, -.45, .45), s=9, color=TESSERA, linewidths=0, alpha=.85, zorder=3)
        c.set_yticks(np.arange(len(groups)), groups)
        c.tick_params(axis='y', length=0)
        c.set_xlabel('Range over three repeats (ARI)')
        record_c = d['repeats']
        bottom = y3 + h3
    cv.resize(bottom)
    # The caption file describes panel c only when this marker exists, so caption and figure stay in step.
    path = save(cv, 'fig-noise', {
        'a': {'datasets': BOUNDED, 'seeds': d['seeds'], 'printed': {'sd_median': sd, 'range_max': largest}},
        'b': {'datasets': d['pair_dataset'], 'seed': d['pair_seed'], 'cpu': d['cpu'], 'gpu': d['gpu'],
              'printed': median},
        'c': record_c, 'panels': 3 if third else 2})
    marker = OUT / NOISE_PANEL_C
    if third:
        marker.write_text('% Panel c (same-seed GPU repeats) is drawn in fig-noise.\n')
    elif marker.exists():
        marker.unlink()
    return path


# ========================================================== fig-concept
CLASS_COLOR = {'tumour': '#e39b7b', 'immune': '#efcd62', 'stroma': '#c3ccd4'}
CLASSES = ['tumour', 'immune', 'stroma']


def synthetic_tissue(nx=15, ny=17, seed=4):
    """A small hexagonal lattice with a tumour nest, an immune-rich margin and stroma (illustration only)."""
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    xy = np.column_stack([(ii + .5 * (jj % 2)).ravel(), (jj * np.sqrt(3) / 2).ravel()])
    centre = np.array([6.6, 7.4])
    rel = xy - centre
    radius, angle = np.hypot(rel[:, 0], rel[:, 1]), np.arctan2(rel[:, 1], rel[:, 0])
    depth = radius - (4.3 + .9 * np.sin(3 * angle + .5) + .45 * np.cos(5 * angle))
    probs = np.where(depth[:, None] < -.8, [.8, .12, .08], np.where(depth[:, None] < 1.7, [.22, .55, .23],
                                                                   [.05, .18, .77]))
    cell = np.array([rng.choice(3, p=p) for p in probs])
    nn = NearestNeighbors(n_neighbors=13).fit(xy)
    _, idx = nn.kneighbors(xy)
    composition = np.stack([(cell[idx] == c).mean(1) for c in range(3)], 1)
    niche = KMeans(3, n_init=10, random_state=0).fit(composition)
    dominant = niche.cluster_centers_.argmax(1)
    if len(set(dominant)) < 3:  # name niches by their most enriched class when two share a dominant class
        dominant = np.argsort(-(niche.cluster_centers_ - composition.mean(0)), axis=0)[0]
    niche_class = dominant[niche.labels_]
    drawn_edge = lambda a: 4.5 + .55 * np.sin(3 * a + .5)  # noqa: E731  the pathologist's smoother outline
    compartment = np.where(radius < drawn_edge(angle), 0, 2)
    theta = np.linspace(0, 2 * np.pi, 200)
    outline = centre + np.column_stack([np.cos(theta), np.sin(theta)]) * drawn_edge(theta)[:, None]
    return dict(xy=xy, cell=cell, niche=niche_class, compartment=compartment, outline=outline)


def draw_map(ax, tissue, labels, x0, y0, size, outline=False):
    xy = tissue['xy']
    span = xy.max(0) - xy.min(0) + 1
    scale = size / span.max()
    to = lambda p: np.column_stack([x0 + (p[:, 0] - xy[:, 0].min() + .5) * scale,  # noqa: E731
                                    y0 + (p[:, 1] - xy[:, 1].min() + .5) * scale])
    pos = to(xy)
    for (px, py), c in zip(pos, labels):
        ax.add_patch(Circle((px, py), .47 * scale, facecolor=CLASS_COLOR[CLASSES[c]], edgecolor='none', zorder=2))
    if outline:
        line = to(tissue['outline'])
        ax.plot(line[:, 0], line[:, 1], color='#3a3a3a', lw=1.1, zorder=3)
    return span * scale


def schematic_axes(cv, x, y, w, h):
    ax = cv.axes(x, y, w, h)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    return ax


def arrow(ax, start, end, colour=INK, lw=.9, scale=7.0, style='-|>', **kwargs):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=scale, color=colour, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=4, **kwargs))


def box(ax, x, y, w, h, text, edge, fill, size=TICK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=3', facecolor=fill, edgecolor=edge,
                                lw=.9, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=size, color=INK, zorder=5)


def swatch(ax, x, y, colour, size=5.0):
    ax.add_patch(Rectangle((x, y - size), size, size, facecolor=colour, edgecolor='none', zorder=3))


def fig_concept():
    m = macros()
    datasets, targets = m['rCalDatasets'], m['rCalTargets']
    units, new_targets = m['rPOneUnits'], m['rPOneTargets']
    tissue = synthetic_tissue()
    cv = Canvas(600, width=W)
    top = 0.0
    gap = 16.0
    wa = 306.0
    h1 = 152.0

    # a: one tissue, three references
    cv.header(0, top, 'a', 'One tissue, three references')
    ax = schematic_axes(cv, 0, top + HEADER, wa, h1 - HEADER)
    size = 88.0
    step = (wa - 3 * size) / 2
    maps = [('niche', 'Computed niche', 'from neighbour composition', 'computed-niche', False),
            ('compartment', 'Pathologist', 'regions drawn', 'pathologist', True),
            ('cell', 'Expression-assigned', 'cell types, own expression', 'expression-assigned', False)]
    for i, (key, name, detail, prov, outline) in enumerate(maps):
        x0 = i * (size + step)
        extent = draw_map(ax, tissue, tissue[key], x0, 2.0, size, outline)
        y_text = 2.0 + extent[1] + 11.0
        swatch(ax, x0 + 1, y_text, PROV_COLOR[prov], 6.0)
        ax.text(x0 + 10, y_text, name, ha='left', va='baseline', fontsize=TICK)
        ax.text(x0 + 1, y_text + 10, detail, ha='left', va='baseline', fontsize=NOTE)
    key_y = h1 - HEADER - 4
    kx = 0.0
    for c in CLASSES:
        ax.add_patch(Circle((kx + 3, key_y - 2.5), 2.6, facecolor=CLASS_COLOR[c], edgecolor='none', zorder=3))
        label = {'tumour': 'Tumour-like', 'immune': 'Immune-like', 'stroma': 'Stroma-like'}[c]
        ax.text(kx + 8, key_y, label, ha='left', va='baseline', fontsize=NOTE)
        kx += 64.0
    ax.plot([kx + 2, kx + 14], [key_y - 2.5, key_y - 2.5], color='#3a3a3a', lw=1.1)
    ax.text(kx + 18, key_y, 'Drawn outline', ha='left', va='baseline', fontsize=NOTE)

    # b: the three label diagnostics as icons
    xb = wa + gap
    wb = W - xb
    cv.header(xb, top, 'b', 'Label diagnostics')
    bx = schematic_axes(cv, xb, top + HEADER, wb, h1 - HEADER)
    slot = wb / 3
    iw, ih, iy = slot - 16, 64.0, 6.0
    for i in range(2):  # the first two icons are curves on axes; the third is a map of spatial blocks
        x0 = i * slot + 8
        bx.plot([x0, x0, x0 + iw], [iy, iy + ih, iy + ih], color=INK, lw=.7, zorder=2)
    # homophily across scale: niches stay high, compartments decline, cell types stay low
    u = np.linspace(0, 1, 50)
    for prov, (start, end) in (('computed-niche', (.92, .66)), ('pathologist', (.74, .36)),
                               ('expression-assigned', (.30, .07))):
        level = start + (end - start) * u ** 1.15
        bx.plot(8 + 3 + u * (iw - 6), iy + ih - level * (ih - 4), color=PROV_COLOR[prov], lw=1.3, zorder=3)
    bx.text(8 + iw, iy + ih + 9, 'k', ha='right', va='baseline', fontsize=NOTE)
    # correlation length: excess same-label share decays with neighbour rank; 1/e marks the length
    x1 = slot + 8
    level = np.exp(-u * 3.2)
    bx.plot(x1 + 3 + u * (iw - 6), iy + ih - level * (ih - 6), color=NICHE, lw=1.3, zorder=3)
    cross = 1 / 3.2
    xe = x1 + 3 + cross * (iw - 6)
    ye = iy + ih - np.exp(-1) * (ih - 6)
    bx.plot([x1, x1 + iw], [ye, ye], color=GUIDE, lw=.7, ls=DASH, zorder=2)
    bx.plot([xe, xe], [ye, iy + ih], color=GUIDE, lw=.7, ls=(0, (1.2, 1.6)), zorder=2)
    bx.text(xe, iy + ih + 9, 'ℓ', ha='center', va='baseline', fontsize=TICK)
    bx.text(x1 + iw, iy + ih + 9, 'rank', ha='right', va='baseline', fontsize=NOTE)
    # own-expression predictability under spatially blocked folds
    x2 = 2 * slot + 8
    n = 4
    cell = (iw - 8) / n
    gy = iy + (ih - n * cell) / 2
    for r in range(n):
        for q in range(n):
            held = (r, q) == (1, 2)
            bx.add_patch(Rectangle((x2 + 4 + q * cell, gy + r * cell), cell - 1.2, cell - 1.2,
                                   facecolor='#5f6b73' if held else '#dfe4e8', edgecolor='none', zorder=3))
    bx.text(x2 + iw / 2, iy + ih + 9, 'held-out block', ha='center', va='baseline', fontsize=NOTE)
    for i, (l1, l2) in enumerate([('Homophily', 'across scale'), ('Correlation', 'length'),
                                  ('Own-expression', 'predictability')]):
        cx = i * slot + 8 + iw / 2
        bx.text(cx, iy + ih + 22, l1, ha='center', va='baseline', fontsize=NOTE)
        bx.text(cx, iy + ih + 31, l2, ha='center', va='baseline', fontsize=NOTE)

    # c: matched attribution on the encoder's own graph
    y2 = top + h1 + 8
    h2 = 136.0
    cv.header(0, y2, 'c', 'Matched attribution')
    cx = schematic_axes(cv, 0, y2 + HEADER, wa, h2 - HEADER)
    nodes = np.array([[14, 30], [30, 16], [30, 44], [48, 30], [62, 14], [62, 46], [44, 58]], float)
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (2, 6), (6, 5), (1, 4)]
    for a_, b_ in edges:
        cx.plot(*nodes[[a_, b_]].T, color='#9aa5ad', lw=1.0, zorder=2)
    for p in nodes:
        cx.add_patch(Circle(p, 4.2, facecolor='#5f6b73', edgecolor='white', lw=.8, zorder=3))
    cx.text(38, 76, 'Encoder graph', ha='center', va='baseline', fontsize=NOTE)
    cx.text(38, 85, 'and depth L', ha='center', va='baseline', fontsize=NOTE)
    bw, bh, bx0 = 112.0, 17.0, 98.0
    rows = [('Encoder, L steps', TESSERA, '#f6e1d7'), ('Propagation, L steps', PROPAGATION, '#d9eeee'),
            ('Same-graph mean', SAME_GRAPH, '#e5eaed')]
    ys = [6.0, 33.0, 60.0]
    for (text, edge, fill), y in zip(rows, ys):
        box(cx, bx0, y, bw, bh, text, edge, fill)
        arrow(cx, (72, 30 + (y + bh / 2 - 39) * .55), (bx0 - 2, y + bh / 2))
    right = bx0 + bw
    cx.plot([right + 8, right + 14, right + 14, right + 8], [ys[0] + bh / 2, ys[0] + bh / 2, ys[2] + bh / 2,
                                                           ys[2] + bh / 2], color=INK, lw=.8, zorder=2)
    arrow(cx, (right + 14, 39), (right + 26, 39))
    cx.text(right + 30, 31, 'Same input', ha='left', va='baseline', fontsize=NOTE)
    cx.text(right + 30, 40, 'and clustering;', ha='left', va='baseline', fontsize=NOTE)
    cx.text(right + 30, 49, 'ARI difference', ha='left', va='baseline', fontsize=NOTE)
    cx.text(right + 30, 58, 'per patient', ha='left', va='baseline', fontsize=NOTE)
    chip_y = h2 - HEADER - 17
    cx.text(0, chip_y + 8.5, 'Removals:', ha='left', va='baseline', fontsize=NOTE)
    chip_x = 40.0
    for text in ('No gate', 'No contrast', 'Self edges only', 'Positives, no messages'):
        width = cv.text_width([text], NOTE) + 8
        cx.add_patch(FancyBboxPatch((chip_x, chip_y), width, 12, boxstyle='round,pad=0,rounding_size=2.5',
                                    facecolor='white', edgecolor=TESSERA, lw=.7, zorder=3))
        cx.text(chip_x + width / 2, chip_y + 6, text, ha='center', va='center', fontsize=NOTE, zorder=4)
        chip_x += width + 5

    # d: the prospective protocol
    cv.header(xb, y2, 'd', 'Prospective protocol')
    dx = schematic_axes(cv, xb, y2 + HEADER, wb, h2 - HEADER)
    steps = [('Calibration', f'{datasets} datasets, {targets} targets'),
             ('Protocol frozen', 'and publicly deposited'),
             ('Three prospective cohorts', 'expression data downloaded after it'),
             ('Prospective tests', f'{units} patients and animals, {new_targets} targets')]
    ys = [8.0, 33.0, 58.0, 83.0]
    dx.plot([9, 9], [ys[0], ys[-1]], color='#9aa5ad', lw=1.2, zorder=1)
    for i in range(3):
        arrow(dx, (9, ys[i] + 9), (9, ys[i + 1] - 5.5), colour='#7d8a93', lw=1.0, scale=6)
    for (l1, l2), y, fill in zip(steps, ys, ['#5f6b73', INK, '#5f6b73', '#5f6b73']):
        dx.add_patch(Circle((9, y), 4.0, facecolor=fill, edgecolor='white', lw=.8, zorder=3))
        dx.text(20, y - 1.0, l1, ha='left', va='baseline', fontsize=TICK)
        dx.text(20, y + 8.0, l2, ha='left', va='baseline', fontsize=NOTE)
    cv.resize(y2 + h2)
    return save(cv, 'fig-concept', {'printed': {'datasets': datasets, 'targets': targets, 'units': units,
                                                'new_targets': new_targets},
                                    'synthetic_cells': int(len(tissue['xy']))})


FIGURES = {'fig-concept': fig_concept, 'fig-diagnostics': fig_diagnostics, 'fig-construction': fig_construction,
           'fig-prospective': fig_prospective, 'fig-mechanism': fig_mechanism, 'fig-encoders': fig_encoders,
           'fig-noise': fig_noise}


def build(names=None):
    for name in names or NAMES:
        drawn = FIGURES[name]()
        print('drew' if drawn else 'skipped (inputs not ready)', name, flush=True)
    return write_record()


if __name__ == '__main__':
    unknown = [n for n in sys.argv[1:] if n not in FIGURES]
    if unknown:
        raise SystemExit(f'unknown figure(s): {unknown}; choose from {NAMES}')
    build(sys.argv[1:])

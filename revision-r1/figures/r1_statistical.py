"""Revision-1 statistical figures, redrawn natively in Arial from the saved inputs.

Data logic is ported verbatim from the submitted producers (render_figures.py,
render_external_figures.py and the 14 Oct Frontiers redraw); only labels, panel
selection and explicit point layout change. Old -> new numbering is in
FIGURE-MAP.md; every plotted array is written to the QA plot-data record.
"""
from __future__ import annotations

from functools import lru_cache
import string

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

import r1_style as S
from r1_style import (BCC, CELL, COLOR, DATA, EXTR, FOLLOW, HEADER, INK, LANE, LED, LEGEND, LIGHT, LUNG, MINUS,
                      NOTE, OBJECTS, RESULTS, SEEDS, SGM, SHORT, TICK, TST, W, X_AXIS, Y_AXIS, Canvas, contrast_label,
                      method_handles, object_rows, plain_log_ticks, save)

plt = S.plt


# ------------------------------------------------------------------ inputs
@lru_cache(maxsize=None)
def summary():
    return S.read_csv(LED / 'primary_summary.csv').set_index(['object', 'arm'])


def ari(name, arm):
    return float(summary().loc[(name, arm), 'ari_mean'])


@lru_cache(maxsize=None)
def nested():
    scale_summary = S.read_json(FOLLOW / 'scale_controls_summary.json')
    return {r['patient']: r for group in ['NSCLC', 'BCC'] for r in scale_summary[group]['per_patient']}


@lru_cache(maxsize=None)
def input_data(name):
    return S.load_npz(DATA / name / 'input.npz'), S.read_json(DATA / name / 'input.json')


@lru_cache(maxsize=None)
def prediction(name, arm, seed=1):
    return S.load_npz(RESULTS / name / f'{arm}-seed{seed}.npz')


@lru_cache(maxsize=None)
def fit_record(name, arm, seed=1):
    return S.read_json(RESULTS / name / f'{arm}-seed{seed}.json')


@lru_cache(maxsize=None)
def hardening():
    return S.read_json(EXTR / 'run-hardening.json')['result']


@lru_cache(maxsize=None)
def landscape():
    return S.read_csv(LED / 'landscape.csv')


# ------------------------------------------------------------ map helpers
DISPLAY_NAMES = {'Bcell': 'B cell', 'Tcell': 'T cell', 'PlasmaCell': 'Plasma cell', 'Cancer.cells': 'Cancer cells',
                 'Normal.Kerat': 'Normal keratinocyte', 'MoMacDC': 'Mono/Mac/DC', 'Tcell_CD4': 'T cell CD4',
                 'Tcell_CD8': 'T cell CD8', 'Imm_other': 'Other immune', 'epi': 'Epithelial', 'imm': 'Immune',
                 'str': 'Stromal'}
UNMATCHED = '#c8ccd0'


def display_name(c):
    return DISPLAY_NAMES.get(c, c.replace('_', ' '))


def display_reference(y, minimum_fraction=.01):
    """Gray <1 % classes by object-level prevalence, without changing score inputs."""
    y = np.asarray(y)
    ok = y >= 0
    cs, counts = np.unique(y[ok], return_counts=True)
    common = cs[counts >= minimum_fraction * ok.sum()]
    return np.where(np.isin(y, common), y, -1)


def visible_classes(data, meta):
    shown = display_reference(data['ref'])
    return set(np.asarray(meta['classes'])[np.unique(shown[shown >= 0])])


def display_match(y, pred):
    shown = display_reference(y)
    cs = np.unique(shown[shown >= 0])
    ps = np.unique(pred)
    if not len(cs):
        return np.full(len(pred), -1, dtype=int), {}
    tab = np.array([[np.sum((y == c) & (pred == p)) for p in ps] for c in cs])
    r, c = linear_sum_assignment(-tab)
    mapping = {int(ps[j]): int(cs[i]) for i, j in zip(r, c)}
    return np.array([mapping.get(int(p), -1) for p in pred]), mapping


def labels_colors(y, classes, union=None):
    union = classes if union is None else union
    return [UNMATCHED if i < 0 else S.CATEGORY[union.index(classes[int(i)]) % len(S.CATEGORY)] for i in y]


def limits(xy, pad=.015):
    lo, hi = xy.min(0), xy.max(0)
    span = hi - lo
    return (lo[0] - pad * span[0], hi[0] + pad * span[0]), (lo[1] - pad * span[1], hi[1] + pad * span[1])


def scatter_map(ax, xy, c, size=2.2):
    ax.scatter(xy[:, 0], xy[:, 1], s=size, c=c, linewidths=0, rasterized=True)


def class_legend(cv, classes, y, max_width=W - 8, visible=None):
    handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=S.CATEGORY[i % len(S.CATEGORY)],
                      markeredgewidth=0, markersize=4.5, label=display_name(c))
               for i, c in enumerate(classes) if visible is None or c in visible]
    if visible is not None:
        handles.append(Line2D([0], [0], marker='o', color='none', markerfacecolor=UNMATCHED, markeredgewidth=0,
                              markersize=4.5, label='Rare / unknown / unmatched'))
    return cv.fitted_legend(handles, W / 2, y, max_width, size=LEGEND, columnspacing=1.1)


def legend_height(cv, legend):
    x0, y0, x1, y1 = cv.extent(legend)
    return y1 - y0


def ari_badge(ax, text):
    ax.text(.985, .015, text, transform=ax.transAxes, fontsize=NOTE, ha='right', va='bottom',
            bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=1.2))


def light_heatmap(name, white_fraction=.55):
    rgba = plt.get_cmap(name)(np.linspace(0, 1, 256))
    rgba[:, :3] = white_fraction + (1 - white_fraction) * rgba[:, :3]
    return LinearSegmentedColormap.from_list(name + '-light', rgba)


def colorbar(cv, mappable, x, y, w, h, label=None, ticks=None, size=NOTE, title=None):
    cax = cv.axes(x, y, w, h)
    bar = cv.fig.colorbar(mappable, cax=cax)
    bar.ax.tick_params(labelsize=size, length=2.5, width=.5, pad=2)
    bar.outline.set_linewidth(.5)
    if ticks is not None:
        bar.set_ticks(ticks)
    if label:
        bar.set_label(label, fontsize=size, labelpad=3)
    if title:
        cv.text(x + w / 2, y - 4, title, size=size, ha='center')
    return bar


def open_marker(ax, x, y, key, marker='o', s=17, lw=1.0, zorder=3):
    ax.scatter(x, y, s=s, marker=marker, facecolors='white', edgecolors=COLOR[key], linewidths=lw, zorder=zorder)


def top_legend(cv, handles, y=1.0, x=W / 2, **kwargs):
    return cv.legend(handles, x, y, ncol=len(handles), size=LEGEND, columnspacing=1.4, handletextpad=.3, **kwargs)


def arm_legend_handles(keys=('PCA50', 'NeighborMean', 'Tuned', 'Tessera')):
    return method_handles(list(keys))


# ============================================================ main Figure 3
def figure_03():
    """Lung and BCC comparisons (old Figure 3; same panel layout, explicit offsets)."""
    cv = Canvas(432)
    top_legend(cv, arm_legend_handles())
    y = np.arange(13)
    names_w = cv.text_width([SHORT[n] for n in OBJECTS], TICK)
    left_a = names_w + 7
    row1, h1, bottom1 = 17, 252, 37
    gap = 6
    wa = 196
    wb = (W - wa - 2 * gap) / 2
    a = cv.panel(0, row1, wa, h1, 'a', 'Partition agreement', left=left_a, bottom=bottom1, right=8)
    b = cv.panel(wa + gap, row1, wb, h1, 'b', 'Subsampled fields', left=6, bottom=bottom1, right=8)
    c = cv.panel(wa + 2 * gap + wb, row1, wb, h1, 'c', 'Whole sections', left=6, bottom=bottom1, right=8)
    for ax in (a, b, c):
        ax.set_ylim(12.6, -.6)
        ax.set_yticks(y)
        ax.tick_params(axis='y', length=0)
    a.set_yticklabels([SHORT[n] for n in OBJECTS])
    for ax in (b, c):
        ax.set_yticklabels([])
    series = {}
    for arm, marker in [('PCA50', 'o'), ('NeighborMean', 's'), ('Tessera', 'o'), ('Tuned', 'D')]:
        values = [nested()[n]['nested_control_ari'] if arm == 'Tuned' and n in nested() else
                  np.nan if arm == 'Tuned' else ari(n, arm) for n in OBJECTS]
        a.scatter(values, y, s=16, marker=marker, color=COLOR[arm], linewidths=0, zorder=3)
        series[arm] = values
    a.set_xlim(0, .56)
    a.set_xticks([0, .2, .4])
    a.set_xlabel('ARI')
    vp = [ari(n, 'Tessera') - ari(n, 'PCA50') for n in OBJECTS[:9]]
    vt = [ari(n, 'Tessera') - nested()[n]['nested_control_ari'] for n in OBJECTS[:9]]
    whole = S.read_json(FOLLOW / 'whole_sections_summary.json')
    wp = whole['NSCLC']['tessera_minus_pca50'] + whole['BCC']['tessera_minus_pca50']
    wt = whole['NSCLC']['tessera_minus_selected_control'] + whole['BCC']['tessera_minus_selected_control']
    for ax, p, t in [(b, vp, vt), (c, wp, wt)]:
        ax.axvline(0, color=INK, lw=.7)
        open_marker(ax, p, np.arange(9), 'PCA50')
        open_marker(ax, t, np.arange(9), 'Tuned', marker='D')
        ax.set_xlim(-.30, .22)
        ax.set_xticks([-.2, 0, .2])
        contrast_label(ax)
    row2, h2 = row1 + h1 + 8, 162
    bottom2 = X_AXIS
    d = cv.panel(0, row2, wa, h2 - 34, 'd', 'Cell-type cluster count', left=left_a, bottom=bottom2, right=8)
    rescore = S.read_csv(FOLLOW / 'rescore.csv')
    r = rescore[rescore.object.isin(OBJECTS[:5])].groupby(['object', 'arm']).ari_celltype_at_celltype_K.mean()
    celltype = {}
    for arm, marker in [('PCA50', 'o'), ('NeighborMean', 's'), ('Tessera', 'o')]:
        values = [r[n, arm] for n in OBJECTS[:5]]
        d.scatter(values, np.arange(5), s=17, marker=marker, color=COLOR[arm], linewidths=0, zorder=3)
        celltype[arm] = values
    d.set_yticks(np.arange(5), [SHORT[n] for n in OBJECTS[:5]])
    d.tick_params(axis='y', length=0)
    d.set_ylim(4.6, -.6)
    d.set_xlim(0, .4)
    d.set_xlabel('Cell-type ARI')
    scales = S.read_csv(FOLLOW / 'scale_controls.csv')
    ks = [6, 10, 15, 30, 50, 100, 200]
    curves = {}
    for i, (letter, text, names, kind) in enumerate([('e', 'Lung neighborhood scale', OBJECTS[:5], 'NeighborMean'),
                                                     ('f', 'BCC neighborhood scale', OBJECTS[5:9], 'BANKSY-style')]):
        x = wa + gap + i * (wb + gap)
        ax = cv.panel(x, row2, wb, h2 - 34, letter, text, left=Y_AXIS - 4, bottom=bottom2, right=8)
        for n, col in zip(names, S.PATIENT_SHADES):
            sub = scales[(scales.object == n) & (scales.kind == kind)]
            if kind == 'BANKSY-style':
                sub = sub[np.isclose(sub.lam, .2)]
            values = sub.groupby('k').ari.mean().reindex(ks)
            ax.plot(ks, values, 'o-', color=col, lw=1.1, ms=2, label=SHORT[n].replace('Lung', 'L').replace('BCC ', 'B'))
            curves[n] = values.tolist()
        ax.set_xscale('log')
        plain_log_ticks(ax.xaxis, [6, 30, 200])
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Neighbors k')
        ax.set_ylabel('ARI')
        box = ax.get_position()
        centre = (box.x0 + box.x1) / 2 * W
        cv.legend(ax.get_legend_handles_labels()[0], centre, row2 + h2 - 30, ncol=2, size=LEGEND,
                  handlelength=1.1, columnspacing=.8, handletextpad=.3)
    cv.resize(row2 + h2)
    return save(cv, 'figure-03', {'a': series, 'bounded_vs_pca': vp, 'bounded_vs_tuned': vt, 'whole_vs_pca': wp,
                                   'whole_vs_tuned': wt, 'd': celltype, 'scale_curves': curves})


# ============================================================ main Figure 4
ABLATION_ARMS = ['NoGate', 'NoScale', 'NoContrast', 'SelfOnly', 'PermutedXY', 'PermutedNoContrast']
ABLATION_LABELS = ['No gate', 'No fusion', 'No contrast', 'Self-only', 'Permuted XY', 'Permuted XY,\nno contrast']


def ablation_matrix():
    nulls = S.read_csv(FOLLOW / 'geometry_nulls.csv').groupby('object').ari.mean()
    return np.array([[ari(n, 'Tessera') - (nulls[n] if a == 'PermutedNoContrast' else ari(n, a)) for a in ABLATION_ARMS]
                     for n in OBJECTS])


def figure_04():
    """Component contrasts (old 4a) and seam discrimination (old 4c)."""
    matrix = ablation_matrix()
    cv = Canvas(244)
    names = [SHORT[n] for n in OBJECTS]
    left = cv.text_width(names, TICK) + 7
    band = cv.text_width(['Permuted XY'], TICK) + 9
    wa, rows_h = 302, 165
    bar_block = 46
    heat_w = wa - left - bar_block - 6
    cv.header(0, 0, 'a', f'{TST} minus component control')
    a = cv.axes(left, HEADER, heat_w, rows_h)
    rgba = plt.get_cmap('RdBu_r')(np.linspace(0, 1, 256))
    rgba[:, :3] = .55 + .45 * rgba[:, :3]
    cmap = LinearSegmentedColormap.from_list('light-diverging', rgba)
    lim = max(.05, np.abs(matrix).max())
    im = a.imshow(matrix, aspect='auto', cmap=cmap, norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim))
    a.set_yticks(np.arange(13), names)
    a.set_xticks(np.arange(6), ABLATION_LABELS, rotation=90, ha='center', va='top', linespacing=1.05)
    a.tick_params(length=0)
    for spine in a.spines.values():
        spine.set_visible(False)
    for i in range(13):
        for j in range(6):
            v = matrix[i, j]
            a.text(j, i, f'{v:+.2f}'.replace('-', MINUS) if abs(v) >= .005 else '0.00', ha='center', va='center',
                   fontsize=NOTE)
    colorbar(cv, im, left + heat_w + 6, HEADER, 6, rows_h, label=f'{TST} {MINUS} control (ARI)')
    gate = S.read_csv(FOLLOW / 'gate_baseline.csv').groupby('object').mean(numeric_only=True)
    xb = wa + 10
    b = cv.panel(xb, 0, W - xb, HEADER + rows_h + X_AXIS + 4, 'b', 'Seam discrimination', left=Y_AXIS,
                 bottom=X_AXIS + 4, right=6)
    b.scatter(gate.expression_difference_auroc, gate.gate_auroc, s=17, color='black', linewidths=0, zorder=3)
    b.plot([.35, .8], [.35, .8], '--', color='#aaaaaa', lw=.7, zorder=1)
    b.axhline(.5, color='#cccccc', lw=.5, zorder=0)
    b.axvline(.5, color='#cccccc', lw=.5, zorder=0)
    b.set_xlim(.35, .8)
    b.set_ylim(.35, .8)
    b.set_xticks([.4, .5, .6, .7, .8])
    b.set_yticks([.4, .5, .6, .7, .8])
    b.set_xlabel('PCA-feature difference AUROC')
    b.set_ylabel('Learned gate AUROC')
    cv.resize(HEADER + rows_h + band + 4)
    return save(cv, 'figure-04', {'full_minus_control': matrix, 'rows': OBJECTS, 'columns': ABLATION_ARMS,
                                   'seam_gate_auroc': gate.gate_auroc.reindex(OBJECTS).tolist(),
                                   'seam_difference_auroc': gate.expression_difference_auroc.reindex(OBJECTS).tolist()})


# ============================================================ main Figure 7
def figure_07():
    """Patient-held-out NSCLC transfer (old Figure 9)."""
    primary = S.read_csv(LED / 'all_transfer_scores.csv').groupby(['patient', 'arm', 'target'])[
        ['accuracy', 'balanced_accuracy']].mean()
    means = S.read_csv(FOLLOW / 'transfer_neighbor_mean.csv').set_index(['patient', 'arm', 'target'])
    mean_names = sorted(set(means.index.get_level_values('arm')), key=lambda a: float(''.join(c for c in a if c.isdigit())))
    assert mean_names == ['NeighborMean-k6', 'NeighborMean-k200'], mean_names
    arms = ['PCA50', 'SelfOnly', mean_names[0], mean_names[1], 'Tessera']
    labels = ['PCA50', 'Self-only', SGM, '200-neighbor mean', TST]
    colors = [COLOR['PCA50'], COLOR['SelfOnly'], COLOR['NeighborMean'], COLOR['Mean200'], COLOR['Tessera']]
    cv = Canvas(400)
    gap_x, gap_y = 10, 10
    wb = (W - gap_x) / 2
    hb = 190
    record = {}
    for i, (target, metric) in enumerate([('region', 'balanced_accuracy'), ('cell_type', 'balanced_accuracy'),
                                          ('region', 'accuracy'), ('cell_type', 'accuracy')]):
        x = 0 if i % 2 == 0 else wb + gap_x
        y = 0 if i < 2 else hb + gap_y
        title = ('Regions' if target == 'region' else 'Cell types') + (': balanced accuracy' if metric == 'balanced_accuracy'
                                                                         else ': accuracy')
        ax = cv.panel(x, y, wb, hb, 'abcd'[i], title, left=Y_AXIS + 6, bottom=52, right=8)
        matrix = np.array([[float((means if a in mean_names else primary).loc[(n, a, target), metric]) for n in OBJECTS[:5]]
                           for a in arms])
        for values in matrix.T:
            ax.plot(np.arange(5), values, color='#ccd0d4', lw=.75, zorder=1)
        for j, col in enumerate(colors):
            ax.scatter(np.full(5, j), matrix[j], s=14, color=col, linewidths=0, zorder=3)
            ax.plot([j - .22, j + .22], [matrix[j].mean()] * 2, color=INK, lw=1.5, zorder=4)
        ax.set_xticks(np.arange(5), labels, rotation=30, ha='right', rotation_mode='anchor')
        ax.set_xlim(-.5, 4.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.set_ylabel('Held-out patient score')
        record[f'{target}_{metric}'] = matrix
    cv.resize(2 * hb + gap_y)
    return save(cv, 'figure-07', {'arms': arms, 'patients': OBJECTS[:5], **record})


# ============================================================ main Figure 8 / S15
def patient_rows(ax, units, extra_gap=False):
    y = np.arange(len(units), dtype=float)
    ax.set_yticks(y, [u[0] for u in units])
    ax.set_ylim(len(units) - .4 + (1.9 if extra_gap else 0), -.6)
    ax.tick_params(axis='y', length=0)
    return y


ARM_DOT_COLUMNS = [('PCA50', 'PCA50', 'o', 20), ('UniformSameGraph', 'NeighborMean', 's', 18),
                   ('NestedControl', 'Tuned', 'D', 20), ('Tessera', 'Tessera', 'o', 28)]


def arm_dots(ax, target, xlim, ticks):
    hard = hardening()
    units = hard['targets']['all']['units']
    per = hard['targets'][target]['per_unit']
    y = patient_rows(ax, units)
    for yy, u in zip(y, units):
        values = [per[u][c] for c, *_ in ARM_DOT_COLUMNS]
        ax.plot([min(values), max(values)], [yy, yy], color=LIGHT, lw=1.2, zorder=1, solid_capstyle='round')
    for c, color, marker, size in ARM_DOT_COLUMNS:
        ax.scatter([per[u][c] for u in units], y, s=size, marker=marker, color=COLOR[color], zorder=3, linewidths=0)
    ax.axvline(0, color=INK, lw=.5)
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.set_xlabel('Adjusted Rand index')
    return {c: [per[u][c] for u in units] for c, *_ in ARM_DOT_COLUMNS}


FOREST_TARGETS = [('All labels', 'all'), ('Compartments', 'compartments'), ('Tumor boundary', 'boundary_T1')]
FOREST_SPEC = [('Tessera_minus_UniformSameGraph', f'{TST} {MINUS} same-graph mean', 'Tessera'),
               ('Tessera_minus_NestedControl', f'{TST} {MINUS} tuned control', 'Tessera'),
               ('UniformSameGraph_minus_NestedControl', f'{SGM} {MINUS} tuned control', 'NeighborMean')]


def forest(ax, left_points):
    """Paired patient contrasts; target headers sit in the label lane above their three rows."""
    hard = hardening()
    ys, labels, headers, row, record = [], [], [], 0., {}
    for header, target in FOREST_TARGETS:
        headers.append((row, header))
        row += 1
        contrasts = hard['targets'][target]['contrasts']
        for key, label, color in FOREST_SPEC:
            c = contrasts[key]
            deltas = np.array(c['deltas'])
            ax.scatter(deltas, np.full(len(deltas), row), s=7, color='#9aa3a9', linewidths=0, zorder=2)
            ax.plot(c['bootstrap95'], [row, row], color=COLOR[color], lw=1.6, zorder=3, solid_capstyle='butt')
            ax.scatter([c['mean']], [row], s=30, marker='D', color=COLOR[color], zorder=4, linewidths=0)
            ax.text(1.02, row, f"+{c['n_positive']}/{MINUS}{c['n_negative']}", transform=ax.get_yaxis_transform(),
                    fontsize=NOTE, va='center', ha='left')
            ys.append(row)
            labels.append(label)
            record[f'{target}:{key}'] = {'deltas': deltas, 'mean': c['mean'], 'bootstrap95': c['bootstrap95'],
                                         'n_positive': c['n_positive'], 'n_negative': c['n_negative']}
            row += 1
        row += .35
    ax.set_yticks(ys, labels)
    ax.set_ylim(row - .75, -.7)
    ax.tick_params(axis='y', length=0)
    ax.axvline(0, color=INK, lw=.6)
    ax.set_xlim(-.26, .2)
    ax.set_xticks([-.2, -.1, 0, .1, .2])
    ax.set_xlabel('ARI difference')
    width = ax.get_position().width * W
    lane = blended_transform_factory(ax.transAxes, ax.transData)
    for yy, header in headers:
        ax.text(-(left_points - 1) / width, yy, header, transform=lane, fontsize=LEGEND, va='center', ha='left')
    return record


SEAM_COLUMNS = [('gate', 'Tessera', 'o', 26), ('raw_difference', 'PCA50', 's', 20),
                ('smoothed_expression_weighted', 'NeighborMean', 'D', 20)]


def seam_dots(ax):
    hard = hardening()
    units = hard['targets']['all']['units']
    seam = hard['seams']['boundary_T1']
    y = patient_rows(ax, units, extra_gap=True)
    for yy, u in zip(y, units):
        values = [seam['per_unit'][u][c] for c, *_ in SEAM_COLUMNS]
        ax.plot([min(values), max(values)], [yy, yy], color=LIGHT, lw=1.2, zorder=1, solid_capstyle='round')
    for c, color, marker, size in SEAM_COLUMNS:
        ax.scatter([seam['per_unit'][u][c] for u in units], y, s=size, marker=marker, color=COLOR[color], zorder=3,
                   linewidths=0)
    mean_row = len(units) + .6
    for j, (c, color, marker, size) in enumerate(SEAM_COLUMNS):
        lo, hi = seam['mean'][c]['bootstrap95']
        yy = mean_row + (j - 1) * .34
        ax.plot([lo, hi], [yy, yy], color=COLOR[color], lw=1.6, zorder=3, solid_capstyle='butt')
        ax.scatter([seam['mean'][c]['mean']], [yy], s=size + 14, marker=marker, color=COLOR[color], zorder=4,
                   linewidths=0)
    ax.set_yticks(list(y) + [mean_row], [u[0] for u in units] + ['Mean'])
    ax.axvline(.5, color=S.GRAY, lw=.8, ls='--', zorder=0)
    ax.set_xlim(.3, .92)
    ax.set_xticks([.3, .4, .5, .6, .7, .8])
    ax.set_xlabel('Seam AUROC on tumor-boundary edges')
    return {c: {'per_unit': [seam['per_unit'][u][c] for u in units], 'mean': seam['mean'][c]} for c, *_ in SEAM_COLUMNS}


def seam_handles():
    return [Line2D([0], [0], marker='o', ls='', color=COLOR['Tessera'], markeredgewidth=0, markersize=5,
                   label='Learned gate'),
            Line2D([0], [0], marker='s', ls='', color=COLOR['PCA50'], markeredgewidth=0, markersize=4.6,
                   label='Feature difference'),
            Line2D([0], [0], marker='D', ls='', color=COLOR['NeighborMean'], markeredgewidth=0, markersize=4.2,
                   label='Smoothed difference')]


def figure_08():
    """Independent pathologist cohort (old Figure 11 panels a, c and d)."""
    cv = Canvas(420)
    top_legend(cv, arm_legend_handles())
    row1, h1 = 17, 206
    wa = 186
    a = cv.panel(0, row1, wa, h1, 'a', 'All labels', left=16, bottom=X_AXIS, right=8)
    rec_a = arm_dots(a, 'all', (-.01, .38), [0, .1, .2, .3])
    labels_w = cv.text_width([s[1] for s in FOREST_SPEC], TICK)
    left_b = labels_w + 8
    xb = wa + 8
    b = cv.panel(xb, row1, W - xb, h1, 'b', 'Paired contrasts', left=left_b, bottom=X_AXIS, right=30)
    rec_b = forest(b, left_b)
    row2, h2 = row1 + h1 + 8, 178
    wc = 312
    c = cv.panel(0, row2, wc, h2, 'c', 'Seam AUROC', left=34, bottom=X_AXIS, right=8)
    rec_c = seam_dots(c)
    cv.legend(seam_handles(), (wc + W) / 2 + 4, row2 + HEADER + (h2 - HEADER - X_AXIS) / 2 - 16, ncol=1, size=LEGEND,
              handletextpad=.4, labelspacing=.6)
    cv.resize(row2 + h2)
    return save(cv, 'figure-08', {'a_all_labels': rec_a, 'b_paired_contrasts': rec_b, 'c_seam_auroc': rec_c})


def replicate_dots(ax):
    hard = hardening()
    rep = S.read_csv(LED / 'replicates.csv')
    rep = rep[rep.target == 'all'].set_index('unit').loc[hard['targets']['all']['units']]
    y = patient_rows(ax, list(rep.index))
    for yy, (_, r) in zip(y, rep.iterrows()):
        runs = [r.run1, r.run2, r.run3]
        ax.plot([min(runs), max(runs)], [yy, yy], color=COLOR['Tessera'], lw=1.3, zorder=2, solid_capstyle='round',
                alpha=.7)
    for column in ('run1', 'run2', 'run3'):
        ax.scatter(rep[column], y, s=15, facecolors='white', edgecolors=COLOR['Tessera'], linewidths=1., zorder=3)
    ax.scatter(rep.same_graph_mean, y, s=20, marker='s', color=COLOR['NeighborMean'], zorder=4, linewidths=0)
    ax.scatter(rep.nested_control, y, s=20, marker='D', color=COLOR['Tuned'], zorder=4, linewidths=0)
    ax.set_xlim(0, .46)
    ax.set_xticks([0, .1, .2, .3, .4])
    ax.set_xlabel('ARI, all labels')
    return {k: rep[k].tolist() for k in ['run1', 'run2', 'run3', 'same_graph_mean', 'nested_control']}


WEIGHT_SPEC = [('UniformSameGraph', 's', COLOR['NeighborMean'], 'full', f'vs {SGM.lower()}'),
               ('WeightShuffled', 's', '#8c8c8c', 'none', 'vs shuffled weights'),
               ('ShuffledCoords', '^', INK, 'full', 'vs shuffled coordinates')]


def weighting_dots(ax):
    hard = hardening()
    units = hard['targets']['all']['units']
    per = hard['targets']['boundary_T1']['per_unit']
    y = patient_rows(ax, units)
    record = {}
    for column, marker, color, fill, _ in WEIGHT_SPEC:
        diff = [per[u]['ExpressionWeighted'] - per[u][column] for u in units]
        ax.scatter(diff, y, s=22, marker=marker, facecolors=color if fill == 'full' else 'white', edgecolors=color,
                   linewidths=1., zorder=3)
        record[column] = diff
    ax.axvline(0, color=INK, lw=.6)
    ax.set_xlim(-.12, .6)
    ax.set_xticks([0, .2, .4, .6])
    ax.set_xlabel('Weighted mean minus comparator (ARI)')
    return record


def figure_S15():
    """Pathologist cohort supplements (old Figure 11 panels b, e and f)."""
    cv = Canvas(300)
    refit = Line2D([0], [0], marker='o', ls='', mfc='white', mec=COLOR['Tessera'], markeredgewidth=1., markersize=4.5,
                   label=f'{TST} refit')
    top_legend(cv, arm_legend_handles() + [refit])
    row, h = 17, 214
    gap = 8
    w = (W - 2 * gap) / 3
    a = cv.panel(0, row, w, h, 'a', 'Tumor boundary', left=16, bottom=X_AXIS, right=8)
    rec_a = arm_dots(a, 'boundary_T1', (-.04, .8), [0, .2, .4, .6, .8])
    b = cv.panel(w + gap, row, w, h, 'b', 'Replicate fits', left=16, bottom=X_AXIS, right=8)
    rec_b = replicate_dots(b)
    c = cv.panel(2 * (w + gap), row, w, h, 'c', 'Weighting and geometry', left=16, bottom=X_AXIS, right=8)
    rec_c = weighting_dots(c)
    handles = [Line2D([0], [0], marker=m, ls='', mfc=col if fill == 'full' else 'white', mec=col, markeredgewidth=1.,
                      markersize=4.5, label=label) for _, m, col, fill, label in WEIGHT_SPEC]
    box = c.get_position()
    cv.legend(handles, (box.x0 + box.x1) / 2 * W, row + h + 6, ncol=1, size=LEGEND, handletextpad=.3, labelspacing=.45)
    cv.resize(row + h + 6 + 36)
    return save(cv, 'figure-S15', {'a_tumor_boundary': rec_a, 'b_replicates': rec_b, 'c_weighting': rec_c})


# ============================================================ main Figure 9 / S16
def landscape_panels(cv, specs, top, height, legend_handles=None):
    """Shared 24-row layout for the landscape contrasts (old Figure 12, upper row)."""
    table = landscape()
    n = len(table)
    labels_w = cv.text_width(table.short_label.tolist(), TICK)
    left0 = labels_w + 7
    gap = 8
    inner = 6
    data_w = (W - left0 - 8 - gap - inner - 8) / 2
    record = {}
    for i, (letter, title, kind) in enumerate(specs):
        x = 0 if i == 0 else left0 + data_w + 8 + gap
        left = left0 if i == 0 else inner
        ax = cv.panel(x, top, left + data_w + 8, height, letter, title, left=left, bottom=37, right=8)
        ax.set_ylim(n - .5, -.5)
        ax.set_yticks(np.arange(n), table.short_label if i == 0 else [])
        ax.tick_params(axis='y', length=0)
        yy = np.arange(n)
        if kind == 'seam':
            ax.scatter(table.gate_auroc, yy, s=16, color=COLOR['Tessera'], linewidths=0, zorder=3)
            open_marker(ax, table.raw_difference_auroc, yy, 'PCA50', marker='s', s=16)
            ax.axvline(.5, color='#aaaaaa', ls='--', lw=.8, zorder=0)
            ax.set_xlim(.2, .9)
            ax.set_xticks([.3, .5, .8])
            ax.set_xlabel('AUROC')
            record['gate_auroc'] = table.gate_auroc.tolist()
            record['raw_difference_auroc'] = table.raw_difference_auroc.tolist()
        else:
            field, marker, key = {'same': ('tessera_minus_same_graph', 's', 'NeighborMean'),
                                  'oracle': ('tessera_minus_oracle', 'D', 'Tuned'),
                                  'pca': ('tessera_minus_pca50', 'o', 'PCA50')}[kind]
            open_marker(ax, table[field], yy, key, marker=marker, s=16)
            ax.axvline(0, color=INK, lw=.6)
            ax.set_xlim(-.4, .5)
            ax.set_xticks([-.3, 0, .3])
            contrast_label(ax)
            record[field] = table[field].tolist()
    return record


def figure_09():
    """Annotation landscape (old Figure 12 panels a, b and e)."""
    table = landscape()
    cv = Canvas(470)
    top, h1 = 0, 318
    record = landscape_panels(cv, [('a', SGM, 'same'), ('b', 'Test-selected', 'oracle')], top, h1)
    row2 = top + h1 + 6
    groups = table.group.drop_duplicates().tolist()
    left = cv.text_width(groups, TICK) + 7
    ax = cv.panel(0, row2, W, 112, 'c', 'Best uniform scale', left=left, bottom=X_AXIS, right=10)
    counts = []
    for r, group in enumerate(groups):
        frequency = table.loc[table.group == group, 'best_uniform_k'].value_counts().sort_index()
        for k, n in frequency.items():
            ax.scatter(k, r, s=42 + 24 * n, color='#657780', edgecolors='white', lw=.6, zorder=3)
            ax.text(k, r, str(n), color='white', fontsize=NOTE, ha='center', va='center', zorder=4)
            counts.append({'group': group, 'k': int(k), 'objects': int(n)})
    assert sum(item['objects'] for item in counts) == 24
    ax.set_yticks(np.arange(len(groups)), groups)
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(len(groups) - .5, -.55)
    ax.set_xscale('log')
    plain_log_ticks(ax.xaxis, [6, 10, 15, 30, 50, 100, 200])
    ax.set_xlim(5, 240)
    ax.set_xlabel('Neighbors k')
    cv.resize(row2 + 112)
    record['best_uniform_scale_multiplicities'] = counts
    return save(cv, 'figure-09', record)


def figure_S16():
    """Landscape supplements (old Figure 12 panels c and d)."""
    cv = Canvas(340)
    key = [Line2D([0], [0], marker='o', ls='', color=COLOR['Tessera'], markeredgewidth=0, markersize=5,
                  label='Learned gate'),
           Line2D([0], [0], marker='s', ls='', markerfacecolor='white', markeredgecolor=COLOR['PCA50'],
                  markeredgewidth=1., markersize=4.5, label='PCA-feature difference')]
    top_legend(cv, key)
    record = landscape_panels(cv, [('a', 'PCA50', 'pca'), ('b', 'Seam AUROC', 'seam')], 17, 318)
    cv.resize(17 + 318)
    return save(cv, 'figure-S16', record)


# ============================================================ S14 (old 4d)
def figure_S14():
    """Last-layer gate density in the fixed lung anchor (old Figure 4 panel d)."""
    n = OBJECTS[0]
    data = input_data(n)[0]
    ref = data['ref']
    pred = prediction(n, 'Tessera', 1)
    edge, gates = pred['edge_index'], pred['last_gate']
    valid = (ref[edge[0]] >= 0) & (ref[edge[1]] >= 0)
    same = ref[edge[0]] == ref[edge[1]]
    bins = np.linspace(0, 1, 31)
    cv = Canvas(196)
    ax = cv.panel(0, 0, W, 196, None, 'Gate density', left=Y_AXIS + 8, bottom=X_AXIS, right=8)
    densities = {}
    for mask, col, label in [(same & valid, COLOR['PCA50'], 'Same region'), (~same & valid, COLOR['Tessera'], 'Different region')]:
        density, _ = np.histogram(gates[mask], bins=bins, density=True)
        ax.stairs(np.log10(1 + density), bins, color=col, lw=1.1, label=label)
        densities[label] = density
    ax.set_xlabel('Gate value')
    ax.set_ylabel('log10(1 + edge density)')
    ax.legend(loc='upper right', frameon=False, fontsize=LEGEND, borderaxespad=.4, labelspacing=.6)
    return save(cv, 'figure-S14', {'anchor': n, 'seed': 1, 'bins': bins, 'histogram_densities': densities})


# ============================================================ map grids (S9, S10, S5, S6)
ARMS_MAP = ['Reference', 'PCA50', 'NeighborMean', 'Tessera']
MAP_TITLES = {'Reference': 'Reference', 'PCA50': 'PCA50', 'NeighborMean': SGM, 'Tessera': TST}


def map_row(cv, y, name, classes, mw, gap, x0, size=2.2):
    d, m = input_data(name)
    ix = d['field'] == m['display_field']
    pts = d['coords'][ix]
    xlim, ylim = limits(pts)
    badges = {}
    for j, arm in enumerate(ARMS_MAP):
        yv = display_reference(d['ref']) if arm == 'Reference' else display_match(d['ref'], prediction(name, arm)['labels'])[0]
        x = x0 + j * (mw + gap)
        cv.text(x, y + 9, MAP_TITLES[arm], size=CELL)
        ax, box = cv.map_axes(x, y + 12, mw, 1e6, xlim, ylim)
        scatter_map(ax, pts, labels_colors(yv[ix], m['classes'], classes), size)
        if arm != 'Reference':
            mean = np.mean([fit_record(name, arm, s)['scores']['ari'] for s in SEEDS])
            ari_badge(ax, f'ARI {mean:.2f}')
            badges[arm] = mean
    bottom = box[1] + box[3]
    cv.text(x0, bottom + 8, f"{m['display_field'].split('|')[-1]}; {int(ix.sum()):,} of {m['n_selected']:,} cells", size=NOTE)
    return bottom + 10, box, badges


def cohort_maps(names, figure_name, height_budget=545):
    metas = [input_data(n)[1] for n in names]
    classes = sorted(set(c for m in metas for c in m['classes']))
    rows = len(names)
    gap = 5
    aspect = np.mean([np.ptp(input_data(n)[0]['coords'][input_data(n)[0]['field'] == m['display_field']][:, 1]) /
                      np.ptp(input_data(n)[0]['coords'][input_data(n)[0]['field'] == m['display_field']][:, 0])
                      for n, m in zip(names, metas)])
    width_limit = (W - LANE - 3 * gap) / 4
    height_limit = (height_budget - 44 - rows * 30) / rows / aspect
    mw = min(width_limit, height_limit)
    x0 = LANE + (W - LANE - 4 * mw - 3 * gap) / 2
    cv = Canvas(1000)
    y = 0
    badges = {}
    for i, n in enumerate(names):
        top = y
        bottom, box, badges[n] = map_row(cv, y, n, classes, mw, gap, x0)
        title = SHORT[n] if rows > 1 else 'Open-ST HNSCC'
        cv.row_letter(x0 - LANE, top + 11, string.ascii_lowercase[i], title, (top + 12 + box[1] + box[3]) / 2)
        y = bottom + 6
    visible = set().union(*(visible_classes(input_data(n)[0], m) for n, m in zip(names, metas)))
    legend = class_legend(cv, classes, y + 1, visible=visible)
    cv.resize(y + 1 + legend_height(cv, legend) + 3)
    return save(cv, figure_name, {'ari_badges_three_seed_mean': badges})


def figure_S9():
    return cohort_maps(OBJECTS[:5], 'figure-S9')


def figure_S10():
    return cohort_maps(OBJECTS[5:9], 'figure-S10')


def figure_S5():
    return cohort_maps(['OpenST_HNSCC26'], 'figure-S5')


def figure_S6():
    n = 'MIBI'
    d, m = input_data(n)
    fields = sorted(set(d['field']))
    cv = Canvas(1000)
    gap, mw = 5, 112
    x0 = LANE + (W - LANE - 4 * mw - 3 * gap) / 2
    y = 0
    preds = {a: (display_reference(d['ref']) if a == 'Reference' else display_match(d['ref'], prediction(n, a)['labels'])[0])
             for a in ARMS_MAP}
    for i, f in enumerate(fields):
        ix = d['field'] == f
        pts = d['coords'][ix]
        xlim, ylim = limits(pts)
        top = y
        for j, a in enumerate(ARMS_MAP):
            x = x0 + j * (mw + gap)
            cv.text(x, y + 9, MAP_TITLES[a], size=CELL)
            ax, box = cv.map_axes(x, y + 12, mw, mw, xlim, ylim)
            scatter_map(ax, pts, labels_colors(preds[a][ix], m['classes']), 3)
        bottom = box[1] + box[3]
        cv.row_letter(x0 - LANE, top + 11, string.ascii_lowercase[i], f.split('|')[-1].replace('point', 'P'),
                      (top + 12 + bottom) / 2)
        y = bottom + 8
    legend = class_legend(cv, m['classes'], y, visible=visible_classes(d, m))
    cv.resize(y + legend_height(cv, legend) + 3)
    return save(cv, 'figure-S6', {'fields': fields})


# ============================================================ S11 (old 7)
def figure_S11():
    """Breast Visium coordinate maps (old Figure 7)."""
    d, m = input_data('Breast_Visium')
    scale = m['scalefactors']['tissue_hires_scalef']
    xy = d['coords'] * scale
    center = np.median(xy, axis=0)
    half = np.ptp(xy, axis=0) * .22
    roi = (center[0] - half[0], center[0] + half[0], center[1] - half[1], center[1] + half[1])
    labels = {'Spot locations': None, 'Reference': display_reference(d['ref']),
              'PCA50': display_match(d['ref'], prediction('Breast_Visium', 'PCA50')['labels'])[0],
              TST: display_match(d['ref'], prediction('Breast_Visium', 'Tessera')['labels'])[0]}
    arm_key = {'PCA50': 'PCA50', TST: 'Tessera'}
    cv = Canvas(1000)
    gap = 5
    mw = (W - LANE - 3 * gap) / 4
    y = 0
    badges = {}
    for zoom, letter, row_title in [(False, 'a', 'Full tissue'), (True, 'b', 'Fixed ROI')]:
        xlim = (roi[0], roi[1]) if zoom else (xy[:, 0].min() - 20, xy[:, 0].max() + 20)
        ylim = (roi[2], roi[3]) if zoom else (xy[:, 1].min() - 20, xy[:, 1].max() + 20)
        top = y
        for j, (title, values) in enumerate(labels.items()):
            x = LANE + j * (mw + gap)
            cv.text(x, y + 9, title, size=CELL)
            ax, box = cv.map_axes(x, y + 12, mw, 1e6, xlim, ylim)
            ax.set_xlim(*xlim)
            ax.set_ylim(ylim[1], ylim[0])
            if values is None:
                ax.scatter(xy[:, 0], xy[:, 1], s=9 if zoom else 2.6, c='#aaaaaa', edgecolors='white', linewidths=.08,
                           rasterized=True)
            else:
                ax.scatter(xy[:, 0], xy[:, 1], s=9 if zoom else 2.6, c=labels_colors(values, m['classes']), alpha=.78,
                           edgecolors='white', linewidths=.08, rasterized=True)
            if not zoom and j == 0:
                ax.add_patch(Rectangle((roi[0], roi[2]), roi[1] - roi[0], roi[3] - roi[2], fill=False,
                                       edgecolor=S.ORANGE, lw=.8))
            if j == 0:
                length_um = 500 if zoom else 2000
                px = length_um / m['um_per_fullres_pixel'] * scale
                xx = xlim[0] + .06 * (xlim[1] - xlim[0])
                yy = ylim[1] - .06 * (ylim[1] - ylim[0])
                ax.plot([xx, xx + px], [yy, yy], color='black', lw=1.8, solid_capstyle='butt')
                ax.text(xx, yy - .03 * (ylim[1] - ylim[0]), f'{length_um / 1000:g} mm', fontsize=NOTE, color='black',
                        va='bottom', bbox=dict(facecolor='white', alpha=.8, edgecolor='none', pad=.8))
            if title in arm_key:
                value = np.mean([fit_record('Breast_Visium', arm_key[title], s)['scores']['ari'] for s in SEEDS])
                ari_badge(ax, f'ARI {value:.2f}')
                badges[arm_key[title]] = value
        bottom = box[1] + box[3]
        cv.row_letter(0, top + 11, letter, row_title, (top + 12 + bottom) / 2)
        y = bottom + 8
    legend = class_legend(cv, m['classes'], y, visible=visible_classes(d, m))
    cv.resize(y + legend_height(cv, legend) + 3)
    return save(cv, 'figure-S11', {'roi_hires_pixels': list(roi), 'ari_badges_three_seed_mean': badges})


# ============================================================ S12 (old 8)
def figure_S12():
    """Breast mixed spots (old Figure 8)."""
    d, m = input_data('Breast_Visium')
    comp, pur = d['composition'], d['purity']
    cols = m['truth_columns']
    tumor = [i for i, c in enumerate(cols) if c in ['Invasive_Tumor', 'Prolif_Invasive_Tumor', 'DCIS_1', 'DCIS_2']]
    fraction = comp[:, tumor].sum(1)
    entropy = -(comp * np.log(comp + 1e-12)).sum(1) / np.log(comp.shape[1])
    full = prediction('Breast_Visium', 'Tessera')
    cv = Canvas(372)
    counts = [int((pur >= v).sum()) for v in [0, .5, .75]]
    a = cv.panel(0, 0, 248, 170, 'a', 'Agreement by spot purity', left=Y_AXIS, bottom=X_AXIS + 10, right=10)
    strata = {}
    for arm in ['PCA50', 'NeighborMean', 'Tessera']:
        vals, sds = [], []
        for th in [0, .5, .75]:
            sc = [fit_record('Breast_Visium', arm, s)['scores']['purity_strata'][str(th)]['ari'] for s in SEEDS]
            vals.append(np.mean(sc))
            sds.append(np.std(sc, ddof=1))
        a.errorbar([0, 1, 2], vals, yerr=sds, fmt='o-', color=COLOR[arm], ms=3.5, capsize=2, lw=1, label=S.ARM_NAME[arm])
        strata[arm] = {'mean': vals, 'sd': sds}
    a.set_xticks([0, 1, 2], [f'All\n{counts[0]:,}', f'≥0.50\n{counts[1]:,}', f'≥0.75\n{counts[2]:,}'])
    a.set_xlim(-.3, 2.3)
    a.set_ylabel('Raw ARI')
    a.set_xlabel('Dominant-category proportion; spots')
    a.legend(frameon=False, fontsize=NOTE, loc='upper left', borderaxespad=0, handletextpad=.3)
    b = cv.panel(252, 0, 244.8, 170, 'b', 'Mixed-spot composition', left=Y_AXIS + 4, bottom=X_AXIS + 10, right=8)
    b.hist(pur, bins=np.linspace(0, 1, 21), color=S.GREEN, edgecolor='white', linewidth=.5)
    b.set_xlabel('Dominant-category proportion')
    b.set_ylabel('Measured spots')
    row, h = 180, 184
    xlim, ylim = limits(d['coords'])
    ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    mh = h - HEADER - 6
    mw = mh / ratio
    bar_x = W - 34
    map_x = bar_x - 6 - mw
    cv.header(map_x, row, 'd', 'Tumor and DCIS fraction')
    md, box = cv.map_axes(map_x, row + HEADER, mw, mh, xlim, ylim)
    scatter_map(md, d['coords'], plt.cm.magma(fraction), 2.5)
    colorbar(cv, plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap='magma'), bar_x, row + HEADER, 5, mh, label='Fraction',
             ticks=[0, .5, 1])
    c = cv.panel(0, row, map_x - 16, h, 'c', 'Seam score and spot mixing', left=Y_AXIS, bottom=X_AXIS, right=4)
    c.scatter(entropy, full['boundary'], s=2.5, color=COLOR['PCA50'], alpha=.35, linewidths=0, rasterized=True)
    rho = spearmanr(entropy, full['boundary']).statistic
    c.set_xlabel('Source composition entropy')
    c.set_ylabel('Gate-derived seam score')
    c.text(.02, .97, f'Spearman ρ = {rho:.2f}'.replace('-', MINUS), transform=c.transAxes, va='top', fontsize=NOTE)
    return save(cv, 'figure-S12', {'purity_strata_ari': strata, 'purity_counts': counts, 'spearman_rho': rho})


# ============================================================ S13 (old 10)
def figure_S13():
    """Optimization stability and target structure (old Figure 10)."""
    st = S.read_csv(LED / 'seed_stability.csv')
    df = S.read_csv(LED / 'all_primary_scores.csv')
    cv = Canvas(276)
    names = [SHORT[n] for n in OBJECTS]
    left = cv.text_width(names, TICK) + 8
    a = cv.panel(0, 0, 248, 270, 'a', 'Partition stability across seeds', left=left, bottom=X_AXIS, right=10)
    y = object_rows(a, OBJECTS)
    stability = {}
    for j, arm in enumerate(['PCA50', 'SelfOnly', 'Tessera']):
        v = st[st.arm == arm].set_index('object').loc[OBJECTS, 'seed_pair_ari_mean']
        a.scatter(v, y + (j - 1) * .2, s=12, color=COLOR[arm], label=S.ARM_NAME[arm], linewidths=0)
        stability[arm] = v.tolist()
    a.set_xlim(0, 1.02)
    a.set_xlabel('Mean ARI between seed partitions')
    a.legend(frameon=False, fontsize=NOTE, loc='lower left', borderaxespad=0, handletextpad=.2)
    hom = df[df.arm == 'Tessera'].groupby('object').reference_edge_homogeneity.mean().reindex(OBJECTS)
    b = cv.panel(252, 0, 244.8, 270, 'b', 'Target smoothness and contrast', left=Y_AXIS + 8, bottom=X_AXIS, right=10)
    markers = {'NSCLC': 'o', 'BCC': 's', 'Other': '^'}
    shades = {'NSCLC': INK, 'BCC': '#6e6e6e', 'Other': '#a9a9a9'}

    def cohort(n):
        return 'NSCLC' if n in LUNG else 'BCC' if n in BCC else 'Other'
    contrast = {n: ari(n, 'Tessera') - ari(n, 'PCA50') for n in OBJECTS}
    for n in OBJECTS:
        b.scatter(hom[n], contrast[n], s=22, marker=markers[cohort(n)], color=shades[cohort(n)], linewidths=0)
    l6 = OBJECTS[1]
    b.annotate('Lung6', (hom[l6], contrast[l6]), xytext=(-28, -10), textcoords='offset points', fontsize=NOTE,
               arrowprops=dict(arrowstyle='-', lw=.5, color=INK))
    b.axhline(0, color=INK, lw=.6)
    b.set_xlabel('Reference edge homogeneity')
    b.set_ylabel(f'{TST} {MINUS} PCA50 (ARI)')
    b.legend(handles=[Line2D([0], [0], marker=markers[k], ls='', color=c, markeredgewidth=0, markersize=5, label=k)
                      for k, c in shades.items()], frameon=False, fontsize=NOTE, loc='upper left', borderaxespad=0,
             handletextpad=.2)
    return save(cv, 'figure-S13', {'seed_pair_ari_mean': stability, 'reference_edge_homogeneity': hom.tolist(),
                                    'tessera_minus_pca50': [contrast[n] for n in OBJECTS]})


# ============================================================ S1 (old S1)
def figure_S1():
    """Scope of the subsampled inputs (old Figure S1)."""
    meta = [input_data(n)[1] for n in OBJECTS]
    cv = Canvas(270)
    names = [SHORT[n] for n in OBJECTS]
    left = cv.text_width(names, TICK) + 8
    block = 244
    yy = np.arange(len(meta))
    source = np.array([m['n_source'] for m in meta])
    sel = np.array([m['n_selected'] for m in meta])
    a = cv.panel(0, 0, 206, block, 'a', 'Analysis scope', left=left, bottom=X_AXIS, right=10)
    a.barh(yy, source, color='#d9dfe3', height=.7, label='Source observations')
    a.barh(yy, sel, color=S.BLUE, height=.43, label='Modeled observations')
    a.set_xscale('log')
    plain_log_ticks(a.xaxis, [10_000, 100_000])
    a.set_yticks(yy, names)
    a.invert_yaxis()
    a.set_xlabel('Observations (log scale)')
    a.tick_params(axis='y', length=0)
    specs = [('b', 'Frame IDs', S.GREEN, [m['n_fields'] for m in meta], 'Retained frames', [1, 3, 5], None),
             ('c', 'Classes', S.SAND, [m['n_classes'] for m in meta], 'Observed classes', None, None),
             ('d', 'Shared input', S.PURPLE, [m['pca_explained_variance'] for m in meta], 'Retained PCA variance', [0, .5, 1], (0, 1))]
    x = 214
    for (letter, title, color, values, xlabel, ticks, xlim), w in zip(specs, [90, 90, 96.8]):
        ax = cv.panel(x, 0, w, block, letter, title, left=6, bottom=X_AXIS, right=10)
        ax.barh(yy, values, color=color, height=.6)
        ax.set_yticks(yy, [])
        ax.tick_params(axis='y', length=0)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        if ticks:
            ax.set_xticks(ticks)
        if xlim:
            ax.set_xlim(*xlim)
        x += w
    handles, _ = a.get_legend_handles_labels()
    box = a.get_position()
    cv.legend(handles, (box.x0 + box.x1) / 2 * W, block + 6, ncol=2, size=LEGEND, columnspacing=.9)
    cv.resize(block + 20)
    return save(cv, 'figure-S1', {'n_source': source, 'n_selected': sel, 'n_fields': [m['n_fields'] for m in meta],
                                   'n_classes': [m['n_classes'] for m in meta],
                                   'pca_explained_variance': [m['pca_explained_variance'] for m in meta]})


# ============================================================ S3 (old S3)
def scale_bar(ax, xlim, ylim, length, label):
    xx = xlim[0] + .04 * (xlim[1] - xlim[0])
    yy = ylim[1] - .05 * (ylim[1] - ylim[0])
    ax.plot([xx, xx + length], [yy, yy], color='black', lw=1.5, solid_capstyle='butt')
    ax.text(xx, yy - .025 * (ylim[1] - ylim[0]), label, fontsize=NOTE, va='bottom',
            bbox=dict(facecolor='white', alpha=.8, edgecolor='none', pad=.8))


def map_matrix_with_bars(cv, specs, bar_label, bar_space=38, gap=8, y0=0):
    """Rows of three equal maps; a colorbar row reserves a bar slot beside every map."""
    mw = (W - LANE - 3 * bar_space - 2 * gap) / 3
    y = y0
    for letter, row_title, cells, with_bars in specs:
        top, bottom = y, y
        for j, (title, xy, draw) in enumerate(cells):
            x = LANE + j * (mw + bar_space + gap)
            cv.text(x, y + 9, title, size=CELL)
            xlim, ylim = limits(xy)
            ax, box = cv.map_axes(x, y + 12, mw, 1e6, xlim, ylim)
            mappable = draw(ax, xlim, ylim)
            if with_bars and mappable is not None:
                colorbar(cv, mappable, box[0] + box[2] + 4, box[1], 5, box[3], label=bar_label, size=NOTE)
            bottom = max(bottom, box[1] + box[3])
        cv.row_letter(0, top + 11, letter, row_title, (top + 12 + bottom) / 2)
        y = bottom + 8
    return y


def figure_S3():
    """Colorectal TVA section 232 (old Figure S3)."""
    n = 'Colorectal_TVA232'
    d, m = input_data(n)
    ix = d['field'] == m['display_field']
    xy = d['coords'][ix]
    coarse = d['secondary_name']
    coarse_classes = sorted(set(coarse[~np.isin(coarse, ['nan', 'unknown'])]))
    coarse_y = np.array([coarse_classes.index(c) if c in coarse_classes else -1 for c in coarse])
    lv2map = {i: pd.Series(coarse[d['ref'] == i]).mode().iloc[0] for i in range(m['n_classes'])}

    def assignment(arm):
        if arm == 'Reference':
            y = coarse_y
        else:
            matched = display_match(d['ref'], prediction(n, arm)['labels'])[0]
            y = np.array([coarse_classes.index(lv2map[v]) if v >= 0 and lv2map[v] in coarse_classes else -1 for v in matched])

        def draw(ax, xlim, ylim):
            scatter_map(ax, xy, labels_colors(y[ix], coarse_classes), 2.5)
            scale_bar(ax, xlim, ylim, 100, '100 µm')
        return draw

    def channel(col):
        values = np.log1p(np.maximum(d['protein'][ix, m['protein_columns'].index(col)], 0))
        norm = Normalize(*np.quantile(values, [.02, .98]))

        def draw(ax, xlim, ylim):
            scatter_map(ax, xy, plt.cm.magma(norm(values)), 2.5)
            scale_bar(ax, xlim, ylim, 100, '100 µm')
            return plt.cm.ScalarMappable(norm=norm, cmap='magma')
        return draw
    names = {'Mean.PanCK': 'PanCK', 'Mean.CD45': 'CD45', 'Mean.DAPI': 'DAPI (nuclear stain)'}
    titles = {'Reference': 'Reference', 'PCA50': 'PCA50', 'Tessera': TST}
    cv = Canvas(1000)
    y = map_matrix_with_bars(cv, [('a', 'Assignments', [(titles[arm] + ' compartments', xy, assignment(arm))
                                                        for arm in ['Reference', 'PCA50', 'Tessera']], False),
                                  ('b', 'Measured channels', [(names[col], xy, channel(col)) for col in names], True)],
                             'log1p intensity')
    legend = class_legend(cv, coarse_classes, y, visible=set(coarse_classes))
    cv.resize(y + legend_height(cv, legend) + 3)
    return save(cv, 'figure-S3', {'display_field': m['display_field'], 'n_cells': int(ix.sum())})


# ============================================================ S4 (old S4)
def figure_S4():
    """Execution and training records (old Figure S4)."""
    df = S.read_csv(LED / 'all_primary_scores.csv')
    cv = Canvas(300)
    n_obs = np.array([input_data(o)[1]['n_selected'] for o in OBJECTS])
    width = (W - LANE - 14) / 2
    axes = []
    for r, (letter, row_title, titles) in enumerate([('a', 'Cost + memory', ['Recorded run time', 'Peak GPU allocation']),
                                                     ('b', 'Training', ['Reconstruction loss', 'Gate-weighted contrastive loss'])]):
        y = r * 148
        row = []
        for j, title in enumerate(titles):
            x = LANE + j * (width + 14)
            cv.text(x, y + 10, title, size=CELL)
            row.append(cv.axes(x + Y_AXIS + 4, y + 16, width - Y_AXIS - 10, 148 - 16 - X_AXIS - 4))
        cv.row_letter(0, y + 11, letter, row_title, y + 16 + (148 - 16 - X_AXIS) / 2)
        axes.append(row)
    record = {}
    for arm in ['PCA50', 'SelfOnly', 'Tessera']:
        g = df[df.arm == arm].groupby('object').agg(t=('total_seconds', 'mean'), mem=('gpu_GiB', 'max')).reindex(OBJECTS)
        axes[0][0].scatter(n_obs, g.t, s=12, label=S.ARM_NAME[arm], color=COLOR[arm], linewidths=0)
        if arm != 'PCA50':
            axes[0][1].scatter(n_obs, g.mem, s=12, label=S.ARM_NAME[arm], color=COLOR[arm], linewidths=0)
        record[arm] = {'seconds': g.t.tolist(), 'GiB': g.mem.tolist()}
    ax = axes[0][0]
    ax.set_yscale('log')
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 4)
    plain_log_ticks(ax.yaxis, [.1, 1])
    ax.set_xlabel('Modeled observations')
    ax.set_ylabel('Seconds')
    ax.legend(frameon=False, fontsize=NOTE, loc='upper left', ncol=3, columnspacing=.8, handletextpad=.15, borderaxespad=0)
    ax = axes[0][1]
    ax.set_xlabel('Modeled observations')
    ax.set_ylabel('GiB')
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 1.25)
    ax.legend(frameon=False, fontsize=NOTE, loc='upper left', ncol=2, borderaxespad=0, handletextpad=.15)
    histories = {}
    for arm in ['Tessera', 'NoContrast', 'SelfOnly']:
        h = fit_record(OBJECTS[0], arm, 1)['history']
        axes[1][0].plot([r['epoch'] for r in h], [r['recon'] for r in h], color=COLOR[arm], label=S.ARM_NAME[arm], lw=1.1)
        histories['recon_' + arm] = [r['recon'] for r in h]
    axes[1][0].set_xlabel('Training epoch')
    axes[1][0].set_ylabel('MSE')
    axes[1][0].legend(frameon=False, fontsize=NOTE, borderaxespad=0)
    for arm in ['Tessera', 'NoGate', 'NoScale']:
        h = fit_record(OBJECTS[0], arm, 1)['history']
        axes[1][1].plot([r['epoch'] for r in h], [r['contrastive'] for r in h], color=COLOR[arm], label=S.ARM_NAME[arm], lw=1.1)
        histories['contrastive_' + arm] = [r['contrastive'] for r in h]
    axes[1][1].set_xlabel('Training epoch')
    axes[1][1].set_ylabel('Loss')
    axes[1][1].legend(frameon=False, fontsize=NOTE, borderaxespad=0)
    return save(cv, 'figure-S4', {'cost': record, 'histories': histories})


# ============================================================ S7 (old S7)
def figure_S7():
    """Measured RNA markers in the display fields (old Figure S7)."""
    def marker(n, gene):
        d, m = input_data(n)
        ix = d['field'] == m['display_field']
        values = d['marker_log'][ix, list(d['marker_gene']).index(gene)]
        norm = Normalize(*np.quantile(values, [0, .98]))
        xy = d['coords'][ix]

        def draw(ax, xlim, ylim):
            order = np.argsort(values)
            ax.scatter(xy[order, 0], xy[order, 1], s=2.5, c=plt.cm.viridis(norm(values[order])), linewidths=0,
                       rasterized=True)
            return plt.cm.ScalarMappable(norm=norm, cmap='viridis')
        return gene, xy, draw
    cv = Canvas(1000)
    y = map_matrix_with_bars(cv, [('a', 'Lung', [marker(OBJECTS[0], g) for g in ['EPCAM', 'COL1A1', 'PTPRC']], True),
                                  ('b', 'BCC', [marker('BCC_PatientA', g) for g in ['KRT14', 'COL1A1', 'PTPRC']], True)],
                             'log expression', bar_space=36)
    cv.resize(y + 2)
    return save(cv, 'figure-S7', {'genes': {'Lung': ['EPCAM', 'COL1A1', 'PTPRC'], 'BCC': ['KRT14', 'COL1A1', 'PTPRC']}})


# ============================================================ S8 (old S8)
TUMOR, OTHER, UNASSIGNED = '#aa4c53', '#78a8ba', '#c8ccd0'


def figure_S8():
    """Tumor-boundary inspection in four breast patients (old Figure S8)."""
    hard = hardening()
    parts = S.read_csv(EXTR / 'hardening/partitions_boundary_T1.csv')
    per = hard['targets']['boundary_T1']['per_unit']
    order = sorted(per, key=lambda u: per[u]['UniformSameGraph'])
    chosen = [order[-1], order[-2], order[1], order[0]]
    columns = [('reference', 'Pathologist', None), ('pca50', 'PCA50', 'PCA50'),
               ('uniform_same_graph', SGM, 'UniformSameGraph'), ('control', 'Tuned control', 'NestedControl'),
               ('tessera', TST, 'Tessera')]
    gap = 5.
    mw = (W - LANE - 4 * gap) / 5
    cv = Canvas(1000)
    y = 0.
    badges = {}
    for i, unit in enumerate(chosen):
        frame = parts[parts.unit == unit]
        xy = frame[['x', 'y']].to_numpy(float)
        xlim, ylim = limits(xy)
        ref = frame.reference.fillna('').astype(str).to_numpy()
        labels = np.where(ref == 'tumor', 1, np.where(ref == 'other tissue', 0, -1))
        top = y
        bottom = top
        for j, (column, title, key) in enumerate(columns):
            x = LANE + j * (mw + gap)
            cv.text(x, y + 9, title, size=CELL)
            ax, box = cv.map_axes(x, y + 12, mw, 1e6, xlim, ylim)
            shown = labels if column == 'reference' else display_match(labels, frame[column].to_numpy())[0]
            colors = [UNASSIGNED if v < 0 else TUMOR if v == 1 else OTHER for v in shown]
            scatter_map(ax, xy, colors, size=7)
            if key:
                ari_badge(ax, f'{per[unit][key]:.2f}'.replace('-', MINUS))
                badges[f'{unit}:{key}'] = per[unit][key]
            bottom = box[1] + box[3]
        cv.row_letter(LANE - 20, top + 11, 'abcd'[i], f'Patient {unit[0]}', (top + 12 + bottom) / 2)
        y = bottom + 8
    handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=TUMOR, markeredgewidth=0, markersize=4.5, label='Tumor'),
               Line2D([0], [0], marker='o', color='none', markerfacecolor=OTHER, markeredgewidth=0, markersize=4.5,
                      label='Other tissue'),
               Line2D([0], [0], marker='o', color='none', markerfacecolor=UNASSIGNED, markeredgewidth=0, markersize=4.5,
                      label='Unassigned or unmatched')]
    legend = cv.fitted_legend(handles, W / 2, y + 1, W - 8, size=LEGEND)
    cv.resize(y + 1 + legend_height(cv, legend) + 3)
    return save(cv, 'figure-S8', {'chosen_units': chosen, 'ari_badges': badges})


STATISTICAL = {'figure-03': figure_03, 'figure-04': figure_04, 'figure-07': figure_07, 'figure-08': figure_08,
               'figure-09': figure_09, 'figure-S1': figure_S1, 'figure-S3': figure_S3, 'figure-S4': figure_S4,
               'figure-S5': figure_S5, 'figure-S6': figure_S6, 'figure-S7': figure_S7, 'figure-S8': figure_S8,
               'figure-S9': figure_S9, 'figure-S10': figure_S10, 'figure-S11': figure_S11, 'figure-S12': figure_S12,
               'figure-S13': figure_S13, 'figure-S14': figure_S14, 'figure-S15': figure_S15, 'figure-S16': figure_S16}

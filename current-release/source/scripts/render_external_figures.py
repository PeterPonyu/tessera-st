#!/usr/bin/env python3
"""Figures for the independent pathologist cohort and the tissue landscape.

Reads saved score tables only; no model is fitted. Run after render_figures.py
and make_external_tables.py. Geometry follows figure_layout.py: every figure is a
496.8-pt canvas placed at 100%, with every object positioned in points.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import render_figures as rf
from common import ROOT, read, dump, sha

plt = rf.plt
Canvas, save = rf.Canvas, rf.save
INK, LIGHT, GRAY, W, LANE = rf.INK, rf.LIGHT, rf.GRAY, rf.W, rf.LANE
X_AXIS, Y_AXIS, TICK = rf.X_AXIS, rf.Y_AXIS, rf.TICK
MC = rf.METHOD_COLORS
EXTR = ROOT / 'validation-extension/results'
LED = rf.LED
TUMOR, OTHER, UNASSIGNED = '#aa4c53', '#78a8ba', '#c8ccd0'
GROUPS = ['Computed regions', 'Cell-type labels', 'Pathologist regions', 'Anatomical regions']
MINUS = '$-$'


def hardening():
    return read(EXTR / 'run-hardening.json')['result']


def patient_rows(ax, units, extra_gap=False):
    y = np.arange(len(units), dtype=float)
    ax.set_yticks(y, [u[0] for u in units])
    ax.set_ylim(len(units) - .4 + (1.9 if extra_gap else 0), -.6)
    ax.tick_params(axis='y', length=0)
    return y


def arm_dots(ax, hard, target, xlim, ticks):
    units = hard['targets']['all']['units']
    per = hard['targets'][target]['per_unit']
    y = patient_rows(ax, units)
    cols = [('PCA50', 'PCA50', 'o', 20), ('UniformSameGraph', 'NeighborMean', 's', 18),
            ('NestedControl', 'Tuned', 'D', 20), ('Tessera', 'Tessera', 'o', 28)]
    for yy, u in zip(y, units):
        values = [per[u][c] for c, *_ in cols]
        ax.plot([min(values), max(values)], [yy, yy], color=LIGHT, lw=1.2, zorder=1, solid_capstyle='round')
    for c, color, marker, size in cols:
        ax.scatter([per[u][c] for u in units], y, s=size, marker=marker, color=MC[color], zorder=3, linewidths=0)
    ax.axvline(0, color=INK, lw=.5)
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.set_xlabel('Adjusted Rand index')


def forest(ax, hard):
    targets = [('All', 'all'), ('Comp.', 'compartments'), ('Tumor', 'boundary_T1')]
    spec = [('Tessera_minus_UniformSameGraph', f'Tessera {MINUS} mean', 'Tessera'),
            ('Tessera_minus_NestedControl', f'Tessera {MINUS} tuned', 'Tessera'),
            ('UniformSameGraph_minus_NestedControl', f'Mean {MINUS} tuned', 'NeighborMean')]
    ys, labels, row = [], [], 0.
    for short, target in targets:
        contrasts = hard['targets'][target]['contrasts']
        for key, label, color in spec:
            c = contrasts[key]
            deltas = np.array(c['deltas'])
            ax.scatter(deltas, np.full(len(deltas), row), s=7, color='#9aa3a9', linewidths=0, zorder=2)
            ax.plot(c['bootstrap95'], [row, row], color=MC[color], lw=1.6, zorder=3, solid_capstyle='butt')
            ax.scatter([c['mean']], [row], s=34, marker='D', color=MC[color], zorder=4, linewidths=0)
            ax.text(1.02, row, f"+{c['n_positive']}/{MINUS}{c['n_negative']}", transform=ax.get_yaxis_transform(),
                    fontsize=7, va='center', ha='left')
            ys.append(row)
            labels.append(f'{short}: {label}')
            row += 1
        row += .55
    ax.set_yticks(ys, labels)
    ax.set_ylim(row - .6, -.7)
    ax.tick_params(axis='y', length=0, labelsize=7.5)
    ax.axvline(0, color=INK, lw=.6)
    ax.set_xlim(-.26, .2)
    ax.set_xticks([-.2, -.1, 0, .1, .2])
    ax.set_xlabel('ARI difference (dots: patients; bar: 95% bootstrap)')


def seam_dots(ax, hard):
    units = hard['targets']['all']['units']
    seam = hard['seams']['boundary_T1']
    cols = [('gate', 'Tessera', 'o', 26), ('raw_difference', 'PCA50', 's', 20), ('smoothed_expression_weighted', 'NeighborMean', 'D', 20)]
    y = patient_rows(ax, units, extra_gap=True)
    for yy, u in zip(y, units):
        values = [seam['per_unit'][u][c] for c, *_ in cols]
        ax.plot([min(values), max(values)], [yy, yy], color=LIGHT, lw=1.2, zorder=1, solid_capstyle='round')
    for c, color, marker, size in cols:
        ax.scatter([seam['per_unit'][u][c] for u in units], y, s=size, marker=marker, color=MC[color], zorder=3, linewidths=0)
    mean_row = len(units) + .6
    for j, (c, color, marker, size) in enumerate(cols):
        lo, hi = seam['mean'][c]['bootstrap95']
        yy = mean_row + (j - 1) * .34
        ax.plot([lo, hi], [yy, yy], color=MC[color], lw=1.6, zorder=3, solid_capstyle='butt')
        ax.scatter([seam['mean'][c]['mean']], [yy], s=size + 14, marker=marker, color=MC[color], zorder=4, linewidths=0)
    ax.set_yticks(list(y) + [mean_row], [u[0] for u in units] + ['Mean'])
    ax.axvline(.5, color=GRAY, lw=.8, ls='--', zorder=0)
    ax.set_xlim(.3, .92)
    ax.set_xticks([.3, .4, .5, .6, .7, .8])
    ax.set_xlabel('Seam AUROC on tumor-boundary edges')


def replicate_dots(ax, hard):
    rep = pd.read_csv(LED / 'replicates.csv')
    rep = rep[rep.target == 'all'].set_index('unit').loc[hard['targets']['all']['units']]
    y = patient_rows(ax, list(rep.index))
    for yy, (_, r) in zip(y, rep.iterrows()):
        runs = [r.run1, r.run2, r.run3]
        ax.plot([min(runs), max(runs)], [yy, yy], color=MC['Tessera'], lw=1.3, zorder=2, solid_capstyle='round', alpha=.7)
    for column in ('run1', 'run2', 'run3'):
        ax.scatter(rep[column], y, s=15, facecolors='white', edgecolors=MC['Tessera'], linewidths=1., zorder=3)
    ax.scatter(rep.same_graph_mean, y, s=20, marker='s', color=MC['NeighborMean'], zorder=4, linewidths=0)
    ax.scatter(rep.nested_control, y, s=20, marker='D', color=MC['Tuned'], zorder=4, linewidths=0)
    ax.set_xlim(0, .46)
    ax.set_xticks([0, .1, .2, .3])
    ax.set_xlabel('ARI, all labels')


def weighting_dots(ax, hard):
    units = hard['targets']['all']['units']
    per = hard['targets']['boundary_T1']['per_unit']
    y = patient_rows(ax, units)
    spec = [('UniformSameGraph', 's', MC['NeighborMean'], 'full'), ('WeightShuffled', 's', '#8c8c8c', 'none'),
            ('ShuffledCoords', '^', INK, 'full')]
    for column, marker, color, fill in spec:
        diff = [per[u]['ExpressionWeighted'] - per[u][column] for u in units]
        ax.scatter(diff, y, s=22, marker=marker, facecolors=color if fill == 'full' else 'white', edgecolors=color,
                   linewidths=1., zorder=3)
    ax.axvline(0, color=INK, lw=.6)
    ax.set_xlim(-.12, .6)
    ax.set_xticks([0, .2, .4, .6])
    ax.set_xlabel('ARI of weighted mean minus comparator')


def external_scores():
    hard = hardening()
    cv = Canvas(plt, 552)
    a = cv.panel(0, 0, 246, 172, 'a', 'All labels: mean leads', left=Y_AXIS - 22, bottom=X_AXIS, right=8)
    arm_dots(a, hard, 'all', (-.01, .38), [0, .1, .2, .3])
    b = cv.panel(250.8, 0, 246, 172, 'b', 'Boundary: mean beats tuned control', left=Y_AXIS - 22, bottom=X_AXIS, right=8)
    arm_dots(b, hard, 'boundary_T1', (-.04, .8), [0, .2, .4, .6, .8])
    c = cv.panel(0, 178, 250, 172, 'c', 'Paired contrasts', left=96, bottom=X_AXIS, right=34)
    forest(c, hard)
    d = cv.panel(254, 178, 242.8, 172, 'd', 'Gate at chance on seams', left=Y_AXIS - 12, bottom=X_AXIS, right=8)
    seam_dots(d, hard)
    e = cv.panel(0, 356, 246, 172, 'e', 'Refits move Tessera only', left=Y_AXIS - 22, bottom=X_AXIS, right=8)
    replicate_dots(e, hard)
    f = cv.panel(250.8, 356, 246, 172, 'f', 'Geometry and weighting', left=Y_AXIS - 22, bottom=X_AXIS, right=8)
    weighting_dots(f, hard)
    methods = [Line2D([0], [0], marker='o', ls='', color=MC['PCA50'], markeredgewidth=0, markersize=5, label='PCA50'),
               Line2D([0], [0], marker='s', ls='', color=MC['NeighborMean'], markeredgewidth=0, markersize=5, label='Same-graph mean'),
               Line2D([0], [0], marker='D', ls='', color=MC['Tuned'], markeredgewidth=0, markersize=4.5, label='Tuned control'),
               Line2D([0], [0], marker='o', ls='', color=MC['Tessera'], markeredgewidth=0, markersize=5.5, label='Tessera')]
    cv.fitted_legend(methods, W / 2, 535, W - 8, size=7.5)
    seam_key = [Line2D([0], [0], marker='o', ls='', color=MC['Tessera'], markeredgewidth=0, markersize=5, label='Learned gate'),
                Line2D([0], [0], marker='s', ls='', color=MC['PCA50'], markeredgewidth=0, markersize=5, label='Feature difference'),
                Line2D([0], [0], marker='D', ls='', color=MC['NeighborMean'], markeredgewidth=0, markersize=4.5, label='Smoothed difference')]
    d.legend(handles=seam_key, frameon=False, fontsize=7, loc='upper right', borderaxespad=0, handletextpad=.2,
             bbox_to_anchor=(1.0, 1.0))
    replicate_key = [Line2D([0], [0], marker='o', ls='', mfc='white', mec=MC['Tessera'], markeredgewidth=1., markersize=4.5, label='Tessera refit'),
                     Line2D([0], [0], marker='s', ls='', color=MC['NeighborMean'], markeredgewidth=0, markersize=4.5, label='Same-graph mean')]
    e.legend(handles=replicate_key, frameon=False, fontsize=7, loc='upper right', borderaxespad=0, handletextpad=.2,
             bbox_to_anchor=(1.0, 1.0))
    weight_key = [Line2D([0], [0], marker='s', ls='', color=MC['NeighborMean'], markeredgewidth=0, markersize=4.5, label='vs unweighted'),
                  Line2D([0], [0], marker='s', ls='', mfc='white', mec='#8c8c8c', markeredgewidth=1., markersize=4.5, label='vs shuffled weights'),
                  Line2D([0], [0], marker='^', ls='', color=INK, markeredgewidth=0, markersize=4.5, label='vs shuffled coordinates')]
    f.legend(handles=weight_key, frameon=False, fontsize=7, loc='center right', borderaxespad=0, handletextpad=.2,
             bbox_to_anchor=(1.0, .5))
    save(cv, 'figure15-external-cohort')


def external_maps():
    hard = hardening()
    parts = pd.read_csv(EXTR / 'hardening/partitions_boundary_T1.csv')
    per = hard['targets']['boundary_T1']['per_unit']
    order = sorted(per, key=lambda u: per[u]['UniformSameGraph'])
    chosen = [order[-1], order[-2], order[1], order[0]]
    columns = [('reference', 'Pathologist', None), ('pca50', 'PCA50', 'PCA50'), ('uniform_same_graph', 'Same-graph mean', 'UniformSameGraph'),
               ('control', 'Tuned control', 'NestedControl'), ('tessera', 'Tessera', 'Tessera')]
    gap = 5.
    mw = (W - LANE - 4 * gap) / 5
    cv = Canvas(plt, 1000)
    y = 0.
    for i, unit in enumerate(chosen):
        frame = parts[parts.unit == unit]
        xy = frame[['x', 'y']].to_numpy(float)
        xlim, ylim = rf.limits(xy)
        ref = frame.reference.fillna('').astype(str).to_numpy()
        labels = np.where(ref == 'tumor', 1, np.where(ref == 'other tissue', 0, -1))
        top = y
        bottom = top
        for j, (column, title, key) in enumerate(columns):
            x = LANE + j * (mw + gap)
            cv.text(x, y + 9, title, size=rf.CELL_TITLE_SIZE)
            ax, box = cv.map_axes(x, y + 12, mw, 1e6, xlim, ylim)
            if column == 'reference':
                shown = labels
            else:
                shown, _ = rf.display_match(labels, frame[column].to_numpy())
            colors = [UNASSIGNED if v < 0 else TUMOR if v == 1 else OTHER for v in shown]
            rf.scatter_map(ax, xy, colors, size=7)
            if key:
                rf.ari_badge(ax, f'{per[unit][key]:.2f}')
            bottom = box[1] + box[3]
        cv.row_marker(LANE - 22, top + 11, 'abcd'[i], f'Patient {unit[0]}', (top + 12 + bottom) / 2)
        y = bottom + 8
    handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=TUMOR, markeredgewidth=0, markersize=4.5, label='Tumor'),
               Line2D([0], [0], marker='o', color='none', markerfacecolor=OTHER, markeredgewidth=0, markersize=4.5, label='Other tissue'),
               Line2D([0], [0], marker='o', color='none', markerfacecolor=UNASSIGNED, markeredgewidth=0, markersize=4.5,
                      label='Unassigned or unmatched')]
    legend = cv.fitted_legend(handles, W / 2, y + 1, W - 8, size=7.5)
    rf.resize(cv, y + 1 + rf.legend_height(cv, legend) + 3)
    save(cv, 'figure16-external-maps')


def landscape_rows(ax, table, labels=True):
    """Row positions with a header row before every annotation-origin group."""
    ys, names, headers, row = [], [], [], 0.
    for group in GROUPS:
        sub = table[table.group == group]
        headers.append((row, group))
        row += 1
        for _, record in sub.iterrows():
            ys.append(row)
            names.append(record.short_label)
            row += 1
    ax.set_ylim(row - .4, -.6)
    ax.tick_params(axis='y', length=0)
    if labels:
        ax.set_yticks(ys, names)
        ax.tick_params(axis='y', labelsize=7)
    else:
        ax.set_yticks([])
    return np.array(ys), headers


def tissue_landscape():
    table = pd.read_csv(LED / 'landscape.csv')
    table['group'] = pd.Categorical(table.group, GROUPS, ordered=True)
    table = table.sort_values('group', kind='stable').reset_index(drop=True)
    cv = Canvas(plt, 1000)
    a = cv.panel(0, 0, 262, 322, 'a', 'Advantage depends on the label', left=70, bottom=X_AXIS, right=8)
    ys, headers = landscape_rows(a, table)
    a.scatter(table.tessera_minus_same_graph, ys, s=26, marker='s', facecolors='white', edgecolors=MC['NeighborMean'], linewidths=1.1, zorder=3)
    a.axvline(0, color=INK, lw=.6)
    for row, name in headers:
        a.text(.02, row, name, transform=a.get_yaxis_transform(), fontsize=7.5, va='center', ha='left', zorder=5)
    a.set_xlim(-.14, .22)
    a.set_xticks([-.1, 0, .1, .2])
    a.set_xlabel(f'Tessera {MINUS} same-graph mean (ARI)')
    b = cv.panel(266, 0, 230.8, 322, 'b', 'Best control beats Tessera', left=8, bottom=X_AXIS, right=8)
    landscape_rows(b, table, labels=False)
    b.scatter(table.tessera_minus_pca50, ys, s=22, facecolors='white', edgecolors=MC['PCA50'], linewidths=1.1, zorder=3)
    b.scatter(table.tessera_minus_oracle, ys, s=24, marker='D', facecolors='white', edgecolors=MC['Tuned'], linewidths=1.1, zorder=3)
    b.axvline(0, color=INK, lw=.6)
    b.set_xlim(-.3, .4)
    b.set_xticks([-.2, 0, .2, .4])
    b.set_xlabel(f'Tessera {MINUS} comparator (ARI)')
    c = cv.panel(0, 346, 246, 150, 'c', 'Best scale differs by label type', left=70, bottom=X_AXIS, right=8)
    d = cv.panel(250.8, 346, 246, 150, 'd', 'Gate AUROC centers on chance', left=8, bottom=X_AXIS, right=8)
    for ax, key in ((c, 'strip_k'), (d, 'strip_gate')):
        ax.set_yticks(range(len(GROUPS)))
        ax.set_yticklabels(GROUPS if ax is c else [])
        ax.set_ylim(len(GROUPS) - .5, -.5)
        ax.tick_params(axis='y', length=0, labelsize=7)
    for gi, group in enumerate(GROUPS):
        sub = table[table.group == group]
        offsets = (np.arange(len(sub)) % 4 - 1.5) * .12
        c.scatter(sub.best_uniform_k, gi + offsets, s=8 + 190 * sub.best_uniform_ari.clip(lower=0), color=MC['NeighborMean'],
                  alpha=.8, linewidths=0, zorder=3)
        d.scatter(sub.raw_difference_auroc, gi + offsets, s=18, marker='s', facecolors='white', edgecolors=MC['PCA50'], linewidths=1., zorder=3)
        d.scatter(sub.gate_auroc, gi + offsets, s=24, color=MC['Tessera'], linewidths=0, zorder=4)
    c.set_xscale('log')
    c.set_xlim(4.5, 280)
    c.set_xticks([6, 10, 30, 100, 200], ['6', '10', '30', '100', '200'])
    c.minorticks_off()
    c.set_xlabel('Neighbors in the best uniform mean (area: its ARI)')
    d.axvline(.5, color=GRAY, lw=.8, ls='--', zorder=0)
    d.set_xlim(.33, .76)
    d.set_xticks([.4, .5, .6, .7])
    d.set_xlabel('Seam AUROC')
    key = [Line2D([0], [0], marker='o', ls='', color=MC['Tessera'], markeredgewidth=0, markersize=5, label='Learned gate'),
           Line2D([0], [0], marker='s', ls='', mfc='white', mec=MC['PCA50'], markeredgewidth=1., markersize=4.5, label='Feature difference')]
    d.legend(handles=key, frameon=False, fontsize=7, loc='lower left', borderaxespad=0, handletextpad=.2, bbox_to_anchor=(0., .0))
    contrasts = [Line2D([0], [0], marker='s', ls='', mfc='white', mec=MC['NeighborMean'], markeredgewidth=1.1, markersize=4.5,
                        label=f'(a) vs same-graph mean'),
                 Line2D([0], [0], marker='D', ls='', mfc='white', mec=MC['Tuned'], markeredgewidth=1.1, markersize=4.5,
                        label='(b) vs best of 30 controls, chosen with labels'),
                 Line2D([0], [0], marker='o', ls='', mfc='white', mec=MC['PCA50'], markeredgewidth=1.1, markersize=4.5,
                        label='(b) vs PCA50')]
    cv.fitted_legend(contrasts, W / 2, 330, W - 8, size=7.5)
    rf.resize(cv, 496)
    save(cv, 'figure17-tissue-landscape')


def main():
    rf.STYLE_AUDIT.update(read(rf.QA / 'statistical_figure_style_audit.json'))
    external_scores()
    external_maps()
    tissue_landscape()
    manifest = read(LED / 'figure_manifest.json')
    manifest['figures'].update({f'{name}.pdf': sha(rf.FIG / f'{name}.pdf') for name in
                                ('figure15-external-cohort', 'figure16-external-maps', 'figure17-tissue-landscape')})
    dump(LED / 'figure_manifest.json', manifest)


if __name__ == '__main__':
    main()

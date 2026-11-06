"""Revision-1 main Figures 5 (training budget and tuning) and 6 (cluster number).

Inputs are the read-only revision-1 summary ledgers; no value is refitted or rescored.
The only arithmetic is display-level: the ledger differences named in the figure
specification (for example tessera_default - nested_control), patient means for the
diamonds and bold lines, and counts of patients above and below zero. Cohort-mean
lines are drawn only at x positions where every patient of the cohort has a value.
Layout uses r1_style (Arial, explicit point offsets, overlap checks).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

import r1_style as S
from r1_style import COLOR, INK, LEGEND, LIGHT, MINUS, NOTE, TICK, TST, W, X_AXIS, Canvas, save

SUMMARY = Path(os.environ.get('TESSERA_R1_SUMMARY') or S.RESULTS / 'revision-r1/summary')
TARGETS = [('NSCLC', 'primary', 'NSCLC regions'), ('BCC', 'primary', 'BCC cell types'),
           ('HER2ST', 'all', 'HER2ST, all labels'), ('HER2ST', 'boundary_T1', 'HER2ST, tumor boundary')]
PATIENT_LINE = '#a9b1b6'
DOT = '#9aa3a9'
GUIDE = '#8c8c8c'
DASH = (0, (4.0, 2.2))
Y_LABEL = f'{TST} {MINUS} tuned control (ARI)'


def subset(table, cohort, target):
    rows = table[(table.cohort == cohort) & (table.target == target)]
    assert len(rows), (cohort, target)
    return rows


def complete_mean(pivot):
    """Patient mean at each column, kept only where every patient has a value (NaN elsewhere)."""
    mean = pivot.mean(axis=0)
    mean[pivot.isna().any(axis=0)] = np.nan
    return mean


def legend_band(cv, handles, y=1.0):
    return cv.legend(handles, W / 2, y, ncol=len(handles), size=LEGEND, columnspacing=1.6, handlelength=2.2,
                     handletextpad=.4)


def paired_rows(ax, groups, rows, lane_points, xlim, ticks, xlabel):
    """Grouped dot rows: gray patient dots, a mean diamond, faint within-patient connectors, +n/-m counts.

    groups: [(header, {row label: patient values in a fixed patient order})]
    rows:   [(row label, colour, filled)]
    """
    y, ticks_y, labels, headers, record = 0., [], [], [], {}
    for header, values in groups:
        headers.append((y, header))
        y += 1
        positions = {}
        for label, colour, filled in rows:
            v = np.asarray(values[label], float)
            assert np.all(np.isfinite(v)) and not np.any(v == 0), (header, label)
            positions[label] = y
            ax.scatter(v, np.full(len(v), y), s=10, color=DOT, linewidths=0, zorder=3)
            ax.scatter([v.mean()], [y], s=36, marker='D', facecolors=colour if filled else 'white', edgecolors=colour,
                       linewidths=1.1, zorder=4)
            ax.text(1.02, y, f'+{int((v > 0).sum())}/{MINUS}{int((v < 0).sum())}', transform=ax.get_yaxis_transform(),
                    fontsize=NOTE, va='center', ha='left')
            ticks_y.append(y)
            labels.append(label)
            record[f'{header}: {label}'] = {'patients': v, 'mean': float(v.mean()), 'n_positive': int((v > 0).sum()),
                                            'n_negative': int((v < 0).sum())}
            y += 1
        first, second = rows[0][0], rows[1][0]
        for a, b in zip(values[first], values[second]):
            ax.plot([a, b], [positions[first], positions[second]], color=LIGHT, lw=.55, zorder=2)
        y += .35
    ax.set_yticks(ticks_y, labels)
    ax.set_ylim(y - .75, -.7)
    ax.tick_params(axis='y', length=0)
    ax.axvline(0, color=INK, lw=.6, zorder=1)
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.set_xlabel(xlabel)
    width = ax.get_position().width * W
    lane = blended_transform_factory(ax.transAxes, ax.transData)
    for yy, header in headers:
        ax.text(-(lane_points - 1) / width, yy, header, transform=lane, fontsize=TICK, va='center', ha='left')
    return record


# ================================================================== Figure 5
def figure_05():
    budget = S.read_csv(SUMMARY / 'fig_budget.csv')
    e2 = S.read_csv(SUMMARY / 'table_E2.csv')
    epochs = [40, 80, 120, 160, 200, 300, 400]
    cv = Canvas(420)
    legend_band(cv, [Line2D([0], [0], color=PATIENT_LINE, lw=.8, label='Patients'),
                     Line2D([0], [0], color=COLOR['Tessera'], lw=2, marker='o', ms=3, label='Cohort mean')])
    row1, axes_h = 17, 126
    left0 = 40   # tick labels plus one rotated label line
    inner, right, gap = 26, 8, 6
    data_w = (W - left0 - 2 * inner - 3 * right - 2 * gap) / 3
    record = {}
    for i, (cohort, target, title) in enumerate(TARGETS[:3]):
        left = left0 if i == 0 else inner
        x = 0 if i == 0 else left0 + data_w + right + gap + (i - 1) * (inner + data_w + right + gap)
        ax = cv.panel(x, row1, left + data_w + right, S.HEADER + axes_h + X_AXIS, 'abc'[i], title, left=left,
                      bottom=X_AXIS, right=right)
        sub = subset(budget, cohort, target)
        pivot = sub.pivot(index='patient', columns='epoch', values='minus_control').reindex(columns=epochs)
        assert pivot.notna().all().all(), (cohort, target)
        for _, values in pivot.iterrows():
            ax.plot(epochs, values.to_numpy(), color=PATIENT_LINE, lw=.75, zorder=2, solid_capstyle='round')
        mean = complete_mean(pivot)
        ax.plot(epochs, mean.to_numpy(), color=COLOR['Tessera'], lw=2.0, marker='o', ms=3, zorder=4)
        ax.axhline(0, color=INK, lw=.6, zorder=1)
        for epoch in (120, 200):
            ax.axvline(epoch, color=GUIDE, lw=.6, ls=(0, (3, 2)), zorder=1)
        ax.set_xlim(20, 420)
        ax.set_xticks([40, 120, 200, 300, 400])
        ax.set_ylim(-.30, .20)
        ax.set_yticks([-.3, -.2, -.1, 0, .1, .2])
        ax.set_xlabel('Training epochs')
        if i == 0:
            ax.set_ylabel(Y_LABEL)
            # Guide labels once, in the empty band above zero (every lung value is below -0.09).
            for epoch, text in [(120, 'evaluation'), (200, 'released\ndefault')]:
                ax.text(epoch + 6, .19, text, rotation=90, ha='left', va='top', fontsize=NOTE, linespacing=1.0,
                        color=INK, zorder=5)
        record[title] = {'patients': list(pivot.index), 'epochs': epochs, 'minus_control': pivot.to_numpy(),
                         'cohort_mean': mean.to_numpy()}
    row2 = row1 + S.HEADER + axes_h + X_AXIS + 10
    lane = cv.text_width([t[2] for t in TARGETS], TICK) + 6
    wb = (W - 8) / 2
    groups_d, groups_e = [], []
    for cohort, target, title in TARGETS:
        sub = subset(e2, cohort, target)
        groups_d.append((title, {'Released': (sub.tessera_default - sub.nested_control).to_numpy(),
                                 'Tuned': (sub.tessera_nested - sub.nested_control).to_numpy()}))
        groups_e.append((title, {'No gate': (sub.nogate_selected - sub.tessera_nested).to_numpy(),
                                 'No contrast': (sub.nocontrast_selected - sub.tessera_nested).to_numpy()}))
    h2 = 206
    # Same bottom allowance as (e), whose label needs two lines, so the rows of (d) and (e) align.
    d = cv.panel(0, row2, wb, h2, 'd', 'Released and tuned', left=lane, bottom=X_AXIS + 10, right=30)
    record['d'] = paired_rows(d, groups_d, [('Released', COLOR['Tessera'], False), ('Tuned', COLOR['Tessera'], True)],
                              lane, (-.28, .18), [-.2, -.1, 0, .1], Y_LABEL)
    e = cv.panel(wb + 8, row2, wb, h2, 'e', 'Components at tuned setting', left=lane, bottom=X_AXIS + 10, right=30)
    record['e'] = paired_rows(e, groups_e, [('No gate', COLOR['NoGate'], True), ('No contrast', COLOR['NoContrast'], True)],
                              lane, (-.28, .18), [-.2, -.1, 0, .1], f'Removed component minus\ntuned {TST} (ARI)')
    cv.resize(row2 + h2)
    return save(cv, 'figure-05', record)


# ================================================================== Figure 6
def figure_06():
    k = S.read_csv(SUMMARY / 'fig_k.csv')
    label_free = S.read_csv(SUMMARY / 'fig_k_label_free.csv')
    column = 'Tessera120-TunedControl'
    cv = Canvas(430)
    legend_band(cv, [Line2D([0], [0], color=PATIENT_LINE, lw=.8, label='Patients, Euclidean'),
                     Line2D([0], [0], color=COLOR['Tessera'], lw=2, marker='o', ms=3, label='Euclidean'),
                     Line2D([0], [0], color=COLOR['Tessera'], lw=2, ls=DASH, marker='o', ms=3, mfc='white',
                            label='Unit length')])
    row1, axes_h = 17, 126
    left0, inner, right, gap = 40, 24, 6, 5
    data_w = (W - left0 - 3 * inner - 4 * right - 3 * gap) / 4
    record = {}
    for i, (cohort, target, title) in enumerate(TARGETS):
        left = left0 if i == 0 else inner
        x = 0 if i == 0 else left0 + data_w + right + gap + (i - 1) * (inner + data_w + right + gap)
        ax = cv.panel(x, row1, left + data_w + right, S.HEADER + axes_h + X_AXIS + 10, 'abcd'[i], title, left=left,
                      bottom=X_AXIS + 10, right=right)
        sub = subset(k, cohort, target)
        offsets = list(range(int(sub.offset.min()), int(sub.offset.max()) + 1))  # every swept cluster number
        pivots = {g: sub[sub.geometry == g].pivot(index='unit', columns='offset', values=column).reindex(columns=offsets)
                  for g in ('euclidean', 'cosine')}
        assert list(pivots['euclidean'].index) == list(pivots['cosine'].index)
        for _, values in pivots['euclidean'].iterrows():
            ax.plot(offsets, values.to_numpy(), color=PATIENT_LINE, lw=.75, zorder=2, solid_capstyle='round')
        means = {g: complete_mean(p) for g, p in pivots.items()}
        ax.plot(offsets, means['euclidean'].to_numpy(), color=COLOR['Tessera'], lw=2.0, marker='o', ms=3, zorder=4)
        ax.plot(offsets, means['cosine'].to_numpy(), color=COLOR['Tessera'], lw=2.0, ls=DASH, marker='o', ms=3,
                mfc='white', zorder=4)
        ax.axhline(0, color=INK, lw=.6, zorder=1)
        ax.axvline(0, color=GUIDE, lw=.6, zorder=1)
        span = offsets[-1] - offsets[0]
        step = 5 if span > 14 else 3 if span > 8 else 2
        ax.set_xlim(offsets[0] - .6, offsets[-1] + .6)
        ax.set_xticks([t for t in range(-30, 31, step) if offsets[0] <= t <= offsets[-1]])
        ax.set_ylim(-.45, .45)
        ax.set_yticks([-.4, -.2, 0, .2, .4])
        ax.set_xlabel('Cluster number\nminus reference', linespacing=1.15)
        if i == 0:
            ax.set_ylabel(Y_LABEL)
        record[title] = {'units': list(pivots['euclidean'].index), 'offsets': offsets,
                         'euclidean': pivots['euclidean'].to_numpy(), 'unit_length': pivots['cosine'].to_numpy(),
                         'mean_euclidean': means['euclidean'].to_numpy(), 'mean_unit_length': means['cosine'].to_numpy(),
                         'patients_per_offset': pivots['euclidean'].notna().sum(axis=0).tolist()}
    row2 = row1 + S.HEADER + axes_h + X_AXIS + 10 + 10
    lane = cv.text_width([t[2] for t in TARGETS], TICK) + 6
    groups = []
    for cohort, target, title in TARGETS:
        sub = subset(label_free, cohort, target)
        eu = sub[sub.geometry == 'euclidean'].set_index('unit').sort_index()  # same patient order as panels a-d
        co = sub[sub.geometry == 'cosine'].set_index('unit').reindex(eu.index)
        assert (eu.K == eu.K_label_free).all() and (co.K == co.K_label_free).all() and (eu.K == co.K).all()
        groups.append((title, {'Euclidean': eu[column].to_numpy(), 'Unit length': co[column].to_numpy()}))
    h2 = 206
    e = cv.panel(0, row2, W, h2, 'e', 'Label-free cluster number', left=lane, bottom=X_AXIS, right=30)
    record['e'] = paired_rows(e, groups, [('Euclidean', COLOR['Tessera'], True), ('Unit length', COLOR['Tessera'], False)],
                              lane, (-.45, .25), [-.4, -.3, -.2, -.1, 0, .1, .2], Y_LABEL)
    cv.resize(row2 + h2)
    return save(cv, 'figure-06', record)


NEW = {'figure-05': figure_05, 'figure-06': figure_06}

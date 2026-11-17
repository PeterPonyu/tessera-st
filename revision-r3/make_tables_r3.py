"""Write the round-3 supplementary tables from saved inputs and results (no typed values).

table-new-cohorts-r3.tex  one row per new and pilot unit: observations, frames, targets with class
                          counts, adjusted homophily at six neighbours and label-selected gain.
table-p3-r3.tex           P3 per encoder, counterpart and target class (written when W4 results exist).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import functools  # noqa: E402
read_csv = functools.partial(pd.read_csv, float_precision="round_trip")  # pandas' default float parser segfaults on some 17-digit values

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_r3 as c3  # noqa: E402

DATA_R3 = c3.c1.PROJECT / 'data-r3'
COHORT_NAME = {'spinal': 'Mouse spinal cord', 'colorectal': 'Colorectal cancer', 'colorectal_pilot': 'Colorectal, pilot',
               'wu_breast': 'Breast cancer'}
TARGET_NAME = {'celltype_l1': 'cell types, level 1', 'celltype_l2': 'cell types, level 2', 'all': 'all labels',
               'tumor': 'tumour/non-tumour'}


def num(v, d=2):
    v = float(v)
    v = 0.0 if round(v, d) == 0 else v      # no negative zero
    s = f'{v:.{d}f}'
    return s.replace('-', '\\ensuremath{-}') if s.startswith('-') else s


def new_cohorts(out: Path):
    w6 = read_csv(c3.OUT / 'W6' / 'targets.csv').set_index(['dataset', 'target'])
    rows = []
    for f in sorted(DATA_R3.glob('*/*/input.json')):
        if f.parts[-3] == 'raw':
            continue
        info = json.loads(f.read_text())
        cohort, dataset = f.parts[-3], f.parts[-2]
        key = f'{cohort}/{dataset}'
        frames = info.get('frames')
        n_frames = len(frames) if isinstance(frames, (dict, list)) else frames
        for t, meta in info['targets'].items():
            r = w6.loc[(key, t)] if (key, t) in w6.index else None
            rows.append((COHORT_NAME[cohort], info.get('unit', dataset), info['n_obs'], n_frames,
                         f"{TARGET_NAME.get(t, t)} ({meta['n_classes']})",
                         num(r.h_adj6, 3) if r is not None else '', num(r.g_star, 3) if r is not None else ''))
    lines = [r'\begin{table}[!htbp]\centering\densetablefont\fontsize{8.5}{10}\selectfont\setlength{\tabcolsep}{3pt}',
             r'\caption{Units of the new cohorts and of the colorectal pilot. Observations and coordinate frames after '
             r'bounding; targets with their class counts; adjusted homophily at six neighbours; and the gain of the best '
             r'uniform neighbourhood mean over expression-only clustering (ARI), which uses the test labels and serves '
             r'only to evaluate the diagnostics.}\label{tab:new-cohorts}',
             r'\begin{tabular}{@{}llrrlrr@{}}\toprule',
             r'Cohort & Unit & Obs. & Frames & Target (classes) & $h_{\mathrm{adj}}(6)$ & Gain \\\midrule']
    last = None
    for c, u, n, fr, t, h, g in rows:
        lines.append(f"{c if c != last else ''} & {u} & {n:,} & {fr} & {t} & {h} & {g} \\\\")
        last = c
    lines += [r'\bottomrule\end{tabular}\end{table}']
    (out / 'table-new-cohorts-r3.tex').write_text('\n'.join(lines) + '\n')


def p3_table(out: Path):
    s = json.loads((c3.OUT / 'summary' / 'summary.json').read_text())
    if 'P3' not in s:
        return
    rows = [('tessera', 'propagation', 'Tessera-ST', 'three-step propagation'),
            ('tessera', 'same_graph_mean', '', 'same-graph mean'),
            ('graphst', 'propagation', 'GraphST', 'two-step propagation'),
            ('graphst', 'same_graph_mean', '', 'same-graph mean'),
            ('stagate', 'same_graph_mean', 'STAGATE', 'same-graph mean (one step)')]   # one step: the same operator
    cls = {'region': 'Regions (10 patients)', 'celltype': 'Cell types (10 mice)'}
    signed = lambda v: ('+' if round(float(v), 3) > 0 else '') + num(v, 3)
    sesoi = ' or '.join(sorted({num(v['sesoi'], 3) for by in s['P3'].values() for v in by.values()}))
    lines = [r'\begin{table}[!htbp]\centering\densetablefont\setlength{\tabcolsep}{3pt}',
             r'\caption{Encoders against parameter-free propagation on their own graphs in the new cohorts. Each encoder, '
             r'at the configuration selected with the unit excluded, minus a counterpart on its own graph: the mean of its '
             r'incoming neighbours (same-graph mean) or that mean repeated as many times as the encoder passes messages. '
             r'Difference, mean over units of the difference in adjusted Rand index (ARI), with seeds averaged within units; interval, 95\% '
             r'$t$-interval over units; positive, units with a positive difference. SESOI, smallest effect size of '
             r'interest: one tenth of the cohort\textquoteright s spatial headroom, at least 0.01 ARI, averaged over units '
             + f'({sesoi} ARI in both classes). '
             r'A side is superior when the one-sided test against the SESOI, Holm-adjusted over the comparisons of a target '
             r'class, falls below 0.025; otherwise the difference is not resolvable. Region targets are pathologist '
             r'annotations of four colorectal and six breast cancer patients; cell-type targets are expression-assigned '
             r'labels of ten mouse spinal cords.}\label{tab:p3}',
             r'\begin{tabular}{@{}lllrrrl@{}}\toprule',
             r'Targets & Encoder & Counterpart & Difference & 95\% interval & Positive & Decision \\\midrule']
    for c, cname in cls.items():
        for e, k, ename, kname in rows:
            t = s['P3'][c][f'{e}|{k}']
            lines.append(f"{cname} & {ename} & {kname} & {signed(t['estimate'])} & "
                         f"{signed(t['ci'][0])} to {signed(t['ci'][1])} & {t['units_positive']}/{t['n_units']} & "
                         f"{t['verdict']} \\\\")
            cname = ''
        if c == 'region':
            lines.append(r'\addlinespace')
    lines += [r'\bottomrule\end{tabular}\end{table}']
    (out / 'table-p3-r3.tex').write_text('\n'.join(lines) + '\n')


def main():
    out = c3.OUT / 'summary'
    new_cohorts(out)
    p3_table(out)
    print('tables ->', out)


if __name__ == '__main__':
    main()

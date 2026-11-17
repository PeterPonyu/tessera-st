"""Write the round-3 supplementary tables from saved inputs and results (no typed values).

table-new-cohorts-r3.tex  one row per new and pilot unit: observations, frames, targets with class
                          counts, adjusted homophily at six neighbours and label-selected gain.
table-p3-r3.tex           P3 per encoder, counterpart and target class (written when W4 results exist).
table-diagnostics-r3.tex  label diagnostics of every target (calibration; new cohorts and pilot).
table-absolute-r3.tex     absolute agreement in the new cohorts (expression only, best mean, encoders, propagation).
table-native-r3.tex       native versus shared input for GraphST and STAGATE (calibration cohorts).
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



PROV_SHORT = {'computed-niche': 'CN', 'expression-assigned': 'EA', 'pathologist': 'PA', 'histology-anatomical': 'HA',
              'dominant-category': 'DC'}
CAL_COHORT = {'NSCLC': 'Lung cancer', 'BCC': 'Basal cell carcinoma', 'HER2ST': 'HER2-positive breast',
              'Breast_Visium': 'Breast', 'Colorectal_TVA232': 'Colorectal adenoma', 'MIBI': 'Development',
              'OpenST_HNSCC26': 'Development', 'DLPFC_151673': 'Development', 'STARmap_visual_cortex': 'Development',
              'MERFISH_hypothalamus': 'Development'}
CAL_UNIT = {'NSCLC_Lung5_Rep2': 'Lung 5, replicate 2', 'NSCLC_Lung6': 'Lung 6', 'NSCLC_Lung9_Rep1': 'Lung 9, replicate 1',
            'NSCLC_Lung12': 'Lung 12', 'NSCLC_Lung13': 'Lung 13', 'Breast_Visium': 'one section',
            'Colorectal_TVA232': 'one section', 'MIBI': 'MIBI', 'OpenST_HNSCC26': 'Open-ST',
            'DLPFC_151673': 'cortex', 'STARmap_visual_cortex': 'visual cortex', 'MERFISH_hypothalamus': 'hypothalamus',
            'BCC_PatientA': 'Patient A', 'BCC_PatientB': 'Patient B', 'BCC_PatientC': 'Patient C', 'BCC_PatientD': 'Patient D'}
CAL_TARGET = {('NSCLC', 'primary'): 'regions', ('NSCLC', 'celltype'): 'cell types', ('BCC', 'primary'): 'cell types',
              ('Breast_Visium', 'primary'): 'majority category', ('Colorectal_TVA232', 'primary'): 'subpopulations',
              ('MIBI', 'primary'): 'molecular clusters', ('OpenST_HNSCC26', 'primary'): 'molecular clusters',
              ('HER2ST', 'all'): 'all labels', ('HER2ST', 'compartments'): 'compartments',
              ('HER2ST', 'boundary_T1'): 'tumour/non-tumour'}
DIAG_CAPTION = (r'{what} Prov., provenance: CN computed niche, EA expression-assigned, PA pathologist, HA histology-anatomical, '
                r'DC dominant category. $h_{{\mathrm{{adj}}}}(6)$, adjusted homophily at six neighbours; LI, label informativeness '
                r'at six neighbours; $\ell$, label correlation length in neighbours, in median nearest-neighbour distances '
                r'($d_1$) and in micrometres where the platform\textquoteright s spacing is documented, with $\geq$ when it '
                r'reached the largest neighbourhood examined; $\kappa$, Cohen\textquoteright s $\kappa$ of predicting the label '
                r'from the observation\textquoteright s own expression under spatially blocked cross-validation, and with '
                r'random folds ($\kappa_{{\mathrm{{r}}}}$, an optimistic bound); BA, the balanced accuracy of that prediction, '
                r'with chance level in parentheses; IG, the largest increase in $\kappa$ when a neighbourhood mean is added '
                r'to the observation\textquoteright s expression, at the neighbourhood size in parentheses; $g^*$, the gain '
                r'in adjusted Rand index of the best uniform neighbourhood mean over expression-only clustering, at the '
                r'size $k^*$ in parentheses, which uses the reference labels. Dashes mark values that could not be '
                r'estimated.')


def _f(v, d=3, sign=False):
    if v is None or pd.isna(v):
        return '--'
    s = num(v, d)
    return ('+' + s) if sign and round(float(v), d) > 0 else s


def _int(v):
    return '--' if v is None or pd.isna(v) else f'{int(round(float(v)))}'


def _diag_row(r, cohort, unit, target):
    cens = r'$\geq$' if bool(r.ell_censored) else ''
    return [cohort, unit, f'{target} ({int(r.K)})', PROV_SHORT.get(r.provenance, r.provenance), _f(r.h_adj6), _f(r.li6),
            (cens + _int(r.ell_k)) if not pd.isna(r.ell_k) else '--', _f(r.ell_d1, 1), _int(r.ell_um),
            _f(r.kappa_self, 2), _f(r.kappa_self_random, 2),
            f'{_f(r.ba_self, 2)} ({_f(r.chance_ba, 2)})' if not pd.isna(r.ba_self) else '--',
            f'{_f(r.info_gain_max, 3, sign=True)} ({_int(r.info_gain_k)})' if not pd.isna(r.info_gain_max) else '--',
            f'{_f(r.g_star, 3, sign=True)} ({_int(r.k_star)})']


def _diag_table(rows, caption, label):
    head = (r'Cohort & Unit & Target ($K$) & Prov. & $h_{\mathrm{adj}}(6)$ & LI & $\ell$ & $\ell$ ($d_1$) & $\ell$ ($\mu$m) & '
            r'$\kappa$ & $\kappa_{\mathrm{r}}$ & BA & IG & $g^*$ ($k^*$) \\\midrule')
    lines = [r'\begin{table}[!htbp]\centering\scriptsize\setlength{\tabcolsep}{2.2pt}\renewcommand{\arraystretch}{1.02}',
             r'\caption{' + caption + r'}\label{' + label + r'}',
             r'\fitwide{\begin{tabular}{@{}lllcrrrrrrrrrr@{}}\toprule', head]
    last = None
    for row in rows:
        cells = list(row)
        if cells[0] == last:
            cells[0] = ''
        else:
            last = cells[0]
        lines.append(' & '.join(c.replace('_', r'\_') for c in cells) + r' \\')
    lines += [r'\bottomrule\end{tabular}}\end{table}']
    return '\n'.join(lines)


def diagnostics_tables(out: Path):
    cal = read_csv(c3.OUT / 'W1' / 'targets.csv')
    new = read_csv(c3.OUT / 'W6' / 'targets.csv')
    rows_c = []
    for r in cal.itertuples():
        target = CAL_TARGET.get((r.cohort, r.target), 'anatomical labels' if r.target == 'anatomy' else r.target)
        rows_c.append(_diag_row(r, CAL_COHORT[r.cohort], CAL_UNIT.get(r.dataset, r.unit), target))
    rows_n = []
    for r in new.itertuples():
        rows_n.append(_diag_row(r, COHORT_NAME[r.cohort], r.unit, TARGET_NAME.get(r.target, r.target)))
    text = (_diag_table(rows_c, DIAG_CAPTION.format(what='Label diagnostics of the calibration targets.'),
                        'tab:diagnostics-calibration') + '\n\n' +
            _diag_table(rows_n, DIAG_CAPTION.format(what='Label diagnostics of the targets of the new cohorts and of the '
                                                          'colorectal pilot.'), 'tab:diagnostics-new'))
    (out / 'table-diagnostics-r3.tex').write_text(text + '\n')


def absolute_table(out: Path):
    w6 = read_csv(c3.OUT / 'W6' / 'targets.csv')
    u = read_csv(c3.OUT / 'summary' / 'p3_units.csv')
    order = [('spinal', 'celltype_l1'), ('spinal', 'celltype_l2'), ('colorectal', 'all'), ('wu_breast', 'all'),
             ('wu_breast', 'tumor')]
    lines = [r'\begin{table}[!htbp]\centering\densetablefont\setlength{\tabcolsep}{3pt}',
             r'\caption{Absolute agreement in the new cohorts. Mean adjusted Rand index (ARI) of K-means clustering over '
             r'units (mice or patients), with seeds averaged within units: expression-only clustering, with the range over '
             r'units; the best uniform neighbourhood mean, which is selected with the reference labels and is shown only '
             r'for reference; each encoder at the configuration selected with the unit excluded; and '
             r'parameter-free propagation on that encoder\textquoteright s own graph, repeated as many times as the '
             r'encoder passes messages ($P^L x$).}\label{tab:absolute}',
             r'\fitwide{\begin{tabular}{@{}llrlrrrrrrr@{}}\toprule',
             r'Cohort & Target & Units & Expression only & Best mean & Tessera-ST & $P^3x$ & GraphST & $P^2x$ & STAGATE & '
             r'$P^1x$ \\\midrule']
    last = None
    for cohort, target in order:
        w = w6[(w6.cohort == cohort) & (w6.target == target)]
        e = u[(u.cohort == cohort) & (u.target == target)]
        cells = [COHORT_NAME[cohort] if cohort != last else '', TARGET_NAME.get(target, target), str(len(w)),
                 f'{_f(w.ari_pca50.mean())} ({_f(w.ari_pca50.min())} to {_f(w.ari_pca50.max())})',
                 _f((w.ari_pca50 + w.g_star).mean())]
        for enc in ('tessera', 'graphst', 'stagate'):
            g = e[e.encoder == enc]
            assert len(g) == len(w), (cohort, target, enc)
            cells += [_f(g.encoder_ari.mean()), _f(g.propagation_ari.mean())]
        lines.append(' & '.join(cells) + r' \\')
        last = cohort
    lines += [r'\bottomrule\end{tabular}}\end{table}']
    (out / 'table-absolute-r3.tex').write_text('\n'.join(lines) + '\n')


def native_table(out: Path):
    s = read_csv(c3.OUT / 'W4' / 'scores.csv')
    s = s[s.arm == 'released'].groupby(['rule', 'encoder', 'cohort', 'dataset', 'target', 'config', 'input_mode', 'seed'],
                                       as_index=False).ari.mean()
    unit = s.groupby(['rule', 'encoder', 'cohort', 'dataset', 'target', 'config', 'input_mode']).ari.mean().unstack('input_mode')
    unit = unit.dropna(subset=['native'])
    assert unit.common.notna().all(), 'a native fit has no shared-input counterpart'
    unit['diff'] = unit.native - unit.common
    names = {('NSCLC', 'primary'): 'lung regions', ('NSCLC', 'cell_type'): 'lung cell types',
             ('HER2ST', 'all'): 'HER2ST all labels', ('HER2ST', 'compartments'): 'HER2ST compartments',
             ('HER2ST', 'boundary_T1'): 'HER2ST tumour/non-tumour'}
    lines = [r'\begin{table}[!htbp]\centering\densetablefont\setlength{\tabcolsep}{4pt}',
             r'\caption{Native versus shared input for GraphST and STAGATE, in the calibration cohorts for which each '
             r'method\textquoteright s own preprocessing could be run. Difference in adjusted Rand index, native '
             r'preprocessing minus the shared input (the first 50 principal components), at each method\textquoteright s '
             r'released settings with five seeds averaged within patients: mean over patients with its range, and the '
             r'number of patients with a positive difference, under K-means and Leiden clustering.}\label{tab:native-r3}',
             r'\begin{tabular}{@{}llrlrlr@{}}\toprule',
             r'Encoder & Target & Patients & K-means & Positive & Leiden & Positive \\\midrule']
    last = None
    for enc, ename in (('graphst', 'GraphST'), ('stagate', 'STAGATE')):
        for key, tname in names.items():
            cells = [ename if enc != last else '', tname]
            for rule in ('kmeans', 'leiden'):
                d = unit.loc[(rule, enc, key[0])]['diff']
                d = d[d.index.get_level_values('target') == key[1]]
                if rule == 'kmeans':
                    cells.append(str(len(d)))
                cells += [f'{_f(d.mean(), sign=True)} ({_f(d.min(), sign=True)} to {_f(d.max(), sign=True)})',
                          f'{int((d > 0).sum())}/{len(d)}']
            lines.append(' & '.join(cells) + r' \\')
            last = enc
    lines += [r'\bottomrule\end{tabular}\end{table}']
    (out / 'table-native-r3.tex').write_text('\n'.join(lines) + '\n')


def main():
    out = c3.OUT / 'summary'
    new_cohorts(out)
    p3_table(out)
    diagnostics_tables(out)
    absolute_table(out)
    native_table(out)
    print('tables ->', out)


if __name__ == '__main__':
    main()

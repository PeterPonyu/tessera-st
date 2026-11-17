"""Write revision-r3-macros.tex: every round-3 number in the manuscript, computed from saved outputs.

Sources: results/revision-r3/{W1, W1b, W2, W3, W6, freeze, summary} and, when present, W4.
Each macro is computed here from a CSV or JSON cell; none is typed by hand. Values are
formatted to three decimals for ARI-scale quantities. Run after analyze_r3.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_r3 as c3  # noqa: E402
import prepare_freeze as pf  # noqa: E402

R = c3.OUT
M: dict[str, str] = {}


def f3(v, sign=False):
    v = float(v)
    v = 0.0 if round(v, 3) == 0 else v      # no negative zero
    s = f'{v:+.3f}' if sign else f'{v:.3f}'
    return s.replace('-', '\\ensuremath{-}') if s.startswith('-') else s


def f2(v, sign=False):
    v = float(v)
    v = 0.0 if round(v, 2) == 0 else v      # no negative zero
    s = f'{v:+.2f}' if sign else f'{v:.2f}'
    return s.replace('-', '\\ensuremath{-}') if s.startswith('-') else s


def put(name, value):
    assert name.isalpha(), name
    M[name] = str(value)


def calibration():
    t = pd.read_csv(R / 'W1' / 'targets.csv')
    put('rCalTargets', len(t))
    put('rCalDatasets', t.dataset.nunique())
    for name, col in [('Hadj', 'h_adj6'), ('Li', 'li6'), ('Gain', 'info_gain_max'), ('Kappa', 'kappa_self'),
                      ('Ell', 'ell_k')]:
        m = t[[col, 'g_star']].dropna()
        put(f'rCalRho{name}', f2(stats.spearmanr(m[col], m.g_star).statistic, sign=False))
    rho_t = [abs(stats.spearmanr(t[[c, 'tessera_minus_sgm']].dropna()[c],
                                 t[[c, 'tessera_minus_sgm']].dropna().tessera_minus_sgm).statistic)
             for c in ('h_adj6', 'li6', 'ell_k', 'kappa_self', 'info_gain_max')]
    put('rCalRhoTessMax', f2(max(rho_t)))
    med = t.groupby('provenance')[['h_adj6', 'kappa_self', 'info_gain_max', 'g_star']].median()
    names = {'computed-niche': 'Niche', 'expression-assigned': 'Expr', 'pathologist': 'Path',
             'histology-anatomical': 'Anat', 'dominant-category': 'Dom'}
    for prov, short in names.items():
        if prov in med.index:
            put(f'rProv{short}Hadj', f2(med.loc[prov, 'h_adj6']))
            put(f'rProv{short}Kappa', f2(med.loc[prov, 'kappa_self']))
            put(f'rProv{short}Gstar', f3(med.loc[prov, 'g_star'], sign=True))
    lung = t[t.cohort == 'NSCLC']
    reg, ct = lung[lung.target == 'primary'], lung[lung.target == 'celltype']
    put('rLungRegHadjMin', f2(reg.h_adj6.min())); put('rLungRegHadjMax', f2(reg.h_adj6.max()))
    put('rLungCtHadjMin', f2(ct.h_adj6.min())); put('rLungCtHadjMax', f2(ct.h_adj6.max()))
    put('rLungCtKappaMin', f2(ct.kappa_self.min())); put('rLungCtKappaMax', f2(ct.kappa_self.max()))
    put('rLungRegKappaMin', f2(reg.kappa_self.min())); put('rLungRegKappaMax', f2(reg.kappa_self.max()))
    put('rLungRegGstarMin', f3(reg.g_star.min())); put('rLungRegGstarMax', f3(reg.g_star.max()))
    put('rLungCtGstarMin', f3(ct.g_star.min(), sign=True)); put('rLungCtGstarMax', f3(ct.g_star.max(), sign=True))
    reg_t = pf.regret_table(t)
    for col, short in [('rule', 'Rule'), ('no smoothing', 'None'), ('k=6', 'Six'), ('k=30', 'Thirty'),
                       ('k=200', 'TwoHundred')]:
        put(f'rCalRegret{short}', f3(reg_t[col].mean()))
    chk = json.loads((R / 'W1' / 'checks.json').read_text())
    put('rReproMaxDiff', f'{chk["bounded_primary_max_abs_diff"]:.4f}')
    put('rReproValues', chk['bounded_primary_values'])


def construction():
    t = pd.read_csv(R / 'W1b' / 'targets.csv')
    rho_e, rho_k = [], []
    for _, g in t.groupby('unit'):
        rho_e.append(stats.spearmanr(g.k_c, g.ell_k).statistic)
        rho_k.append(stats.spearmanr(g.k_c, g.k_star).statistic)
    put('rPTwoRhoEllMin', f2(min(rho_e))); put('rPTwoRhoEllMax', f2(max(rho_e)))
    put('rPTwoRhoKMin', f2(min(rho_k))); put('rPTwoRhoKMax', f2(max(rho_k)))
    put('rPTwoEllPass', sum(r >= 0.9 for r in rho_e)); put('rPTwoKPass', sum(r >= 0.8 for r in rho_k))
    m = t.groupby('k_c').tessera_minus_sgm.mean()
    put('rPTwoMarginMin', f3(m.min(), sign=True)); put('rPTwoMarginMax', f3(m.max(), sign=True))
    peak = t.groupby('k_c').info_gain_k.median()
    put('rPTwoPeakMatches', int(sum(abs(np.log2(peak[k] / k)) <= 0.6 for k in peak.index if k <= 200)))
    put('rPTwoScales', int((peak.index <= 200).sum()))
    g = t.groupby('k_c').g_star.mean()
    tess = t.groupby('k_c').ari_tessera.mean()
    put('rPTwoTessSmall', f3(tess.loc[tess.index.min()])); put('rPTwoTessLarge', f3(tess.loc[tess.index.max()]))
    put('rPTwoGainSmall', f3(g.loc[g.index.min()])); put('rPTwoGainLarge', f3(g.max()))


def leiden():
    s = pd.read_csv(R / 'W2' / 'selection.csv')
    cohorts = {('NSCLC', 'primary'): 'Lung', ('BCC', 'primary'): 'Bcc', ('HER2ST', 'all'): 'Her',
               ('HER2ST', 'boundary_T1'): 'Tumor'}
    rules = {('kmeans', 'kmeans'): 'KK', ('kmeans', 'leiden'): 'KL', ('leiden', 'kmeans'): 'LK',
             ('leiden', 'leiden'): 'LL'}
    for (cohort, target), short in cohorts.items():
        for (sel, sco), r in rules.items():
            g = s[(s.cohort == cohort) & (s.target == target) & (s.select_rule == sel) & (s.score_rule == sco)]
            put(f'rLeiden{short}{r}', f3(g.tessera_minus_control.mean(), sign=True))
            put(f'rLeiden{short}{r}Below', int((g.tessera_minus_control < 0).sum()))
            put(f'rLeiden{short}N', len(g))


def noise():
    s = pd.read_csv(R / 'W2' / 'seeds.csv')
    g = s.groupby('dataset').ari.agg(['std', 'min', 'max'])
    put('rSeedSdMedian', f3(g['std'].median())); put('rSeedRangeMax', f3((g['max'] - g['min']).max()))
    ps = json.loads((R / 'W1' / 'per_seed.json').read_text())
    d = [r.ari - ps[f'{r.dataset}|primary|tessera'][r.seed - 1] for r in s[s.seed <= 3].itertuples()]
    put('rCpuGpuMedian', f'{np.median(np.abs(d)):.4f}'); put('rCpuGpuMax', f3(np.max(np.abs(d))))
    dm = pd.Series(d, index=s[s.seed <= 3].dataset.values).groupby(level=0).mean()
    put('rCpuGpuDatasetMax', f3(dm.abs().max()))


def mechanism():
    w3 = pd.read_csv(R / 'W3' / 'scores.csv')
    t = pd.read_csv(R / 'W1' / 'targets.csv').set_index(['dataset', 'target'])
    km = w3[w3.rule == 'kmeans']
    cn = km[km.arm == 'ContrastNoMP'].groupby(['dataset', 'target']).ari.mean()
    p3 = km[(km.arm == 'Propagate') & (km.L == 3)].groupby(['dataset', 'target']).ari.mean()
    d_cn = (t.ari_tessera - cn).dropna()
    d_p3 = (t.ari_tessera - p3).dropna()
    put('rCnMedian', f3(d_cn.median(), sign=True)); put('rCnWithin', int((d_cn.abs() <= 0.02).sum()))
    put('rCnN', len(d_cn))
    lung = [i for i in t.index if i[0].startswith('NSCLC') and i[1] == 'primary']
    rec = [(cn[i] - t.ari_pca50[i]) / (t.ari_tessera[i] - t.ari_pca50[i]) for i in lung
           if t.ari_tessera[i] > t.ari_pca50[i]]
    put('rCnRecoverMin', f'{100 * min(rec):.0f}'); put('rCnRecoverMax', f'{100 * max(rec):.0f}')
    put('rCnRecoverN', len(rec))
    put('rPThreeMedian', f3(d_p3.median(), sign=True)); put('rPThreeBelow', int((d_p3 < 0).sum()))
    lct = d_p3[[i for i in d_p3.index if i[0].startswith('NSCLC') and i[1] == 'celltype']]
    put('rLungCtPThreeMin', f3(lct.min(), sign=True)); put('rLungCtPThreeMax', f3(lct.max(), sign=True))
    put('rPThreeN', len(d_p3))


def prospective():
    s = json.loads((R / 'summary' / 'summary.json').read_text())
    p1 = s['P1']
    put('rPOneUnits', p1['n_units']); put('rPOneTargets', p1['n_targets'])
    for col, short in [('rule', 'Rule'), ('no smoothing', 'None'), ('k=6', 'Six'), ('k=30', 'Thirty'),
                       ('k=200', 'TwoHundred')]:
        put(f'rPOneRegret{short}', f3(p1['mean_regret'][col]))
    for col, short in [('no smoothing', 'None'), ('k=6', 'Six'), ('k=30', 'Thirty'), ('k=200', 'TwoHundred')]:
        put(f'rPOneBetter{short}', p1['rule_better_units'][col])
        p = p1['p_holm'][col]
        put(f'rPOneP{short}', f'{p:.3f}' if p >= 0.001 else '<0.001')
    w6 = pd.read_csv(R / 'W6' / 'targets.csv')
    new = w6[w6.cohort.isin(['spinal', 'colorectal', 'wu_breast'])]
    smooth_helps = new.g_star > 0
    put('rPOneHadjCorrect', int(((new.h_adj6 > 0.5) == smooth_helps).sum()))
    put('rPOneHelped', int(smooth_helps.sum()))
    wu = new[new.cohort == 'wu_breast']
    no_smooth = [pf.rule_choice(r) == 0 for r in wu.itertuples()]
    put('rPOneWuMissed', int(sum(ns and g > 0 for ns, g in zip(no_smooth, wu.g_star))))
    bc = p1['by_cohort']['wu_breast']
    put('rPOneWuRegretRule', f3(bc['rule']))
    fixed = [bc[k] for k in ('k=6', 'k=30', 'k=200')]
    put('rPOneWuRegretFixedMin', f3(min(fixed))); put('rPOneWuRegretFixedMax', f3(max(fixed)))
    for cohort, short in [('spinal', 'Spinal'), ('colorectal', 'Crc'), ('wu_breast', 'Wu')]:
        g = new[new.cohort == cohort]
        put(f'rNew{short}HadjMin', f3(g.h_adj6.min())); put(f'rNew{short}HadjMax', f3(g.h_adj6.max()))
        put(f'rNew{short}GstarMin', f3(g.g_star.min(), sign=True))
        put(f'rNew{short}GstarMax', f3(g.g_star.max(), sign=True))
        put(f'rNew{short}Units', g.dataset.nunique()); put(f'rNew{short}Targets', len(g))
    for cohort, short in [('spinal', 'Spinal'), ('colorectal', 'Crc'), ('wu_breast', 'Wu')]:
        if cohort in s.get('sesoi_new', {}):
            put(f'rSesoi{short}', f3(s['sesoi_new'][cohort]))


def attribution():
    s = json.loads((R / 'summary' / 'summary.json').read_text())
    if 'P3' not in s:
        return
    enc = {'tessera': 'Tess', 'graphst': 'Graph', 'stagate': 'Stag'}
    cp = {'propagation': 'Prop', 'same_graph_mean': 'Mean'}
    cls = {'region': 'Reg', 'celltype': 'Ct'}
    for c, cs in cls.items():
        for key, test in s['P3'].get(c, {}).items():
            e, k = key.split('|')
            base = f'rPThree{cls[c]}{enc[e]}{cp[k]}'
            put(base + 'Est', f3(test['estimate'], sign=True))
            put(base + 'Lo', f3(test['ci'][0], sign=True)); put(base + 'Hi', f3(test['ci'][1], sign=True))
            put(base + 'Pos', test['units_positive']); put(base + 'N', test['n_units'])
            put(base + 'Verdict', test['verdict'])


def frozen():
    fz = json.loads((R / 'freeze' / 'freeze_numbers.json').read_text())
    names = {'NSCLC|primary': 'Nsclc', 'BCC|primary': 'Bcc', 'HER2ST|all': 'HerAll', 'HER2ST|boundary_T1': 'HerTumor'}
    for key, short in names.items():
        put(f'rSesoi{short}', f3(fz['sesoi'][key]['sesoi']))
    hw = fz['power']['Tessera-ST minus same-graph mean']['half_width']   # the column quoted in the deposited protocol
    for key, short in [('spinal cord (mice)', 'Spinal'), ('colorectal, new patients', 'CrcNew'),
                       ('colorectal, all patients', 'CrcAll'), ('breast Visium (Wu)', 'Wu'),
                       ('pooled new cancer units', 'Cancer'), ('pooled new units', 'All')]:
        put(f'rPower{short}', f3(hw[key]))


def main():
    for step in (calibration, construction, leiden, noise, mechanism, prospective, attribution, frozen):
        step()
    lines = ['% Generated by revision-r3/make_macros_r3.py from results/revision-r3; do not edit by hand.']
    lines += [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(M.items())]
    out = R / 'summary' / 'revision-r3-macros.tex'
    out.write_text('\n'.join(lines) + '\n')
    print(f'{len(M)} macros -> {out}')


if __name__ == '__main__':
    main()

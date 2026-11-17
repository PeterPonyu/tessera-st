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
import functools  # noqa: E402
read_csv = functools.partial(pd.read_csv, float_precision="round_trip")  # pandas' default float parser segfaults on some 17-digit values
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
    s = f'{v:+.3f}' if sign and v != 0 else f'{v:.3f}'
    return s.replace('-', '\\ensuremath{-}') if s.startswith('-') else s


def f2(v, sign=False):
    v = float(v)
    v = 0.0 if round(v, 2) == 0 else v      # no negative zero
    s = f'{v:+.2f}' if sign and v != 0 else f'{v:.2f}'
    return s.replace('-', '\\ensuremath{-}') if s.startswith('-') else s


def put(name, value):
    assert name.isalpha(), name
    M[name] = str(value)


def calibration():
    t = read_csv(R / 'W1' / 'targets.csv')
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
    put('rLungRegKstarMin', int(reg.k_star.min())); put('rLungRegKstarMax', int(reg.k_star.max()))
    put('rLungCtGstarMin', f3(ct.g_star.min(), sign=True)); put('rLungCtGstarMax', f3(ct.g_star.max(), sign=True))
    reg_t = pf.regret_table(t)
    for col, short in [('rule', 'Rule'), ('no smoothing', 'None'), ('k=6', 'Six'), ('k=30', 'Thirty'),
                       ('k=200', 'TwoHundred')]:
        put(f'rCalRegret{short}', f3(reg_t[col].mean()))
    chk = json.loads((R / 'W1' / 'checks.json').read_text())
    put('rReproMaxDiff', f'{chk["bounded_primary_max_abs_diff"]:.4f}')
    put('rReproValues', chk['bounded_primary_values'])


def construction():
    t = read_csv(R / 'W1b' / 'targets.csv')
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
    s = read_csv(R / 'W2' / 'selection.csv')
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
    s = read_csv(R / 'W2' / 'seeds.csv')
    g = s.groupby('dataset').ari.agg(['std', 'min', 'max'])
    put('rSeedSdMedian', f3(g['std'].median())); put('rSeedRangeMax', f3((g['max'] - g['min']).max()))
    put('rSeedDatasets', len(g))
    ps = json.loads((R / 'W1' / 'per_seed.json').read_text())
    d = [r.ari - ps[f'{r.dataset}|primary|tessera'][r.seed - 1] for r in s[s.seed <= 3].itertuples()]
    put('rCpuGpuMedian', f'{np.median(np.abs(d)):.4f}'); put('rCpuGpuMax', f3(np.max(np.abs(d))))
    dm = pd.Series(d, index=s[s.seed <= 3].dataset.values).groupby(level=0).mean()
    put('rCpuGpuDatasetMax', f3(dm.abs().max()))


def mechanism():
    w3 = read_csv(R / 'W3' / 'scores.csv')
    t = read_csv(R / 'W1' / 'targets.csv').set_index(['dataset', 'target'])
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
    w6 = read_csv(R / 'W6' / 'targets.csv')
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
            for cohort, short in (('colorectal', 'Crc'), ('wu_breast', 'Wu')):   # cohort means behind a pooled class
                if cohort in test.get('cohort_means', {}) and test['n_cohorts'] > 1:
                    put(base + short, f3(test['cohort_means'][cohort], sign=True))


def attribution_descriptive():
    s = json.loads((R / 'summary' / 'summary.json').read_text())
    if 'P3_descriptive' not in s:
        return
    d = s['P3_descriptive']
    enc = {'tessera': 'Tess', 'graphst': 'Graph', 'stagate': 'Stag'}
    cls = {'region': 'Reg', 'celltype': 'Ct'}
    arm = {'no_gate': 'NoGate', 'no_contrast': 'NoContrast', 'self_only': 'SelfOnly',
           'permuted_xy_no_contrast': 'PermNoContrast', 'contrast_no_mp': 'ContrastNoMp',
           'identity_encoder': 'IdentityEncoder', 'permuted_xy': 'PermutedXy',
           'uniform_attention': 'UniformAttention', 'identity_graph': 'IdentityGraph'}
    word = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 8: 'Eight'}
    for c, by in d['arms'].items():
        for e, arms in by.items():
            for a, t in arms.items():
                base = f'rPThreeArm{cls[c]}{enc[e]}{arm[a]}'
                put(base + 'Est', f3(t['estimate'], sign=True)); put(base + 'Pos', t['units_positive'])
                put(base + 'N', t['n_units'])
    for c, by in d['depth'].items():
        for e, depths in by.items():
            for L, t in depths.items():
                put(f'rPThreeDepth{cls[c]}{enc[e]}{word[int(L)]}', f3(t['estimate'], sign=True))
            est = [t['estimate'] for t in depths.values()]          # range over one to eight steps
            put(f'rPThreeDepth{cls[c]}{enc[e]}Min', f3(min(est), sign=True))
            put(f'rPThreeDepth{cls[c]}{enc[e]}Max', f3(max(est), sign=True))
        est = [t['estimate'] for depths in by.values() for t in depths.values()]   # every encoder and depth
        put(f'rPThreeDepth{cls[c]}Min', f3(min(est), sign=True)); put(f'rPThreeDepth{cls[c]}Max', f3(max(est), sign=True))
    cp = {'propagation': 'Prop', 'same_graph_mean': 'Mean'}
    for c, by in d['leiden'].items():
        for e, tests in by.items():
            for k, t in tests.items():
                base = f'rPThreeLeiden{cls[c]}{enc[e]}{cp[k]}'
                put(base + 'Est', f3(t['estimate'], sign=True))
                put(base + 'Lo', f3(t['ci'][0], sign=True)); put(base + 'Hi', f3(t['ci'][1], sign=True))
                put(base + 'Pos', t['units_positive']); put(base + 'N', t['n_units'])
    for c, by in d.get('selfloop', {}).items():
        for L, t in by.items():
            base = f'rPThreeSelf{cls[c]}Tess{word[int(L)]}'
            put(base + 'Est', f3(t['estimate'], sign=True))
            put(base + 'Lo', f3(t['ci'][0], sign=True)); put(base + 'Hi', f3(t['ci'][1], sign=True))
            put(base + 'Pos', t['units_positive']); put(base + 'N', t['n_units'])
    for g, by in d.get('calibration_selfloop', {}).items():
        for L, t in by.items():
            base = f'rPThreeCalSelf{g}Tess{word[int(L)]}'
            put(base + 'Est', f3(t['estimate'], sign=True)); put(base + 'Pos', t['units_positive'])
            put(base + 'N', t['n_units'])
    for g, by in d['calibration'].items():
        for e, tests in by.items():
            for k, t in tests.items():
                base = f'rPThreeCal{g}{enc[e]}{cp[k]}'
                put(base + 'Est', f3(t['estimate'], sign=True)); put(base + 'Pos', t['units_positive'])
                put(base + 'N', t['n_units'])
    other = [t['propagation']['estimate'] for g in ('Bcc', 'HerAll', 'HerTumor') for t in d['calibration'].get(g, {}).values()]
    if other:   # BCC cell types and the two HER2ST targets: largest distance of an encoder mean from its propagation
        put('rPThreeCalOtherMaxAbs', f3(max(abs(v) for v in other)))


def secondary():
    path = R / 'W7' / 'summary.json'
    if not path.exists():
        return
    s = json.loads(path.read_text())
    lf, mg = s['labelfree'], s['magnitude']
    t = read_csv(R / 'W7' / 'labelfree.csv')
    put('rSecFreeSmoothed', lf['free_smoothed_targets']); put('rSecFreeTargets', lf['n_targets'])
    put('rSecFreeRegret', f3(lf['mean_regret']['label-free rule']))
    put('rSecFreeRegretWu', f3(lf['by_cohort']['wu_breast']['label-free rule']))
    put('rSecFreeKappaMin', f2(t.kappa_self_free.min())); put('rSecFreeKappaMax', f2(t.kappa_self_free.max()))
    put('rSecFreeGainMax', f3(t.info_gain_max_free.max(), sign=True))
    put('rSecMagTrain', mg['n_train'])
    for name, short in (('model', 'Model'), ('provenance class', 'Class'), ('K and class balance', 'Kb')):
        put(f'rSecMag{short}Mae', f2(mg[name]['mae'])); put(f'rSecMag{short}Rho', f2(mg[name]['spearman']))


def checks():
    """Numbers requested by the independent critique: unit-level uncertainty of the calibration correlations,
    within-class correlations, the fallback threshold on the calibration targets, the construction-scale
    positive control, P1 ties, the breast cohort, the mechanism exception and single-patient Leiden effects."""
    t = read_csv(R / 'W1' / 'targets.csv')
    rng = np.random.default_rng(0)
    units = t.unit.unique()
    for name, col in [('Hadj', 'h_adj6'), ('Li', 'li6'), ('Gain', 'info_gain_max'), ('Kappa', 'kappa_self')]:
        m = t[['unit', col, 'g_star']].dropna()
        put(f'rCalRho{name}N', len(m))
        boots = []
        for _ in range(2000):
            pick = rng.choice(units, len(units), replace=True)
            b = pd.concat([m[m.unit == u] for u in pick])
            if b[col].nunique() > 1:
                boots.append(stats.spearmanr(b[col], b.g_star).statistic)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        put(f'rCalRho{name}Lo', f2(lo)); put(f'rCalRho{name}Hi', f2(hi))
    put('rCalUnits', len(units))
    for prov, short in (('pathologist', 'Path'), ('expression-assigned', 'Expr')):
        g = t[t.provenance == prov][['h_adj6', 'g_star']].dropna()
        put(f'rCalWithin{short}Rho', f2(stats.spearmanr(g.h_adj6, g.g_star).statistic)); put(f'rCalWithin{short}N', len(g))
    put('rCalHadjCorrect', int(((t.h_adj6 > 0.5) == (t.g_star > 0)).sum()))
    hc = t.groupby('provenance').h_adj6.agg(['min', 'max'])
    put('rCalExprHadjMin', f2(hc.loc['expression-assigned', 'min'])); put('rCalExprHadjMax', f2(hc.loc['expression-assigned', 'max']))
    put('rCalPathHadjMin', f2(hc.loc['pathologist', 'min'])); put('rCalPathHadjMax', f2(hc.loc['pathologist', 'max']))
    # construction scale: positive control
    b = read_csv(R / 'W1b' / 'targets.csv')
    rb = b.pivot_table(index='unit', columns='k_c', values='ari_rebuilt_vs_source')
    put('rPTwoRebuiltMin', f2(rb[200].min())); put('rPTwoRebuiltMax', f2(rb[200].max()))
    put('rPTwoRebuiltPeakHundred', int((rb.idxmax(axis=1) == 100).sum()))
    ratio = b[b.k_c == 10].ell_k / 10
    put('rPTwoEllRatioMin', f'{ratio.min():.1f}'); put('rPTwoEllRatioMax', f'{ratio.max():.1f}')
    put('rPTwoEllCensoredHundred', int(b[(b.k_c == 100)].ell_censored.astype(bool).sum()))
    mg = b.pivot_table(index='unit', columns='k_c', values='tessera_minus_sgm')
    put('rPTwoMarginPatientMin', f3(mg.values.min(), sign=True)); put('rPTwoMarginPatientMax', f3(mg.values.max(), sign=True))
    sesoi = json.loads((R / 'freeze' / 'freeze_numbers.json').read_text())['sesoi']['NSCLC|primary']['sesoi']
    put('rPTwoAboveSesoiLarge', int(((mg[200] > sesoi) & (mg[400] > sesoi)).sum()))
    low = mg.mean(axis=1).idxmin()
    put('rPTwoLowPatientMean', f3(mg.loc[low].mean(), sign=True))
    gs = b.groupby('k_c').g_star.mean()
    put('rPTwoGainPeakKc', int(gs.idxmax())); put('rPTwoGainLargestKc', f3(gs.loc[gs.index.max()]))
    put('rPTwoKcMax', int(gs.index.max()))
    # P1: units better, worse and tied against never smoothing
    w6 = read_csv(R / 'W6' / 'targets.csv')
    new = w6[w6.cohort.isin(['spinal', 'colorectal', 'wu_breast'])].reset_index(drop=True)
    reg = pf.regret_table(new); reg['dataset'] = new.dataset.values
    u = reg.groupby('dataset')[['rule', 'no smoothing']].mean()
    diff = u['rule'] - u['no smoothing']
    put('rPOneWorseNone', int((diff > 1e-12).sum())); put('rPOneTiedNone', int((diff.abs() <= 1e-12).sum()))
    put('rPOneLosingUnits', int((u['no smoothing'] <= 1e-12).sum()))
    wu = new[new.cohort == 'wu_breast']
    put('rWuRhoGainGstar', f2(stats.spearmanr(wu.info_gain_max, wu.g_star).statistic, sign=True))
    put('rWuRhoEllKstar', f2(stats.spearmanr(wu.ell_k, wu.k_star).statistic, sign=True))
    put('rWuEllCensored', int(wu.ell_censored.astype(bool).sum()))
    put('rNewCrcHadjNearest', f3(new[new.cohort == 'colorectal'].h_adj6.max()))
    # mechanism: recovery by the variant without message passing on every target with a gain
    w3 = read_csv(R / 'W3' / 'scores.csv')
    cn = w3[(w3.rule == 'kmeans') & (w3.arm == 'ContrastNoMP')].groupby(['dataset', 'target']).ari.mean()
    tt = t.set_index(['dataset', 'target'])
    gain = (tt.ari_tessera - tt.ari_pca50)
    idx = [i for i in gain.index if gain[i] > 0.02 and i in cn.index]
    rec = pd.Series({i: (cn[i] - tt.ari_pca50[i]) / gain[i] for i in idx})
    put('rCnGainTargets', len(idx)); put('rCnRecoverHalf', int((rec >= 0.5).sum()))
    worst = rec.idxmin()
    put('rCnExceptionRecover', f'{100 * rec.min():.0f}'); put('rCnExceptionGain', f3(gain[worst]))
    put('rCnExceptionTess', f3(tt.ari_tessera[worst])); put('rCnExceptionVariant', f3(cn[worst]))
    rank = list(gain[idx].sort_values(ascending=False).index).index(worst) + 1
    put('rCnExceptionRank', {1: 'largest', 2: 'second-largest', 3: 'third-largest'}.get(rank, f'{rank}th-largest'))
    top = gain[idx].idxmax()
    put('rCnTopGain', f3(gain[top])); put('rCnTopRecover', f'{100 * rec[top]:.0f}')
    # lung cell types: the saved Tessera-ST fits against self-inclusive three-step propagation (W8, k = 6)
    w8p = R / 'W8' / 'selfloop.csv'
    if w8p.exists():
        w8 = read_csv(w8p)
        s3 = w8[(w8.k_graph == 6) & (w8.L == 3)].groupby(['dataset', 'target']).ari.mean()
        lung_ct = [d for d in tt.index if d[0].startswith('NSCLC') and d[1] == 'celltype']
        dd = pd.Series({d: tt.ari_tessera[d] - s3[(d[0], 'cell_type')] for d in lung_ct})
        put('rLungCtSelfMin', f3(dd.min(), sign=True)); put('rLungCtSelfMax', f3(dd.max(), sign=True))
        put('rLungCtPcaBest', int(sum(tt.ari_pca50[d] > max(tt.ari_tessera[d], s3[(d[0], 'cell_type')]) for d in lung_ct)))
    # Leiden, rule-matched selection: single-patient effects
    s = read_csv(R / 'W2' / 'selection.csv')
    def sel(cohort, target, rule):
        g = s[(s.cohort == cohort) & (s.target == target) & (s.select_rule == rule) & (s.score_rule == rule)]
        return g.set_index('unit').tessera_minus_control
    bcc = sel('BCC', 'primary', 'leiden')
    put('rLeidenBccLLMedian', f3(bcc.median(), sign=True)); put('rLeidenBccLLTop', f3(bcc.max(), sign=True))
    tum = sel('HER2ST', 'boundary_T1', 'leiden')
    put('rLeidenTumorLLTop', f3(tum.max(), sign=True)); put('rLeidenTumorLLRest', f3(tum.drop(tum.idxmax()).mean(), sign=True))
    tk = sel('HER2ST', 'boundary_T1', 'kmeans')
    put('rLeidenTumorKKMedian', f3(tk.median(), sign=True)); put('rLeidenTumorKKAbove', int((tk > 0).sum()))


def absolute():
    """Absolute ARI in the new cohorts (supplementary table of absolute agreement)."""
    w6 = read_csv(R / 'W6' / 'targets.csv')
    up = R / 'summary' / 'p3_units.csv'
    if not up.exists():
        return
    u = read_csv(up)
    for cohort, short in (('spinal', 'Spinal'), ('colorectal', 'Crc'), ('wu_breast', 'Wu')):
        w = w6[w6.cohort == cohort]
        put(f'rAbs{short}PcaMin', f3(w.ari_pca50.min())); put(f'rAbs{short}PcaMax', f3(w.ari_pca50.max()))
    sp = w6[w6.cohort == 'spinal'].groupby('target').ari_pca50.mean()
    put('rAbsSpinalPcaLOne', f3(sp['celltype_l1'])); put('rAbsSpinalPcaLTwo', f3(sp['celltype_l2']))
    enc = u[u.cohort == 'spinal'].groupby(['encoder', 'target']).encoder_ari.mean()
    put('rAbsSpinalEncMin', f3(enc.min())); put('rAbsSpinalEncMax', f3(enc.max()))
    wu = w6[w6.cohort == 'wu_breast']
    put('rAbsWuNearChance', wu[wu.ari_pca50 < 0.05].unit.nunique())


def native():
    """Native minus shared input for GraphST and STAGATE (K-means; supplementary table of native input)."""
    s = read_csv(R / 'W4' / 'scores.csv')
    s = s[(s.arm == 'released') & (s.rule == 'kmeans')].groupby(
        ['encoder', 'cohort', 'dataset', 'target', 'config', 'input_mode', 'seed'], as_index=False).ari.mean()
    u = s.groupby(['encoder', 'cohort', 'dataset', 'target', 'config', 'input_mode']).ari.mean().unstack('input_mode')
    u = u.dropna(subset=['native'])
    if u.empty:
        return
    d = (u.native - u.common).groupby(level=['encoder', 'cohort', 'target'])
    for (enc, cohort, target), short in ((('graphst', 'HER2ST', 'all'), 'GraphHerAll'),
                                         (('stagate', 'HER2ST', 'boundary_T1'), 'StagHerTumor')):
        g = d.get_group((enc, cohort, target))
        put(f'rNative{short}Est', f3(g.mean(), sign=True)); put(f'rNative{short}N', len(g))
        put(f'rNative{short}Pos', int((g > 0).sum())); put(f'rNative{short}Max', f3(g.max(), sign=True))
    put('rNativeMaxAbsMean', f3(d.mean().abs().max()))


def gpu_repeats():
    path = R / 'W4' / 'gpu_repeats' / 'gpu_repeats.csv'
    if not path.exists():
        return
    g = read_csv(path).groupby(['dataset', 'target', 'seed']).ari.agg(lambda a: a.max() - a.min())
    put('rGpuRepN', len(g)); put('rGpuRepDatasets', g.index.get_level_values(0).nunique())
    put('rGpuRepMedian', f'{g.median():.4f}'); put('rGpuRepMax', f3(g.max()))
    put('rGpuRepWithin', int((g <= 0.02).sum())); put('rGpuRepOver', int((g > 0.05).sum()))
    put('rGpuRepAboveTwo', int((g > 0.02).sum())); put('rGpuRepAboveTwoMin', f3(g[g > 0.02].min()))
    put('rGpuRepOverHer', int((g[g > 0.05].index.get_level_values(0).str.startswith('HER2ST')).sum()))


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
    for step in (calibration, construction, leiden, noise, mechanism, prospective, attribution, attribution_descriptive,
                 secondary, checks, gpu_repeats, absolute, native, frozen):
        step()
    lines = ['% Generated by revision-r3/make_macros_r3.py from results/revision-r3; do not edit by hand.']
    lines += [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(M.items())]
    out = R / 'summary' / 'revision-r3-macros.tex'
    out.write_text('\n'.join(lines) + '\n')
    print(f'{len(M)} macros -> {out}')


if __name__ == '__main__':
    main()

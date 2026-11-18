"""Checks for the round-3 main figures; run after drawing them:

  /home/zeyufu/miniconda3/envs/dl/bin/python figures_r3.py
  /home/zeyufu/miniconda3/envs/dl/bin/python -m pytest -q test_figures_r3.py

Values: the plot-data record written at draw time is compared with values recomputed here from
the result tables and the summary record (the frozen regret function is the only shared code),
and every number printed on a figure is found in the PDF text and equals the manuscript macro.
Presentation: seven figures as vector PDF and 300-dpi PNG, 180 mm wide, Arial only and embedded,
no unrenamed method label, no overlapping text; the encoder figure is checked once summary.json
has P3 and is skipped before. Captions: one complete figure environment per command,
self-contained, under about 130 words, and the caption file compiles with the macros and the
figures as drawn.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import functools  # noqa: E402
read_csv = functools.partial(pd.read_csv, float_precision="round_trip")  # pandas' default float parser segfaults on some 17-digit values
import pytest
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figures_r3 as F  # noqa: E402  (paths and names only; expected values are recomputed below)
import prepare_freeze as pf  # noqa: E402  frozen regret definition

R3 = F.R3
OUT = F.OUT
SOURCE = OUT.parent   # the manuscript source folder that holds figures-r3/ and figures-r3.tex
CAPTIONS = SOURCE / 'figures-r3.tex'
WIDTH_PT = 180 / 25.4 * 72
MINUS = '\u2212'
LUNG = F.LUNG
NEW = ('spinal', 'colorectal', 'wu_breast')
COMMANDS = {'FigureConcept': ('fig-concept', 'fig:concept'), 'FigureDiagnostics': ('fig-diagnostics', 'fig:diagnostics'),
            'FigureConstruction': ('fig-construction', 'fig:construction'),
            'FigureProspective': ('fig-prospective', 'fig:prospective'),
            'FigureMechanism': ('fig-mechanism', 'fig:mechanism'), 'FigureEncoders': ('fig-encoders', 'fig:encoders'),
            'FigureNoise': ('fig-noise', 'fig:noise')}
ENCODERS_READY = 'P3' in json.loads((R3 / 'summary/summary.json').read_text())  # else fig-encoders is skipped


# ------------------------------------------------------------------ helpers
def drawn(name):
    if name == 'fig-encoders' and not ENCODERS_READY:
        pytest.skip('summary.json has no P3 yet, so the encoder figure is not drawn')


def signed(value):
    """The macros' signed three-decimal format: no negative zero and no sign on zero."""
    value = 0.0 if round(value, 3) == 0 else value
    return (f'{value:+.3f}' if value != 0 else f'{value:.3f}').replace('-', MINUS)


def close(a, b, atol=1e-12):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.allclose(a, b, rtol=0, atol=atol, equal_nan=True), np.nanmax(np.abs(a - b))


def pdf_text(name):
    out = subprocess.run(['pdftotext', '-enc', 'UTF-8', str(OUT / f'{name}.pdf'), '-'], check=True,
                         capture_output=True, text=True).stdout
    return re.sub(r'\s+', ' ', out)


def csv(relative):
    return read_csv(R3 / relative)


@pytest.fixture(scope='module')
def plot():
    path = F.QA / 'plot-data.json'
    assert path.exists(), f'draw the figures first: {path} is missing'
    return json.loads(path.read_text())


@pytest.fixture(scope='module')
def macros():
    text = (R3 / 'summary/revision-r3-macros.tex').read_text()
    return {k: v.replace('\\ensuremath{-}', MINUS) for k, v in
            re.findall(r'^\\newcommand\{\\(\w+)\}\{(.*)\}$', text, re.M)}


@pytest.fixture(scope='module')
def summary():
    return json.loads((R3 / 'summary/summary.json').read_text())


# ------------------------------------------------------------ presentation
@pytest.mark.parametrize('name', F.NAMES)
def test_files_and_geometry(name):
    from PIL import Image
    drawn(name)
    info = subprocess.run(['pdfinfo', str(OUT / f'{name}.pdf')], check=True, capture_output=True, text=True).stdout
    width, height = map(float, re.search(r'Page size:\s+([\d.]+) x ([\d.]+) pts', info).groups())
    assert abs(width - WIDTH_PT) < .05 and 150 < height < 640, (name, width, height)
    assert re.search(r'Pages:\s+1\b', info), name
    with Image.open(OUT / f'{name}.png') as im:
        assert tuple(round(v) for v in im.info['dpi']) == (300, 300), (name, im.info.get('dpi'))
        assert abs(im.width - WIDTH_PT / 72 * 300) <= 1, (name, im.width)
        assert abs(im.height - height / 72 * 300) <= 1, (name, im.height)


@pytest.mark.parametrize('name', F.NAMES)
def test_fonts_are_embedded_arial(name):
    drawn(name)
    out = subprocess.run(['pdffonts', str(OUT / f'{name}.pdf')], check=True, text=True, capture_output=True).stdout
    rows = [line.split() for line in out.splitlines()[2:] if line.strip()]
    assert rows, name
    for row in rows:
        assert row[0].split('+')[-1] in {'ArialMT', 'Arial-BoldMT'}, (name, row)
        assert row[-5] == 'yes', (name, row)  # embedded


@pytest.mark.parametrize('name', F.NAMES)
def test_text_rules(name, plot):
    drawn(name)
    text = pdf_text(name)
    assert not re.search(r'Tessera(?!-ST)', text), name
    assert not re.search(r'Neighbou?r mean|[Ss]ix-neighbou?r|\bMean 6\b|_|\$', text), name
    audit = plot['audit'][name]
    assert audit['overlapping_text_runs'] == 0 and audit['text_outside_canvas'] == 0, audit
    assert audit['min_text_points'] >= 7 and audit['letters'] == sorted(audit['letters'])
    assert audit['letters'] == [chr(97 + i) for i in range(len(audit['letters']))], audit['letters']


# ----------------------------------------------------------- fig-diagnostics
def test_diagnostics_values(plot, macros):
    cal = csv('W1/targets.csv')
    new = csv('W6/targets.csv')
    new = new[new.cohort.isin(NEW)]
    p = plot['plot_data']['fig-diagnostics']
    assert p['a']['calibration']['n_lines'] == len(cal) == int(macros['rCalTargets'])
    assert p['a']['new']['n_lines'] == len(new) == int(macros['rPOneTargets'])
    both = pd.concat([cal, new], ignore_index=True)
    finite = both[np.isfinite(both.kappa_self)]
    assert p['b']['n_points'] == len(finite) and p['b']['n_calibration'] == int(np.isfinite(cal.kappa_self).sum())
    close(p['b']['h_adj6'], finite.h_adj6)
    close(p['b']['kappa'], finite.kappa_self)
    assert p['c']['n_points'] == len(both) == 81
    close(p['c']['h_adj6'], both.h_adj6)
    close(p['c']['g_star'], both.g_star)
    rho = stats.spearmanr(cal.h_adj6, cal.g_star).statistic
    assert f'{rho:.2f}' == p['c']['rho_printed'] == macros['rCalRhoHadj']
    assert f"ρ = {macros['rCalRhoHadj']}" in pdf_text('fig-diagnostics')
    assert p['c']['threshold'] == 0.5
    diag = pd.concat([csv('W1/diagnostics.csv'), csv('W6/diagnostics.csv')]).set_index(['dataset', 'target']).sort_index()
    for key, curve in p['curves'].items():
        dataset, target = key.rsplit('|', 1)
        rows = diag.loc[(dataset, target)].sort_values('k')
        rows = rows[np.isfinite(rows.h_adj)]
        assert curve['k'] == rows.k.tolist(), key
        close(curve['h_adj'], rows.h_adj)
        six = both[(both.dataset == dataset) & (both.target == target)].h_adj6.item()
        assert abs(curve['h_adj'][0] - six) < 1e-12 and curve['k'][0] == 6, key
    lung = cal[cal.cohort == 'NSCLC'].set_index(['dataset', 'target'])
    for metric in ('h_adj6', 'kappa_self', 'g_star'):
        expected = [[lung.loc[(u, 'primary'), metric], lung.loc[(u, 'celltype'), metric]] for u in LUNG]
        close(p['d'][metric], expected)
    region = np.array(p['d']['h_adj6'])[:, 0]
    assert f'{region.min():.2f}' == macros['rLungRegHadjMin'] and f'{region.max():.2f}' == macros['rLungRegHadjMax']


# ---------------------------------------------------------- fig-construction
def test_construction_values(plot, macros, summary):
    t = csv('W1b/targets.csv')
    p = plot['plot_data']['fig-construction']
    kcs = sorted(t.k_c.unique().tolist())
    assert p['k_c'] == kcs == [10, 25, 50, 100, 200, 400]
    for u in LUNG:
        g = t[t.unit == u].sort_values('k_c')
        close(p['a'][u], g.ell_k)
        assert p['a_censored'][u] == g.ell_censored.astype(bool).tolist()
        close(p['b'][u], g.k_star)
        close(p['c'][u], g.info_gain_k)
        close(p['e']['patients'][u], g.tessera_minus_sgm)
    mean = t.groupby('k_c').tessera_minus_sgm.mean().loc[kcs]
    close(p['e']['mean'], mean)
    close(p['e']['mean'], [summary['P2']['margin_by_kc'][str(k)] for k in kcs])
    assert f'{mean.min():+.3f}'.replace('-', MINUS) == macros['rPTwoMarginMin']
    assert f'{mean.max():+.3f}'.replace('-', MINUS) == macros['rPTwoMarginMax']
    band = summary['P2']['sesoi']
    assert p['e']['band'] == band == summary['freeze']['sesoi']['NSCLC|primary']['sesoi']
    assert p['e']['band_printed'] == '±0.029' == f'±{band:.3f}' and '±0.029' in pdf_text('fig-construction')
    grid = [6, 10, 15, 30, 50, 100, 200]
    curves = t.groupby('k_c')[[f'ari_mean{k}' for k in grid]].mean().loc[kcs].to_numpy()
    close(p['d']['mean_ari'], curves)
    assert p['d']['peak_k'] == [grid[i] for i in curves.argmax(1)]
    peaks = t.groupby('k_c').info_gain_k.median()
    assert all(peaks[k] == summary['P2']['gain_peak_k'][str(k)] for k in kcs)
    assert int(macros['rPTwoPeakMatches']) == int(macros['rPTwoScales']) == sum(peaks[k] == k or
                                                                                 abs(np.log2(peaks[k] / k)) <= .6
                                                                                 for k in kcs if k <= 200)


# ---------------------------------------------------------- fig-prospective
def test_prospective_values(plot, macros, summary):
    w6 = csv('W6/targets.csv')
    t = w6[w6.cohort.isin(NEW)].reset_index(drop=True)
    regret = pf.regret_table(t)
    regret['unit'], regret['cohort'] = t.dataset.values, t.cohort.values
    unit = regret.groupby(['cohort', 'unit']).mean(numeric_only=True)
    p = plot['plot_data']['fig-prospective']
    assert len(unit) == 20 == int(macros['rPOneUnits']) and len(t) == 36 == int(macros['rPOneTargets'])
    assert p['a']['units'] == [u for _, u in unit.index] and p['a']['cohort'] == [c for c, _ in unit.index]
    names = {'rule': 'Rule', 'no smoothing': 'None', 'k=6': 'Six', 'k=30': 'Thirty', 'k=200': 'TwoHundred'}
    text = pdf_text('fig-prospective')
    for c, short in names.items():
        close(p['a']['regret'][c], unit[c])
        assert abs(p['a']['mean'][c] - summary['P1']['mean_regret'][c]) < 1e-12
        assert p['a']['printed'][c] == f'{unit[c].mean():.3f}' == macros[f'rPOneRegret{short}']
        assert macros[f'rPOneRegret{short}'] in text
        for k in NEW:
            assert abs(p['c'][k][c] - summary['P1']['by_cohort'][k][c]) < 5e-5, (k, c)
    for k, short, word in (('spinal', 'Spinal', 'mice'), ('colorectal', 'Crc', 'patients'), ('wu_breast', 'Wu', 'patients')):
        assert p['units_per_cohort'][k] == int(macros[f'rNew{short}Units'])
        assert f"({macros[f'rNew{short}Units']} {word})" in text
    assert p['b']['n_points'] == 36
    close(p['b']['h_adj6'], t.h_adj6)
    close(p['b']['g_star'], t.g_star)
    assert p['b']['rule_choice'] == [pf.rule_choice(r) for r in t.itertuples()]
    correct = int(((t.h_adj6 > p['b']['threshold']) == (t.g_star > 0)).sum())
    assert correct == int(macros['rPOneHadjCorrect'])


# ------------------------------------------------------------ fig-mechanism
def test_mechanism_values(plot, macros):
    t = csv('W1/targets.csv').set_index(['dataset', 'target'])
    w3 = csv('W3/scores.csv')
    km = w3[w3.rule == 'kmeans']
    p = plot['plot_data']['fig-mechanism']
    index = [tuple(k.split('|')) for k in p['a']['targets']]
    assert len(index) == 45 and set(index) == set(t.index)
    no_mp = km[km.arm == 'ContrastNoMP'].groupby(['dataset', 'target']).ari.mean()
    assert (km[km.arm == 'ContrastNoMP'].groupby(['dataset', 'target']).size() == 3).all()
    close(p['a']['no_message_passing'], no_mp.loc[index])
    close(p['a']['tessera'], t.ari_tessera.loc[index])
    diff = t.ari_tessera.loc[index].to_numpy() - no_mp.loc[index].to_numpy()
    assert p['a']['within_002'] == int((np.abs(diff) <= .02).sum()) == int(macros['rCnWithin'])
    assert f"Within ±0.02: {macros['rCnWithin']} of {macros['rCnN']}" in pdf_text('fig-mechanism')
    prop = km[km.arm == 'Propagate'].groupby(['dataset', 'target', 'L']).ari.mean().unstack('L')
    prov = t.provenance.loc[index]
    for L in (1, 2, 3, 4, 8):
        d = t.ari_tessera.loc[index] - prop.loc[index, float(L)]
        close(p['b']['minus_propagation'][str(L)], d)
        for group, g in d.groupby(prov.values):
            i = [1, 2, 3, 4, 8].index(L)
            assert abs(p['b']['groups'][group]['median'][i] - g.median()) < 1e-12
            assert abs(p['b']['groups'][group]['q1'][i] - g.quantile(.25)) < 1e-12
            assert abs(p['b']['groups'][group]['q3'][i] - g.quantile(.75)) < 1e-12
            assert p['b']['groups'][group]['n'] == len(g)
        if L == 3:
            assert f'{d.median():+.3f}'.replace('-', MINUS) == macros['rPThreeMedian']
            assert int((d < 0).sum()) == int(macros['rPThreeBelow'])
    s = csv('W2/selection.csv')
    text = pdf_text('fig-mechanism')
    targets = {'NSCLC regions': ('NSCLC', 'primary', 'Lung'), 'BCC cell types': ('BCC', 'primary', 'Bcc'),
               'HER2ST all labels': ('HER2ST', 'all', 'Her'), 'HER2ST tumour/non-tumour': ('HER2ST', 'boundary_T1', 'Tumor')}
    pairings = {'K/K': ('kmeans', 'kmeans'), 'K/L': ('kmeans', 'leiden'), 'L/K': ('leiden', 'kmeans'),
                'L/L': ('leiden', 'leiden')}
    printed = []
    for title, (cohort, target, short) in targets.items():
        rec = p['c'][title]
        for label, (sel, sco) in pairings.items():
            g = s[(s.cohort == cohort) & (s.target == target) & (s.select_rule == sel) & (s.score_rule == sco)]
            g = g.set_index('unit').loc[rec['units']]
            close(rec['values'][label], g.tessera_minus_control)
            macro = macros[f'rLeiden{short}{label.replace("/", "")}']
            assert rec['printed'][label] == macro == f'{g.tessera_minus_control.mean():+.3f}'.replace('-', MINUS)
            assert int(macros[f'rLeiden{short}{label.replace("/", "")}Below']) == int((g.tessera_minus_control < 0).sum())
            printed.append(macro)
        assert len(rec['units']) == int(macros[f'rLeiden{short}N'])
        for clipped in rec['clipped']:
            assert clipped['value'] > .43 and clipped['printed'] == f"{clipped['value']:.2f}"
            assert clipped['printed'] in text
    for value in printed:
        assert value in text, value


# ------------------------------------------------------------- fig-encoders
P3_ROWS = [('tessera', 'propagation'), ('tessera', 'same_graph_mean'), ('graphst', 'propagation'),
           ('graphst', 'same_graph_mean'), ('stagate', 'same_graph_mean')]
P3_SHORT = {'region': 'Reg', 'celltype': 'Ct', 'tessera': 'Tess', 'graphst': 'Graph', 'stagate': 'Stag',
            'propagation': 'Prop', 'same_graph_mean': 'Mean'}
DEPTH = {'tessera': 3, 'graphst': 2, 'stagate': 1}
CALIBRATION = {'Lung regions': ('NSCLC', 'primary', 'LungReg'), 'Lung cell types': ('NSCLC', 'cell_type', 'LungCt'),
               'BCC cell types': ('BCC', 'primary', 'Bcc'), 'HER2ST all labels': ('HER2ST', 'all', 'HerAll'),
               'HER2ST tumour/non-tumour': ('HER2ST', 'boundary_T1', 'HerTumor')}


def test_encoders_values(plot, macros, summary):
    """Panel a against summary.json, the macros and p3_units.csv; panels b-d recomputed from the W4 tables."""
    drawn('fig-encoders')
    p = plot['plot_data']['fig-encoders']
    units = csv('summary/p3_units.csv')
    text = pdf_text('fig-encoders')
    for cls in ('region', 'celltype'):
        assert list(p['a'][cls]) == [f'{e}|{c}' for e, c in P3_ROWS], cls
        for enc, comp in P3_ROWS:
            rec, test = p['a'][cls][f'{enc}|{comp}'], summary['P3'][cls][f'{enc}|{comp}']
            unit = units[(units.encoder == enc) & (units.target_class == cls)].groupby(['cohort', 'dataset'])[
                f'minus_{comp}'].mean().dropna()
            assert rec['cohort'] == unit.index.get_level_values(0).tolist()
            assert rec['unit'] == unit.index.get_level_values(1).tolist()
            close(rec['values'], unit)
            half = stats.t.ppf(.975, len(unit) - 1) * unit.std(ddof=1) / np.sqrt(len(unit))  # t-interval over units
            sesoi = np.mean([summary['sesoi_new'][c] for c in unit.index.get_level_values(0)])  # unit-weighted
            close([test['estimate'], *test['ci'], test['sesoi']], [unit.mean(), unit.mean() - half, unit.mean() + half,
                                                                     sesoi])
            close([rec['estimate'], *rec['ci'], rec['sesoi']], [test['estimate'], *test['ci'], test['sesoi']])
            base = f'rPThree{P3_SHORT[cls]}{P3_SHORT[enc]}{P3_SHORT[comp]}'
            assert rec['macro'] == base + 'Est' and rec['printed'] == signed(test['estimate']) == macros[base + 'Est']
            assert signed(rec['ci'][0]) == macros[base + 'Lo'] and signed(rec['ci'][1]) == macros[base + 'Hi']
            assert len(unit) == int(macros[base + 'N']) and int((unit > 0).sum()) == int(macros[base + 'Pos'])
            assert rec['printed'] in text, rec['printed']
        stagate = summary['P3'][cls]['stagate|propagation']  # depth one: the same operator as the same-graph mean
        assert stagate['estimate'] == summary['P3'][cls]['stagate|same_graph_mean']['estimate']

    # b-d: seeds averaged per configuration, joined to each unit's selected configuration
    s = csv('W4/scores.csv')
    fit = s[(s.rule == 'kmeans') & (s.input_mode == 'common')].groupby(
        ['encoder', 'dataset', 'target', 'config', 'arm'], as_index=False).ari.mean()
    cp = csv('W4/counterparts.csv')
    prop = cp[cp.rule == 'kmeans'].groupby(['encoder', 'dataset', 'target', 'config', 'L'], as_index=False).ari.mean()
    sel = csv('W4/selection.csv').rename(columns={'selected': 'config'})
    keys = ['encoder', 'dataset', 'target', 'config']
    base = sel.merge(fit[fit.arm == 'released'].drop(columns='arm'), on=keys)
    cls_of = units.set_index(['encoder', 'dataset', 'target']).target_class
    new = base[base.cohort.isin(NEW)].copy()
    new['cls'] = [cls_of[k] for k in zip(new.encoder, new.dataset, new.target)]
    depth = new.merge(prop, on=keys, suffixes=('', '_prop'))
    depth['diff'] = depth.ari - depth.ari_prop
    for enc in DEPTH:
        Ls = sorted({1, 2, 4, 8, DEPTH[enc]})
        assert p['b'][enc]['L'] == Ls
        for cls in ('region', 'celltype'):
            g = depth[(depth.encoder == enc) & (depth.cls == cls)]
            unit = g.groupby(['cohort', 'dataset', 'L'])['diff'].mean().unstack('L')[Ls]
            assert p['b'][enc][cls]['n'] == len(unit) > 0, (enc, cls)
            close(p['b'][enc][cls]['median'], unit.median())
            close(p['b'][enc][cls]['q1'], unit.quantile(.25))
            close(p['b'][enc][cls]['q3'], unit.quantile(.75))
    arms = new.merge(fit[fit.arm != 'released'], on=keys, suffixes=('', '_arm'))
    arms['diff'] = arms.ari_arm - arms.ari
    described = summary.get('P3_descriptive', {}).get('arms', {})
    expected_arms = {'tessera': ['no_gate', 'no_contrast', 'self_only', 'permuted_xy_no_contrast', 'contrast_no_mp'],
                     'graphst': ['no_contrast', 'identity_encoder', 'permuted_xy'],
                     'stagate': ['uniform_attention', 'identity_graph', 'permuted_xy']}
    for enc, names in expected_arms.items():
        assert list(p['c'][enc]) == names, enc
        for arm in names:
            rec = p['c'][enc][arm]
            unit = arms[(arms.encoder == enc) & (arms.arm == arm)].groupby(['cls', 'cohort', 'dataset'])['diff'].mean()
            assert rec['group'] == unit.index.get_level_values(0).tolist()
            assert rec['unit'] == unit.index.get_level_values(2).tolist()
            close(rec['values'], unit)
            for cls in ('region', 'celltype'):
                assert abs(rec['mean'][cls] - unit.loc[cls].mean()) < 1e-12, (enc, arm, cls)  # over units, as in a
                if described:  # its pooled estimate weights cohorts equally; cohort means and counts must agree
                    t = described[cls][enc][arm]
                    assert t['n_units'] == len(unit.loc[cls]) and t['units_positive'] == int((unit.loc[cls] > 0).sum())
                    for cohort, m in t['cohort_means'].items():
                        assert abs(unit.loc[cls].loc[cohort].mean() - m) < 1e-12, (enc, arm, cls, cohort)
    calibration = summary.get('P3_descriptive', {}).get('calibration', {})
    cal = base.merge(prop, on=keys, suffixes=('', '_prop'))
    assert set(p['d']) == {'graphst', 'stagate'}  # Tessera-ST was not refitted on the calibration cohorts
    for title, (cohort, target, group) in CALIBRATION.items():
        for enc in ('graphst', 'stagate'):
            g = cal[(cal.cohort == cohort) & (cal.target == target) & (cal.encoder == enc) & (cal.L == DEPTH[enc])]
            expected = (g.ari - g.ari_prop).set_axis(g.dataset)  # one patient per dataset
            rec = p['d'][enc][title]
            assert sorted(rec['unit']) == sorted(expected.index) and expected.index.is_unique, (title, enc)
            close(pd.Series(rec['values'], index=rec['unit']).loc[expected.index], expected)
            assert abs(rec['mean'] - expected.mean()) < 1e-12
            if calibration:
                assert abs(calibration[group][enc]['propagation']['estimate'] - rec['mean']) < 1e-12, (title, enc)
    assert p['d_provenance'] == {'Lung regions': 'computed-niche', 'Lung cell types': 'expression-assigned',
                                 'BCC cell types': 'expression-assigned', 'HER2ST all labels': 'pathologist',
                                 'HER2ST tumour/non-tumour': 'pathologist'}


def test_encoders_skipped_without_p3(tmp_path):
    """Before the three-encoder results are in, the build skips fig-encoders and still succeeds."""
    (tmp_path / 'res/summary').mkdir(parents=True)
    (tmp_path / 'res/summary/summary.json').write_text('{}\n')
    env = dict(os.environ, TESSERA_R3_OUT=str(tmp_path / 'res'), R3_FIG_OUT=str(tmp_path / 'out'),
               R3_FIG_QA=str(tmp_path / 'qa'))
    run = subprocess.run([sys.executable, str(HERE / 'figures_r3.py'), 'fig-encoders'], env=env, cwd=tmp_path,
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-2000:]
    assert 'skipped (inputs not ready) fig-encoders' in run.stdout
    assert not (tmp_path / 'out/fig-encoders.pdf').exists()
    assert 'fig-encoders' not in json.loads((tmp_path / 'qa/plot-data.json').read_text()).get('plot_data', {})


# ---------------------------------------------------------------- fig-noise
def test_noise_values(plot, macros):
    s = csv('W2/seeds.csv')
    saved = json.loads((R3 / 'W1/per_seed.json').read_text())
    p = plot['plot_data']['fig-noise']
    assert len(p['a']['datasets']) == 13
    for n in p['a']['datasets']:
        close(p['a']['seeds'][n], s[s.dataset == n].sort_values('seed').ari)
        assert len(p['a']['seeds'][n]) == 10
    spread = s.groupby('dataset').ari.agg(['std', 'min', 'max'])
    assert p['a']['printed']['sd_median'] == f"{spread['std'].median():.3f}" == macros['rSeedSdMedian']
    assert p['a']['printed']['range_max'] == f"{(spread['max'] - spread['min']).max():.3f}" == macros['rSeedRangeMax']
    cpu = s[s.seed <= 3].sort_values(['dataset', 'seed'])
    gpu = [saved[f'{r.dataset}|primary|tessera'][r.seed - 1] for r in cpu.itertuples()]
    assert len(gpu) == 39
    close(p['b']['cpu'], cpu.ari)
    close(p['b']['gpu'], gpu)
    assert p['b']['printed'] == f'{np.median(np.abs(cpu.ari.to_numpy() - np.array(gpu))):.4f}' == macros['rCpuGpuMedian']
    text = pdf_text('fig-noise')
    for value in (macros['rSeedSdMedian'], macros['rSeedRangeMax'], macros['rCpuGpuMedian']):
        assert value in text
    assert p['panels'] == (3 if F.GPU_REPEATS.exists() else 2)
    assert (OUT / F.NOISE_PANEL_C).exists() == F.GPU_REPEATS.exists()


def test_noise_gains_panel_c_when_repeats_appear(tmp_path):
    """A same-seed GPU repeat table (synthetic here) makes the build add panel c and its caption marker."""
    rows = [dict(dataset=d, target=t, seed=s, repeat=r, ari=.2 + .001 * r + .01 * s)
            for d, targets in (('NSCLC_Lung6', ('primary', 'cell_type')), ('HER2ST-A1', ('all', 'compartments', 'boundary_T1')))
            for t in targets for s in (1, 2, 3) for r in (1, 2, 3)]
    table = tmp_path / 'gpu_repeats.csv'
    pd.DataFrame(rows).to_csv(table, index=False)
    env = dict(os.environ, R3_GPU_REPEATS=str(table), R3_FIG_OUT=str(tmp_path / 'out'), R3_FIG_QA=str(tmp_path / 'qa'))
    subprocess.run([sys.executable, str(HERE / 'figures_r3.py'), 'fig-noise'], check=True, env=env, cwd=tmp_path,
                   capture_output=True, text=True)
    record = json.loads((tmp_path / 'qa/plot-data.json').read_text())['plot_data']['fig-noise']
    assert record['panels'] == 3 and set(record['c']) == {'NSCLC regions', 'NSCLC cell types', 'HER2ST all labels',
                                                          'HER2ST compartments', 'HER2ST tumour/non-tumour'}
    assert all(abs(v - .002) < 1e-9 for values in record['c'].values() for v in values)
    assert (tmp_path / 'out' / F.NOISE_PANEL_C).exists()
    text = subprocess.run(['pdftotext', str(tmp_path / 'out/fig-noise.pdf'), '-'], check=True, capture_output=True,
                          text=True).stdout
    assert 'Same-seed GPU repeats' in text


# -------------------------------------------------------------- fig-concept
def test_concept_numbers(plot, macros):
    p = plot['plot_data']['fig-concept']['printed']
    assert (p['datasets'], p['targets'], p['units'], p['new_targets']) == (
        macros['rCalDatasets'], macros['rCalTargets'], macros['rPOneUnits'], macros['rPOneTargets'])
    text = pdf_text('fig-concept')
    assert f"{macros['rCalDatasets']} datasets, {macros['rCalTargets']} targets" in text
    assert f"{macros['rPOneUnits']} patients and animals, {macros['rPOneTargets']} targets" in text


# ------------------------------------------------------------------ captions
def caption_parts():
    src = CAPTIONS.read_text()
    out = {}
    for command in COMMANDS:
        m = re.search(r'^\\newcommand\{\\%s\}\{(.*)\}$' % command, src, re.M)
        assert m, command
        out[command] = m.group(1)
    return src, out


def caption_text(body):
    i = body.index('\\caption{') + len('\\caption{')
    depth, j = 1, i
    while depth:
        depth += {'{': 1, '}': -1}.get(body[j], 0)
        j += 1
    return body[i:j - 1], body[j:]


def words(text):
    text = re.sub(r'\\mbox\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\[A-Za-z]+\{\}', 'N', text)
    text = re.sub(r'\\[A-Za-z]+', ' ', text).replace('~', ' ')
    return [w for w in text.split() if re.search(r'[A-Za-z0-9]', w)]


def test_caption_environments():
    src, parts = caption_parts()
    for command, (stem, label) in COMMANDS.items():
        body = parts[command]
        assert body.startswith('\\begin{figure}[!htbp]\\centering\\includegraphics[width=\\linewidth]'
                               f'{{figures-r3/{stem}.pdf}}\\caption{{'), command
        caption, rest = caption_text(body)
        assert rest == f'\\label{{{label}}}\\end{{figure}}', (command, rest)
        assert body.count('{') == body.count('}'), command
        assert (OUT / f'{stem}.pdf').exists() or (stem == 'fig-encoders' and not ENCODERS_READY), stem
        n = len(words(caption))
        assert n <= 130, (command, n)
        assert not re.search(r'\\(cite|ref|eqref|autoref)|[Ss]ee Fig|Fig\.|Figure|\.pdf|\.csv|\.json|_', caption), command
        assert not re.search(r'\b(W1b?|W[2-6]|P[123]|SESOI|ContrastNoMP|h_adj|g_star|kappa_self|tessera_)\b', caption), command
        letters = list(dict.fromkeys(re.findall(r'\(([a-e])\)', caption)))  # first mention of each panel
        assert letters == sorted(letters), (command, letters)
    assert not re.search(r'^%\s*\\newcommand\{\\FigureEncoders\}', src, re.M)  # the template line is replaced


def test_caption_panels_match_figures(plot):
    _, parts = caption_parts()
    for command, (stem, _) in COMMANDS.items():
        if stem == 'fig-encoders' and not ENCODERS_READY:
            continue
        caption, _ = caption_text(parts[command])
        described = set(re.findall(r'\(([a-e])\)', caption))
        drawn = set(plot['audit'][stem]['letters'])
        if stem == 'fig-noise' and (OUT / F.NOISE_PANEL_C).exists():
            described.add('c')  # described by the conditional sentence defined above the figure commands
        assert described == drawn, (stem, drawn, described)


@pytest.mark.skipif(shutil.which('pdflatex') is None, reason='pdflatex is not installed')
def test_captions_compile(tmp_path):
    # figures-r3/ resolves to the folder the figures were drawn into, with its noise-panel marker
    (tmp_path / 'figures-r3').symlink_to(OUT, target_is_directory=True)
    doc = tmp_path / 'check.tex'
    commands = ''.join(f'\\{c}\\clearpage\n' for c, (stem, _) in COMMANDS.items()
                       if stem != 'fig-encoders' or ENCODERS_READY)
    doc.write_text('\\documentclass{article}\\usepackage[a4paper,margin=15mm]{geometry}\\usepackage{graphicx}\n'
                   f'\\input{{{R3 / "summary/revision-r3-macros.tex"}}}\n\\input{{{CAPTIONS}}}\n'
                   f'\\begin{{document}}\n{commands}\\end{{document}}\n')
    result = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', f'-output-directory={tmp_path}',
                             str(doc)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:]
    log = (tmp_path / 'check.log').read_text(errors='replace')
    assert 'Undefined control sequence' not in log and 'Missing' not in log

#!/usr/bin/env python3
"""TeX macros and table rows for the 2026-10-10 follow-up analyses.

Every number cited from the follow-up experiments is generated here from the
saved JSON/CSV outputs in results/followup-20261010, never typed by hand.
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from common import ROOT, OBJECTS, read, dump, sha

F = ROOT / 'results/followup-20261010'
LED = ROOT / 'source/manuscript/ledgers'
LUNG, BCC = OBJECTS[:5], OBJECTS[5:9]
SHORT = {n: n.replace('NSCLC_', '').replace('BCC_Patient', 'BCC ').replace('Breast_Visium', 'Breast Visium')
         .replace('Colorectal_TVA232', 'Colorectal TVA').replace('OpenST_HNSCC26', 'Open-ST HNSCC') for n in OBJECTS}


def f3(v): return f'{v:+.3f}' if v < 0 or v > 0 else '0.000'
def u3(v): return f'{v:.3f}'
def tex(s): return str(s).replace('_', r'\_').replace('&', r'\&')


def sign_flip(values):
    values = np.asarray(values, float); observed = abs(values.mean())
    return float(np.mean([abs((values * np.array(s)).mean()) >= observed - 1e-12
                          for s in itertools.product([-1, 1], repeat=len(values))]))


def config_label(config):
    kind, k, lam = config.split('|')
    k = int(float(k)) if k not in ('nan', 'None') else None
    if kind == 'NeighborMean': return f'neighbour mean, $k={k}$'
    if kind == 'BANKSY-style': return f'BANKSY-style, $k={k}$, $\\lambda={float(lam):g}$'
    if kind == 'Composition': return 'composition, $k=200$'
    return 'PCA50'


def main():
    primary = pd.read_csv(LED / 'primary_summary.csv').pivot(index='object', columns='arm', values='ari_mean')
    scale = read(F / 'scale_controls_summary.json'); rescore = read(F / 'rescore_summary.json')
    nulls = read(F / 'geometry_nulls_summary.json'); whole = read(F / 'whole_sections_summary.json')
    transfer = read(F / 'transfer_neighbor_mean_summary.json')
    m = {}
    for cohort, names, tag in [('NSCLC', LUNG, 'Lung'), ('BCC', BCC, 'Bcc')]:
        vs_pca = primary.loc[names, 'Tessera'] - primary.loc[names, 'PCA50']
        m[f'{tag}VsPcaPositive'] = str(int((vs_pca > 0).sum()))
        m[f'{tag}VsPcaP'] = f'{sign_flip(vs_pca):.3g}'
        nested = scale[cohort]
        delta = [r['tessera_ari'] - r['nested_control_ari'] for r in nested['per_patient']]
        m[f'{tag}NestedDelta'] = f3(np.mean(delta))
        m[f'{tag}NestedTesseraBetter'] = str(nested['tessera_better'])
        m[f'{tag}NestedP'] = f"{nested['exact_sign_flip_p']:.3g}"
        m[f'{tag}NestedMin'] = f3(min(delta)); m[f'{tag}NestedMax'] = f3(max(delta))
        m[f'{tag}NestedControlMean'] = u3(np.mean([r['nested_control_ari'] for r in nested['per_patient']]))
        w = whole[cohort]
        m[f'{tag}WholeVsControlMean'] = f3(np.mean(w['tessera_minus_selected_control']))
        m[f'{tag}WholeVsControlBetter'] = str(int(sum(v > 0 for v in w['tessera_minus_selected_control'])))
        m[f'{tag}WholeVsControlP'] = f"{w['sign_flip_p_vs_control']:.3g}"
        m[f'{tag}WholeVsPcaMean'] = f3(np.mean(w['tessera_minus_pca50']))
        m[f'{tag}WholeVsPcaPositive'] = str(int(sum(v > 0 for v in w['tessera_minus_pca50'])))
        m[f'{tag}WholeVsPcaP'] = f"{w['sign_flip_p_vs_pca50']:.3g}"
    cells = whole['n_cells']
    m['WholeCellsMin'] = f'{min(cells.values()):,}'; m['WholeCellsMax'] = f'{max(cells.values()):,}'
    lt = rescore['lung_celltypes']
    m['CellTypeTessera'] = u3(np.mean(lt['tessera'])); m['CellTypePca'] = u3(np.mean(lt['pca50']))
    m['CellTypeMean'] = u3(np.mean(lt['neighbor_mean']))
    m['CellTypeTesseraLower'] = str(int(sum(t < p for t, p in zip(lt['tessera'], lt['pca50']))))
    gate = rescore['gate']
    m['GateBeaten'] = str(rescore['gate_beaten_by_expression_difference']); m['GateObjects'] = str(rescore['gate_objects'])
    au = [g['gate_auroc'] for g in gate.values()]; ex = [g['expression_difference_auroc'] for g in gate.values()]
    m['GateAurocMin'] = u3(min(au)); m['GateAurocMax'] = u3(max(au))
    m['GateExprAurocMedian'] = u3(np.median(ex)); m['GateAurocMedian'] = u3(np.median(au))
    med = [gate[n]['gate_median'] for n in LUNG]; low = [gate[n]['gate_fraction_below_0_05'] for n in LUNG]
    m['LungGateMedianMin'] = u3(min(med)); m['LungGateMedianMax'] = u3(max(med))
    m['LungGateLowMin'] = f'{100 * min(low):.0f}'; m['LungGateLowMax'] = f'{100 * max(low):.0f}'
    nogate = primary.loc[LUNG, 'NoGate'] - primary.loc[LUNG, 'Tessera']
    m['NoGateAtLeast'] = str(int((nogate >= -.0015).sum())); m['NoGateMean'] = f3(nogate.mean())
    ns = {n: v for n, v in nulls.items() if isinstance(v, dict)}
    m['BelowSelfOnly'] = str(nulls['tessera_minus_self_only_negative_objects'])
    lung_null = [ns[n]['permuted_no_contrast'] - ns[n]['self_only'] for n in LUNG]
    m['PermNoContrastMaxGap'] = u3(max(abs(v) for v in lung_null))
    nc_self = [abs(primary.loc[n, 'NoContrast'] - primary.loc[n, 'SelfOnly']) for n in LUNG]
    m['NoContrastSelfMaxGap'] = u3(max(nc_self))
    st = pd.read_csv(LED / 'seed_stability.csv').groupby('arm').seed_pair_ari_mean
    m['StabilityTesseraMin'] = f'{st.min()["Tessera"]:.2f}'; m['StabilityTesseraMax'] = f'{st.max()["Tessera"]:.2f}'
    m['StabilityPcaMin'] = f'{st.min()["PCA50"]:.2f}'
    for key, metric in [('Region', 'region_balanced_accuracy'), ('Cell', 'cell_type_balanced_accuracy')]:
        t = transfer[metric]
        for arm, tag in [('PCA50', 'Pca'), ('Tessera', 'Full'), ('NeighborMean-k6', 'MeanSix'), ('NeighborMean-k200', 'MeanTwoHundred')]:
            m[f'Transfer{key}{tag}'] = u3(np.mean(t[arm]))
        m[f'Transfer{key}TesseraBelowPca'] = str(int(sum(a < b for a, b in zip(t['Tessera'], t['PCA50']))))
    common = rescore['common_K']
    delta_common = dict(zip(common['objects'], common['tessera_minus_pca50']))
    for tag, names in [('Lung', LUNG), ('Bcc', BCC)]:
        m[f'{tag}CommonKBetter'] = str(sum(delta_common[n] > 0 for n in names))
        m[f'{tag}CommonKDelta'] = f3(np.mean([delta_common[n] for n in names]))
    weights = pd.read_csv(F / 'gate_weight_diagnostics-20261011.csv')
    wmean = weights.groupby('object').mean(numeric_only=True)
    m['LungWeightTvdMin'] = u3(wmean.loc[LUNG, 'tvd_median'].min())
    m['LungWeightTvdMax'] = u3(wmean.loc[LUNG, 'tvd_median'].max())
    macros = ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in sorted(m.items()))
    (LED / 'followup_numbers.tex').write_text(macros)
    # Supplementary table: nested selection, capped and whole-section results.
    rows = []
    for cohort, names in [('NSCLC', LUNG), ('BCC', BCC)]:
        per = {r['patient']: r for r in scale[cohort]['per_patient']}
        control = 'NeighborMean|200|None' if cohort == 'NSCLC' else 'BANKSY-style|100|0.2'
        for n in names:
            r = per[n]; w = whole['mean_ari'][n]
            rows.append(' & '.join([tex(SHORT[n]), config_label(r['selected']), u3(r['pca50_ari']), u3(r['nested_control_ari']),
                u3(r['tessera_ari']), f"{cells[n]:,}", u3(w['PCA50|None|None']), u3(w[control]), u3(w['Tessera'])]) + r'\\')
    (LED / 'followup_rows.tex').write_text('\n'.join(rows) + '\n')
    gate_rows = []
    for n in OBJECTS:
        g = gate[n]
        gate_rows.append(' & '.join([tex(SHORT[n]), u3(g['gate_auroc']), u3(g['expression_difference_auroc']),
            u3(g['gate_median']), f"{100 * g['gate_fraction_below_0_05']:.0f}"]) + r'\\')
    (LED / 'gate_rows.tex').write_text('\n'.join(gate_rows) + '\n')
    common_rows = []
    for i,n in enumerate(common['objects']):
        common_rows.append(' & '.join([tex(SHORT[n]), str(rescore['k_primary'][n]), str(rescore['k_common'][n]),
            u3(common['pca50'][i]), u3(common['neighbor_mean'][i]), u3(common['tessera'][i]), f3(delta_common[n])]) + r'\\')
    (LED / 'common_k_rows.tex').write_text('\n'.join(common_rows) + '\n')
    weight_rows = []
    for n in OBJECTS:
        w = wmean.loc[n]
        attenuation = weights[weights.object == n].epsilon_attenuation_max.max()
        weight_rows.append(' & '.join([tex(SHORT[n]), u3(w.tvd_median), u3(w.tvd_p90),
            u3(w.ess_fraction_median), f'{attenuation:.2g}']) + r'\\')
    (LED / 'gate_weight_rows.tex').write_text('\n'.join(weight_rows) + '\n')
    dump(LED / 'followup_numbers.json', {'macros': m, 'sources': {p.name: sha(p) for p in sorted(F.iterdir())
        if p.is_file() and p.suffix in {'.csv', '.json'}}})
    print(f'{len(m)} follow-up macros; {len(rows)} nested/whole rows; {len(gate_rows)} gate rows')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""TeX macros, table rows and figure tables for the independent-cohort analyses.

Every number cited from validation-external/ and validation-extension/ is
generated here from the saved JSON/CSV outputs, never typed by hand.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import run_experiments as rx
from common import ROOT, OBJECTS, read, dump, sha

EXT = ROOT / 'validation-extension'
EXTERNAL = ROOT / 'validation-external'
FOLLOW = ROOT / 'results/followup-20261010'
LED = ROOT / 'source/manuscript/ledgers'
LUNG, BCC, OTHER = OBJECTS[:5], OBJECTS[5:9], OBJECTS[9:]
HER_TARGETS = {'All': 'all', 'Comp': 'compartments', 'Bnd': 'boundary_T1', 'BndTwo': 'boundary_T2', 'BndThree': 'boundary_T3'}
GROUP_OF = {**{n: 'Computed regions' for n in LUNG}, **{n: 'Cell-type labels' for n in BCC + OTHER},
            'DLPFC_151673': 'Anatomical regions', 'STARmap_visual_cortex': 'Anatomical regions',
            'MERFISH_hypothalamus': 'Anatomical regions'}
SHORT_BREADTH = {'DLPFC_151673': 'Human cortex (Visium)', 'STARmap_visual_cortex': 'Mouse cortex (STARmap)',
                 'MERFISH_hypothalamus': 'Hypothalamus (MERFISH)'}
FIGURE_LABEL = {'DLPFC_151673': 'Human cortex', 'STARmap_visual_cortex': 'Mouse cortex', 'MERFISH_hypothalamus': 'Hypothalamus'}
SHORT = {n: n.replace('NSCLC_', '').replace('BCC_Patient', 'BCC ').replace('Breast_Visium', 'Breast Visium')
         .replace('Colorectal_TVA232', 'Colorectal TVA').replace('OpenST_HNSCC26', 'Open-ST HNSCC') for n in OBJECTS}


def f3(v):
    return f'{v:+.3f}' if v else '0.000'


def u3(v):
    return f'{v:.3f}'


def pv(v):
    return f'{v:.3g}'


def tex(s):
    return str(s).replace('_', r'\_').replace('&', r'\&')


def config_label(config):
    kind, k, lam = config.split('|')
    k = int(float(k)) if k not in ('nan', 'None') else None
    if kind == 'NeighborMean':
        return f'neighbor mean, $k={k}$'
    if kind == 'BANKSY-style':
        return f'BANKSY-style, $k={k}$, $\\lambda={float(lam):g}$'
    if kind == 'Composition':
        return 'composition, $k=200$'
    return 'PCA50'


def best_by_object(grid, units, column='unit'):
    means = grid.groupby([column, 'config']).ari.mean().unstack()
    best_any = {u: float(means.loc[u].max()) for u in units}
    uniform = [c for c in means.columns if str(c).startswith('NeighborMean')]
    best_k = {u: int(float(str(means.loc[u, uniform].idxmax()).split('|')[1])) for u in units}
    best_u = {u: float(means.loc[u, uniform].max()) for u in units}
    return best_any, best_k, best_u


def contiguity_from_partitions(partitions):
    """Reference-edge homogeneity on the six-neighbor graph, valid endpoints only."""
    out = {}
    for unit, frame in partitions.groupby('unit', sort=False):
        coords = frame[['x', 'y']].to_numpy(float)
        edge, _ = rx.graph(coords, np.full(len(frame), unit), 'Tessera', 1)
        ref = frame.reference.fillna('').astype(str).to_numpy()
        valid = (ref[edge[0]] != '') & (ref[edge[1]] != '')
        out[unit] = float((ref[edge[0]][valid] == ref[edge[1]][valid]).mean())
    return out


def landscape(hard, dev, breadth):
    """One row per object: the comparison table behind the tissue-landscape figure."""
    primary = pd.read_csv(LED / 'all_primary_scores.csv')
    means = primary.groupby(['object', 'arm']).ari.mean().unstack()
    contiguity = primary.groupby('object').reference_edge_homogeneity.mean()
    oracle = read(FOLLOW / 'scale_controls_summary.json')['oracle_best_control_by_object']
    scale = pd.read_csv(FOLLOW / 'scale_controls.csv')
    scale['config'] = scale.kind + '|' + scale.k.astype('Int64').astype(str).replace('<NA>', 'None') + '|' + scale.lam.astype(str).replace('nan', 'None')
    gate = pd.read_csv(FOLLOW / 'gate_baseline.csv').groupby('object')[['gate_auroc', 'expression_difference_auroc']].mean()
    rows = []
    nested = {r['patient']: r['nested_control_ari'] for c in ('NSCLC', 'BCC') for r in read(FOLLOW / 'scale_controls_summary.json')[c]['per_patient']}
    dev_uniform = scale[scale.kind == 'NeighborMean'].groupby(['object', 'k']).ari.mean().unstack()
    for name in OBJECTS:
        k_best = int(dev_uniform.loc[name].idxmax()) if name in dev_uniform.index else None
        rows.append({'object': name, 'label': SHORT[name], 'group': GROUP_OF[name], 'source': 'development',
                     'tessera': float(means.loc[name, 'Tessera']), 'same_graph_mean': float(means.loc[name, 'NeighborMean']),
                     'pca50': float(means.loc[name, 'PCA50']), 'oracle_control': float(oracle[name]['ari']),
                     'nested_control': nested.get(name, np.nan), 'gate_auroc': float(gate.loc[name, 'gate_auroc']),
                     'raw_difference_auroc': float(gate.loc[name, 'expression_difference_auroc']),
                     'smoothed_difference_auroc': np.nan, 'contiguity': float(contiguity[name]),
                     'best_uniform_k': k_best, 'best_uniform_ari': float(dev_uniform.loc[name].max())})
    grid = pd.read_csv(EXT / 'results/hardening/grid.csv')
    parts = pd.read_csv(EXT / 'results/hardening/partitions_compartments.csv')
    cont = contiguity_from_partitions(parts)
    seams = pd.read_csv(EXT / 'results/hardening/seams.csv')
    sub = grid[grid.target == 'all']
    units = sorted(sub.unit.unique())
    best_any, best_k, best_u = best_by_object(sub, units)
    per_unit = hard['targets']['all']['per_unit']
    seam_c = seams[seams.target == 'compartments'].groupby('unit')[['gate', 'raw_difference', 'smoothed_expression_weighted']].mean()
    for u in units:
        rows.append({'object': f'HER2ST_{u}', 'label': f'HER2ST {u[0]}', 'group': 'Pathologist regions', 'source': 'independent cohort',
                     'tessera': per_unit[u]['Tessera'], 'same_graph_mean': per_unit[u]['UniformSameGraph'],
                     'pca50': per_unit[u]['PCA50'], 'oracle_control': best_any[u], 'nested_control': per_unit[u]['NestedControl'],
                     'gate_auroc': float(seam_c.loc[u, 'gate']), 'raw_difference_auroc': float(seam_c.loc[u, 'raw_difference']),
                     'smoothed_difference_auroc': float(seam_c.loc[u, 'smoothed_expression_weighted']),
                     'contiguity': cont[u], 'best_uniform_k': best_k[u], 'best_uniform_ari': best_u[u]})
    bgrid = pd.read_csv(EXT / 'results/breadth/grid.csv')
    bunits = list(breadth['units'])
    best_any, best_k, best_u = best_by_object(bgrid, bunits)
    for u in bunits:
        v = breadth['per_unit'][u]
        s = breadth['seams'][u]
        rows.append({'object': u, 'label': SHORT_BREADTH[u], 'group': GROUP_OF[u], 'source': 'further tissue',
                     'tessera': v['Tessera'], 'same_graph_mean': v['UniformSameGraph'], 'pca50': v['PCA50'],
                     'oracle_control': best_any[u], 'nested_control': v['NestedControl'], 'gate_auroc': s['gate'],
                     'raw_difference_auroc': s['raw_difference'], 'smoothed_difference_auroc': s['smoothed_expression_weighted'],
                     'contiguity': breadth['included'][u]['gt_contiguity'], 'best_uniform_k': best_k[u],
                     'best_uniform_ari': best_u[u]})
    frame = pd.DataFrame(rows)
    frame['short_label'] = [FIGURE_LABEL.get(o, l) for o, l in zip(frame.object, frame.label)]
    frame['tessera_minus_same_graph'] = frame.tessera - frame.same_graph_mean
    frame['tessera_minus_oracle'] = frame.tessera - frame.oracle_control
    frame['tessera_minus_pca50'] = frame.tessera - frame.pca50
    return frame


def replicate_table(hard):
    run1 = pd.read_csv(EXT / 'replicates/run1/primary_patients.csv').set_index('patient')
    run2 = pd.read_csv(EXTERNAL / 'results/primary_patients.csv').set_index('patient')
    enh1 = pd.read_csv(EXT / 'replicates/run1/enhanced/primary.csv')
    enh2 = pd.read_csv(EXTERNAL / 'results/enhanced/primary.csv')
    rows = []
    for target, label, e1, e2 in (('all', 'All labels', None, None), ('compartments', 'Compartments', 'compartments', 'compartments'),
                                   ('boundary_T1', 'Tumor boundary', 'boundary', 'boundary')):
        per = hard['targets'][target]['per_unit']
        for unit, v in per.items():
            p = unit[0]
            if target == 'all':
                t1, t2 = float(run1.loc[p, 'tessera_ari']), float(run2.loc[p, 'tessera_ari'])
            else:
                t1 = float(enh1[(enh1.target == e1) & (enh1.patient == p)].tessera_ari.iloc[0])
                t2 = float(enh2[(enh2.target == e2) & (enh2.patient == p)].tessera_ari.iloc[0])
            runs = [t1, t2, v['Tessera']]
            rows.append({'target': target, 'target_label': label, 'unit': unit, 'run1': runs[0], 'run2': runs[1], 'run3': runs[2],
                         'range': max(runs) - min(runs), 'sd': float(np.std(runs, ddof=1)),
                         'same_graph_mean': v['UniformSameGraph'], 'nested_control': v['NestedControl'],
                         'sign_vs_control_changes': len({r > v['NestedControl'] for r in runs}) > 1,
                         'sign_vs_same_graph_changes': len({r > v['UniformSameGraph'] for r in runs}) > 1})
    return pd.DataFrame(rows)


def add_contrast(m, prefix, c, with_interval=True):
    m[prefix + 'Pos'] = str(c['n_positive'])
    m[prefix + 'Neg'] = str(c['n_negative'])
    m[prefix + 'Mean'] = f3(c['mean'])
    m[prefix + 'P'] = pv(c['exact_sign_flip_p'])
    if with_interval and 'bootstrap95' in c:
        m[prefix + 'Lo'] = f3(c['bootstrap95'][0])
        m[prefix + 'Hi'] = f3(c['bootstrap95'][1])


def main():
    hard = read(EXT / 'results/run-hardening.json')['result']
    dev = read(EXT / 'results/run-dev.json')['result']
    breadth = read(EXT / 'results/run-breadth.json')['result']
    m = {}
    included = hard['included']
    m['HerPatients'] = str(len(included))
    spots = [v['n_analysis'] for v in included.values()]
    m['HerSpotsMin'], m['HerSpotsMax'] = str(min(spots)), str(max(spots))
    classes = pd.read_csv(EXT / 'results/hardening/partitions_all.csv').groupby('unit').reference.nunique()
    m['HerClassesMin'], m['HerClassesMax'] = str(int(classes.min())), str(int(classes.max()))
    for tag, target in HER_TARGETS.items():
        c = hard['targets'][target]['contrasts']
        add_contrast(m, f'Her{tag}TessUSix', c['Tessera_minus_UniformSameGraph'])
        add_contrast(m, f'Her{tag}TessNested', c['Tessera_minus_NestedControl'])
        add_contrast(m, f'Her{tag}TessPca', c['Tessera_minus_PCA50'])
        add_contrast(m, f'Her{tag}USixNested', c['UniformSameGraph_minus_NestedControl'])
        add_contrast(m, f'Her{tag}EUSix', c['ExpressionWeighted_minus_UniformSameGraph'])
        add_contrast(m, f'Her{tag}EShuf', c['ExpressionWeighted_minus_WeightShuffled'])
        add_contrast(m, f'Her{tag}ECoord', c['ExpressionWeighted_minus_ShuffledCoords'])
        add_contrast(m, f'Her{tag}ENested', c['ExpressionWeighted_minus_NestedControl'])
    for tag, target in (('Comp', 'compartments'), ('Bnd', 'boundary_T1')):
        s = hard['seams'][target]
        for key, name in (('gate', 'Gate'), ('raw_difference', 'Raw'), ('smoothed_expression_weighted', 'Smooth'), ('smoothed_uniform', 'Plain')):
            m[f'Her{tag}Seam{name}'] = u3(s['mean'][key]['mean'])
            m[f'Her{tag}Seam{name}Lo'] = u3(s['mean'][key]['bootstrap95'][0])
            m[f'Her{tag}Seam{name}Hi'] = u3(s['mean'][key]['bootstrap95'][1])
        add_contrast(m, f'Her{tag}SeamSmoothGate', s['differences']['smoothed_expression_weighted_minus_gate'])
        add_contrast(m, f'Her{tag}SeamRawGate', s['differences']['raw_difference_minus_gate'])
        add_contrast(m, f'Her{tag}SeamSmoothRaw', s['differences']['smoothed_expression_weighted_minus_raw_difference'])
        add_contrast(m, f'Her{tag}SeamSmoothPlain', s['differences']['smoothed_expression_weighted_minus_smoothed_uniform'])
        m[f'Her{tag}GateAboveChance'] = str(s['gate_above_chance'])
        m[f'Her{tag}GateOverRaw'] = str(s['gate_exceeds_raw'])
    m['HerHOneVerdict'] = 'was met' if hard['H1_weighting']['holds'] else 'was not met'
    m['HerHTwoVerdict'] = 'was met' if hard['H2_gate']['holds'] else 'was not met'
    g = hard['patient_G_section_identity']
    m['HerGOwn'] = f"{100 * g['G2']['fraction']:.1f}"
    m['HerGOther'] = f"{100 * g['G1']['fraction']:.1f}"
    # Replicate fits of the frozen encoder.
    rep = replicate_table(hard)
    rep.to_csv(LED / 'replicates.csv', index=False)
    m['RepPairs'] = str(len(rep))
    m['RepMaxRange'] = u3(rep['range'].max())
    m['RepMedianRange'] = u3(rep['range'].median())
    m['RepSignChangesControl'] = str(int(rep.sign_vs_control_changes.sum()))
    m['RepSignChangesSame'] = str(int(rep.sign_vs_same_graph_changes.sum()))
    m['RepAllMaxRange'] = u3(rep[rep.target == 'all']['range'].max())
    # Development objects: weighting ablation on the saved inputs.
    for cohort, tag in (('NSCLC', 'Lung'), ('BCC', 'Bcc'), ('other', 'Oth'), ('all13', 'AllThirteen')):
        c = dev['cohorts'][cohort]
        add_contrast(m, f'Dev{tag}EMean', c['E6_minus_NeighborMean'], with_interval=False)
        add_contrast(m, f'Dev{tag}EShuf', c['E6_minus_WeightShuffled'], with_interval=False)
        add_contrast(m, f'Dev{tag}ECoord', c['E6_minus_ShuffledCoords'], with_interval=False)
    # Three further tissues.
    per = breadth['per_unit']
    tags = {'DLPFC_151673': 'Cortex', 'STARmap_visual_cortex': 'Starmap', 'MERFISH_hypothalamus': 'Merfish'}
    for unit, tag in tags.items():
        v = per[unit]
        m[f'Br{tag}Tess'] = u3(v['Tessera'])
        m[f'Br{tag}Pca'] = u3(v['PCA50'])
        m[f'Br{tag}Mean'] = u3(v['UniformSameGraph'])
        m[f'Br{tag}Nested'] = u3(v['NestedControl'])
        m[f'Br{tag}TessNested'] = f3(v['Tessera'] - v['NestedControl'])
        m[f'Br{tag}TessPca'] = f3(v['Tessera'] - v['PCA50'])
        m[f'Br{tag}Gate'] = u3(breadth['seams'][unit]['gate'])
        m[f'Br{tag}Contig'] = u3(breadth['included'][unit]['gt_contiguity'])
        o = breadth['oracle_not_for_inference'][unit]
        m[f'Br{tag}Oracle'] = u3(o['ari'])
        m[f'Br{tag}TessOracle'] = f3(v['Tessera'] - o['ari'])
        m[f'Br{tag}BestK'] = o['best_uniform_scale'].split('|')[1]
    pca_gain = [per[u]['Tessera'] - per[u]['PCA50'] for u in per]
    m['BrPcaGainMin'], m['BrPcaGainMax'] = u3(min(pca_gain)), u3(max(pca_gain))
    gates = [breadth['seams'][u]['gate'] for u in per]
    m['BrGateMin'], m['BrGateMax'] = u3(min(gates)), u3(max(gates))
    # Landscape over every object.
    table = landscape(hard, dev, breadth)
    table.to_csv(LED / 'landscape.csv', index=False)
    m['LandN'] = str(len(table))
    m['LandTessSamePos'] = str(int((table.tessera_minus_same_graph > 0).sum()))
    m['LandTessOraclePos'] = str(int((table.tessera_minus_oracle > 0).sum()))
    m['LandTessPcaPos'] = str(int((table.tessera_minus_pca50 > 0).sum()))
    for group, tag in (('Computed regions', 'Computed'), ('Cell-type labels', 'Cell'), ('Pathologist regions', 'Path'), ('Anatomical regions', 'Anat')):
        sub = table[table.group == group]
        m[f'Land{tag}N'] = str(len(sub))
        m[f'Land{tag}SamePos'] = str(int((sub.tessera_minus_same_graph > 0).sum()))
        m[f'Land{tag}SameMean'] = f3(sub.tessera_minus_same_graph.mean())
        m[f'Land{tag}OraclePos'] = str(int((sub.tessera_minus_oracle > 0).sum()))
        m[f'Land{tag}KMin'], m[f'Land{tag}KMax'] = str(int(sub.best_uniform_k.min())), str(int(sub.best_uniform_k.max()))
    m['LandGateMax'] = u3(table.gate_auroc.max())
    m['LandGateMedian'] = u3(table.gate_auroc.median())
    m['LandGateAbove'] = str(int((table.gate_auroc > .5).sum()))
    m['LandRawOverGate'] = str(int((table.raw_difference_auroc > table.gate_auroc).sum()))
    # Macros and JSON.
    for name in m:
        assert name.isalpha(), name
    (LED / 'external_numbers.tex').write_text('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(m.items())) + '\n')
    dump(LED / 'external_numbers.json', dict(sorted(m.items())))
    # Main table: HER2ST patients.
    rows = []
    for unit in hard['targets']['all']['units']:
        a = hard['targets']['all']['per_unit'][unit]
        b = hard['targets']['boundary_T1']['per_unit'][unit]
        rows.append(' & '.join([f'{unit[0]} ({unit})', str(included[unit]['n_analysis']),
                                str(int(classes.loc[unit])), u3(a['PCA50']), u3(a['UniformSameGraph']), u3(a['NestedControl']), u3(a['Tessera']),
                                u3(b['UniformSameGraph']), u3(b['NestedControl']), u3(b['Tessera'])]) + r'\\')
    (LED / 'external_rows.tex').write_text('\n'.join(rows) + '\n')
    # Supplementary table: composition and joins.
    parts = pd.read_csv(EXT / 'results/hardening/partitions_all.csv')
    order = ['invasive cancer', 'cancer in situ', 'connective tissue', 'adipose tissue', 'immune infiltrate', 'breast glands', 'undetermined']
    rows = []
    for unit in hard['targets']['all']['units']:
        counts = parts[parts.unit == unit].reference.value_counts()
        base = included[unit]
        rows.append(' & '.join([unit, str(base['n_label_rows']), f"{100 * base['match_fraction']:.1f}"] + [str(int(counts.get(c, 0))) for c in order]) + r'\\')
    (LED / 'external_composition_rows.tex').write_text('\n'.join(rows) + '\n')
    # Supplementary table: contrasts per target.
    labels = {'all': 'All labels', 'compartments': 'Named compartments', 'boundary_T1': 'Tumor boundary',
              'boundary_T2': 'Invasive cancer only', 'boundary_T3': 'Without breast glands'}
    names = [('Tessera_minus_UniformSameGraph', 'Tessera $-$ same-graph mean'), ('Tessera_minus_NestedControl', 'Tessera $-$ tuned control'),
             ('UniformSameGraph_minus_NestedControl', 'Same-graph mean $-$ tuned control'),
             ('ExpressionWeighted_minus_UniformSameGraph', 'Weighted $-$ same-graph mean'),
             ('ExpressionWeighted_minus_WeightShuffled', 'Weighted $-$ shuffled weights'),
             ('ExpressionWeighted_minus_ShuffledCoords', 'Weighted $-$ shuffled coordinates')]
    rows = []
    for target in labels:
        c = hard['targets'][target]['contrasts']
        for key, label in names:
            d = c[key]
            rows.append(' & '.join([labels[target] if key == names[0][0] else '', label, f"{d['n_positive']}/{d['n_negative']}",
                                    f3(d['mean']), f"[{f3(d['bootstrap95'][0])}, {f3(d['bootstrap95'][1])}]", pv(d['exact_sign_flip_p'])]) + r'\\')
        rows.append(r'\addlinespace')
    (LED / 'external_contrast_rows.tex').write_text('\n'.join(rows) + '\n')
    # Supplementary table: seam AUROC per patient.
    rows = []
    for unit in hard['targets']['all']['units']:
        cells = [unit]
        for target in ('boundary_T1', 'compartments'):
            s = hard['seams'][target]['per_unit'][unit]
            cells += [u3(s['gate']), u3(s['raw_difference']), u3(s['smoothed_uniform']), u3(s['smoothed_expression_weighted'])]
        rows.append(' & '.join(cells) + r'\\')
    (LED / 'external_seam_rows.tex').write_text('\n'.join(rows) + '\n')
    # Supplementary table: three further tissues.
    rows = []
    for unit in breadth['units']:
        v = per[unit]
        o = breadth['oracle_not_for_inference'][unit]
        info = breadth['included'][unit]
        rows.append(' & '.join([tex(SHORT_BREADTH[unit]), str(info['n_valid']), str(info['n_classes']), u3(info['gt_contiguity']),
                                u3(v['PCA50']), u3(v['UniformSameGraph']), u3(v['NestedControl']), u3(o['ari']), u3(v['Tessera']),
                                u3(breadth['seams'][unit]['gate']), u3(breadth['seams'][unit]['smoothed_expression_weighted'])]) + r'\\')
    (LED / 'breadth_rows.tex').write_text('\n'.join(rows) + '\n')
    # Supplementary table: replicate fits.
    rows = []
    for target in ('all', 'compartments', 'boundary_T1'):
        sub = rep[rep.target == target]
        for _, r in sub.iterrows():
            rows.append(' & '.join([r.target_label if r.unit == sub.unit.iloc[0] else '', r.unit, u3(r.run1), u3(r.run2), u3(r.run3), u3(r['range']),
                                    u3(r.same_graph_mean), u3(r.nested_control)]) + r'\\')
        rows.append(r'\addlinespace')
    (LED / 'replicate_rows.tex').write_text('\n'.join(rows) + '\n')
    print(f'{len(m)} external macros; {len(table)} landscape rows; replicate pairs {len(rep)}')


if __name__ == '__main__':
    main()

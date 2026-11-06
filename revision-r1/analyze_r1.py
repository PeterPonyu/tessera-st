#!/usr/bin/env python3
"""Summaries, decision rules (DR-1 to DR-4) and manuscript macros for the revision R1 analyses.

Reads only the CSV/JSON outputs under results/revision-r1/ and the saved comparator tables of the
original analyses; writes results/revision-r1/summary/ and the revision packet's macro file.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import common_r1 as c

E1, E2, E3, E4 = (c.OUT / e for e in ['E1', 'E2', 'E3', 'E4'])
SUMMARY = Path(os.environ.get('TESSERA_R1_SUMMARY') or c.OUT / 'summary')
EXT = c.VALIDATION / 'validation-extension' / 'results'
FOLLOW = c.RESULTS / 'followup-20261010'
_PACKET_SOURCE = c.PROJECT / 'submissions/frontiers-in-genetics/revision-1-2028706/source'
MACROS = Path(os.environ.get('TESSERA_MACROS') or
              (_PACKET_SOURCE if _PACKET_SOURCE.exists() else SUMMARY) / 'revision-macros.tex')
COHORTS = [('NSCLC', 'primary'), ('BCC', 'primary'), ('HER2ST', 'all'), ('HER2ST', 'boundary_T1')]


def seed_mean(path_fmt, name, arm):
    return float(np.mean([json.loads((c.RESULTS / name / f'{arm}-seed{s}.json').read_text())['scores']['ari']
                          for s in c.SEEDS]))


def comparators() -> pd.DataFrame:
    """Saved comparator ARIs (seed means) at the reference cluster number."""
    rows = []
    summary = json.loads((FOLLOW / 'scale_controls_summary.json').read_text())
    nested = {r['patient']: r['nested_control_ari'] for k in ['NSCLC', 'BCC'] for r in summary[k]['per_patient']}
    rescore = pd.read_csv(FOLLOW / 'rescore.csv')
    for name in c.OBJECTS:
        cohort = json.loads((c.DATA / name / 'input.json').read_text())['cohort']
        values = {'PCA50': seed_mean(None, name, 'PCA50'), 'SameGraphMean': seed_mean(None, name, 'NeighborMean'),
                  'Tessera_original': seed_mean(None, name, 'Tessera')}
        if name in nested:
            values['NestedControl'] = nested[name]
        for comp, v in values.items():
            rows.append(dict(cohort=cohort, patient=name, target='primary', comparator=comp, ari=v))
        if name in c.LUNG:
            part = rescore[rescore.object == name]
            for comp, arm in [('PCA50', 'PCA50'), ('SameGraphMean', 'NeighborMean'), ('Tessera_original', 'Tessera')]:
                rows.append(dict(cohort=cohort, patient=name, target='cell_type', comparator=comp,
                                 ari=float(part[part.arm == arm].ari_celltype_at_celltype_K.mean())))
    hard = json.loads((EXT / 'run-hardening.json').read_text())['result']['targets']
    for target in ['all', 'compartments', 'boundary_T1']:
        for unit, v in hard[target]['per_unit'].items():
            for comp, key in [('PCA50', 'PCA50'), ('SameGraphMean', 'UniformSameGraph'),
                              ('NestedControl', 'NestedControl'), ('Tessera_original', 'Tessera')]:
                rows.append(dict(cohort='HER2ST', patient=unit, target=target, comparator=comp, ari=v[key]))
    breadth = json.loads((EXT / 'run-breadth.json').read_text())['result']['per_unit']
    for unit, v in breadth.items():
        for comp, key in [('PCA50', 'PCA50'), ('SameGraphMean', 'UniformSameGraph'),
                          ('NestedControl', 'NestedControl'), ('Tessera_original', 'Tessera')]:
            rows.append(dict(cohort='tissue', patient=unit, target='primary', comparator=comp, ari=v[key]))
    return pd.DataFrame(rows)


def comp_lookup(comps, cohort, target):
    part = comps[(comps.cohort == cohort) & (comps.target == target)]
    return part.pivot(index='patient', columns='comparator', values='ari')


# ------------------------------------------------------------------ E1
def e1(comps):
    scores = pd.concat([pd.read_csv(E1 / 'E1_scores_gpu.csv'), pd.read_csv(E1 / 'E1_scores_her2st.csv')])
    scores['cohort'] = scores.cohort.replace({'Breast': 'Breast', 'tissue': 'tissue'})
    per = scores.groupby(['cohort', 'object', 'target', 'arm', 'epoch']).agg(ari=('ari', 'mean'),
                                                                            seam=('seam_auroc', 'mean')).reset_index()
    out, tables = {}, []
    for cohort, target in COHORTS + [('HER2ST', 'compartments'), ('NSCLC', 'cell_type'), ('tissue', 'primary')]:
        t = per[(per.cohort == cohort) & (per.target == target) & (per.arm == 'Tessera')]
        wide = t.pivot(index='object', columns='epoch', values='ari')
        cmp_ = comp_lookup(comps, cohort, target).reindex(wide.index)
        nested_epoch = {p: int(c.select_nested(wide, p)) for p in wide.index}
        rec = pd.DataFrame({'cohort': cohort, 'target': target, 'patient': wide.index,
                            'ari_120': wide[120].values, 'ari_200': wide[200].values, 'ari_400': wide[400].values,
                            'nested_epoch': [nested_epoch[p] for p in wide.index],
                            'ari_nested_epoch': [wide.loc[p, nested_epoch[p]] for p in wide.index],
                            'oracle_epoch': wide.idxmax(axis=1).values, 'ari_oracle_epoch': wide.max(axis=1).values})
        for comp in ['PCA50', 'SameGraphMean', 'NestedControl', 'Tessera_original']:
            if comp in cmp_:
                rec[comp] = cmp_[comp].values
        tables.append(rec)
        key = f'{cohort}|{target}'
        out[key] = dict(n=len(rec), budget_200_minus_120=c.summarize(rec.ari_200 - rec.ari_120),
                        mean_ari={e: float(wide[e].mean()) for e in wide.columns})
        for comp in ['NestedControl', 'SameGraphMean', 'PCA50']:
            if comp in rec:
                out[key][f'vs_{comp}'] = {lab: c.summarize(rec[col] - rec[comp], bootstrap=len(rec) >= 8)
                                          for lab, col in [('e120', 'ari_120'), ('e200', 'ari_200'),
                                                           ('nested_epoch', 'ari_nested_epoch'),
                                                           ('oracle_epoch', 'ari_oracle_epoch')]}
        if 'Tessera_original' in rec:
            out[key]['refit120_minus_original'] = dict(
                mean=float((rec.ari_120 - rec.Tessera_original).mean()),
                max_abs=float((rec.ari_120 - rec.Tessera_original).abs().max()))
        arms = per[(per.cohort == cohort) & (per.target == target)].pivot_table(
            index=['object', 'epoch'], columns='arm', values='ari').reset_index()
        out[key]['components'] = {
            int(e): {f'Tessera_minus_{a}': c.summarize(g.Tessera - g[a])
                     for a in ['NoGate', 'NoContrast', 'SelfOnly', 'PermutedXY-NoContrast'] if a in g}
            for e, g in arms.groupby('epoch') if e in (120, 200, 400)}
        seams = per[(per.cohort == cohort) & (per.target == target) & (per.arm == 'Tessera')]
        out[key]['seam_auroc'] = {int(e): float(g.seam.mean()) for e, g in seams.groupby('epoch')
                                  if e in (120, 200, 400) and g.seam.notna().any()}
    table = pd.concat(tables)
    table.to_csv(SUMMARY / 'table_E1.csv', index=False)
    transfer = pd.read_csv(E1 / 'E1b_transfer.csv') if (E1 / 'E1b_transfer.csv').exists() else None
    if transfer is not None:
        m = transfer.groupby(['task', 'epoch', 'target']).balanced_accuracy.mean().reset_index()
        out['transfer'] = {f'{task}|{int(e)}': float(g.balanced_accuracy.mean())
                           for (task, e), g in m.groupby(['task', 'epoch'])}
        w = m.pivot_table(index=['task', 'target'], columns='epoch', values='balanced_accuracy').reset_index()
        out['transfer_200_minus_120'] = {task: c.summarize(g[200] - g[120]) for task, g in w.groupby('task')}
    return out, table


# ------------------------------------------------------------------ E2
def saved_grid(cohort, target):
    if cohort in ('NSCLC', 'BCC'):
        g = pd.read_csv(FOLLOW / 'scale_controls.csv')
        names = c.LUNG if cohort == 'NSCLC' else c.BCC
        g = g[g.object.isin(names)].rename(columns={'object': 'patient'})
        g['config'] = g.kind + '|' + g.k.astype(str) + '|' + g.lam.astype(str)
    else:
        g = pd.read_csv(EXT / 'hardening' / 'grid.csv')
        g = g[g.target == target].rename(columns={'unit': 'patient'})
    return g


def e2(comps):
    grid = pd.concat([pd.read_csv(E2 / 'E2_grid_gpu.csv'), pd.read_csv(E2 / 'E2_grid_her2st.csv')])
    sel = pd.read_csv(E2 / 'E2_selection.csv')
    mech = pd.read_csv(E2 / 'E2_mechanism.csv')
    out, rows = {}, []
    for cohort, target in COHORTS:
        s = sel[(sel.cohort == cohort) & (sel.target == target)].set_index('patient')
        cmp_ = comp_lookup(comps, cohort, target).reindex(s.index)
        g = saved_grid(cohort, target)
        means = g.groupby(['patient', 'config']).ari.mean().unstack()
        small = [col for col in means.columns if not str(col).startswith('PCA50') and
                 str(col).split('|')[1] not in ('None', 'nan') and float(str(col).split('|')[1]) <= 30]
        matched = {p: float(means.loc[p, c.select_nested(means[small], p)]) for p in s.index}
        oracle_control = means.max(axis=1)
        m = mech[(mech.cohort == cohort) & (mech.target == target)].groupby(['patient', 'arm']).ari.mean().unstack()
        mean_col = [a for a in m.columns if a.startswith('SameGraphMean')]
        sgm = m[mean_col].bfill(axis=1).iloc[:, 0] if mean_col else None
        rec = pd.DataFrame({'cohort': cohort, 'target': target, 'patient': s.index, 'selected': s.selected.values,
                            'tessera_nested': s.nested_tessera_ari.values, 'tessera_default': s.default_tessera_ari.values,
                            'nested_control': cmp_['NestedControl'].values, 'same_graph_mean_k6': cmp_['SameGraphMean'].values,
                            'same_graph_mean_selected_k': sgm.reindex(s.index).values,
                            'matched_scale_control': [matched[p] for p in s.index],
                            'nogate_selected': m['NoGate'].reindex(s.index).values,
                            'nocontrast_selected': m['NoContrast'].reindex(s.index).values,
                            'tessera_oracle': s.oracle_tessera_ari.values,
                            'control_oracle': oracle_control.reindex(s.index).values})
        rows.append(rec)
        key = f'{cohort}|{target}'
        out[key] = dict(
            DR1_vs_nested_control=c.summarize(rec.tessera_nested - rec.nested_control, bootstrap=len(rec) >= 8),
            DR1_vs_same_graph_mean_selected_k=c.summarize(rec.tessera_nested - rec.same_graph_mean_selected_k,
                                                          bootstrap=len(rec) >= 8),
            tuned_minus_default=c.summarize(rec.tessera_nested - rec.tessera_default),
            vs_matched_scale_control=c.summarize(rec.tessera_nested - rec.matched_scale_control),
            nogate_minus_tessera=c.summarize(rec.nogate_selected - rec.tessera_nested),
            nocontrast_minus_tessera=c.summarize(rec.nocontrast_selected - rec.tessera_nested),
            oracle_tessera_minus_oracle_control=c.summarize(rec.tessera_oracle - rec.control_oracle),
            selected_configs=s.selected.value_counts().to_dict(),
            selected_k_values=sorted({int(x.split('|')[0][1:]) for x in s.selected}))
    table = pd.concat(rows)
    table.to_csv(SUMMARY / 'table_E2.csv', index=False)
    return out, table


# ------------------------------------------------------------------ E3
def e3(comps):
    sweep = pd.read_csv(E3 / 'E3_sweep.csv')
    agg = sweep.groupby(['cohort', 'unit', 'target', 'K_ref', 'geometry', 'K', 'representation']).ari.mean().unstack()
    agg = agg.reset_index()
    agg['offset'] = agg.K - agg.K_ref
    lf = pd.read_csv(E3 / 'E3_label_free_K.csv').set_index(['unit', 'target']).K_label_free
    out, rows = {}, []
    for cohort, target in COHORTS + [('HER2ST', 'compartments'), ('tissue', 'anatomy')]:
        part = agg[(agg.cohort == cohort) & (agg.target == target)].copy()
        for tess in ['Tessera120', 'Tessera200']:
            for comp in ['TunedControl', 'SameGraphMean', 'PCA50']:
                if comp not in part:
                    continue
                part[f'{tess}-{comp}'] = part[tess] - part[comp]
        rows.append(part)
        key = f'{cohort}|{target}'
        res = {}
        for geometry in ['euclidean', 'cosine']:
            g = part[part.geometry == geometry]
            by_offset = {}
            for off, gg in g.groupby('offset'):
                if gg.unit.nunique() == part.unit.nunique():
                    by_offset[int(off)] = {col: c.summarize(gg[col]) for col in gg.columns
                                           if col.startswith('Tessera1') or col.startswith('Tessera2')
                                           if '-' in col}
            lf_rows = pd.concat([g[(g.unit == u) & (g.K == lf.get((u, target), -1))] for u in part.unit.unique()])
            res[geometry] = dict(by_offset=by_offset,
                                 label_free={col: c.summarize(lf_rows[col]) for col in lf_rows.columns
                                             if '-' in col and col.startswith('Tessera')} if len(lf_rows) else None)
        window = [o for o in range(-2, 3) if all(o in res[g]['by_offset'] for g in res)]
        ref_sign = {g: np.sign(res[g]['by_offset'][0]['Tessera120-TunedControl']['mean'])
                    for g in res if 'Tessera120-TunedControl' in res[g]['by_offset'].get(0, {})}

        def holds(summary, sign):
            maj = summary['n_neg'] if sign < 0 else summary['n_pos']
            return np.sign(summary['mean']) == sign and maj > summary['n'] / 2
        if ref_sign:
            dr2 = all(holds(res[g]['by_offset'][o]['Tessera120-TunedControl'], ref_sign[g]) for g in ref_sign
                      for o in window) and all(res[g]['label_free'] is not None and
                                               holds(res[g]['label_free']['Tessera120-TunedControl'], ref_sign[g])
                                               for g in ref_sign)
            held = {g: sorted(o for o, s in res[g]['by_offset'].items()
                              if 'Tessera120-TunedControl' in s and holds(s['Tessera120-TunedControl'], ref_sign[g]))
                    for g in ref_sign}
            res['DR2'] = dict(robust=bool(dr2), window=window, offsets_where_ordering_holds=held,
                              reference_sign={g: int(v) for g, v in ref_sign.items()})
        out[key] = res
    table = pd.concat(rows)
    table.to_csv(SUMMARY / 'table_E3.csv', index=False)
    extra = {}
    if (E3 / 'E3_harmonization.csv').exists():
        h = pd.read_csv(E3 / 'E3_harmonization.csv').groupby(['unit', 'target'])[['ari_per_k_head', 'ari_fixed_head']].mean()
        h = h.reset_index()
        nested = comps[(comps.cohort == 'HER2ST') & (comps.comparator == 'NestedControl')].set_index(['patient', 'target']).ari
        h['nested'] = [nested.get((u, t), np.nan) for u, t in zip(h.unit, h.target)]
        h['flip'] = np.sign(h.ari_per_k_head - h.nested) != np.sign(h.ari_fixed_head - h.nested)
        h.to_csv(SUMMARY / 'table_E3_harmonization.csv', index=False)
        extra['harmonization'] = dict(mean_abs_difference=float((h.ari_per_k_head - h.ari_fixed_head).abs().mean()),
                                      max_abs_difference=float((h.ari_per_k_head - h.ari_fixed_head).abs().max()),
                                      sign_flips=h[h.flip][['unit', 'target']].values.tolist(), n_pairs=len(h))
    if (E3 / 'E3_refinement.csv').exists():
        r = pd.read_csv(E3 / 'E3_refinement.csv').groupby(['cohort', 'unit', 'target', 'representation'])[
            ['ari', 'ari_refined']].mean().reset_index()
        r.to_csv(SUMMARY / 'table_E3_refinement.csv', index=False)
        ref = {}
        for cohort, target in COHORTS:
            p = r[(r.cohort == cohort) & (r.target == target)].pivot(index='unit', columns='representation',
                                                                      values='ari_refined')
            if 'TunedControl' in p:
                ref[f'{cohort}|{target}'] = dict(
                    tessera120_minus_control_refined=c.summarize(p.Tessera120 - p.TunedControl),
                    tessera120_minus_mean_refined=c.summarize(p.Tessera120 - p.SameGraphMean))
        gain = r.assign(gain=r.ari_refined - r.ari).groupby('representation').gain.mean().to_dict()
        extra['refinement'] = dict(by_cohort=ref, mean_gain_by_representation=gain)
    if (E3 / 'E3_leiden.csv').exists():
        le = pd.read_csv(E3 / 'E3_leiden.csv')
        le['offset'] = le.K_target - le.K_ref
        w = le.pivot_table(index=['cohort', 'unit', 'target', 'offset'], columns='representation', values='ari').reset_index()
        extra['leiden'] = {f'{co}|{t}|{int(o)}|{comp}': c.summarize(g.Tessera120 - g[comp])
                           for (co, t, o), g in w.groupby(['cohort', 'target', 'offset'])
                           for comp in ['TunedControl', 'SameGraphMean', 'PCA50'] if comp in g}
        extra['leiden_hit_rate'] = float((le.K_achieved == le.K_target).mean())
    out['extra'] = extra
    return out, table


# ------------------------------------------------------------------ E4
def e4(comps):
    path = E4 / 'E4_scores.csv'
    if not path.exists():
        return {'status': 'not available'}, None
    s = pd.read_csv(path)
    m = s.groupby(['method', 'variant', 'cohort', 'patient', 'target', 'clustering']).ari.mean().reset_index()
    out = {}
    target_map = {('NSCLC', 'regions'): 'primary', ('NSCLC', 'cell_types'): 'cell_type',
                  ('HER2ST', 'all'): 'all', ('HER2ST', 'boundary_T1'): 'boundary_T1'}
    rows = []
    for (method, variant, cohort, target, clustering), g in m.groupby(['method', 'variant', 'cohort', 'target', 'clustering']):
        ct = target_map.get((cohort, target), target)
        cmp_ = comp_lookup(comps, cohort, ct)
        g = g.set_index('patient')
        rec = {'method': method, 'variant': variant, 'cohort': cohort, 'target': ct, 'clustering': clustering,
               'n': len(g), 'mean_ari': float(g.ari.mean())}
        for comp in ['NestedControl', 'SameGraphMean', 'PCA50', 'Tessera_original']:
            if comp in cmp_:
                d = g.ari - cmp_[comp].reindex(g.index)
                rec[f'minus_{comp}'] = c.summarize(d.dropna())
        rows.append(rec)
    out['by_setting'] = rows
    for method in m.method.unique():
        verdict = []
        for cohort, target in [('NSCLC', 'primary'), ('HER2ST', 'all')]:
            for clustering in ['kmeans_matched', 'native']:
                cand = [r for r in rows if r['method'] == method and r['cohort'] == cohort and r['target'] == target
                        and r['clustering'] == clustering and 'minus_NestedControl' in r]
                verdict += [r['minus_NestedControl']['outcome'] == 'retained' for r in cand]
        out[f'DR4|{method}'] = bool(verdict) and all(verdict)
    return out, pd.DataFrame(rows)


# ------------------------------------------------------------------ macros
def fmt(x, signed=False, digits=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 'NA'
    s = f'{x:+.{digits}f}' if signed else f'{x:.{digits}f}'
    return s.replace('-', '\\ensuremath{-}') if s.startswith('-') else s


WORD = {120: 'OneTwenty', 200: 'TwoHundred', 400: 'FourHundred'}


def write_macros(summary):
    lines = ['% Generated by analyze_r1.py from results/revision-r1; do not edit by hand.']

    def mac(name, value):
        lines.append(f'\\newcommand{{\\{name}}}{{{value}}}')
    for key, tag in [('NSCLC|primary', 'Lung'), ('BCC|primary', 'Bcc'), ('HER2ST|all', 'HerAll'),
                     ('HER2ST|boundary_T1', 'HerBnd')]:
        e1s = summary['E1'][key]
        b = e1s['budget_200_minus_120']
        mac(f'R{tag}BudgetMean', fmt(b['mean'], True))
        mac(f'R{tag}BudgetPos', b['n_pos'])
        mac(f'R{tag}BudgetNeg', b['n_neg'])
        mac(f'R{tag}N', b['n'])
        for lab in ['e120', 'e200', 'nested_epoch', 'oracle_epoch']:
            v = e1s['vs_NestedControl'][lab]
            tagl = {'e120': 'OneTwenty', 'e200': 'TwoHundred', 'nested_epoch': 'NestedEpoch',
                    'oracle_epoch': 'OracleEpoch'}[lab]
            mac(f'R{tag}{tagl}VsCtlMean', fmt(v['mean'], True))
            mac(f'R{tag}{tagl}VsCtlPos', v['n_pos'])
            mac(f'R{tag}{tagl}VsCtlNeg', v['n_neg'])
        for lab, tagl in [('e120', 'OneTwenty'), ('e200', 'TwoHundred')]:
            v = e1s['vs_SameGraphMean'][lab]
            mac(f'R{tag}{tagl}VsMeanMean', fmt(v['mean'], True))
            mac(f'R{tag}{tagl}VsMeanPos', v['n_pos'])
            mac(f'R{tag}{tagl}VsMeanNeg', v['n_neg'])
        if 'E2' in summary and key in summary['E2']:
            e2s = summary['E2'][key]
            for lab, tagl in [('DR1_vs_nested_control', 'TunedVsCtl'), ('DR1_vs_same_graph_mean_selected_k', 'TunedVsMean'),
                              ('tuned_minus_default', 'TunedVsDefault'), ('vs_matched_scale_control', 'TunedVsMatched'),
                              ('nogate_minus_tessera', 'NoGateAtSel'), ('nocontrast_minus_tessera', 'NoContrastAtSel'),
                              ('oracle_tessera_minus_oracle_control', 'OracleVsOracle')]:
                v = e2s[lab]
                mac(f'R{tag}{tagl}Mean', fmt(v['mean'], True))
                mac(f'R{tag}{tagl}Pos', v['n_pos'])
                mac(f'R{tag}{tagl}Neg', v['n_neg'])
                mac(f'R{tag}{tagl}P', fmt(v['p_sign_flip'], digits=3))
                mac(f'R{tag}{tagl}Outcome', v['outcome'])
                if 'bootstrap_95' in v:
                    mac(f'R{tag}{tagl}Lo', fmt(v['bootstrap_95'][0], True))
                    mac(f'R{tag}{tagl}Hi', fmt(v['bootstrap_95'][1], True))
            mac(f'R{tag}SelectedK', ', '.join(str(k) for k in e2s['selected_k_values']))
        if 'E3' in summary and key in summary['E3'] and 'DR2' in summary['E3'][key]:
            d = summary['E3'][key]['DR2']
            mac(f'R{tag}KRobust', 'held' if d['robust'] else 'did not hold')
            for geometry in ['euclidean', 'cosine']:
                held = d['offsets_where_ordering_holds'].get(geometry, [])
                mac(f'R{tag}KHeld{geometry.capitalize()}', f'{len(held)}')
                k0 = summary['E3'][key][geometry]['by_offset'][0]['Tessera120-TunedControl']
                mac(f'R{tag}KRef{geometry.capitalize()}Mean', fmt(k0['mean'], True))
                lf = summary['E3'][key][geometry]['label_free']
                if lf:
                    mac(f'R{tag}KFree{geometry.capitalize()}Mean', fmt(lf['Tessera120-TunedControl']['mean'], True))
                    mac(f'R{tag}KFree{geometry.capitalize()}Neg', lf['Tessera120-TunedControl']['n_neg'])
    for key, tag in [('NSCLC|primary', 'Lung'), ('BCC|primary', 'Bcc'), ('HER2ST|all', 'HerAll')]:
        comp = summary['E1'][key]['components']
        for e in (120, 200, 400):
            for arm, tagl in [('NoGate', 'NoGate'), ('NoContrast', 'NoContrast'), ('SelfOnly', 'SelfOnly')]:
                v = comp.get(e, comp.get(str(e), {})).get(f'Tessera_minus_{arm}')
                if v:
                    mac(f'R{tag}MinusArm{tagl}{WORD[e]}Mean', fmt(v['mean'], True))
                    mac(f'R{tag}MinusArm{tagl}{WORD[e]}Pos', v['n_pos'])
        seam = summary['E1'][key]['seam_auroc']
        for e in (120, 200, 400):
            if e in seam or str(e) in seam:
                mac(f'R{tag}Seam{WORD[e]}', fmt(seam.get(e, seam.get(str(e)))))
        r = summary['E1'][key].get('refit120_minus_original')
        if r:
            mac(f'R{tag}RefitMaxAbs', fmt(r['max_abs']))
            mac(f'R{tag}RefitMean', fmt(r['mean'], True))
    if 'transfer' in summary['E1']:
        for task, tagl in [('region', 'Region'), ('cell_type', 'Cell')]:
            for e in (120, 200):
                mac(f'RTransfer{tagl}{WORD[e]}', fmt(summary['E1']['transfer'][f'{task}|{e}']))
            v = summary['E1']['transfer_200_minus_120'][task]
            mac(f'RTransfer{tagl}DeltaMean', fmt(v['mean'], True))
            mac(f'RTransfer{tagl}DeltaPos', v['n_pos'])
    extra = summary.get('E3', {}).get('extra', {})
    if 'harmonization' in extra:
        h = extra['harmonization']
        mac('RHarmMeanAbs', fmt(h['mean_abs_difference']))
        mac('RHarmMaxAbs', fmt(h['max_abs_difference']))
        mac('RHarmFlips', len(h['sign_flips']))
        mac('RHarmPairs', h['n_pairs'])
    if 'refinement' in extra:
        for key, tag in [('NSCLC|primary', 'Lung'), ('HER2ST|all', 'HerAll')]:
            v = extra['refinement']['by_cohort'].get(key)
            if v:
                mac(f'R{tag}RefinedVsCtlMean', fmt(v['tessera120_minus_control_refined']['mean'], True))
                mac(f'R{tag}RefinedVsCtlNeg', v['tessera120_minus_control_refined']['n_neg'])
    if 'leiden_hit_rate' in extra:
        mac('RLeidenHit', fmt(100 * extra['leiden_hit_rate'], digits=0))
        for key, tag in [('NSCLC|primary', 'Lung'), ('HER2ST|all', 'HerAll')]:
            for comp, tagc in [('TunedControl', 'Ctl'), ('SameGraphMean', 'Mean'), ('PCA50', 'Pca')]:
                v = extra['leiden'].get(f'{key}|0|{comp}')
                if v:
                    mac(f'R{tag}Leiden{tagc}Mean', fmt(v['mean'], True))
                    mac(f'R{tag}Leiden{tagc}Neg', v['n_neg'])
    lf = pd.read_csv(E3 / 'E3_label_free_K.csv') if (E3 / 'E3_label_free_K.csv').exists() else None
    if lf is not None:
        groups = {'Lung': (lf.unit.isin(c.LUNG)) & (lf.target == 'primary'),
                  'Bcc': (lf.unit.isin(c.BCC)) & (lf.target == 'primary'),
                  'HerAll': (lf.unit.isin(c.HER2ST)) & (lf.target == 'all')}
        for tag, mask in groups.items():
            ks = lf[mask].K_label_free
            mac(f'R{tag}KFreeMin', int(ks.min()))
            mac(f'R{tag}KFreeMax', int(ks.max()))
    MACROS.write_text('\n'.join(lines) + '\n')


def figure_data(comps) -> None:
    """Long-format tables for the new main figures (budget, tuning, cluster number)."""
    scores = pd.concat([pd.read_csv(E1 / 'E1_scores_gpu.csv'), pd.read_csv(E1 / 'E1_scores_her2st.csv')])
    per = scores[scores.arm == 'Tessera'].groupby(['cohort', 'object', 'target', 'epoch']).ari.mean().reset_index()
    rows = []
    for cohort, target in COHORTS:
        cmp_ = comp_lookup(comps, cohort, target)
        part = per[(per.cohort == cohort) & (per.target == target)]
        for r in part.itertuples():
            rows.append(dict(cohort=cohort, target=target, patient=r.object, epoch=r.epoch, tessera=r.ari,
                             tuned_control=cmp_.loc[r.object, 'NestedControl'],
                             same_graph_mean=cmp_.loc[r.object, 'SameGraphMean']))
    budget = pd.DataFrame(rows)
    budget['minus_control'] = budget.tessera - budget.tuned_control
    budget.to_csv(SUMMARY / 'fig_budget.csv', index=False)
    t3 = pd.read_csv(SUMMARY / 'table_E3.csv')
    keep = [c_ for c_ in t3.columns if c_ in ('cohort', 'unit', 'target', 'K_ref', 'geometry', 'K', 'offset')
            or c_.startswith('Tessera120-') or c_.startswith('Tessera200-')]
    t3[keep].to_csv(SUMMARY / 'fig_k.csv', index=False)
    lf = pd.read_csv(E3 / 'E3_label_free_K.csv')
    t3m = t3.merge(lf, left_on=['unit', 'target', 'K'], right_on=['unit', 'target', 'K_label_free'])
    t3m[keep + ['K_label_free']].to_csv(SUMMARY / 'fig_k_label_free.csv', index=False)


PATIENT_LABEL = {'NSCLC_Lung5_Rep2': 'Lung5\\_Rep2', 'NSCLC_Lung6': 'Lung6', 'NSCLC_Lung9_Rep1': 'Lung9\\_Rep1',
                 'NSCLC_Lung12': 'Lung12', 'NSCLC_Lung13': 'Lung13', 'BCC_PatientA': 'BCC A', 'BCC_PatientB': 'BCC B',
                 'BCC_PatientC': 'BCC C', 'BCC_PatientD': 'BCC D'}
BLOCK = {('NSCLC', 'primary'): 'NSCLC regions', ('BCC', 'primary'): 'BCC cell types',
         ('HER2ST', 'all'): 'HER2ST, all labels', ('HER2ST', 'boundary_T1'): 'HER2ST, tumor boundary'}


def n3(x):
    return 'NA' if pd.isna(x) else f'{x:.3f}'


def write_tables(tables: dict) -> None:
    src = MACROS.parent
    t1 = tables['E1']
    lines = []
    for (cohort, target), label in BLOCK.items():
        part = t1[(t1.cohort == cohort) & (t1.target == target)]
        lines.append(f'\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{label}}}}}\\\\')
        for r in part.itertuples():
            name = PATIENT_LABEL.get(r.patient, r.patient)
            lines.append(f'{name} & {n3(r.ari_120)} & {n3(r.ari_200)} & {n3(r.ari_400)} & '
                         f'{n3(r.ari_nested_epoch)} ({int(r.nested_epoch)}) & {n3(r.ari_oracle_epoch)} ({int(r.oracle_epoch)}) & '
                         f'{n3(r.NestedControl)} & {n3(r.SameGraphMean)}\\\\')
    bodies = {'RTableBudget': '\n'.join(lines)}
    t2 = tables['E2']
    lines = []
    for (cohort, target), label in BLOCK.items():
        part = t2[(t2.cohort == cohort) & (t2.target == target)]
        lines.append(f'\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{label}}}}}\\\\')
        for r in part.itertuples():
            k, w, tmp, e = r.selected.split('|')
            cfg = f'{k[1:]}/{w[1:]}/{tmp[1:]}/{e[1:]}'
            lines.append(f'{PATIENT_LABEL.get(r.patient, r.patient)} & {cfg} & {n3(r.tessera_nested)} & {n3(r.tessera_default)} & '
                         f'{n3(r.nested_control)} & {n3(r.same_graph_mean_selected_k)} & {n3(r.matched_scale_control)} & '
                         f'{n3(r.nogate_selected)} & {n3(r.nocontrast_selected)}\\\\')
    bodies['RTableTuning'] = '\n'.join(lines)
    bodies['RTableK'] = tables.get('E3summary') or ''
    if tables.get('E4') is not None and len(tables['E4']):
        t4 = tables['E4']
        lines = []
        for r in t4.sort_values(['method', 'variant', 'cohort', 'target', 'clustering']).itertuples():
            ctl = getattr(r, 'minus_NestedControl', None)
            mean = getattr(r, 'minus_SameGraphMean', None)
            ctl_s = f"{ctl['mean']:+.3f} ({ctl['n_neg']}/{ctl['n']})" if isinstance(ctl, dict) else '--'
            mean_s = f"{mean['mean']:+.3f} ({mean['n_neg']}/{mean['n']})" if isinstance(mean, dict) else '--'
            tgt = {'primary': 'regions', 'cell_type': 'cell types', 'all': 'all labels', 'boundary_T1': 'boundary'}[r.target]
            clus = 'K-means' if r.clustering == 'kmeans_matched' else 'native'
            variant = '' if r.variant in ('', 'default', 'nan') or pd.isna(r.variant) else f' ({r.variant})'
            lines.append(f'{r.method}{variant} & {r.cohort} {tgt} & {clus} & {r.mean_ari:.3f} & '
                         f'{ctl_s} & {mean_s}\\\\')
        bodies['RTableNative'] = '\n'.join(lines)
    bodies.setdefault('RTableNative', '')
    write_bodies(src, bodies)


def write_bodies(src, bodies):
    out = ['% Generated by analyze_r1.py; table bodies as macros so they expand inside tabular.']
    for name, body in bodies.items():
        out.append(f'\\newcommand{{\\{name}}}{{%\n{body.strip()}\n}}')
    (src / 'revision-tables.tex').write_text('\n'.join(out) + '\n')


def k_table(summary) -> str:
    lines = []
    for (cohort, target), label in BLOCK.items():
        s = summary['E3'].get(f'{cohort}|{target}')
        if not s:
            continue
        for geometry, gl in [('euclidean', 'Euclidean'), ('cosine', 'Unit length')]:
            cells = []
            for off in range(-2, 3):
                v = s[geometry]['by_offset'].get(off, {}).get('Tessera120-TunedControl')
                cells.append('--' if v is None else f"{v['mean']:+.3f} ({v['n_neg']}/{v['n']})")
            lf = s[geometry]['label_free']
            v = lf['Tessera120-TunedControl'] if lf else None
            cells.append('--' if v is None else f"{v['mean']:+.3f} ({v['n_neg']}/{v['n']})")
            lines.append(f'{label} & {gl} & ' + ' & '.join(cells) + '\\\\')
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parts', nargs='+', default=['E1', 'E2', 'E3', 'E4'])
    args = p.parse_args()
    SUMMARY.mkdir(parents=True, exist_ok=True)
    comps = comparators()
    comps.to_csv(SUMMARY / 'comparators.csv', index=False)
    summary, tables = {}, {}
    for part, fn in [('E1', e1), ('E2', e2), ('E3', e3), ('E4', e4)]:
        if part in args.parts:
            summary[part], tables[part] = fn(comps)
    if 'E3' in summary:
        tables['E3summary'] = k_table(summary)
    (SUMMARY / 'R1_summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
    write_macros(summary)
    if {'E1', 'E2'} <= set(tables):
        write_tables(tables)
    if {'E1', 'E3'} <= set(summary):
        figure_data(comps)
    print('wrote', SUMMARY / 'R1_summary.json', 'and', MACROS)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Bind a manually checked figure-to-computation map to current source hashes.

This checks coverage and identities, not mathematical truth by string matching.
The definitions were checked against the frozen runners and display code.
"""
import re
from common import ROOT, read, dump, sha

EQUATIONS = 32  # 28 main Methods identities, 4 supplementary display definitions


def item(definition, *equations):
    return {'definition': definition, 'equations': list(equations)}


MAP = {
    'figure01-workflow': [
        item('Recorded assays, frames and patient identities; conceptual glyphs'),
        item('Common molecular input and frame-restricted graph', 'normalization', 'pca', 'graph'),
        item('Matched representation branches', 'controls', 'input', 'fusion'),
        item('Specified component interventions', 'gate', 'fusion', 'loss', 'controls'),
        item('Source-only transform, frozen encoder and source centroids', 'transfer', 'centroid'),
        item('Distinct partition, transfer and registered-image readouts', 'partition', 'accuracy', 'registration')],
    'figure13-architecture': [
        item('Common PCA and directed frame graph', 'normalization', 'pca', 'graph'),
        item('Input projection, edge MLP and normalized residual update', 'input', 'gate', 'aggregate'),
        item('Depth attention and embedding projection', 'fusion'),
        item('Reconstruction and detached per-edge contrastive weighting', 'reconstruction', 'contrast', 'loss'),
        item('KMeans versus source centroids; uncalibrated seam', 'kmeans', 'centroid', 'seam')],
    'figure03-comparison': [
        item('Seed-mean ARI of PCA50, six-neighbor mean, nested tuned control and Tessera', 'controls', 'tuned', 'nested', 'partition', 'seed-summary'),
        item('Patient contrasts against PCA50 and the nested tuned control, bounded fields', 'nested', 'seed-summary', 'patient'),
        item('Uncapped same-patient sensitivity contrasts; fixed cohort-modal, not nested controls', 'tuned', 'seed-summary', 'patient'),
        item('Saved lung embeddings re-clustered at the cell-type K', 'kmeans', 'partition')],
    'figure04-ablations': [
        item('Control minus full seed-mean ARI, including the no-contrast permuted-graph null', 'controls', 'loss', 'partition', 'seed-summary'),
        item('Gate versus PCA-feature-difference seam AUROC on identical edges', 'gate', 'auroc'),
        item('Fixed-anchor last-layer gate densities on valid same/different-label edges', 'gate', 'aggregate'),
        item('Full Tessera minus self-only seed-mean ARI', 'controls', 'partition', 'seed-summary')],
    'figure05-nsclc-maps': [item('Fixed largest field, seed 1; display-only matching; three-seed object ARI',
        'graph', 'display-match', 'partition', 'seed-summary') for _ in range(5)],
    'figure06-bcc-maps': [item('Fixed largest field per patient, seed 1; display-only matching; three-seed object ARI',
        'graph', 'display-match', 'partition', 'seed-summary') for _ in range(4)],
    'figure07-breast-histology': [
        item('Author-generated full-tissue spot-location and annotation maps using recorded coordinate calibration; no 10x breast H&E background', 'registration', 'display-match'),
        item('Median-centered fixed coordinate-map ROI and source-calibrated bars; no image background', 'registration', 'display-match')],
    'figure08-breast-mixtures': [
        item('Same fitted predictions at purity thresholds; sample seed SD', 'composition', 'seed-summary'),
        item('Source dominant-category proportion histogram', 'composition'),
        item('Source entropy and fixed-seed node seam; tied-rank Spearman', 'composition', 'seam'),
        item('Named source tumor/DCIS columns; fixed fraction color scale', 'composition')],
    'figure10-patient-transfer': [
        item('Region balanced recall; three-seed means and deterministic neighbor means', 'transfer', 'centroid', 'accuracy', 'tuned'),
        item('Cell-type balanced recall', 'transfer', 'centroid', 'accuracy'),
        item('Region accuracy', 'accuracy'),
        item('Cell-type accuracy', 'accuracy')],
    'figure11-stability': [
        item('Mean ARI over all three seed pairs on complete identical rows', 'stability'),
        item('Valid-reference-edge homogeneity versus method contrast', 'homogeneity', 'seed-summary')],
    'figure15-external-cohort': [
        item('Three-seed ARI of PCA50, same-graph mean, nested tuned control and Tessera on all labels', 'neighborhood', 'controls', 'nested', 'partition', 'seed-summary'),
        item('The same four arms on the tumor-boundary target', 'neighborhood', 'nested', 'partition', 'seed-summary'),
        item('Patient contrasts with sign counts and patient-level bootstrap intervals', 'neighborhood', 'patient', 'seed-summary'),
        item('Edge-level seam AUROC of gate, feature difference and smoothed difference with patient bootstrap', 'neighborhood', 'auroc'),
        item('Three replicate fits of the frozen encoder against the deterministic controls', 'partition', 'seed-summary'),
        item('Weighted mean minus unweighted, shuffled-weight and shuffled-coordinate means', 'neighborhood', 'partition')],
    'figure16-external-maps': [item('Seed 1 partitions on one frame per patient; display-only matching; three-seed ARI badges',
        'neighborhood', 'display-match', 'partition', 'seed-summary') for _ in range(4)],
    'figure17-tissue-landscape': [
        item('Tessera minus the same-graph six-neighbor mean for every object', 'neighborhood', 'partition', 'seed-summary'),
        item('Tessera minus the label-chosen best simple control and minus PCA50', 'tuned', 'partition', 'seed-summary'),
        item('Neighborhood size of the best uniform mean with its ARI', 'tuned', 'partition'),
        item('Gate and feature-difference seam AUROC for every object', 'auroc')],
    'figure02-cohort': [
        item('Source and retained cardinalities from input.json; no fitted-score selection'),
        item('Distinct retained frame identities', 'graph'),
        item('Valid reference-class cardinality', 'kmeans'),
        item('Returned PCA explained-variance ratio sum', 'variance')],
    'figure14-target-lineage': [
        item('Source annotation constructions are distinct', 'composition'),
        item('Within-object reference class count and label-free encoder fit', 'kmeans', 'loss'),
        item('Patient-disjoint fitted parameters; observed target geometry', 'transfer', 'centroid'),
        item('Registered image and measured-channel inspection', 'registration', 'display-normalization'),
        item('Patient contrasts, optimization spread and gate readouts', 'patient', 'seed-summary', 'seam')],
    'figure09-colorectal-proteins': [
        item('Display-matched lv2 labels mapped to modal supplied compartments', 'display-match', 'partition'),
        item('Actual intensities; nonnegative log1p; field 2nd/98th percentiles; millimeters x1000',
             'display-normalization', 'registration')],
    'figure12-efficiency-mechanism': [
        item('Mean saved total elapsed time; max peak allocated GiB'),
        item('Fixed anchor reconstruction and contrastive loss histories', 'reconstruction', 'contrast', 'loss')],
    'supplement01-hnscc': [item('Complete selected 6000 rows; rare/unknown/unmatched gray; original-score labels preserved; seed 1', 'display-match', 'partition')],
    'supplement02-mibi': [item('Three measured fields; signed source exception; display-only matching',
        'normalization', 'display-match', 'partition') for _ in range(3)],
    'supplement03-rna-markers': [item('Saved marker expression, not re-logged; field 0th/98th percentiles',
        'normalization', 'display-normalization') for _ in range(2)],
    'supplement05-scale-controls': [
        item('Neighbor-mean ARI across neighborhood sizes per lung patient', 'tuned', 'partition', 'seed-summary'),
        item('BANKSY-style ARI across neighborhood sizes per BCC patient', 'tuned', 'partition', 'seed-summary')],
}


def main():
    manuscript = ROOT/'source/manuscript/paper.tex'
    supporting = ROOT/'source/manuscript/supporting.tex'
    methods = manuscript.read_text().split(r'\section{Materials and methods}')[1].split(r'\section{Results}')[0]
    supplementary = supporting.read_text().split(r'\section*{Supplementary methods}')[1].split(r'\section*{Complete')[0]
    labels = re.findall(r'\\label\{eq:([^}]+)\}', methods) + re.findall(r'\\label\{eq:([^}]+)\}', supplementary)
    assert len(labels) == len(set(labels)) == EQUATIONS, len(labels)
    catalog = read(ROOT/'source/manuscript/ledgers/figure_catalog.json')['figures']
    assert {f['filename'][:-4] for f in catalog} == set(MAP)
    records = []
    for figure in catalog:
        panels = MAP[figure['filename'][:-4]]
        assert len(panels) == figure['panel_labels']
        assert all(set(p['equations']) <= set(labels) for p in panels)
        records.append({'figure': figure['number'], 'filename': figure['filename'],
            'pdf_sha256': figure['pdf_sha256'],
            'panel_methods': {chr(97+i): p for i,p in enumerate(panels)}})
    code = ['source/scripts/prepare_data.py', 'source/scripts/run_experiments.py',
            'source/scripts/run_transfer.py', 'source/scripts/run_followup.py',
            'source/scripts/render_figures.py', 'source/scripts/figure_layout.py',
            'source/scripts/artwork.py', 'source/scripts/audit_followup.py',
            'source/scripts/render_external_figures.py', 'source/scripts/make_external_tables.py',
            'validation-extension/run_extension.py', 'validation-external/run_her2st_validation.py',
            'validation-external/run_enhanced_targets.py']
    dump(ROOT/'qa/figure_methods_audit.json', {
        'manuscript_sha256': sha(manuscript), 'supporting_sha256': sha(supporting), 'equation_count': len(labels),
        'equation_labels': labels, 'figures': records,
        'computation_source_sha256': {p:sha(ROOT/p) for p in code},
        'scope': 'Manual implementation comparison plus automated figure/equation/hash coverage; not independent scientific validation.'})
    print(f'VERIFIED: {len(records)} figures / {sum(len(r["panel_methods"]) for r in records)} semantic sections mapped to Methods; {len(labels)} equation identities')


if __name__ == '__main__':
    main()

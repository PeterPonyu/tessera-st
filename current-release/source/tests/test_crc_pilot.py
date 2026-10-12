"""Bind paper-facing CRC claims to the separate external pilot and its targets."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from hashlib import sha256

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / 'validation-independent-crc'
LED = ROOT / 'source/manuscript/ledgers'


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def test_crc_receipts_bind_frozen_code_and_every_output():
    p = read(PILOT/'results/run-receipt.json')
    q = read(PILOT/'annotation-qc/run-receipt.json')
    assert p['bindings']['protocol_sha256'] == digest(PILOT/'PROTOCOL.md') == read(PILOT/'protocol-lock.json')['protocol_sha256']
    assert p['bindings']['script_sha256'] == digest(PILOT/'run_pilot.py')
    assert q['primary_receipt_sha256'] == digest(PILOT/'results/run-receipt.json')
    assert q['protocol_sha256'] == digest(PILOT/'ANNOTATION_QC_PROTOCOL.md')
    assert q['script_sha256'] == digest(PILOT/'analyze_annotation_qc.py')
    for name, expected in p['bindings']['frozen_sha256'].items():
        assert digest(ROOT/name) == expected
    for receipt in [p,q]:
        for name, expected in receipt['output_sha256'].items():
            assert digest(PILOT/name) == expected


def test_crc_is_three_patients_without_tuning_or_patient_pseudoreplication():
    r = read(PILOT/'results/run-receipt.json')
    assert r['patients'] == 3 and r['fit_count'] == 18 and r['score_rows'] == 36
    assert set(r['unique_patient_ids']) == {'A120838','A595688','A551763'}
    assert read(PILOT/'cohort-audit.json')['untouched_within_audited_scope']
    assert read(PILOT/'results/execution.json')['torch_num_threads'] == 1
    for s in r['statistics'].values():
        assert s['n_independent_patients'] == 3
    table=pd.read_csv(PILOT/'results/arms.csv')
    assert set(table.arm) == {'PCA50','UniformSameGraph','Tessera','NoGate'}
    for path in (PILOT/'results').glob('*/*-seed*.json'):
        assert read(path)['auxiliary_fit_K'] == 2


def test_crc_qc_amendment_keeps_graph_spots_and_only_removes_the_qc_target():
    for path in sorted((PILOT/'inputs').glob('*/input.npz')):
        d=np.load(path,allow_pickle=False)
        q=np.load(PILOT/'annotation-qc'/path.parent.name/'target.npz',allow_pickle=False)
        is_qc=np.char.lower(np.char.strip(d['ref_name'])) == 'exclude'
        assert (q['ref'][is_qc] == -1).all()
        assert np.array_equal(q['ref']>=0,(d['ref']>=0)&~is_qc)
        assert len(q['ref']) == len(d['x'])
    assert read(PILOT/'annotation-qc/run-receipt.json')['new_model_fits'] == 0


def test_crc_macro_values_equal_saved_patient_contrasts():
    ledger=read(LED/'crc_numbers.json')
    assert ledger['source_receipt_sha256'] == digest(PILOT/'annotation-qc/run-receipt.json')
    assert ledger['script_sha256'] == digest(ROOT/'source/scripts/make_crc_tables.py')
    r=read(PILOT/'annotation-qc/run-receipt.json')
    n=ledger['numbers']
    pairs={'CrcGate':'Gate','CrcRaw':'RawFeature','CrcGateRawDelta':'Gate_minus_RawFeature',
           'CrcTessMeanDelta':'Tessera_minus_UniformSameGraph','CrcNoGateDelta':'NoGate_minus_Tessera'}
    for macro, metric in pairs.items():
        assert abs(float(n[macro])-r['statistics'][metric]['mean']) <= .0005
    assert n['CrcPatients'] == '3' and n['CrcTissueSpots'] == '1934' and n['CrcQcSpots'] == '1758'
    assert set(re.findall(r'\\newcommand\{\\(Crc[A-Za-z]+)\}',(LED/'crc_numbers.tex').read_text())) == set(n)


def test_crc_claims_report_weak_gate_information_and_unsupported_added_value():
    r=read(PILOT/'annotation-qc/run-receipt.json')
    assert r['pilot_criteria']['gate_discrimination']
    assert not r['pilot_criteria']['gate_advantage_over_raw']
    assert not r['pilot_criteria']['learned_ari_gain_over_mean_margin_0_02']
    text=(ROOT/'source/manuscript/paper.tex').read_text()
    assert 'not universally devoid of seam information' in text
    assert 'valdeolivas2024crc' in text
    assert 'disclosed QC sensitivity' in text
    assert 'auxiliary head' in text
    assert 'Calibration and selective abstention would require a separate protocol' in text
    assert r'\input{ledgers/crc_numbers.tex}' in text


def test_crc_nulls_and_original_target_are_both_reported():
    for folder in ['results','annotation-qc']:
        table=pd.read_csv(PILOT/folder/'label_shuffle.csv')
        assert len(table)==9000
        med=table.groupby(['section','score']).auroc.median()
        assert med.between(.45,.55).all()
    text=(ROOT/'source/manuscript/supporting.tex').read_text()
    assert r'\tableinput ledgers/crc_rows.tex' in text
    assert r'\tableinput ledgers/crc_raw_rows.tex' in text
    assert 'do not interpret QC seams as biological boundaries' in text

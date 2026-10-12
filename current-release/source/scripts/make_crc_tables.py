#!/usr/bin/env python3
"""Generate CRC pilot paper values from verified, separately receipted outputs."""
from pathlib import Path
import pandas as pd
from common import ROOT, read, sha, dump


def main():
    pilot = ROOT / 'validation-independent-crc'
    ledger = ROOT / 'source/manuscript/ledgers'
    result = read(pilot / 'annotation-qc/run-receipt.json')
    for name, digest in result['output_sha256'].items():
        assert sha(pilot / name) == digest
    assert result['script_sha256'] == sha(pilot / 'analyze_annotation_qc.py')
    ari = pd.read_csv(pilot / 'annotation-qc/patients.csv').set_index('patient')
    seam = pd.read_csv(pilot / 'annotation-qc/patient_seams.csv').set_index('patient')
    raw_ari = pd.read_csv(pilot / 'results/patients.csv').set_index('patient')
    raw_seam = pd.read_csv(pilot / 'results/patient_seams.csv').set_index('patient')
    stats = result['statistics']
    metadata = {p['patient']:p for p in [read(f) for f in sorted((pilot/'inputs').glob('*/input.json'))]}
    numbers = dict(CrcPatients=str(len(ari)), CrcTissueSpots=str(sum(p['n_tissue'] for p in metadata.values())),
        CrcLabeledSpots=str(sum(p['n_annotated'] for p in metadata.values())),
        CrcQcExcluded=str(sum(p['class_counts'].get('exclude',0) for p in metadata.values())),
        CrcQcSpots=str(sum(p['n_annotated']-p['class_counts'].get('exclude',0) for p in metadata.values())),
        CrcTessMeanNeg=str(int((ari.Tessera<ari.UniformSameGraph).sum())),
        CrcRawTessMeanNeg=str(int((raw_ari.Tessera<raw_ari.UniformSameGraph).sum())),
        CrcTessMeanDelta=f'{stats["Tessera_minus_UniformSameGraph"]["mean"]:+.3f}',
        CrcTessMeanLow=f'{stats["Tessera_minus_UniformSameGraph"]["lower"]:+.3f}',
        CrcTessMeanHigh=f'{stats["Tessera_minus_UniformSameGraph"]["upper"]:+.3f}',
        CrcGate=f'{stats["Gate"]["mean"]:.3f}', CrcGateLow=f'{stats["Gate"]["lower"]:.3f}',
        CrcGateHigh=f'{stats["Gate"]["upper"]:.3f}', CrcRaw=f'{stats["RawFeature"]["mean"]:.3f}',
        CrcGateRawDelta=f'{stats["Gate_minus_RawFeature"]["mean"]:+.3f}',
        CrcGateRawLow=f'{stats["Gate_minus_RawFeature"]["lower"]:+.3f}',
        CrcGateRawHigh=f'{stats["Gate_minus_RawFeature"]["upper"]:+.3f}',
        CrcNoGateDelta=f'{stats["NoGate_minus_Tessera"]["mean"]:+.3f}',
        CrcNoGateLow=f'{stats["NoGate_minus_Tessera"]["lower"]:+.3f}',
        CrcNoGateHigh=f'{stats["NoGate_minus_Tessera"]["upper"]:+.3f}',
        CrcFitSeconds=f'{read(pilot/"results/run-receipt.json")["total_fit_seconds"]:.0f}')
    (ledger/'crc_numbers.tex').write_text(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k,v in numbers.items()))
    rows=[]
    for patient, row in ari.iterrows():
        m=metadata[patient];s=seam.loc[patient]
        count=m['n_annotated']-m['class_counts'].get('exclude',0)
        rows.append(f'{patient} & {m["n_tissue"]} & {count} & {row.PCA50:.3f} & {row.UniformSameGraph:.3f} & {row.Tessera:.3f} & {row.NoGate:.3f} & {s.Gate:.3f} & {s.RawFeature:.3f} & {s.UniformMeanFeature:.3f} \\\\\n')
    (ledger/'crc_rows.tex').write_text(''.join(rows))
    rows=[]
    for patient in ari.index:
        rows.append(f'{patient} & {raw_ari.loc[patient,"Tessera"]:.3f} & {raw_ari.loc[patient,"UniformSameGraph"]:.3f} & {raw_seam.loc[patient,"Gate"]:.3f} & {raw_seam.loc[patient,"RawFeature"]:.3f} \\\\\n')
    (ledger/'crc_raw_rows.tex').write_text(''.join(rows))
    dump(ledger/'crc_numbers.json', dict(numbers=numbers,
        source_receipt_sha256=sha(pilot/'annotation-qc/run-receipt.json'),
        primary_receipt_sha256=sha(pilot/'results/run-receipt.json'),
        script_sha256=sha(Path(__file__)), n_independent_patients=3))
    print('CRC TABLES',len(ari),'patients',len(numbers),'bound values')


if __name__ == '__main__':
    main()

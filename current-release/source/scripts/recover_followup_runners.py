#!/usr/bin/env python3
"""Recover historical runners only when their exact bytes match saved receipts.

The retained runner added whole-section and transfer analyses successively.
Removing those extensions recovers the earlier versions without guessing any
analysis implementation. --verify checks the recipe and saved files read-only.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from common import ROOT, RESULTS, read, dump, sha

ARCHIVE = ROOT / 'source/provenance/followup-runners'


def remove_extension(text, mode):
    if mode == 'whole':
        start = text.index('# ------------------------------------------------------------------ whole\n')
        end = text.index('# --------------------------------------------------------------- transfer\n')
        text = text[:start] + text[end:]
        start = text.index('  whole     ')
        end = text.index('  transfer  ')
        text = text[:start] + text[end:]
        text = text.replace(", 'whole'", '')
        text = text.replace("    if args.analysis in ('whole', 'all'): record['whole'] = whole_sections(args.device)\n", '')
    elif mode == 'transfer':
        start = text.index('# --------------------------------------------------------------- transfer\n')
        end = text.index('def main():', start)
        text = text[:start] + text[end:]
        start = text.index('  transfer  ')
        end = text.index('"""', start)
        text = text[:start] + text[end:]
        text = text.replace(", 'transfer'", '')
        text = text.replace("    if args.analysis in ('transfer', 'all'): record['transfer'] = transfer()\n", '')
    else:
        raise ValueError(mode)
    return text


def preserve(blob, expected, filename, verify):
    if hashlib.sha256(blob).hexdigest() != expected:
        raise ValueError(f'Recovered {filename} does not match its original receipt')
    target = ARCHIVE / expected / filename
    if target.exists():
        if target.read_bytes() != blob:
            raise ValueError(f'Conflicting snapshot: {target}')
    elif verify:
        raise FileNotFoundError(target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
    return str(target.relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    source = ROOT / 'source/scripts/run_followup.py'
    followup = RESULTS / 'followup-20261010'
    original = source.read_text()
    transfer_source = remove_extension(original, 'whole')
    first_source = remove_extension(transfer_source, 'transfer')
    records = []
    for modes, text, removed in [
        (['transfer'], transfer_source, ['whole']),
        (['scale', 'rescore', 'nulls'], first_source, ['whole', 'transfer'])]:
        expected = read(followup / f'run-{modes[0]}.json')['script_sha256']
        for mode in modes:
            if read(followup / f'run-{mode}.json')['script_sha256'] != expected:
                raise ValueError('Historical modes no longer share their recorded producer')
        path = preserve(text.encode(), expected, 'run_followup.py', args.verify)
        records.append(dict(modes=modes, sha256=expected, path=path,
                            removed_extensions=removed, verification='exact SHA256 match to original receipts'))
    # Preserve the existing gate-diagnostic producer before extending its audit.
    gate_receipt = read(followup / 'run-gate-weights-20261011.json')
    gate_sha = gate_receipt['script_sha256']
    audit = ROOT / 'source/scripts/audit_followup.py'
    saved_audit = ARCHIVE / gate_sha / 'audit_followup.py'
    blob = audit.read_bytes() if sha(audit) == gate_sha else saved_audit.read_bytes()
    gate_path = preserve(blob, gate_sha, 'audit_followup.py', args.verify)
    record = dict(recovery_date='2026-10-11', retained_runner_sha256=sha(source),
                  recovery_script_sha256=sha(__file__), snapshots=records,
                  gate_diagnostic_producer=dict(path=gate_path, sha256=gate_sha),
                  scope='Byte-exact source recovery; original receipts and numerical outputs unchanged; no rerun.')
    manifest = ARCHIVE / 'recovery.json'
    if args.verify:
        if read(manifest) != record:
            raise ValueError('Recovery record differs from the verified recipe')
    else:
        dump(manifest, record)
    for item in records:
        print('VERIFIED RECOVERED SOURCE', ', '.join(item['modes']), item['sha256'])
    print('VERIFIED preserved gate-diagnostic producer', gate_sha)


if __name__ == '__main__':
    main()

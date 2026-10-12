#!/usr/bin/env python3
"""Package editable document sources; exclude data, cached fits and stale art."""
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET
import zipfile
from common import ROOT, dump, read, sha


def main():
    required = ['qa/build_manifest.json', 'qa/document_qa_manifest.json',
                'qa/visual_review.json', 'qa/regression-tests.xml',
                'qa/statistical_figure_style_audit.json', 'qa/schematic_style_audit.json',
                'qa/figure_contract_audit.json', 'FIGURE_STYLE_AND_VALIDATION_PLAN.md',
                'qa/figure_methods_audit.json',
                'qa/manuscript_structure_audit.json',
                'qa/submission_number_audit.json',
                'qa/scientific-reports/format-check.json',
                'qa/resumed-document-review/render-manifest.json',
                'qa/resumed-document-review/visual-review.json',
                'qa/closeout/crc-reconciliation.json',
                'qa/closeout/resumed-crc-original-verification.json',
                'SUBMISSION_CLOSEOUT.md', 'COVER_LETTER_DRAFT.md', 'CITATION.cff',
                'source/scientific-reports.json',
                'qa/followup_evidence_audit.json', 'FOLLOWUP_PROVENANCE.md',
                'REVISION_REPORT.md', 'EDITABLE_SOURCE_README.md',
                'validation-extension/RESULT_EXTENSION.md']
    for filename in required:
        if not (ROOT / filename).is_file():
            raise FileNotFoundError(filename)
    build_record=read(ROOT/'qa/build_manifest.json')
    build=build_record['documents']
    assert build_record['tests_executed']
    assert build_record['regression_xml_sha256']==sha(ROOT/'qa/regression-tests.xml')
    for filename,expected in build_record['test_source_sha256'].items():
        assert sha(ROOT/filename)==expected
    renders=read(ROOT/'qa/document_qa_manifest.json')['documents']
    review=read(ROOT/'qa/visual_review.json')
    assert review['status']=='completed'
    methods=read(ROOT/'qa/figure_methods_audit.json')
    assert methods['manuscript_sha256']==sha(ROOT/'source/manuscript/paper.tex')
    assert methods['supporting_sha256']==sha(ROOT/'source/manuscript/supporting.tex')
    for filename,expected in methods['computation_source_sha256'].items():
        assert sha(ROOT/filename)==expected
    structure=read(ROOT/'qa/manuscript_structure_audit.json')
    assert structure['status']=='PASS'
    assert not structure['counts']['BLOCKER'] and not structure['counts']['MAJOR']
    assert structure['manuscript_sha256']==sha(ROOT/'source/manuscript/paper.tex')
    assert structure['supporting_sha256']==sha(ROOT/'source/manuscript/supporting.tex')
    assert structure['audit_source_sha256']==sha(ROOT/'source/scripts/audit_manuscript_structure.py')
    numbers=read(ROOT/'qa/submission_number_audit.json')
    assert numbers['status']=='PASS'
    assert numbers['audit_source_sha256']==sha(ROOT/'source/scripts/audit_submission_numbers.py')
    for filename,expected in numbers['input_sha256'].items():
        assert sha(ROOT/filename)==expected
    assert review['individual_figures_inspected']==21
    journal=read(ROOT/'qa/scientific-reports/format-check.json')
    assert journal['status']=='PASS'
    assert journal['canonical_source_sha256']==sha(ROOT/'source/manuscript/paper.tex')
    assert journal['canonical_supplement_sha256']==sha(ROOT/'source/manuscript/supporting.tex')
    assert journal['configuration_sha256']==sha(ROOT/'source/scientific-reports.json')
    assert journal['generator_sha256']==sha(ROOT/'source/scripts/build_scientific_reports.py')
    assert journal['abstract_words']<=200 and journal['main_figures']+journal['main_tables']<=8
    for key,doc in journal['documents'].items():
        assert sha(ROOT/'output/scientific-reports'/('scientific-reports-'+key+'.pdf'))==doc['sha256']
    journal_review = read(ROOT/'qa/resumed-document-review/visual-review.json')
    assert journal_review['status'] == 'PASS'
    assert journal_review['render_manifest_sha256'] == sha(ROOT/'qa/resumed-document-review/render-manifest.json')
    for key, doc in journal['documents'].items():
        assert journal_review['documents'][key]['sha256'] == doc['sha256']
        assert journal_review['documents'][key]['pages_inspected_in_overview'] == doc['pages']
    crc = read(ROOT/'validation-independent-crc/annotation-qc/run-receipt.json')
    assert crc['patients'] == 3 and crc['new_model_fits'] == 0
    for relative, expected in crc['output_sha256'].items():
        assert sha(ROOT/'validation-independent-crc'/relative) == expected
    followup_audit=read(ROOT/'qa/followup_evidence_audit.json')
    assert followup_audit['audit_source_sha256']==sha(ROOT/'source/scripts/audit_followup.py')
    assert followup_audit['unrecovered_historical_modes']==[]
    recovery=read(ROOT/'source/provenance/followup-runners/recovery.json')
    assert recovery['retained_runner_sha256']==sha(ROOT/'source/scripts/run_followup.py')
    assert recovery['recovery_script_sha256']==sha(ROOT/'source/scripts/recover_followup_runners.py')
    for snapshot in recovery['snapshots']:
        assert sha(ROOT/snapshot['path'])==snapshot['sha256']
    for mode,binding in followup_audit['historical_run_receipts'].items():
        assert binding['historical_source_available']
        assert sha(ROOT/binding['historical_source_path'])==binding['recorded_producer_sha256']
    for name,expected in followup_audit['current_output_sha256'].items():
        assert sha(ROOT/'results/followup-20261010'/name)==expected
    suites=ET.parse(ROOT/'qa/regression-tests.xml').getroot().iter('testsuite')
    for suite in suites:
        assert all(int(suite.get(key,'0'))==0 for key in ['failures','errors','skipped'])
        assert int(suite.get('tests','0'))>=51
    for name,expected in build.items():
        assert sha(ROOT/'output/pdf'/name)==expected
        assert renders[name]['sha256']==review['documents'][name]['sha256']==expected
        assert review['documents'][name]['all_pages_inspected']
    sources = []
    for path in sorted((ROOT / 'source').rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(p in {'__pycache__', '.omc', '.omx', '.cursor.cloud'} for p in relative.parts):
            continue
        if (path.suffix in {'.pyc', '.bak','.aux','.log','.fls','.fdb_latexmk'} or '.bak-' in path.name
                or path.name.endswith('.synctex.gz')
                or path.name == 'figure01-workflow.tex'):
            continue
        if path.parent.name == 'assets' and path.suffix == '.pdf':
            continue
        sources.append(path)
    sources.extend(ROOT / name for name in required + ['PROTOCOL.md', 'results/run-lock.json',
                       'results/transfer/run-lock.json'])
    # Follow-up analysis outputs (small tables and summaries; rebuilt whole-section inputs are excluded).
    followup = ROOT / 'results/followup-20261010'
    sources.extend(sorted(p for p in followup.iterdir() if p.is_file() and p.suffix in {'.json', '.csv', '.log'}))
    sources.extend(sorted((ROOT / 'qa/independent-review-20261010').glob('*.md')))
    # Independent-cohort and extension analyses: protocols, runners, receipts and small result tables.
    # Public count matrices and label tables (validation-external/raw) are excluded and re-fetched by the runner.
    for folder in ('validation-external', 'validation-extension'):
        for path in sorted((ROOT / folder).rglob('*')):
            relative = path.relative_to(ROOT)
            if not path.is_file() or any(p in {'raw', 'logs', '__pycache__'} for p in relative.parts):
                continue
            if path.suffix in {'.pyc', '.npz', '.part'} or path in sources:
                continue
            sources.append(path)
    # Retain compact pilot inputs and predictions; public count archives stay excluded.
    for path in sorted((ROOT/'validation-independent-crc').rglob('*')):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(p in {'raw', 'logs', '__pycache__', 'runtime-fourthread-partial'} for p in relative.parts):
            continue
        if path.suffix == '.pyc':
            continue
        sources.append(path)
    target = ROOT / 'output/tessera-scientific-reports-source-resumed-20261011.zip'
    sources=list(dict.fromkeys(sources))
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    dump(ROOT / 'qa/source_archive_contents.json', {'files': manifest,
         'excluded': 'large matrices, cached primary fits, model weights, rebuilt whole-section inputs, public count archives, backups, old workflow TeX, stale icon PDFs, caches and compiler intermediates; compact CRC PCA inputs and saved predictions are retained'})
    sources.append(ROOT / 'qa/source_archive_contents.json')
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sources:
            archive.write(path, path.relative_to(ROOT))
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(sources)
        assert set(archive.namelist())==set(manifest)|{'qa/source_archive_contents.json'}
        for name,expected in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==expected,name
    print('PACKAGED', target, len(sources), 'files', sha(target))


if __name__ == '__main__':
    main()

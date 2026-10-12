#!/usr/bin/env python3
"""Bind the manual overview-sheet inspection of the Scientific Reports PDFs to their hashes.

Run only after the overview sheets written by render_srep_qa.py for the CURRENT build have been opened
(every page of every changed PDF). Unchanged documents keep their earlier attestation.
"""
from common import ROOT, dump, read, sha

OBSERVED = {
    'manuscript': 'All 21 pages viewed on the overview sheets of this build (text pages 1-14; figures, legends and the table '
                  'after the references on pages 15-21). No clipping, overlap, stranded heading or unresolved text observed.',
    'manuscript-inline': 'All 20 pages viewed on the overview sheets of this build. Figure 1 and Table 1 sit on page 3 (after the '
                         'paragraph that first cites them); Figures 2-7 follow their first citation on the same page or within '
                         'two pages; Figure 7 occupies a float page that ends about 20% short of the bottom. No clipping, overlap, '
                         'stranded heading or unresolved text observed.',
}


def main():
    path = ROOT / 'qa/resumed-document-review/visual-review.json'
    review = read(path)
    manifest = read(ROOT / 'qa/resumed-document-review/render-manifest.json')
    for key, doc in manifest['documents'].items():
        record = review['documents'].get(key)
        if record and record['sha256'] == doc['sha256']:
            continue
        assert key in OBSERVED, f'{key} changed but has no recorded inspection'
        review['documents'][key] = {
            'sha256': doc['sha256'], 'pages_inspected_in_overview': doc['pages'],
            'overview_sheet_sha256': [s['sha256'] for s in doc['overview_sheets']],
            'full_size_pages': [], 'result': OBSERVED[key]}
    review.update({'status': 'PASS', 'date': '2026-10-11', 'reviewer': 'operator; no independent peer review',
                   'render_manifest_sha256': sha(ROOT / 'qa/resumed-document-review/render-manifest.json')})
    dump(path, review)
    print({k: v['pages_inspected_in_overview'] for k, v in review['documents'].items()})


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Read-only final journal PDF text and panel-label QA; no document rebuilding."""
from pathlib import Path
import hashlib
import json
import re
import fitz

ROOT = Path(__file__).resolve().parents[1]


def main():
    documents = {}
    for name in ['scientific-reports-manuscript.pdf', 'scientific-reports-supplement.pdf']:
        path = ROOT / 'output/scientific-reports' / name
        markers = {}
        with fitz.open(path) as document:
            for page_index, page in enumerate(document):
                text = page.get_text()
                assert text.strip(), 'Blank text page'
                assert '\ufffd' not in text, 'Replacement glyph'
                found = []
                for block in page.get_text('dict')['blocks']:
                    for line in block.get('lines', []):
                        for span in line['spans']:
                            if 'bold' in span['font'].lower() and re.fullmatch('[a-z]', span['text'].strip()):
                                found.append({'letter': span['text'].strip(), 'size_points': span['size'],
                                              'font': span['font'], 'bbox': list(span['bbox'])})
                if found:
                    assert all(item['size_points'] >= 12 for item in found), (name, page_index + 1, found)
                    markers[str(page_index + 1)] = found
            documents[name] = {'path': str(path.relative_to(ROOT)),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'page_count': len(document),
                'statistical_panel_markers_at_final_placement': markers}
    count = sum(len(values) for document in documents.values()
                for values in document['statistical_panel_markers_at_final_placement'].values())
    assert count == 75, count
    print(json.dumps({'status': 'passed', 'documents': documents, 'panel_markers': count,
                     'text_checks': 'nonblank text on every page; no replacement glyphs',
                     'scope': 'Main journal manuscript plus supplement; inline copy has identical scientific content and is separately hash-bound by the parent format check. Visual review is separate.'}, indent=2))


if __name__ == '__main__':
    main()

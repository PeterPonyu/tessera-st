#!/usr/bin/env python3
"""Render every deliverable page and bind visual-review sheets to PDF hashes."""
from __future__ import annotations
import re
import subprocess
import pdfplumber
from PIL import Image, ImageDraw
from build_revision import PANELS
from common import ROOT, dump, sha


def section_words(text, section, following):
    text = re.sub(r'(?<!\\)%[^\n]*', '', text)
    body = text.split(r'\section{' + section + '}')[1].split(following)[0]
    body = re.sub(r'\\(?:cite\w*|ref)\{[^}]+\}', '', body)
    body = re.sub(r'\\[A-Za-z]+(?:\{[^}]*\})?', ' ', body)
    return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", body))


def main():
    record = {}
    for pdf in sorted((ROOT / 'output/pdf').glob('*.pdf')):
        out = ROOT / 'qa/final-pages' / (pdf.stem + '-' + sha(pdf)[:12])
        out.mkdir(parents=True, exist_ok=True)
        prefix = out / pdf.stem
        subprocess.run(['pdftoppm', '-r', '110', '-jpeg', '-jpegopt', 'quality=90',
                        str(pdf), str(prefix)], check=True, capture_output=True)
        pages = sorted((p for p in out.glob(pdf.stem + '-*.jpg')
                        if p.stem.rsplit('-', 1)[1].isdigit()),
                       key=lambda p: int(p.stem.rsplit('-', 1)[1]))
        with pdfplumber.open(pdf) as doc:
            assert len(pages) == len(doc.pages)
            extracted = [p.extract_text() or '' for p in doc.pages]
            assert all(t.strip() for t in extracted), f'Blank text page: {pdf}'
            assert not any('\ufffd' in t for t in extracted), f'Replacement glyph: {pdf}'
            page_markers = {}
            if pdf.stem != 'tessera-figure-atlas':
                for i,page in enumerate(doc.pages):
                    if not re.search(r'Figure\s*(S?\d+)\s*:',extracted[i]):continue
                    markers=[{'letter':w['text'],'size_points':w['size']}
                        for w in page.extract_words(extra_attrs=['fontname','size'])
                        if 'bold' in w['fontname'].lower() and re.fullmatch('[a-z]',w['text'])]
                    if markers:
                        assert all(m['size_points']>=12 for m in markers),(pdf,i+1,markers)
                        page_markers[str(i+1)]=markers
        contacts = []
        for start in range(0, len(pages), 4):
            selected = pages[start:start + 4]
            width, height = 700, 940
            sheet = Image.new('RGB', (2 * width, 2 * height), '#e7eaed')
            draw = ImageDraw.Draw(sheet)
            for i, page in enumerate(selected):
                image = Image.open(page).convert('RGB')
                image.thumbnail((width - 12, height - 32))
                x, y = (i % 2) * width, (i // 2) * height
                draw.text((x + 10, y + 8), f'{pdf.stem} | page {start+i+1}', fill='#263744')
                sheet.paste(image, (x + (width - image.width) // 2, y + 28))
            target = out / (pdf.stem + f'-contact{start//4+1:02d}.jpg')
            sheet.save(target, quality=68, optimize=True)
            contacts.append({'path': str(target.relative_to(ROOT)),
                             'pages': list(range(start + 1, start + 1 + len(selected))),
                             'sha256': sha(target)})
        record[pdf.name] = {'sha256': sha(pdf), 'page_count': len(pages),
                           'page_images': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p)} for p in pages],
                           'contact_sheets': contacts,
                           'visual_review_status': 'rendered; manual inspection is separate'}
        if pdf.stem != 'tessera-figure-atlas':
            record[pdf.name]['statistical_panel_markers_at_final_placement']=page_markers
            record[pdf.name]['figure_caption_pages'] = {
                number: i+1 for i,text in enumerate(extracted)
                for number in re.findall(r'Figure\s*(S?\d+)\s*:',text)}
    assert sum(len(markers) for document in record.values()
        for markers in document.get('statistical_panel_markers_at_final_placement',{}).values())==sum(PANELS)-16
    paper = (ROOT / 'source/manuscript/paper.tex').read_text()
    citations = sorted(set(k.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}', paper) for k in group.split(',')))
    dump(ROOT / 'qa/document_qa_manifest.json', {'documents': record,
         'section_word_counts': {'introduction': section_words(paper, 'Introduction', r'\begin{figure}'),
                                 'discussion': section_words(paper, 'Discussion', r'\section*')},
         'cited_reference_count': len(citations), 'citation_keys': citations,
         'numbered_equation_count': len(re.findall(r'\\label\{eq:', paper)),
         'renderer': 'pdftoppm 110 dpi JPEG; PIL contact sheets',
         'text_checks': 'nonblank text on every page; no replacement glyphs'})
    for name, item in record.items():
        print(name, item['page_count'], 'pages', len(item['contact_sheets']), 'review sheets')


if __name__ == '__main__':
    main()

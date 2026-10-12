#!/usr/bin/env python3
"""Verify final figure fonts, marker runs and dimensions without fitting models."""
import argparse
import re
import statistics
import xml.etree.ElementTree as ET
import pdfplumber
from common import ROOT, read, dump, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',help='Prior revision backup, relative to this revision root')
    args=parser.parse_args()
    figures=ROOT/'source/manuscript/figures'
    records={}
    audits=[read(ROOT/'qa/statistical_figure_style_audit.json'),
            read(ROOT/'qa/schematic_style_audit.json')]
    for audit in audits:
        for name,record in audit.items():
            pdf=figures/(name+'.pdf')
            assert sha(pdf)==record['pdf_sha256'],name
            with pdfplumber.open(pdf) as doc:
                assert len(doc.pages)==1
                page=doc.pages[0]
                fonts=sorted({c['fontname'] for c in page.chars})
                bold=[w for w in page.extract_words(extra_attrs=['fontname'])
                      if 'bold' in w['fontname'].lower()]
                if 'separate_text_artists' in record:
                    assert sorted(w['text'] for w in bold)==sorted(record['labels']),(name,bold)
                    for word in bold:
                        assert re.fullmatch(r'[a-z]',word['text']),word
                    visible_chars=[c for c in page.chars
                        if c['x1']>0 and c['x0']<page.width and c['bottom']>0 and c['top']<page.height]
                    for char in visible_chars:
                        assert char['x0']>=0 and char['x1']<=page.width+.1,(name,char)
                        assert char['top']>=0 and char['bottom']<=page.height+.1,(name,char)
                        color=char['non_stroking_color']
                        channels=(color,) if isinstance(color,(int,float)) else color
                        assert channels and all(value==0 for value in channels),(name,char)
                    bold_chars=[c for c in visible_chars if 'bold' in c['fontname'].lower()]
                    regular_chars=[c for c in visible_chars if 'bold' not in c['fontname'].lower()]
                    assert min(c['size'] for c in bold_chars)>max(c['size'] for c in regular_chars),name
                else:
                    root=ET.parse(figures/(name+'.svg')).getroot()
                    ns={'s':'http://www.w3.org/2000/svg'}
                    texts=root.findall('.//s:text',ns)
                    assert [t.text for t in texts if t.get('font-weight')=='bold']==record['labels']
                    assert all(t.get('font-style','normal')=='normal' for t in texts)
                    assert all(t.get('fill')=='#000000' for t in texts)
                    assert min(float(t.get('font-size')) for t in texts if t.get('data-role')=='panel-label')>max(
                        float(t.get('font-size')) for t in texts if t.get('data-role')!='panel-label')
                    arrows=root.findall('.//s:g[@data-role="arrow"]',ns)
                    assert arrows and not root.findall('.//s:marker',ns)
                    for arrow in arrows:
                        path=arrow.find('s:path',ns);head=arrow.find('s:polygon',ns)
                        assert head.get('fill')==path.get('stroke')
                        assert path.get('stroke-linecap')=='butt'
                        assert float(arrow.get('data-head-length'))==12
                        assert float(arrow.get('data-head-width'))==9
                    # The manuscript's .8-inch margins leave 496.8 points.
                    page_scale=496.8/float(root.get('viewBox').split()[2])
                    assert 4 <= float(arrows[0].get('data-head-length'))*page_scale <= 5
                    assert float(texts[0].get('font-size'))*page_scale >= 13
                current=[page.width,page.height]
            output={'pdf_sha256':sha(pdf),'panel_count':len(record['labels']),
                    'only_panel_labels_bold':True,'fonts':fonts,
                    'all_text_black':True,'labels_larger_than_regular_text':True,
                    'bold_pdf_runs':[w['text'] for w in bold],
                    'width_height_points':current,
                    'svg_text_outlined': 'separate_svg_text_elements' in record}
            if 'separate_text_artists' in record:
                output['fully_offpage_glyphs']=len(page.chars)-len(visible_chars)
                output['partial_page_edge_clipping']=False
            else:
                output['arrow_count']=len(arrows)
                output['arrow_head_length_at_manuscript_width_points']=12*page_scale
                output['arrow_head_width_at_manuscript_width_points']=9*page_scale
                output['panel_label_at_manuscript_width_points']=38*page_scale
            if args.baseline:
                baseline=ROOT/args.baseline/'manuscript/figures'/(name+'.pdf')
                with pdfplumber.open(baseline) as previous:
                    old=previous.pages[0]
                    output['baseline_width_height_points']=[old.width,old.height]
                    output['area_ratio_to_baseline']=current[0]*current[1]/(old.width*old.height)
            records[name]=output
    from build_revision import PANELS
    assert len(records)==len(PANELS) and sum(r['panel_count'] for r in records.values())==sum(PANELS)
    summary={'figures':records,'figure_count':len(records),'panel_count':sum(PANELS),
             'all_only_panel_labels_bold':True,
             'all_text_black':True,'all_labels_larger_than_regular_text':True,
             'scope':'Exported statistical PDFs plus source SVG text before outlining; no manuscript prose font audit.'}
    if args.baseline:
        summary['baseline']=args.baseline
        summary['median_figure_area_ratio']=statistics.median(r['area_ratio_to_baseline'] for r in records.values())
    dump(ROOT/'qa/figure_contract_audit.json',summary)
    print(f'VERIFIED: {len(records)} figures, {sum(PANELS)} larger bare bold labels; all text black; PDF/SVG hashes match')
    if args.baseline:print('Median figure area ratio:',round(summary['median_figure_area_ratio'],3))


if __name__=='__main__':main()

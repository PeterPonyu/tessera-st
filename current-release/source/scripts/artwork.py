#!/usr/bin/env python3
"""Editable, concept-first SVG diagrams of the unchanged experimental design.

All motifs are schematic, never fabricated measurements. PDF companions are
exports of these SVGs, not independently redrawn versions of the diagrams.
"""
from html import escape
from pathlib import Path
import subprocess
import re
import xml.etree.ElementTree as ET
from common import ROOT, dump, sha

MAN = ROOT / 'source/manuscript'
FIG = MAN / 'figures'
ASSET = MAN / 'assets'
INK = '#000000'
BLUE = '#176b93'
ORANGE = '#bd5b30'
GREEN = '#527762'
PURPLE = '#82618e'
GRAY = '#9eabb4'
PALE = '#e8eef1'
ARROW_LENGTH = 12
ARROW_HALF_WIDTH = 4.5
ARROW_STROKE = 2.8
PANEL_LABEL_SIZE = 38


def text(x, y, value, size=23, color=INK, weight='normal', anchor='start', role=None):
    # Bold is reserved for the standalone panel marker, including legacy calls.
    weight = 'bold' if role == 'panel-label' else 'normal'
    color = INK
    tag = f' data-role="{role}"' if role else ''
    return (f'<text x="{x}" y="{y}" font-family="TeX Gyre Termes, Times New Roman, serif" '
            f'font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}"{tag}>{escape(value)}</text>')


def rect(x, y, w, h, fill='white', stroke=GRAY, radius=8, width=2):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'


def arrow_geometry(points):
    """Return an explicit shaft/head join, independent of SVG marker export."""
    x=y=0
    for match in re.finditer(r'([MHV])([^MHV]+)', points):
        command=match[1];values=[float(v) for v in match[2].split()]
        previous=(x,y)
        if command=='M':x,y=values
        elif command=='H':x,=values
        else:y,=values
    if command not in ('H','V'):
        raise ValueError('An arrow must end in an orthogonal H or V segment')
    index=0 if command=='H' else 1
    end=(x,y)[index];delta=end-previous[index]
    if abs(delta)<ARROW_LENGTH:
        raise ValueError('Terminal arrow segment is shorter than its head')
    shortened=end-ARROW_LENGTH*(1 if delta>0 else -1)
    shaft = points[:match.start()]+f'{command}{shortened:g}'
    tip = (x, y)
    base = (shortened, y) if command == 'H' else (x, shortened)
    half = ARROW_HALF_WIDTH
    corners = [(base[0], base[1]-half), tip, (base[0], base[1]+half)] if command == 'H' else [
        (base[0]-half, base[1]), tip, (base[0]+half, base[1])]
    return shaft, corners, tip, base


def arrow_shaft(points):
    """End an orthogonal shaft at the head's base; keep its original tip fixed."""
    return arrow_geometry(points)[0]


def line(points, color=GRAY, width=2, arrow=False, dash=False):
    if arrow:
        shaft, corners, tip, base = arrow_geometry(points)
        color = INK if color == GRAY else color
        head = ' '.join(f'{x:g},{y:g}' for x,y in corners)
        dashed = ' stroke-dasharray="7 5"' if dash else ''
        return (f'<g data-role="arrow" data-tip="{tip[0]:g},{tip[1]:g}" '
                f'data-base="{base[0]:g},{base[1]:g}" data-head-length="{ARROW_LENGTH}" '
                f'data-head-width="{2*ARROW_HALF_WIDTH:g}">'
                f'<path d="{shaft}" fill="none" stroke="{color}" stroke-width="{ARROW_STROKE}" '
                f'stroke-linecap="butt" stroke-linejoin="round" data-role="arrow-body"{dashed}/>'
                f'<polygon points="{head}" fill="{color}" stroke="none" data-role="arrow-head"/></g>')
    tail = ' stroke-dasharray="7 5"' if dash else ''
    return f'<path d="{points}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="butt" stroke-linejoin="round"{tail}/>'


def circle(x, y, r=8, color=BLUE, stroke='white', width=1.5):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" stroke="{stroke}" stroke-width="{width}"/>'


def matrix(x, y, rows=6, cols=7, size=12, color=BLUE):
    out = []
    for i in range(rows):
        for j in range(cols):
            opacity = .15 + .8 * ((i * 3 + j * 5 + i * j) % 11) / 10
            out.append(f'<rect x="{x+j*size}" y="{y+i*size}" width="{size-2}" height="{size-2}" fill="{color}" opacity="{opacity}"/>')
    return ''.join(out)


def network(x, y, scale=1, color=BLUE, gated=False):
    nodes = [(0, 30), (40, 0), (40, 60), (80, 30), (115, 0), (115, 60)]
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5)]
    items = []
    for k, (a, b) in enumerate(edges):
        ax, ay = nodes[a]; bx, by = nodes[b]
        mx = (ax + bx) / 2
        items.append(line(f'M{x+ax*scale} {y+ay*scale} H{x+mx*scale} V{y+by*scale} H{x+bx*scale}',
                          color if gated else GRAY, 1.6 + (k % 3)*1.3 if gated else 2))
    items += [circle(x+a*scale, y+b*scale, 9*scale, color) for a, b in nodes]
    return ''.join(items)


def patient(x, y, color=BLUE, scale=1):
    return (circle(x+15*scale, y+12*scale, 10*scale, color) +
            f'<path d="M{x} {y+53*scale} V{y+33*scale} Q{x+15*scale} {y+17*scale} {x+30*scale} {y+33*scale} V{y+53*scale} Z" fill="{color}" opacity=".8"/>')


def lock(x, y, color=GREEN):
    return (line(f'M{x+8} {y+20} V{y+9} Q{x+22} {y-9} {x+36} {y+9} V{y+20}', color, 5) +
            rect(x, y+19, 44, 37, color, color, 6) + circle(x+22, y+34, 4, 'white'))


def mosaic(x, y, scale=1):
    tiles = [(0,0,BLUE),(1,0,GREEN),(2,0,BLUE),(0,1,PURPLE),(1,1,BLUE),(2,1,ORANGE)]
    return ''.join(rect(x+i*18*scale,y+j*18*scale,15*scale,15*scale,c,c,2) for i,j,c in tiles)


def heading(x, y, letter, title, color=BLUE):
    return (text(x-12,y,letter.lower(),PANEL_LABEL_SIZE,role='panel-label')+
            text(x+34,y,title,25,role='panel-title'))


def chart(x, y, w=145, h=85, mode='points', color=BLUE):
    items = [line(f'M{x} {y} V{y+h} H{x+w}', INK, 2)]
    if mode == 'points':
        for i in range(6):
            yy = y+h-15-((i*17)%60)
            xx = x+15+i*21
            items += [line(f'M{xx} {yy-8} V{yy+8}',color,2),circle(xx,yy,4,color)]
    elif mode == 'bars':
        items += [rect(x+12+i*29,y+h-v,19,v,color,color,1) for i,v in enumerate([25,50,37,66])]
    else:
        items += [line(f'M{x+12} {y+12} H{x+34} V{y+37} H{x+70} V{y+55} H{x+120} V{y+63}',color,3)]
    return ''.join(items)


def document(height, title, body):
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="1356" height="'+str(height-136)+'" viewBox="20 83 1356 '+str(height-136)+'">'
            '<title>'+escape(title)+'</title>'+rect(0,0,1400,height,'white','none',0)+
            ''.join(body)+'</svg>')


def workflow():
    b = []
    for x,y,l,t,c in [(35,116,'A','Assays and patients',BLUE),(495,116,'B','Identity before geometry',GREEN),
                       (955,116,'C','Matched representations',ORANGE)]:
        b.append(heading(x,y,l,t,c))
    # Footprints, not a stock photograph of tissue.
    b += [rect(55,145,180,110,PALE,'none'),rect(250,145,155,110,'#f2e8dd','none')]
    for i in range(5):
        for j in range(6):b.append(circle(73+j*27,163+i*18,4,BLUE if (i+j)%3 else GREEN))
    for i in range(3):
        for j in range(4):b.append(circle(274+j*34,166+i*31,12,ORANGE if j%2 else BLUE))
    b += [text(145,280,'Cells',24,anchor='middle'),text(326,280,'Mixed spots',24,anchor='middle')]
    for i in range(9):b.append(patient(65+i*37,306,BLUE if i<5 else ORANGE,.8))
    b += [text(245,386,'5 lung + 4 BCC patients',24,anchor='middle')]
    b += [matrix(530,158,7,8,17),text(595,311,'Molecular matrix',24,anchor='middle')]
    for y in [159,242]:
        b += [rect(715,y,190,72,'#f4f7f8',GRAY),network(735,y+8,.55,GREEN)]
    b += [text(811,342,'Separate FOVs',24,anchor='middle'),text(690,386,'Shared PCA; no cross-frame edges',23,anchor='middle')]
    for y,c,label in [(155,BLUE,'PCA50 / PCA30'),(245,GREEN,'Spatial controls'),(335,ORANGE,'Tessera encoder')]:
        glyph=mosaic(1025,y,1.1) if label=='Tessera encoder' else matrix(1025,y,3,4,13,c)
        b += [line(f'M971 {y+26} H1010',c,3,True),glyph,text(1100,y+28,label,23)]
    b += [line('M445 263 H478',INK,3,True),line('M917 263 H942',INK,3,True)]
    top=b
    b=[heading(35,490,'D','Specified interventions',PURPLE),
       heading(495,490,'E','Source-only transfer',GREEN),
       heading(955,490,'F','Distinct evidence channels',BLUE)]
    for x,c,label in [(57,BLUE,'Gate'),(169,GREEN,'Depth'),(281,ORANGE,'Contrast')]:
        b += [network(x,532,.63,c,True),line(f'M{x+16} 613 H{x+70}',c,3),circle(x+38,613,9,'white',c,3),
              text(x+37,654,label,24,anchor='middle')]
    b += [line('M80 700 H140 V725 H188',PURPLE,2,True),circle(80,700,6,PURPLE),circle(200,725,6,PURPLE),
          text(241,716,'Self only',23),text(241,751,'Permute XY',23)]
    for i in range(4):b.append(patient(524+i*53,541,GREEN))
    b += [patient(829,541,GRAY),lock(752,525),line('M718 573 H746',GREEN,3,True),
          line('M802 573 H824',GREEN,3,True),text(620,637,'Four source patients',23,anchor='middle'),
          text(844,637,'Held out',23,anchor='middle'),matrix(537,668,3,5,14,GREEN),line('M617 690 H695',GREEN,2,True),
          lock(702,658),line('M759 690 H820',GREEN,2,True),matrix(835,668,3,4,14,GRAY),text(703,751,'Fit → freeze → predict',24,anchor='middle')]
    b += [chart(991,536,140,76),text(1060,648,'Partition scores',23,anchor='middle'),
          chart(1193,536,133,76,'bars',ORANGE),text(1253,648,'Patient transfer',23,anchor='middle'),
          rect(1000,678,131,63,PALE,GRAY),matrix(1009,686,3,6,14,PURPLE),
          line('M1155 688 V737 H1309',INK,2),circle(1188,707,9,BLUE),circle(1228,722,9,GREEN),circle(1274,699,9,ORANGE),
          text(1155,781,'Registered tissue inspection',24,anchor='middle')]
    b += [line('M445 637 H478',INK,3,True),line('M917 637 H943',INK,3,True)]
    return document(808,'Matched spatial evaluation',top+
        [line('M1160 400 V403 H239 V425',INK,2.5,True),
         '<g transform="translate(0,-36)">'+''.join(b)+'</g>'])


def architecture():
    b = [heading(35,114,'A','Inputs and frame graph'),heading(540,114,'B','Gated residual layer')]
    b += [matrix(55,149,5,6,16),text(105,259,'x : q ≤ 50',25,anchor='middle'),
          network(233,161,.9,BLUE),text(291,259,'Within-frame E',25,anchor='middle'),
          line('M166 198 H206',BLUE,3,True),text(235,305,'No unrelated-field edges',23,anchor='middle')]
    b += [network(572,150,.8,BLUE,True),text(623,253,'h and distance',23,anchor='middle'),line('M704 183 H753',BLUE,3,True),
          rect(773,140,191,92,'#f3edf5',PURPLE),text(868,177,'Edge MLP',25,PURPLE,'bold','middle'),
          text(868,212,'sigmoid gate',23,anchor='middle'),line('M966 183 H1014',PURPLE,3,True),
          matrix(1031,143,4,4,18,GREEN),text(1081,254,'Weighted mean',23,anchor='middle'),
          circle(1263,179,23,'white',BLUE,3),text(1263,188,'+',30,BLUE,'bold','middle'),
          line('M1120 183 H1226',GREEN,3,True),line('M624 136 V123 H1263 V147',BLUE,2,True),
          text(1262,254,'Residual',23,anchor='middle'),text(975,305,'Normalize by gate sum → ReLU → dropout → LayerNorm',23,anchor='middle')]
    top=b
    b = [heading(35,375,'C','Three depths → attention → embedding')]
    xs=[62,264,466,668]
    for i,x in enumerate(xs):
        c=BLUE if i==0 else [GREEN,ORANGE,PURPLE][i-1]
        b += [matrix(x,425,5,4,16,c),text(x+32,585,'Input' if i==0 else 'Depth '+str(i),24,anchor='middle')]
        if i<3:b.append(line(f'M{x+78} 467 H{x+179}',c,3,True))
    for x,c,y in zip(xs[1:],[GREEN,ORANGE,PURPLE],[523,538,553]):
        b.append(line(f'M{x+35} 513 V{y} H881',c,2,True))
    b += [rect(898,415,185,150,'#edf4ef',GREEN),text(990,452,'Depth attention',25,GREEN,'bold','middle')]
    for i,c in enumerate([GREEN,ORANGE,PURPLE]):b.append(rect(930+i*44,484,25,28+i*13,c,c,2))
    b += [line('M1086 470 H1160',GREEN,3,True),matrix(1179,429,5,3,17,BLUE),text(1203,541,'z : 32',25,anchor='middle'),
          text(455,611,'Three 64-dimensional hidden states',23,anchor='middle')]
    middle=b
    b = [heading(35,682,'D','Two training objectives'),heading(927,682,'E','Separate predictions')]
    b += [matrix(58,736,4,3,17,BLUE),line('M120 769 H167',BLUE,3,True),rect(185,720,165,88,PALE,BLUE),
          text(268,754,'Decoder',25,BLUE,'bold','middle'),text(268,790,'32 → 64 → q',23,anchor='middle'),
          line('M352 769 H395',BLUE,3,True),matrix(413,736,4,5,17,BLUE),text(272,848,'Reconstruction MSE',24,anchor='middle')]
    for x,y,c in [(611,748,GREEN),(681,748,GREEN),(651,817,GRAY),(725,817,GRAY),(796,817,GRAY)]:b.append(circle(x,y,14,c))
    b += [line('M627 748 H665',GREEN,3),line('M611 766 V791 H796',GRAY,2),
          text(704,883,'10 random negative draws',23,anchor='middle'),text(445,967,'L = reconstruction + 0.5 × contrast',26,INK,'bold','middle'),
          text(704,919,'Detached gate weights; τ = 0.2',22,anchor='middle')]
    b += [matrix(942,744,4,3,14,BLUE),line('M995 771 H1026 V726 H1088',BLUE,2.5,True),
          line('M1026 771 V824 H1088',GREEN,2.5,True)]
    for i,(y,label,c) in enumerate([(730,'KMeans',BLUE),(828,'Source centroids',GREEN)]):
        for j in range(4):b.append(circle(1111+j*49,y-11,7,c if j%2 else ORANGE))
        b.append(text(1200,y+26,label,24,anchor='middle'))
    b += [text(1146,916,'Seam score ≠ calibrated error',23,ORANGE,anchor='middle')]
    return document(990,'Encoder, loss and prediction branches',top+
        ['<g transform="translate(0,-22)">'+''.join(middle)+'</g>',
         '<g transform="translate(0,-44)">'+''.join(b)+'</g>'])


def lineage():
    b = [heading(35,114,'A','Reference construction'),heading(502,114,'B','Within-object fit'),heading(970,114,'C','Patient-disjoint fit')]
    for i,(label,c) in enumerate([('Cell identity',BLUE),('Spatial region',GREEN),('Spot mixture',ORANGE)]):
        y=158+i*81
        for j in range(6):b.append(circle(64+j*22,y,7,c if i!=2 else [BLUE,GREEN,ORANGE][j%3]))
        b.append(text(221,y+8,label,25))
    b += [text(236,437,'Different targets; no pooled truth',23,anchor='middle')]
    b += [matrix(534,153,5,5,16,BLUE),line('M626 192 H663',BLUE,3,True),network(685,168,.75,ORANGE,True),
          text(652,283,'Encoder uses X and E',24,anchor='middle'),
          # One clear row: the reference contributes only the class count K.
          text(534,352,'Reference',23),line('M640 344 H700',GRAY,2,True),text(712,352,'K = class count',23),
          text(652,404,'Labels score the partition',23,anchor='middle')]
    for i in range(4):b.append(patient(1000+i*52,157,GREEN))
    b += [patient(1280,157,GRAY),lock(1174,241),line('M1104 221 V270 H1162',GREEN,2,True),
          line('M1228 270 H1295 V221',GREEN,2,True),
          text(1179,348,'Source transforms and weights',23,anchor='middle'),text(1179,399,'Target geometry is observed',23,anchor='middle')]
    top=b
    b = [heading(35,526,'D','Inspection, not independent truth'),heading(735,526,'E','Units of variability')]
    b += [rect(59,571,180,119,PALE,GRAY),matrix(69,581,6,9,16,PURPLE),
          circle(93,599,6,ORANGE),circle(146,641,6,BLUE),circle(201,610,6,GREEN),
          text(149,735,'Registered image',25,anchor='middle'),
          rect(300,571,165,119,'#f7f3ed',GRAY)]
    for i in range(4):
        for j in range(5):b.append(circle(317+j*30,590+i*27,6,[ORANGE,GREEN,PURPLE][(i+j)%3]))
    b += [text(384,735,'Measured proteins',25,anchor='middle'),
          text(294,789,'No new pathology labels',24,ORANGE,'bold','middle'),text(294,829,'No invented tissue texture',23,anchor='middle')]
    for i,(y,c,label) in enumerate([(585,BLUE,'Patient contrasts'),(685,GREEN,'Seed variability'),(785,ORANGE,'Descriptive gate')]):
        if i==0:
            for j in range(4):b.append(patient(760+j*38,y,c,.65))
            b += [line(f'M931 {y+25} H992',c,2),circle(964,y+25,8,c)]
        elif i==1:
            b += [text(783,y+28,'1',27,c,'bold'),text(838,y+28,'2',27,c,'bold'),text(893,y+28,'3',27,c,'bold'),line(f'M927 {y+25} H992',c,2)]
        else:b += [network(764,y,.75,c,True),line(f'M885 {y+24} H992',c,2)]
        b.append(text(1020,y+30,label,25))
    return document(883,'Target lineage and information access',top+
        ['<g transform="translate(0,-32)">'+''.join(b)+'</g>'])


def main():
    FIG.mkdir(parents=True,exist_ok=True);ASSET.mkdir(parents=True,exist_ok=True)
    icons={'slide':rect(12,16,78,68,PALE,INK)+matrix(19,23,4,5,12,PURPLE),
           'graph':network(8,20,.65), 'encoder':mosaic(20,20,1.2),
           'control':chart(14,15,72,64,'bars',ORANGE),
           'patients':patient(12,22,GREEN)+patient(56,22,BLUE),
           'report':chart(16,14,73,69),'tessera-logo':mosaic(12,20,1.5)}
    for name,body in icons.items():
        (ASSET/(name+'.svg')).write_text('<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">'+body+'</svg>\n')
    audit={}
    for name,svg in [('figure01-workflow',workflow()),('figure13-architecture',architecture()),('figure14-target-lineage',lineage())]:
        target=FIG/(name+'.svg');target.write_text(svg+'\n');root=ET.parse(target).getroot()
        texts=list(root.iter('{http://www.w3.org/2000/svg}text'))
        labels=[t.text for t in texts if t.get('data-role')=='panel-label']
        titles=[t.text for t in texts if t.get('data-role')=='panel-title']
        for t in texts:
            expected='bold' if t.get('data-role')=='panel-label' else 'normal'
            assert t.get('font-weight')==expected
            if expected=='bold':assert re.fullmatch(r'[a-z]',t.text)
            assert t.get('fill')==INK
        assert len(labels)==len(titles)
        label_size=min(float(t.get('font-size')) for t in texts if t.get('data-role')=='panel-label')
        regular_size=max(float(t.get('font-size')) for t in texts if t.get('data-role')!='panel-label')
        assert label_size>regular_size
        subprocess.run(['inkscape',str(target),'--export-text-to-path','--export-type=pdf',
                        '--export-filename='+str(FIG/(name+'.pdf'))],check=True,capture_output=True)
        subprocess.run(['pdftoppm','-singlefile','-r','110','-jpeg',str(FIG/(name+'.pdf')),
                        str(ROOT/'qa'/name)],check=True,capture_output=True)
        audit[name]={'labels':labels,'titles':titles,'separate_svg_text_elements':True,
            'only_panel_labels_bold':True,'font_family':'TeX Gyre Termes, Times New Roman, serif',
            'all_text_black':True,'panel_label_size_svg_units':label_size,
            'maximum_regular_text_size_svg_units':regular_size,
            'arrow_length_svg_units':ARROW_LENGTH,'arrow_width_svg_units':2*ARROW_HALF_WIDTH,
            'arrow_stroke_svg_units':ARROW_STROKE,'shaft_ends_at_head_base':True,
            'arrow_rendering':'explicit filled polygon plus orthogonal shaft; no SVG markers',
            'title_alignment':'left; bare marker shifted left of regular title','viewBox':root.get('viewBox'),
            'svg_sha256':sha(target),'pdf_sha256':sha(FIG/(name+'.pdf'))}
        print('SVG/PDF',name,flush=True)
    dump(ROOT/'qa/schematic_style_audit.json',audit)


if __name__ == '__main__':
    main()

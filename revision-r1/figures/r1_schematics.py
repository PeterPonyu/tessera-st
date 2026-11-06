"""Revision-1 schematics (main Figures 1 and 2, Supplementary Figure S2) as editable SVG.

Ported from artwork.py (which reproduces the submitted SVG sources byte for byte).
Changes: native Arial lettering (no horizontally compressed substitution), the
method name Tessera-ST, a 496.8-pt physical width so LaTeX places the PDF at
100 %, letter/title sizes matched to the statistical figures, and explicit
coordinate nudges where wider Arial text would otherwise crowd. All motifs stay
conceptual; no measurement is drawn. `check_svg_text` rejects overlapping or
off-canvas text using the Arial advance widths.
"""
from __future__ import annotations

from html import escape
import itertools
import re
import subprocess
import xml.etree.ElementTree as ET

import r1_style as S

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
UNIT = S.W / 1356          # points per SVG unit at 100 % placement
LETTER_UNITS = round(S.LETTER / UNIT, 1)   # 12-pt bold panel letter
TITLE_UNITS = 24           # 8.8-pt panel title
FONT = 'Arial'
SVG_NS = '{http://www.w3.org/2000/svg}'


def text(x, y, value, size=23, color=INK, weight='normal', anchor='start', role=None):
    weight = 'bold' if role == 'panel-label' else 'normal'
    tag = f' data-role="{role}"' if role else ''
    return (f'<text x="{x:g}" y="{y:g}" font-family="{FONT}" font-size="{size:g}" fill="{INK}" '
            f'font-weight="{weight}" text-anchor="{anchor}"{tag}>{escape(value)}</text>')


def rect(x, y, w, h, fill='white', stroke=GRAY, radius=8, width=2):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'


def arrow_geometry(points):
    x = y = 0
    for match in re.finditer(r'([MHV])([^MHV]+)', points):
        command = match[1]
        values = [float(v) for v in match[2].split()]
        previous = (x, y)
        if command == 'M':
            x, y = values
        elif command == 'H':
            x, = values
        else:
            y, = values
    if command not in ('H', 'V'):
        raise ValueError('An arrow must end in an orthogonal H or V segment')
    index = 0 if command == 'H' else 1
    end = (x, y)[index]
    delta = end - previous[index]
    if abs(delta) < ARROW_LENGTH:
        raise ValueError('Terminal arrow segment is shorter than its head')
    shortened = end - ARROW_LENGTH * (1 if delta > 0 else -1)
    shaft = points[:match.start()] + f'{command}{shortened:g}'
    tip = (x, y)
    base = (shortened, y) if command == 'H' else (x, shortened)
    half = ARROW_HALF_WIDTH
    corners = [(base[0], base[1] - half), tip, (base[0], base[1] + half)] if command == 'H' else [
        (base[0] - half, base[1]), tip, (base[0] + half, base[1])]
    return shaft, corners, tip, base


def line(points, color=GRAY, width=2, arrow=False, dash=False):
    if arrow:
        shaft, corners, tip, base = arrow_geometry(points)
        color = INK if color == GRAY else color
        head = ' '.join(f'{x:g},{y:g}' for x, y in corners)
        dashed = ' stroke-dasharray="7 5"' if dash else ''
        return (f'<g data-role="arrow"><path d="{shaft}" fill="none" stroke="{color}" stroke-width="{ARROW_STROKE}" '
                f'stroke-linecap="butt" stroke-linejoin="round"{dashed}/>'
                f'<polygon points="{head}" fill="{color}" stroke="none"/></g>')
    tail = ' stroke-dasharray="7 5"' if dash else ''
    return (f'<path d="{points}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="butt" '
            f'stroke-linejoin="round"{tail}/>')


def circle(x, y, r=8, color=BLUE, stroke='white', width=1.5):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" stroke="{stroke}" stroke-width="{width}"/>'


def matrix(x, y, rows=6, cols=7, size=12, color=BLUE):
    out = []
    for i in range(rows):
        for j in range(cols):
            opacity = .15 + .8 * ((i * 3 + j * 5 + i * j) % 11) / 10
            out.append(f'<rect x="{x + j * size}" y="{y + i * size}" width="{size - 2}" height="{size - 2}" '
                       f'fill="{color}" opacity="{opacity}"/>')
    return ''.join(out)


def network(x, y, scale=1, color=BLUE, gated=False):
    nodes = [(0, 30), (40, 0), (40, 60), (80, 30), (115, 0), (115, 60)]
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5)]
    items = []
    for k, (a, b) in enumerate(edges):
        ax, ay = nodes[a]
        bx, by = nodes[b]
        mx = (ax + bx) / 2
        items.append(line(f'M{x + ax * scale} {y + ay * scale} H{x + mx * scale} V{y + by * scale} H{x + bx * scale}',
                          color if gated else GRAY, 1.6 + (k % 3) * 1.3 if gated else 2))
    items += [circle(x + a * scale, y + b * scale, 9 * scale, color) for a, b in nodes]
    return ''.join(items)


def patient(x, y, color=BLUE, scale=1):
    return (circle(x + 15 * scale, y + 12 * scale, 10 * scale, color) +
            f'<path d="M{x} {y + 53 * scale} V{y + 33 * scale} Q{x + 15 * scale} {y + 17 * scale} {x + 30 * scale} '
            f'{y + 33 * scale} V{y + 53 * scale} Z" fill="{color}" opacity=".8"/>')


def lock(x, y, color=GREEN):
    return (line(f'M{x + 8} {y + 20} V{y + 9} Q{x + 22} {y - 9} {x + 36} {y + 9} V{y + 20}', color, 5) +
            rect(x, y + 19, 44, 37, color, color, 6) + circle(x + 22, y + 34, 4, 'white'))


def mosaic(x, y, scale=1):
    tiles = [(0, 0, BLUE), (1, 0, GREEN), (2, 0, BLUE), (0, 1, PURPLE), (1, 1, BLUE), (2, 1, ORANGE)]
    return ''.join(rect(x + i * 18 * scale, y + j * 18 * scale, 15 * scale, 15 * scale, c, c, 2) for i, j, c in tiles)


def heading(x, y, letter, title):
    """Bold lowercase letter in the header margin; the short title sits on its baseline just right of it."""
    return (text(x - 12, y, letter.lower(), LETTER_UNITS, role='panel-label') +
            text(x + 26, y, title, TITLE_UNITS, role='panel-title'))


def chart(x, y, w=145, h=85, mode='points', color=BLUE):
    items = [line(f'M{x} {y} V{y + h} H{x + w}', INK, 2)]
    if mode == 'points':
        for i in range(6):
            yy = y + h - 15 - ((i * 17) % 60)
            xx = x + 15 + i * 21
            items += [line(f'M{xx} {yy - 8} V{yy + 8}', color, 2), circle(xx, yy, 4, color)]
    elif mode == 'bars':
        items += [rect(x + 12 + i * 29, y + h - v, 19, v, color, color, 1) for i, v in enumerate([25, 50, 37, 66])]
    return ''.join(items)


def document(view_y, view_h, title, body):
    """SVG with a 1356-unit-wide viewBox and a 496.8-pt physical width (100 % in the manuscript)."""
    width_pt = S.W
    height_pt = view_h * UNIT
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_pt:g}pt" height="{height_pt:.3f}pt" '
            f'viewBox="20 {view_y} 1356 {view_h}"><title>{escape(title)}</title>'
            + rect(0, 0, 1400, view_y + view_h + 50, 'white', 'none', 0) + ''.join(body) + '</svg>')


# ------------------------------------------------------------------ Figure 1
def workflow():
    b = [heading(35, 116, 'a', 'Assays and patients'), heading(495, 116, 'b', 'Identity before geometry'),
         heading(955, 116, 'c', 'Matched representations')]
    b += [rect(55, 145, 180, 110, PALE, 'none'), rect(250, 145, 155, 110, '#f2e8dd', 'none')]
    for i in range(5):
        for j in range(6):
            b.append(circle(73 + j * 27, 163 + i * 18, 4, BLUE if (i + j) % 3 else GREEN))
    for i in range(3):
        for j in range(4):
            b.append(circle(274 + j * 34, 166 + i * 31, 12, ORANGE if j % 2 else BLUE))
    b += [text(145, 280, 'Cells', 22, anchor='middle'), text(327, 280, 'Mixed spots', 22, anchor='middle')]
    for i in range(9):
        b.append(patient(65 + i * 37, 306, BLUE if i < 5 else ORANGE, .8))
    b += [text(230, 386, '5 lung + 4 BCC patients', 22, anchor='middle')]
    b += [matrix(530, 158, 7, 8, 17), text(595, 311, 'Molecular matrix', 22, anchor='middle')]
    for y in [159, 242]:
        b += [rect(715, y, 190, 72, '#f4f7f8', GRAY), network(735, y + 8, .55, GREEN)]
    b += [text(811, 342, 'Separate FOVs', 22, anchor='middle'),
          text(706, 386, 'Shared PCA; no cross-frame edges', 21, anchor='middle')]
    for y, c, label in [(155, BLUE, 'PCA50 / PCA30'), (245, GREEN, 'Spatial controls'), (335, ORANGE, f'{S.TST} encoder')]:
        glyph = mosaic(1025, y, 1.1) if label.endswith('encoder') else matrix(1025, y, 3, 4, 13, c)
        b += [line(f'M971 {y + 26} H1010', c, 3, True), glyph, text(1095, y + 28, label, 22)]
    b += [line('M445 263 H478', INK, 3, True), line('M917 263 H942', INK, 3, True)]
    top = b
    b = [heading(35, 490, 'd', 'Specified interventions'), heading(495, 490, 'e', 'Source-only transfer'),
         heading(955, 490, 'f', 'Distinct evidence channels')]
    for x, c, label in [(57, BLUE, 'Gate'), (169, GREEN, 'Depth'), (281, ORANGE, 'Contrast')]:
        b += [network(x, 532, .63, c, True), line(f'M{x + 16} 613 H{x + 70}', c, 3), circle(x + 38, 613, 9, 'white', c, 3),
              text(x + 37, 654, label, 22, anchor='middle')]
    b += [line('M80 700 H140 V725 H188', PURPLE, 2, True), circle(80, 700, 6, PURPLE), circle(200, 725, 6, PURPLE),
          text(232, 716, 'Self-only', 22), text(232, 751, 'Permuted XY', 22)]
    for i in range(4):
        b.append(patient(524 + i * 53, 541, GREEN))
    b += [patient(829, 541, GRAY), lock(752, 525), line('M718 573 H746', GREEN, 3, True),
          line('M802 573 H824', GREEN, 3, True), text(611, 637, 'Four source patients', 21, anchor='middle'),
          text(846, 637, 'Held out', 21, anchor='middle'), matrix(537, 668, 3, 5, 14, GREEN),
          line('M617 690 H695', GREEN, 2, True), lock(702, 658), line('M759 690 H820', GREEN, 2, True),
          matrix(835, 668, 3, 4, 14, GRAY), text(703, 751, 'Fit → freeze → predict', 22, anchor='middle')]
    b += [chart(991, 536, 140, 76), text(1060, 648, 'Partition scores', 21, anchor='middle'),
          chart(1193, 536, 133, 76, 'bars', ORANGE), text(1259, 648, 'Patient transfer', 21, anchor='middle'),
          rect(1000, 678, 131, 63, PALE, GRAY), matrix(1009, 686, 3, 6, 14, PURPLE),
          line('M1155 688 V737 H1309', INK, 2), circle(1188, 707, 9, BLUE), circle(1228, 722, 9, GREEN),
          circle(1274, 699, 9, ORANGE), text(1155, 781, 'Registered tissue inspection', 22, anchor='middle')]
    b += [line('M445 637 H478', INK, 3, True), line('M917 637 H943', INK, 3, True)]
    return document(83, 672, 'Matched spatial evaluation', top +
                    [line('M1160 400 V403 H239 V425', INK, 2.5, True), '<g transform="translate(0,-36)">' + ''.join(b) + '</g>'])


# ------------------------------------------------------------------ Figure 2
def architecture():
    b = [heading(35, 114, 'a', 'Inputs and frame graph'), heading(540, 114, 'b', 'Gated residual layer')]
    b += [matrix(55, 149, 5, 6, 16), text(105, 259, 'x : q ≤ 50', 23, anchor='middle'),
          network(233, 161, .9, BLUE), text(291, 259, 'Within-frame E', 23, anchor='middle'),
          line('M166 198 H206', BLUE, 3, True), text(235, 305, 'No unrelated-field edges', 22, anchor='middle')]
    b += [network(572, 150, .8, BLUE, True), text(623, 253, 'h and distance', 22, anchor='middle'),
          line('M704 183 H753', BLUE, 3, True),
          rect(773, 140, 191, 92, '#f3edf5', PURPLE), text(868, 177, 'Edge MLP', 23, anchor='middle'),
          text(868, 212, 'sigmoid gate', 22, anchor='middle'), line('M966 183 H1014', PURPLE, 3, True),
          matrix(1031, 143, 4, 4, 18, GREEN), text(1081, 254, 'Weighted mean', 22, anchor='middle'),
          circle(1263, 179, 23, 'white', BLUE, 3), text(1263, 188, '+', 30, anchor='middle'),
          line('M1120 183 H1226', GREEN, 3, True), line('M624 136 V129 H1263 V147', BLUE, 2, True),
          text(1263, 254, 'Residual', 22, anchor='middle'),
          text(975, 305, 'Normalize by gate sum → ReLU → dropout → LayerNorm', 21, anchor='middle')]
    top = b
    b = [heading(35, 375, 'c', 'Three depths → attention → embedding')]
    xs = [62, 264, 466, 668]
    for i, x in enumerate(xs):
        c = BLUE if i == 0 else [GREEN, ORANGE, PURPLE][i - 1]
        b += [matrix(x, 425, 5, 4, 16, c), text(x + 32, 585, 'Input' if i == 0 else 'Depth ' + str(i), 22, anchor='middle')]
        if i < 3:
            b.append(line(f'M{x + 78} 467 H{x + 179}', c, 3, True))
    for x, c, y in zip(xs[1:], [GREEN, ORANGE, PURPLE], [523, 538, 553]):
        b.append(line(f'M{x + 35} 513 V{y} H881', c, 2, True))
    b += [rect(898, 415, 185, 150, '#edf4ef', GREEN), text(990, 452, 'Depth attention', 23, anchor='middle')]
    for i, c in enumerate([GREEN, ORANGE, PURPLE]):
        b.append(rect(930 + i * 44, 484, 25, 28 + i * 13, c, c, 2))
    b += [line('M1086 470 H1160', GREEN, 3, True), matrix(1179, 429, 5, 3, 17, BLUE),
          text(1203, 541, 'z : 32', 23, anchor='middle'),
          text(498, 619, 'Three 64-dimensional hidden states', 21, anchor='middle')]
    middle = b
    b = [heading(35, 682, 'd', 'Two training objectives'), heading(927, 682, 'e', 'Separate predictions')]
    b += [matrix(58, 736, 4, 3, 17, BLUE), line('M120 769 H167', BLUE, 3, True), rect(185, 720, 165, 88, PALE, BLUE),
          text(268, 754, 'Decoder', 23, anchor='middle'), text(268, 790, '32 → 64 → q', 22, anchor='middle'),
          line('M352 769 H395', BLUE, 3, True), matrix(413, 736, 4, 5, 17, BLUE),
          text(272, 848, 'Reconstruction MSE', 22, anchor='middle')]
    for x, y, c in [(611, 748, GREEN), (681, 748, GREEN), (651, 817, GRAY), (725, 817, GRAY), (796, 817, GRAY)]:
        b.append(circle(x, y, 14, c))
    b += [line('M627 748 H665', GREEN, 3), line('M611 766 V791 H796', GRAY, 2),
          text(704, 883, '10 random negative draws', 22, anchor='middle'),
          text(445, 967, 'L = reconstruction + 0.5 × contrast', 24, anchor='middle'),
          text(704, 919, 'Detached gate weights; τ = 0.2', 21, anchor='middle')]
    b += [matrix(942, 744, 4, 3, 14, BLUE), line('M995 771 H1026 V726 H1088', BLUE, 2.5, True),
          line('M1026 771 V824 H1088', GREEN, 2.5, True)]
    for y, label, c in [(730, 'K-means', BLUE), (828, 'Source centroids', GREEN)]:
        for j in range(4):
            b.append(circle(1111 + j * 49, y - 11, 7, c if j % 2 else ORANGE))
        b.append(text(1200, y + 26, label, 22, anchor='middle'))
    b += [text(1170, 916, 'Seam score ≠ calibrated error', 21, anchor='middle')]
    # The objective baseline is y=923 after the -44 shift; the view ends 12 units below its descender.
    return document(83, 857, 'Encoder, loss and prediction branches', top +
                    ['<g transform="translate(0,-22)">' + ''.join(middle) + '</g>',
                     '<g transform="translate(0,-44)">' + ''.join(b) + '</g>'])


# ------------------------------------------------------------- Figure S2
def lineage():
    b = [heading(35, 114, 'a', 'Reference construction'), heading(502, 114, 'b', 'Within-object fit'),
         heading(970, 114, 'c', 'Patient-disjoint fit')]
    for i, (label, c) in enumerate([('Cell identity', BLUE), ('Spatial region', GREEN), ('Spot mixture', ORANGE)]):
        y = 158 + i * 81
        for j in range(6):
            b.append(circle(64 + j * 22, y, 7, c if i != 2 else [BLUE, GREEN, ORANGE][j % 3]))
        b.append(text(221, y + 8, label, 23))
    b += [text(236, 437, 'Different targets; no pooled truth', 21, anchor='middle')]
    b += [matrix(534, 153, 5, 5, 16, BLUE), line('M626 192 H663', BLUE, 3, True), network(685, 168, .75, ORANGE, True),
          text(652, 283, 'Encoder uses X and E', 22, anchor='middle'),
          text(534, 352, 'Reference', 21), line('M640 344 H700', GRAY, 2, True), text(712, 352, 'K = class count', 21),
          text(652, 404, 'Labels score the partition', 21, anchor='middle')]
    for i in range(4):
        b.append(patient(1000 + i * 52, 157, GREEN))
    b += [patient(1280, 157, GRAY), lock(1174, 241), line('M1104 221 V270 H1162', GREEN, 2, True),
          line('M1228 270 H1295 V221', GREEN, 2, True),
          text(1179, 348, 'Source transforms and weights', 21, anchor='middle'),
          text(1179, 399, 'Target geometry is observed', 21, anchor='middle')]
    top = b
    b = [heading(35, 526, 'd', 'Inspection, not independent truth'), heading(735, 526, 'e', 'Units of variability')]
    b += [rect(59, 571, 180, 119, PALE, GRAY), matrix(69, 581, 6, 9, 16, PURPLE),
          circle(93, 599, 6, ORANGE), circle(146, 641, 6, BLUE), circle(201, 610, 6, GREEN),
          text(149, 735, 'Registered image', 23, anchor='middle'),
          rect(300, 571, 165, 119, '#f7f3ed', GRAY)]
    for i in range(4):
        for j in range(5):
            b.append(circle(317 + j * 30, 590 + i * 27, 6, [ORANGE, GREEN, PURPLE][(i + j) % 3]))
    b += [text(384, 735, 'Measured proteins', 23, anchor='middle'),
          text(262, 789, 'No new pathology labels', 22, anchor='middle'),
          text(262, 829, 'No invented tissue texture', 21, anchor='middle')]
    for i, (y, c, label) in enumerate([(585, BLUE, 'Patient contrasts'), (685, GREEN, 'Seed variability'),
                                       (785, ORANGE, 'Descriptive gate')]):
        if i == 0:
            for j in range(4):
                b.append(patient(760 + j * 38, y, c, .65))
            b += [line(f'M931 {y + 25} H992', c, 2), circle(964, y + 25, 8, c)]
        elif i == 1:
            b += [text(783, y + 28, '1', 25), text(838, y + 28, '2', 25), text(893, y + 28, '3', 25),
                  line(f'M927 {y + 25} H992', c, 2)]
        else:
            b += [network(764, y, .75, c, True), line(f'M885 {y + 24} H992', c, 2)]
        b.append(text(1020, y + 30, label, 23))
    return document(83, 747, 'Target lineage and information access', top +
                    ['<g transform="translate(0,-32)">' + ''.join(b) + '</g>'])


# ------------------------------------------------------------------ checks
def _offset(element, parents):
    dy = dx = 0.0
    node = element
    while node is not None:
        transform = node.get('transform', '')
        m = re.fullmatch(r'translate\(([-\d.]+),([-\d.]+)\)', transform)
        if m:
            dx += float(m[1])
            dy += float(m[2])
        node = parents.get(node)
    return dx, dy


class _Advance:
    """Arial advance widths from the embedded font files (fontTools; no kerning)."""

    def __init__(self, path):
        from fontTools.ttLib import TTFont
        font = TTFont(str(path))
        self.cmap = font.getBestCmap()
        self.hmtx = font['hmtx'].metrics
        self.em = font['head'].unitsPerEm

    def text_length(self, value, fontsize):
        return sum(self.hmtx[self.cmap[ord(c)]][0] for c in value) * fontsize / self.em


def text_boxes(svg_text):
    fonts = {False: _Advance(S.FONT_DIR / 'Arial.ttf'), True: _Advance(S.FONT_DIR / 'Arial Bold.ttf')}
    root = ET.fromstring(svg_text)
    parents = {child: parent for parent in root.iter() for child in parent}
    boxes = []
    for e in root.iter(SVG_NS + 'text'):
        value = ''.join(e.itertext())
        size = float(e.get('font-size'))
        bold = e.get('font-weight') == 'bold'
        width = fonts[bold].text_length(value, fontsize=size)
        dx, dy = _offset(e, parents)
        x = float(e.get('x')) + dx
        y = float(e.get('y')) + dy
        x0 = x - {'start': 0, 'middle': width / 2, 'end': width}[e.get('text-anchor', 'start')]
        boxes.append((value, (x0, y - .73 * size, x0 + width, y + .21 * size), size, e.get('data-role')))
    view = [float(v) for v in root.get('viewBox').split()]
    return boxes, view


def check_svg_text(name, svg_text, gap=2.0):
    boxes, (vx, vy, vw, vh) = text_boxes(svg_text)
    problems = []
    for value, (x0, y0, x1, y1), _, _ in boxes:
        if x0 < vx or y0 < vy or x1 > vx + vw or y1 > vy + vh:
            problems.append(('outside', value))
    for (a, p, _, _), (b, q, _, _) in itertools.combinations(boxes, 2):
        if min(p[2], q[2]) - max(p[0], q[0]) > -gap and min(p[3], q[3]) - max(p[1], q[1]) > -gap:
            problems.append(('overlap', a, b))
    for value, *_ in boxes:
        for pattern in S.BANNED:
            if re.search(pattern, value):
                problems.append(('banned', value))
    if problems:
        raise AssertionError(f'{name}: {problems}')
    sizes = [b[2] for b in boxes if b[3] != 'panel-label']
    return {'text_elements': len(boxes), 'min_text_points': round(min(sizes) * UNIT, 2),
            'letter_points': round(LETTER_UNITS * UNIT, 2), 'overlapping_text': 0, 'text_outside_viewbox': 0}


SVG_OUT = S.HERE / 'svg'   # editable schematic sources stay with the generator; the packet gets PDFs only


def export(name, svg_text):
    audit = check_svg_text(name, svg_text)
    source = SVG_OUT / (name + '.svg')
    target = S.PDF_OUT / (name + '.pdf')
    S.PDF_OUT.mkdir(parents=True, exist_ok=True)
    SVG_OUT.mkdir(parents=True, exist_ok=True)
    source.write_text(svg_text + '\n')
    subprocess.run(['inkscape', str(source), '--export-type=pdf', '--export-filename=' + str(target)], check=True,
                   capture_output=True)
    S.AUDIT[name] = {**audit, 'svg': source.name, 'export': 'Inkscape SVG to PDF, live Arial text'}
    return target


SCHEMATICS = {'figure-01': workflow, 'figure-02': architecture, 'figure-S2': lineage}


def build(names=None):
    return [export(name, SCHEMATICS[name]()) for name in (names or SCHEMATICS)]

#!/usr/bin/env python3
"""Build document-only deliverables; never fit models or download measurements.

Uses the installed LaTeX skill compiler. Offline figure regeneration is optional;
network bibliography refresh is deliberately not part of this command.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from common import ROOT, dump, sha

MAN = ROOT / 'source/manuscript'
BUILD = ROOT / 'build'
OUT = ROOT / 'output/pdf'
COMPILER = Path('/home/zeyufu/.codex/plugins/cache/openai-bundled/latex/0.2.8/scripts/compile_latex.py')
DOCUMENTS = {'paper': 'tessera-revised-manuscript',
             'supporting': 'tessera-supporting-information',
             'atlas': 'tessera-figure-atlas'}
# Panel labels describe semantic sections. Dense repeated rows use one label
# per row, while the catalog also records the number of plotted cells.
PANELS = [6, 5, 4, 4, 5, 4, 2, 4, 4, 2, 6, 4, 4, 5, 2, 2, 1, 3, 2, 2, 4]
PLOT_PANELS = [6, 5, 4, 4, 20, 16, 8, 4, 4, 2, 6, 4, 4, 5, 6, 4, 4, 12, 6, 2, 20]
MAIN_FIGURES = 12


def figure_blocks(path):
    result = []
    for block in re.findall(r'\\begin\{figure\}.*?\\end\{figure\}', path.read_text(), re.S):
        filename = re.search(r'\\includegraphics(?:\[[^]]+\])?\{([^}]+)\}', block).group(1)
        caption = re.search(r'\\caption\{(.*?)\}(?=\\label\{|\\end\{figure\})', block, re.S).group(1)
        result.append((filename, caption))
    return result


def make_atlas():
    entries = figure_blocks(MAN / 'paper.tex') + figure_blocks(MAN / 'supporting.tex')
    assert len(entries) == len(PANELS) == 21
    preamble = r'''\documentclass[11pt]{article}
\usepackage[letterpaper,margin=.65in]{geometry}
\input{paper_fonts.tex}
\usepackage{graphicx,microtype,caption}
\usepackage[hidelinks]{hyperref}
\captionsetup{font=small,labelfont=normalfont}
\graphicspath{{figures/}}
\emergencystretch=2em
\hypersetup{pdftitle={Tessera figure atlas},pdfauthor={Zeyu Fu}}
\begin{document}\raggedbottom
'''
    pages = []
    catalog = []
    for i, ((filename, caption), panels, plot_panels) in enumerate(zip(entries, PANELS, PLOT_PANELS)):
        number = str(i + 1) if i < MAIN_FIGURES else 'S' + str(i - MAIN_FIGURES + 1)
        pages.append(r'\noindent{\Large Tessera\quad Figure ' + number + r'}\hfill 11 October 2026\par\medskip' + '\n' +
                     # Statistical figures are drawn at text width: show them at true print size.
                     (r'\begin{center}\includegraphics[width=\linewidth,height=.79\textheight,keepaspectratio]{' if (MAN / 'figures' / filename).with_suffix('.svg').exists()
                      else r'\begin{center}\includegraphics{') + filename + r'}\end{center}' + '\n' +
                     r'\noindent{\small Figure ' + number + r'. ' + caption + r'\par}' + '\n' + r'\clearpage' + '\n')
        item = {'number': number, 'filename': filename, 'panels': panels,
                'panel_labels': panels, 'plot_panels': plot_panels,
                'caption': caption, 'pdf_sha256': sha(MAN / 'figures' / filename)}
        svg = (MAN / 'figures' / filename).with_suffix('.svg')
        if svg.exists():
            item.update(svg_filename=svg.name, svg_sha256=sha(svg),
                        svg_export='Inkscape vector PDF from the editable SVG, text converted to paths')
        catalog.append(item)
    (MAN / 'atlas.tex').write_text(preamble + ''.join(pages) + r'\end{document}' + '\n')
    dump(MAN / 'ledgers/figure_catalog.json', {'figures': catalog,
         'main_figures': MAIN_FIGURES, 'supplementary_figures': len(entries) - MAIN_FIGURES,
         'main_panel_labels': sum(PANELS[:MAIN_FIGURES]), 'supplementary_panel_labels': sum(PANELS[MAIN_FIGURES:]),
         'main_plot_panels': sum(PLOT_PANELS[:MAIN_FIGURES]), 'supplementary_plot_panels': sum(PLOT_PANELS[MAIN_FIGURES:])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-figures', action='store_true')
    parser.add_argument('--compiler-script', type=Path, default=COMPILER)
    parser.add_argument('--skip-tests', action='store_true', help='Build only; final verification still requires tests.')
    args = parser.parse_args()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    if args.render_figures:
        subprocess.run([sys.executable, str(ROOT / 'source/scripts/audit_followup.py')], env=env, check=True)
        for script in ['artwork.py', 'render_figures.py', 'make_external_tables.py', 'render_external_figures.py',
                       'make_tables.py', 'make_followup_tables.py', 'audit_figure_contract.py']:
            subprocess.run([sys.executable, str(ROOT / 'source/scripts' / script)], env=env, check=True)
    else:
        subprocess.run([sys.executable, str(ROOT / 'source/scripts/make_followup_tables.py')], env=env, check=True)
        subprocess.run([sys.executable, str(ROOT / 'source/scripts/make_external_tables.py')], env=env, check=True)
    subprocess.run([sys.executable, str(ROOT / 'source/scripts/make_crc_tables.py')], env=env, check=True)
    make_atlas()
    subprocess.run([sys.executable, str(ROOT/'source/scripts/audit_submission_numbers.py')], env=env, check=True)
    subprocess.run([sys.executable, str(ROOT/'source/scripts/audit_figure_methods.py')], env=env, check=True)
    subprocess.run([sys.executable, str(ROOT/'source/scripts/audit_manuscript_structure.py')], env=env, check=True)
    BUILD.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for stem, name in DOCUMENTS.items():
        command = [sys.executable, str(args.compiler_script), str(MAN / (stem + '.tex')),
                   '--compiler', 'texlive', '--output-directory', str(BUILD), '--json']
        run = subprocess.run(command, env=env, text=True, capture_output=True)
        (BUILD / (stem + '-compiler.json')).write_text(run.stdout)
        if run.returncode:
            raise RuntimeError(f'{stem} compilation failed; inspect {BUILD / (stem + "-compiler.json")}')
        log = (BUILD / (stem + '.log')).read_text(errors='replace')
        defects = re.findall(r'Overfull[^\n]*|LaTeX Warning: (?:Float too large|.*undefined)[^\n]*|There were undefined[^\n]*', log)
        if defects:
            raise RuntimeError(f'{stem} layout/reference defects: {defects}')
        shutil.copy2(BUILD / (stem + '.pdf'), OUT / (name + '.pdf'))
        print(f'BUILT {name}.pdf')
    # Placement tests must see the newly built PDFs, not a previous render manifest.
    subprocess.run([sys.executable, str(ROOT / 'source/scripts/render_document_qa.py')], env=env, check=True)
    if not args.skip_tests:
        subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                        '--junitxml', str(ROOT / 'qa/regression-tests.xml'),
                        str(ROOT / 'source/tests')], env=env, check=True)
    dump(ROOT / 'qa/build_manifest.json', {'documents': {
         p.name: sha(p) for p in sorted(OUT.glob('*.pdf'))},
         'compiler': 'installed TeX Live via LaTeX skill compiler',
         'rendered_figures': args.render_figures,
         'tests_executed': not args.skip_tests,
         'regression_xml_sha256': sha(ROOT/'qa/regression-tests.xml') if not args.skip_tests else None,
         'test_source_sha256': {str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'source/tests').glob('test_*.py'))} if not args.skip_tests else {},
         'visual_review': 'required separately; compilation does not certify visual quality'})


if __name__ == '__main__':
    main()

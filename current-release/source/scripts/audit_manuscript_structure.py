#!/usr/bin/env python3
"""Structure, panel-citation and cross-reference audit of the manuscript sources.

Checks (all mechanical, no model fitting):
  * only Results and Methods have subsections; Introduction and Discussion are continuous prose;
  * every panel of every main data figure is cited in Results at panel-letter level, no citation
    names a panel that does not exist, and lettered citations stay inside Results;
  * Results sentences never cite a main figure without a panel letter;
  * figures and tables are first cited in numerical order and every table is cited;
  * supplementary figure/table numbers cited in the paper exist, and every supplementary item
    is cited from the main text;
  * no body text contains dates, hashes, file names or internal process language.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
from common import ROOT, read, dump, sha

MAN = ROOT / 'source/manuscript'
SCHEMATIC_FIGURES = {'figure01-workflow', 'figure13-architecture', 'figure14-target-lineage'}
PROCESS = re.compile(r'(?i)\b(locked|receipt|runner|retrospective|recovered|byte for byte|local revision|'
                     r'this revision|unuploaded|unpublished|handoff|author-side|ledger|sha-?256|hashes?)\b')
DATE = re.compile(r'\b\d{1,2} (January|February|March|April|May|June|July|August|September|October|'
                  r'November|December) 20\d\d\b')


def strip_comments(text):
    return re.sub(r'(?<!\\)%.*', '', text)


def panel_letters(spec):
    """Expand 'a--c', 'a,b', 'a, b--d' into a set of letters."""
    letters = set()
    spec = spec.replace('\\,', ',').replace('~', ' ')
    for part in re.split(r'\s*,\s*|\s+and\s+', spec.strip()):
        m = re.fullmatch(r'([a-z])\s*(?:--|-|\u2013)\s*([a-z])', part.strip())
        if m:
            if m[1] > m[2]:
                raise ValueError(f'descending panel range: {spec!r}')
            letters |= {chr(c) for c in range(ord(m[1]), ord(m[2]) + 1)}
        elif re.fullmatch(r'[a-z]', part.strip()):
            letters.add(part.strip())
        elif part.strip():
            raise ValueError(f'unparsable panel specification: {spec!r}')
    return letters


def figure_blocks(text):
    out = []
    for block in re.findall(r'\\begin\{figure\}.*?\\end\{figure\}', text, re.S):
        match = re.search(r'\\includegraphics(?:\[[^]]+\])?\{([^}]+)\}', block)
        assert match, 'figure without includegraphics'
        stem = Path(match.group(1)).stem
        label = re.search(r'\\label\{([^}]+)\}', block)
        out.append((stem, label.group(1) if label else None))
    return out


def table_labels(text):
    return [m for m in re.findall(r'\\begin\{table\}.*?\\label\{([^}]+)\}', text, re.S)]


def sections(text):
    """[(level, title, start, end)] for \\section and \\subsection."""
    marks = [(m.start(), 'section' if m[1] == 'section' else 'subsection' if m[1] == 'subsection' else 'subsub',
              m[3]) for m in re.finditer(r'\\(section|subsection|subsubsection)(\*?)\{([^}]*)\}', text)]
    marks.append((len(text), 'end', ''))
    return [(lvl, title, a, marks[i + 1][0]) for i, (a, lvl, title) in enumerate(marks[:-1])]


def supplementary_references(text, kind, label_numbers=None):
    """Read singular/plural citations, treating ranges and lists differently."""
    noun = r'(?:Figures?|Figs?\.?)' if kind == 'figure' else r'Tables?'
    pattern = (r'Supplementary\s+' + noun + r'\s*~?\s*'
               r'(S\d+(?:\s*(?:--|-|\u2013|to|and|,)\s*S?\d+)*)')
    numbers = set()
    for match in re.finditer(pattern, text):
        spec = match[1]
        numbers.update(int(n) for n in re.findall(r'S?(\d+)', spec))
        for start, end in re.findall(r'S?(\d+)\s*(?:--|-|\u2013|to)\s*S?(\d+)', spec):
            if int(start) > int(end):
                raise ValueError(f'descending supplementary range: {spec!r}')
            numbers.update(range(int(start), int(end) + 1))
    for match in re.finditer(r'Supplementary\s+' + noun + r'\s*~?\s*\\ref\{([^}]+)\}', text):
        if label_numbers is None or match[1] not in label_numbers:
            raise ValueError(f'unknown supplementary {kind} label: {match[1]!r}')
        numbers.add(label_numbers[match[1]])
    return numbers


def audit(verbose=False, paper_text=None, support_text=None):
    paper = strip_comments((MAN / 'paper.tex').read_text() if paper_text is None else paper_text)
    support = strip_comments((MAN / 'supporting.tex').read_text() if support_text is None else support_text)
    body_end = paper.index(r'\bibliographystyle')
    secs = sections(paper[:body_end])
    findings = []

    def add(severity, where, message):
        findings.append({'severity': severity, 'where': where, 'message': message})

    # ---- 1. structure
    top = [(t, a, b) for lvl, t, a, b in secs if lvl == 'section']
    names = [t for t, _, _ in top]
    for required in ['Introduction', 'Materials and methods', 'Results', 'Discussion']:
        if required not in names:
            add('BLOCKER', 'structure', f'missing section {required!r}')
    for lvl, title, a, b in secs:
        chunk = paper[a:b]
        if title in ('Introduction', 'Discussion'):
            if re.search(r'\\(sub)+section|\\paragraph|\\begin\{(itemize|enumerate|description)\}', chunk):
                add('BLOCKER', title, 'subsection, paragraph heading or list inside a continuous-prose section')
    # the first subsection after \section{Introduction}/\section{Discussion} belongs to the next section
    for i, (lvl, title, a, b) in enumerate(secs):
        if lvl == 'subsection':
            parent = [t for l, t, a2, b2 in secs[:i] if l == 'section'][-1]
            if parent not in ('Materials and methods', 'Results'):
                add('BLOCKER', parent, f'subsection {title!r} outside Methods/Results')

    # ---- 2. panel citations
    letters_by_stem = {}
    for name in ('statistical_figure_style_audit.json', 'schematic_style_audit.json'):
        for stem, rec in read(ROOT / 'qa' / name).items():
            letters_by_stem[stem] = set(rec['labels'])
    fig_order = figure_blocks(paper)
    label_to_stem = {lab: stem for stem, lab in fig_order if lab}
    stems_main = [stem for stem, _ in fig_order]
    sup_fig_order = [stem for stem, _ in figure_blocks(support)]
    sup_tab_count = len(re.findall(r'\\begin\{table\}', support))

    def section_of(pos):
        """Title of the top-level section containing character offset pos."""
        title = None
        for lvl, t, a, b in secs:
            if lvl == 'section' and a <= pos:
                title = t
        return title

    cites = {}                 # stem -> {letter: [subsections]}
    sup_cites = {}
    first_cite_pos = {}
    for m in re.finditer(r'\\panelref\{([^}]+)\}\{([^}]+)\}', paper[:body_end]):
        label, spec = m[1], m[2]
        sec = section_of(m.start())
        if label not in label_to_stem:
            add('BLOCKER', sec, f'panel citation of unknown figure label {label!r}')
            continue
        stem = label_to_stem[label]
        try:
            letters = panel_letters(spec)
        except ValueError as exc:
            add('BLOCKER', sec, str(exc)); continue
        missing = letters - letters_by_stem.get(stem, set())
        if missing:
            add('BLOCKER', sec, f'{label}: cited panels {sorted(missing)} do not exist in {stem}')
        if sec != 'Results':
            add('MAJOR', sec, f'lettered panel citation {label}({spec}) outside Results')
        if len(letters) >= 4 and letters >= letters_by_stem.get(stem, set()):
            add('MINOR', sec, f'{label}({spec}) cites every panel at once; cite the panels that support the claim')
        for letter in letters:
            cites.setdefault(stem, {}).setdefault(letter, []).append(sec)
        first_cite_pos.setdefault(label, m.start())
    # bare figure refs
    for m in re.finditer(r'Fig\.~\\ref\{([^}]+)\}(?!\()', paper[:body_end]):
        sec = section_of(m.start())
        label = m[1]
        if label in label_to_stem:
            first_cite_pos.setdefault(label, m.start())
            if sec == 'Results' and label_to_stem[label] not in SCHEMATIC_FIGURES:
                add('MAJOR', sec, f'Results cite {label} without a panel letter')
    for stem in stems_main:
        if stem in SCHEMATIC_FIGURES or stem not in letters_by_stem:
            continue
        for letter in sorted(letters_by_stem[stem]):
            if letter not in cites.get(stem, {}):
                add('MAJOR', 'Results', f'panel {stem}({letter}) is never cited in Results')
    # ---- 3. first-citation order
    def order_of(labels):
        return [l for l, _ in sorted(((l, first_cite_pos[l]) for l in labels if l in first_cite_pos), key=lambda x: x[1])]
    fig_labels = [lab for _, lab in fig_order if lab]
    if order_of(fig_labels) != [l for l in fig_labels if l in first_cite_pos]:
        add('MAJOR', 'figures', f'figures are not first cited in numerical order: {order_of(fig_labels)}')
    for lab in fig_labels:
        if lab not in first_cite_pos:
            add('BLOCKER', 'figures', f'figure {lab} is never cited')
    tabs = table_labels(paper)
    tab_pos = {}
    for lab in tabs:
        m = re.search(r'\\ref\{' + re.escape(lab) + r'\}', paper[:body_end])
        if m:
            tab_pos[lab] = m.start()
            if section_of(m.start()) not in ('Results', 'Materials and methods'):
                add('MINOR', lab, f'first cited outside Results/Methods ({section_of(m.start())})')
        else:
            add('BLOCKER', 'tables', f'table {lab} is never cited')
    if [t for t, _ in sorted(tab_pos.items(), key=lambda x: x[1])] != [t for t in tabs if t in tab_pos]:
        add('MAJOR', 'tables', 'tables are not first cited in numerical order')
    # ---- 4. supplementary references
    try:
        cited_sfig = supplementary_references(paper, 'figure')
        supplement_table_labels = {}
        for number, block in enumerate(re.findall(r'\\begin\{table\}.*?\\end\{table\}', support, re.S), 1):
            label = re.search(r'\\label\{([^}]+)\}', block)
            if label:
                supplement_table_labels[label[1]] = number
        cited_stab = supplementary_references(paper, 'table', supplement_table_labels)
    except ValueError as exc:
        add('BLOCKER', 'supplement', str(exc))
        cited_sfig, cited_stab = set(), set()
    for m in re.finditer(r'\\suppanelref\{S(\d+)\}\{([^}]+)\}', paper[:body_end]):
        number = int(m[1])
        cited_sfig.add(number)
        if not 1 <= number <= len(sup_fig_order):
            continue
        stem = sup_fig_order[number - 1]
        try:
            letters = panel_letters(m[2])
        except ValueError as exc:
            add('BLOCKER', 'supplement', str(exc))
            continue
        missing = letters - letters_by_stem.get(stem, set())
        if missing:
            add('BLOCKER', 'supplement', f'S{number}: cited panels {sorted(missing)} do not exist')
        sec = section_of(m.start())
        if sec != 'Results':
            add('MAJOR', sec, f'lettered supplementary citation S{number}({m[2]}) outside Results')
        else:
            sup_cites.setdefault(stem, set()).update(letters)
    for n in cited_sfig:
        if not 1 <= n <= len(sup_fig_order):
            add('BLOCKER', 'supplement', f'Supplementary Fig. S{n} does not exist ({len(sup_fig_order)} supplementary figures)')
    for n in cited_stab:
        if not 1 <= n <= sup_tab_count:
            add('BLOCKER', 'supplement', f'Supplementary Table S{n} does not exist ({sup_tab_count} supplementary tables)')
    for n in range(1, len(sup_fig_order) + 1):
        if n not in cited_sfig:
            add('MAJOR', 'supplement', f'Supplementary Fig. S{n} ({sup_fig_order[n-1]}) is never cited from the main text')
    for n in range(1, sup_tab_count + 1):
        if n not in cited_stab:
            add('MAJOR', 'supplement', f'Supplementary Table S{n} is never cited from the main text')
    for number, stem in enumerate(sup_fig_order, 1):
        if stem in SCHEMATIC_FIGURES:
            continue
        for letter in sorted(letters_by_stem.get(stem, set()) - sup_cites.get(stem, set())):
            add('MAJOR', 'Results', f'panel Supplementary Fig. S{number}({letter}) is never cited in Results')
    # ---- 5. hygiene
    body = paper[paper.index(r'\title'):body_end]
    for m in DATE.finditer(body):
        add('MAJOR', 'hygiene', f'date in manuscript body or front matter: {m[0]!r}')
    for m in PROCESS.finditer(body):
        ctx = body[max(0, m.start() - 50):m.end() + 50].replace('\n', ' ')
        add('MAJOR', 'hygiene', f'process language {m[0]!r}: ...{ctx}...')
    return findings, cites, letters_by_stem


def main():
    findings, cites, letters = audit()
    for f in findings:
        print(f"{f['severity']:8s} [{f['where']}] {f['message']}")
    counts = {s: sum(f['severity'] == s for f in findings) for s in ('BLOCKER', 'MAJOR', 'MINOR')}
    print('\nSUMMARY', counts)
    print('\nPanel citation matrix (letter: n citations in Results)')
    for stem, ls in cites.items():
        print(f'  {stem:30s}', ' '.join(f'{l}:{len(ls.get(l, []))}' for l in sorted(set(letters.get(stem, set())) | set(ls))))
    dump(ROOT / 'qa/manuscript_structure_audit.json', {
        'status': 'PASS' if not counts['BLOCKER'] and not counts['MAJOR'] else 'FAIL',
        'counts': counts, 'findings': findings,
        'manuscript_sha256': sha(MAN / 'paper.tex'),
        'supporting_sha256': sha(MAN / 'supporting.tex'),
        'audit_source_sha256': sha(__file__),
        'main_panel_citations': {stem: {letter: len(positions) for letter, positions in panels.items()}
                                for stem, panels in cites.items()},
        'scope': 'Mechanical source structure, citation coverage and hygiene. Not scientific peer review.'})
    return 1 if counts['BLOCKER'] or counts['MAJOR'] else 0


if __name__ == '__main__':
    sys.exit(main())

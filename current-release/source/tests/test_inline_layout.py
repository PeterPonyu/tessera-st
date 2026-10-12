"""Unit tests for the reading-layout (inline figure) generator of the Scientific Reports builder."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('build_scientific_reports', ROOT / 'source/scripts/build_scientific_reports.py')
builder = importlib.util.module_from_spec(spec)
sys.modules['build_scientific_reports'] = builder
spec.loader.exec_module(builder)

FIG = r'\begin{figure}[!htbp]\centering X\caption{c}\label{%s}\end{figure}'
BODY = ('\\section{Results}\nFirst paragraph cites \\panelref{fig:a}{a--b}.\n\nSecond paragraph cites '
        'Table~\\ref{tab:t} and \\panelref{fig:b}{c}.\n\nThird paragraph.\n\n\\bibliographystyle{x}\n')


def item(kind, label):
    return {'kind': kind, 'label': label,
            'source': (FIG if kind == 'figure' else FIG.replace('figure', 'table')) % label}


class InlineLayout(unittest.TestCase):
    kept = [item('figure', 'fig:a'), item('table', 'tab:t'), item('figure', 'fig:b')]

    def test_each_display_follows_the_paragraph_that_first_cites_it(self):
        out = builder.place_inline(BODY, self.kept)
        self.assertLess(out.index('First paragraph'), out.index('{fig:a}\\end'))
        self.assertLess(out.index('{fig:a}\\end'), out.index('Second paragraph'))
        self.assertLess(out.index('Second paragraph'), out.index('{tab:t}\\end'))
        self.assertLess(out.index('{tab:t}\\end'), out.index('{fig:b}\\end'))
        self.assertLess(out.index('{fig:b}\\end'), out.index('Third paragraph'))
        self.assertEqual(out.count('\\begin{figure}[tbp]'), 2)
        self.assertEqual(out.count('\\begin{table}[tbp]'), 1)

    def test_display_order_must_match_first_citation_order(self):
        with self.assertRaises(ValueError):
            builder.place_inline(BODY, [self.kept[2], self.kept[0], self.kept[1]])

    def test_uncited_display_is_rejected(self):
        with self.assertRaises(ValueError):
            builder.place_inline(BODY, self.kept + [item('figure', 'fig:never')])

    def test_overrides_must_match_exactly_once(self):
        self.assertEqual(builder.apply_overrides('a b a', [{'find': 'b', 'replace': 'c'}]), 'a c a')
        for find in ('a', 'z'):
            with self.assertRaises(ValueError):
                builder.apply_overrides('a b a', [{'find': find, 'replace': 'c'}])

    def test_equation_style_follows_the_journal(self):
        self.assertEqual(builder.journal_equation_style('in Eq.~\\eqref{x}. Eq.~\\eqref{y} holds'),
                         'in equation~\\eqref{x}. Equation~\\eqref{y} holds')

    def test_configuration_declares_the_inline_limits(self):
        cfg = json.loads((ROOT / 'source/scientific-reports.json').read_text())
        self.assertLessEqual(cfg['inline_max_pages_after_citation'], 2)
        self.assertEqual(cfg['inline_float_spec'], 'tbp')


if __name__ == '__main__':
    unittest.main()

"""Regression checks for submission defects and the audit's citation parser."""
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source/scripts"), str(ROOT / "source/src"),
                str(ROOT / "validation-extension"), str(ROOT / "validation-external")]
from audit_manuscript_structure import audit, panel_letters, supplementary_references
from run_extension import weighted_features


def manuscript():
    return (ROOT / "source/manuscript/paper.tex").read_text()


def test_current_manuscript_passes_source_structure_audit():
    findings, _, _ = audit()
    assert not findings


def test_missing_input_table_reference_is_rejected():
    text = manuscript().replace(r"Table~\ref{tab:inputs}", "Input coverage")
    findings, _, _ = audit(paper_text=text)
    assert any(f["severity"] == "BLOCKER" and "tab:inputs" in f["message"] for f in findings)


def test_omitting_a_breast_data_panel_is_rejected():
    text = manuscript().replace(r"\panelref{fig:mixtures}{c--d}", r"\panelref{fig:mixtures}{c}")
    findings, _, _ = audit(paper_text=text)
    assert any("figure08-breast-mixtures(d)" in f["message"] for f in findings)


def test_supplementary_lists_do_not_invent_intervening_citations():
    assert supplementary_references("Supplementary Tables S1 and S4", "table") == {1, 4}
    assert supplementary_references("Supplementary Tables S9--S13", "table") == set(range(9, 14))
    assert supplementary_references("Supplementary Figs.~S1, S2 and S7", "figure") == {1, 2, 7}


def test_supplementary_table_labels_resolve_and_unknown_labels_fail():
    text = r"Supplementary Table~\ref{tab:crc-pilot}"
    assert supplementary_references(text, "table", {"tab:crc-pilot": 14}) == {14}
    with pytest.raises(ValueError, match="unknown supplementary table label"):
        supplementary_references(text, "table", {})


def test_invalid_supplementary_panel_is_rejected():
    text = manuscript().replace(r"\suppanelref{S9}{a--d}", r"\suppanelref{S9}{a--e}")
    findings, _, _ = audit(paper_text=text)
    assert any(f["severity"] == "BLOCKER" and "S9" in f["message"] for f in findings)


def test_descending_citation_ranges_are_rejected():
    with pytest.raises(ValueError):
        panel_letters("d--a")
    with pytest.raises(ValueError):
        supplementary_references("Supplementary Tables S13--S9", "table")


def test_rendered_abstract_fits_the_journal_200_word_limit():
    text = subprocess.run(["pdftotext", "-nopgbrk",
                           str(ROOT / "output/scientific-reports/scientific-reports-manuscript.pdf"), "-"],
                          check=True, capture_output=True, text=True).stdout
    abstract = text.split("Abstract", 1)[1].split("Keywords:", 1)[0]
    assert 100 <= len(re.findall(r"\S+", abstract)) <= 200


def test_zero_feature_distance_and_isolated_frame_remain_finite():
    x = np.array([[1, 2], [1, 2], [5, 8]], dtype=np.float32)
    xy = np.array([[0, 0], [1, 0], [0, 0]], dtype=float)
    fields = np.array(["pair", "pair", "singleton"])
    value, _ = weighted_features(x, xy, fields, "expression")
    assert np.isfinite(value).all()
    np.testing.assert_array_equal(value[:2], x[:2])
    np.testing.assert_array_equal(value[2], [0, 0])

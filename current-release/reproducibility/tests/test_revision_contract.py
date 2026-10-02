"""Current manuscript checks reject mismatched seeds and fabricated summaries."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

CAP = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("claim_contract", CAP / "manuscript/verify_committed_claims.py")
claims = importlib.util.module_from_spec(spec)
spec.loader.exec_module(claims)


def example():
    raw = [-.2, -.1, -.3]
    summary = dict(status="complete", seeds=[1, 2, 3], raw_per_seed=raw,
                   raw_mean=float(np.mean(raw)), raw_sd=float(np.std(raw, ddof=1)),
                   delta_ari_vs_expression=float(np.mean(raw)) - .1)
    records = [dict(status="complete", seed=i, raw_ari=v) for i, v in enumerate(raw, 1)]
    return summary, records


def test_negative_result_is_not_an_acceptance_failure():
    summary, records = example()
    claims.check_seed_summary(summary, records, .1)


@pytest.mark.parametrize("fault", ["order", "missing", "failed", "mean", "sd", "delta", "partial_diagnostic"])
def test_changed_or_incomplete_seed_summary_fails(fault):
    summary, records = copy.deepcopy(example())
    if fault == "order":
        records = records[::-1]
    elif fault == "missing":
        records.pop()
    elif fault == "failed":
        records[1]["status"] = "failed"
    elif fault in {"mean", "sd"}:
        summary["raw_" + fault] += .1
    elif fault == "delta":
        summary["delta_ari_vs_expression"] += .1
    else:
        records[0]["diagnostic_only_refined_control_ari"] = .7
    with pytest.raises(ValueError):
        claims.check_seed_summary(summary, records, .1)


def test_persisted_current_source_contract():
    claims.verify(source_only=True)
    for filename in ["REPRODUCIBILITY-CURRENT.md", "reproducibility/METHOD-AUDIT.md"]:
        text = (CAP / filename).read_text(encoding="utf-8")
        current, historical = text.split("## Archived pre-integration", 1)
        assert "147" in current and "visual_review=NOT_PERFORMED" in current
        assert "HISTORICAL ONLY" in historical
        assert not any(line.startswith("# ") or line.startswith("Status:")
                       for line in historical.splitlines())
        assert "> Historical status at that time:" in historical

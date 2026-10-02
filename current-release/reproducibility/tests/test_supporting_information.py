"""Readable SI must reuse complete committed records and stay in Results."""
import importlib.util
import json
from pathlib import Path
import re
import shutil

import pytest

CAP = Path(__file__).resolve().parents[2]
MAN = CAP / "manuscript"
SPEC = importlib.util.spec_from_file_location("tessera_si", MAN / "scripts/build_supporting_information.py")
SI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SI)


def isolated_export(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    for name in ["protocol.json", "summary.json", "QA.json", "estimand-audit.json"]:
        shutil.copy2(SI.LEDGER / name, ledger / name)
    monkeypatch.setattr(SI, "LEDGER", ledger)
    return ledger


def test_current_si_tables_are_explicitly_referenced_in_results():
    body = (MAN / "paper.tex").read_text().split(r"\section{Results}")[1].split(r"\section{Discussion}")[0]
    for number in range(1, 6):
        assert re.search(r"(?:SI\s+)?Table~S" + str(number), body)
    supplement = (MAN / "SI.tex").read_text()
    assert "new independent" not in supplement.lower()
    assert "adds no tissue" in supplement
    assert len(re.findall(r"\\caption\{", supplement)) == 5


def test_exported_table_identities_and_full_seed_inventory(tmp_path, monkeypatch):
    ledger = isolated_export(tmp_path, monkeypatch)
    SI.main()
    text = (ledger / "supporting_tables.tex").read_text()
    audit = json.loads((ledger / "estimand-audit.json").read_text())
    for row in audit["generator_rows"]:
        cell = f"{row['substrate']} & {row['seed']} & {row['level']:.0f} & ${row['expression_moran']:.6f}$"
        assert cell in text
    provenance = json.loads((ledger / "supporting_tables-provenance.json").read_text())
    assert provenance["completed_method_seeds"] == 147
    assert provenance["generator_endpoint_rows"] == 20
    assert provenance["scientific_ready_attested"] is False
    assert provenance["generated_sha256"] == SI.sha(ledger / "supporting_tables.tex")


@pytest.mark.parametrize("fault", ["incomplete_run", "missing_endpoint", "duplicate_endpoint", "changed_audit_source"])
def test_si_export_fails_closed_before_authoring_on_changed_evidence(tmp_path, monkeypatch, fault):
    ledger = isolated_export(tmp_path, monkeypatch)
    if fault == "incomplete_run":
        path = ledger / "QA.json"
        record = json.loads(path.read_text())
        record["completed_method_seeds"] = 146
    else:
        path = ledger / "estimand-audit.json"
        record = json.loads(path.read_text())
        if fault == "missing_endpoint":
            record["generator_rows"].pop()
        elif fault == "duplicate_endpoint":
            record["generator_rows"][-1] = record["generator_rows"][0]
        else:
            key = next(iter(record["source_sha256"]))
            record["source_sha256"][key] = "0" * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        SI.main()
    assert not (ledger / "supporting_tables.tex").exists()


def test_build_and_release_include_both_current_documents():
    build = (MAN / "scripts/rebuild_current.sh").read_text()
    assert build.index("build_supporting_information.py") < build.index("latexmk")
    assert "-halt-on-error SI.tex" in build
    release = json.loads((CAP / "RELEASE-MANIFEST.json").read_text())
    documents = {row["path"]: row for row in release["files"]
                 if row["path"] in {"manuscript/paper.pdf", "manuscript/SI.pdf"}}
    assert set(documents) == {"manuscript/paper.pdf", "manuscript/SI.pdf"}
    for path, row in documents.items():
        assert SI.sha(CAP / path) == row["sha256"]

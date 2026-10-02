#!/usr/bin/env python3
"""Bind the repaired working face, never promote it or mutate historical kits.

--bind-method is an explicit reviewed-source refresh, not a default build step.
The default rebuild verifies those locked inputs before rendering/compiling and
uses --attest-build only after the source, PDF and layout checks have succeeded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile
import xml.etree.ElementTree as ET

CAP = Path(__file__).resolve().parents[2]
MAN = CAP / "manuscript"
RUN = CAP / "experiments/generated/field-identified-fixed-backend-20260923-v1"
LEDGER = MAN / "ledgers/revision_20260923"
METHOD = CAP / "reproducibility/method-contract.json"
HOLD = "SCIENTIFIC_SUBMISSION_HOLD"
BLOCKERS = [
    "SpaceFlow, GraphST and SpaGCN not run under the corrected spatial protocol; no scores imputed.",
    "Field provenance unestablished here for seqFISH, osmFISH, BRCA, IMC, Slide-seqV2, st_COAD, st_LIHC and st_OV.",
    "Historical n11/n12/n14 oracle, proxy, backend, winner/rank and tissue-ablation results are archived outside the main inference, not repaired.",
    "CODEX SpatialLeiden is not count-matched to 100 classes; no equal-K superiority conclusion is licensed.",
    "Legacy confidence/ECE claims withdrawn; no calibrated confidence model for returned KMeans labels.",
    "Independent scientific review of revised estimands, field-aware adapters and annotation scope pending.",
    "Machine build does not attest visual acceptance; the separately hash-bound final-package review supplies its own visual record.",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def item(path, base):
    return {"path": str(path.relative_to(base)), "sha256": sha(path)}


def verify_items(rows, base):
    wrong = [r["path"] for r in rows if not (base / r["path"]).is_file()
             or sha(base / r["path"]) != r["sha256"]]
    if wrong:
        raise ValueError("Bound identity changed: " + ", ".join(wrong))


def protocol_check():
    protocol = read(RUN / "protocol.json")
    expected = "ed165f146b42da3ff09019246cbb8107ba279f5f29412df8cf9a3dde0c95a5f9"
    if sha(RUN / "protocol.json") != expected:
        raise ValueError("Do not rebind frozen training protocol")
    for name, digest in protocol["code_sha256"].items():
        path = Path(name)
        # External sources are historical recorded identities, not build dependencies.
        if path.is_relative_to(CAP) and sha(path) != digest:
            raise ValueError(f"Frozen training source changed: {path}")
    return protocol


def bind_method():
    protocol = protocol_check()
    # Preserve earlier bound legacy implementations as audited historical inputs.
    old = read(CAP / "reproducibility/backups/20260923-status/method-contract.json")
    paths = {CAP / row["path"] for row in old["locked_files"]}
    paths |= {Path(p) for p in protocol["code_sha256"] if Path(p).is_relative_to(CAP)}
    paths |= {CAP / p for p in [
        "experiments/verify_field_aware_panel_20260923.py",
        "experiments/summarize_field_aware_panel_20260923.py",
        "experiments/audit_estimands_20260923.py", "experiments/stat_rigor.py",
        "experiments/protocol_guard.py", "experiments/coordinate_guard.py",
        "reproducibility/verify_method_contract.py", "REPRODUCIBILITY-CURRENT.md",
        "manuscript/verify_committed_claims.py",
        "manuscript/Makefile", "manuscript/scripts/rebuild_current.sh"]}
    paths |= set((CAP / "reproducibility/tests").glob("test_*.py"))
    paths |= set((MAN / "scripts").glob("*.py"))
    paths |= {p for p in RUN.rglob("*") if p.is_file() and p.suffix in {".json", ".npz"}}
    paths |= {LEDGER / p for p in ["summary.json", "QA.json", "protocol.json", "estimand-audit.json"]}
    for p in paths:
        if not p.is_file() or not p.resolve().is_relative_to(CAP):
            raise ValueError(f"Missing or external method dependency: {p}")
    contract = {"schema_version": "field-identified-method-2", "paper": "Tessera",
        "status": HOLD, "scientific_ready_attested": False, "visual_review": "NOT_PERFORMED",
        "scope": "Completed seven-object/seven-procedure/three-seed reanalysis; bounded protocol audit and specified-intervention total effects.",
        "protocol_id": protocol["protocol_id"], "protocol_sha256": sha(RUN / "protocol.json"),
        "objects": list(protocol["retained_inputs"]), "methods_run": protocol["methods"],
        "seeds": protocol["seeds"], "completed_runs": 147, "failed_runs": [],
        "methods_not_run": protocol["unavailable_methods"], "remaining_blockers": BLOCKERS,
        "external_source_identities_recorded_not_build_dependencies": {
            p: h for p, h in protocol["code_sha256"].items() if not Path(p).is_relative_to(CAP)},
        "source_identity_note": "Original protocol/QA contain raw-HDF5 identities; a hash match is not scientific clearance.",
        "locked_files": [item(p, CAP) for p in sorted(paths)]}
    write(METHOD, contract)
    print(f"Bound {len(paths)} local method/evidence files; training protocol unchanged; HOLD retained")


def historical_record(archive=None):
    manifest = MAN / "attestation/historical-snapshots-20260923.json"
    if archive:
        rows = []
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                name = member.name.removeprefix("./")
                if not name.startswith(("manuscript/submission_flat/", "manuscript/submission_flat_candidate/")):
                    continue
                if not member.isfile():
                    continue
                stream = tar.extractfile(member)
                digest = hashlib.sha256(stream.read()).hexdigest()
                p = CAP / name
                if not p.is_file() or sha(p) != digest:
                    raise ValueError(f"Historical snapshot differs from preintegration backup: {name}")
                rows.append({"path": name, "sha256": digest})
        if not rows:
            raise ValueError("No historical snapshot members in backup")
        baseline = {"scope": "Both historical submission directories compared byte-for-byte with preintegration backup",
                    "backup_path": str(archive), "backup_sha256": sha(archive),
                    "files": sorted(rows, key=lambda x: x["path"])}
        write(manifest, baseline)
    baseline = read(manifest)
    verify_items(baseline["files"], CAP)
    current = {str(p.relative_to(CAP)) for folder in [MAN / "submission_flat", MAN / "submission_flat_candidate"]
               for p in folder.rglob("*") if p.is_file()}
    if current != {row["path"] for row in baseline["files"]}:
        raise ValueError("Historical snapshot file set differs from baseline")
    return item(manifest, MAN), len(baseline["files"])


def build_inputs():
    paths = {MAN / p for p in ["paper.tex", "SI.tex", "refs.bib", "paper.bbl", "Makefile", "README.md",
                             "FIGURE-MANIFEST.md", "verify_committed_claims.py"]}
    paths |= {p for p in (MAN / "scripts").iterdir() if p.suffix in {".py", ".sh"}}
    paths |= {p for p in (MAN / "ledgers").rglob("*") if p.is_file()}
    # Include only actual manuscript figure dependencies, not every draft image.
    source = (MAN / "paper.tex").read_text() + (LEDGER / "field_results.tex").read_text()
    for p in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source):
        paths.add(MAN / p)
    for p in re.findall(r"\\input\{([^}]+)\}", source):
        paths.add(MAN / (p if p.endswith(".tex") else p + ".tex"))
    for p in paths:
        if not p.resolve().is_relative_to(MAN):
            raise ValueError(f"External manuscript dependency: {p}")
    return [item(p, MAN) for p in sorted(paths)]


def attest():
    protocol = protocol_check()
    contract = read(METHOD)
    verify_items(contract["locked_files"], CAP)
    qa = read(MAN / "attestation/current-pdf-qa.json")
    if qa["status"] != "passed" or qa["paper"]["sha256"] != sha(MAN / "paper.pdf"):
        raise ValueError("No current passing machine PDF QA")
    if qa["latex_log_sha256"] != sha(MAN / "paper.log") or qa["rendered_pages"] != len(qa["paper"]["pages"]):
        raise ValueError("Current build log or page render identity missing")
    si = qa["supporting_information"]
    if (si["sha256"] != sha(MAN / "SI.pdf") or si["errors"] or
            qa["si_log_sha256"] != sha(MAN / "SI.log") or
            qa["rendered_si_pages"] != len(si["pages"])):
        raise ValueError("Current SI build, bounds or page render identity missing")
    history, count = historical_record()
    tests_path = MAN / "attestation/revision-tests.xml"
    suites = ET.parse(tests_path).getroot().findall("testsuite")
    tests = {key: sum(int(s.get(key, "0")) for s in suites)
             for key in ["tests", "failures", "errors", "skipped"]}
    if tests != {"tests": 34, "failures": 0, "errors": 0, "skipped": 0}:
        raise ValueError("Expected complete 34-test regression evidence")
    release = {"schema_version": "field-identified-working-face-2", "entrypoint": "scripts/rebuild_current.sh",
        "status": HOLD, "scientific_ready_attested": False, "visual_review_attested": False,
        "visual_review": "NOT_PERFORMED",
        "working_face": "paper.tex -> paper.pdf; original manuscript integrated in place",
        "historical_faces": ["submission_flat/", "submission_flat_candidate/"],
        "scope": {"protocol_id": protocol["protocol_id"], "protocol_sha256": sha(RUN / "protocol.json"),
                  "objects": list(protocol["retained_inputs"]), "methods_run": protocol["methods"],
                  "seeds": protocol["seeds"], "completed_runs": 147, "failed_runs": [],
                  "methods_not_run": protocol["unavailable_methods"], "model_reruns_in_rebuild": False},
        "remaining_blockers": BLOCKERS,
        "method_contract": item(METHOD, CAP), "method_contract_path_root": "capsule",
        "build_inputs": build_inputs(),
        "documents": [{**item(MAN / "paper.pdf", MAN), "pages": len(qa["paper"]["pages"])},
                      {**item(MAN / "SI.pdf", MAN), "pages": len(si["pages"])}],
        "machine_pdf_qa": item(MAN / "attestation/current-pdf-qa.json", MAN),
        "regression_tests": {**item(tests_path, MAN), **tests,
                             "scope": "Saved 34-test run; ordinary manuscript rebuild does not refit models or rerun tests"},
        "historical_preservation": {**history, "files_compared": count}}
    for name in ["release-current.json", "release-spec.json"]:
        write(MAN / name, release)
    (MAN / "working-face.sha256").write_text(f"{sha(MAN / 'paper.pdf')}  paper.pdf\n", encoding="utf-8")
    attestation = {"schema_version": 2, "scope": "Native latexmk working-face build; no training; no submission promotion",
        "status": HOLD, "scientific_ready_attested": False, "visual_review_attested": False,
        "visual_review": "NOT_PERFORMED",
        "source": item(MAN / "paper.tex", MAN), "document": release["documents"][0],
        "supporting_information": release["documents"][1],
        "release_current_sha256": sha(MAN / "release-current.json"),
        "method_contract_sha256": sha(METHOD), "log": item(MAN / "paper.log", MAN),
        "machine_pdf_qa": release["machine_pdf_qa"],
        "regression_tests": release["regression_tests"],
        "command": "SOURCE_DATE_EPOCH=0 FORCE_SOURCE_DATE=1 TZ=UTC latexmk -g -pdf -interaction=nonstopmode -halt-on-error paper.tex",
        "wrapper_note": "Earlier wrapper log UnicodeDecodeError was not a TeX or scientific failure; native latexmk succeeded.",
        "historical_preservation": release["historical_preservation"], "remaining_blockers": BLOCKERS}
    write(MAN / "attestation/current-revision-build.json", attestation)
    print(f"Current release bound: {release['documents'][0]['pages']} pages, {len(release['build_inputs'])} build inputs; "
          f"{count} historical files unchanged; HOLD remains")


def check():
    protocol_check()
    record = read(MAN / "release-current.json")
    if record != read(MAN / "release-spec.json") or record["status"] != HOLD or record["scientific_ready_attested"]:
        raise ValueError("Current release contract/status differs")
    verify_items(record["build_inputs"] + record["documents"] +
                 [record["machine_pdf_qa"], record["regression_tests"]], MAN)
    verify_items([record["method_contract"]], CAP)
    verify_items(read(METHOD)["locked_files"], CAP)
    history, count = historical_record()
    if history["sha256"] != record["historical_preservation"]["sha256"]:
        raise ValueError("Historical preservation record changed")
    face = (MAN / "working-face.sha256").read_text().split()[0]
    att = read(MAN / "attestation/current-revision-build.json")
    if face != sha(MAN / "paper.pdf") or att["release_current_sha256"] != sha(MAN / "release-current.json"):
        raise ValueError("Current face/attestation identity changed")
    verify_items([att["source"], att["document"], att["supporting_information"],
                  att["log"], att["machine_pdf_qa"]], MAN)
    print(f"CURRENT RELEASE CONTRACT PASS; {count} preserved historical files; scientific/visual HOLD retained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-method", action="store_true")
    parser.add_argument("--attest-build", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--historical-backup", type=Path)
    args = parser.parse_args()
    if args.historical_backup:
        _, count = historical_record(args.historical_backup)
        print(f"Historical preintegration comparison PASS: {count} files")
    if args.bind_method:
        bind_method()
    if args.attest_build:
        attest()
    if args.check:
        check()
    if not any(vars(args).values()):
        parser.error("Select a binding/check operation explicitly")

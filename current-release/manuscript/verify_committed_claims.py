"""Check current claims against frozen records, without fitting or rescoring.

The independent raw-HDF5/prediction rescore is recorded in QA.json. This check
binds that evidence to seed summaries, figures and the working face, not to any
historical submission snapshot. It does not attest scientific readiness.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
CAP = HERE.parent
PROTOCOL = "field-identified-fixed-backend-20260923-v1"
PROTOCOL_SHA = "ed165f146b42da3ff09019246cbb8107ba279f5f29412df8cf9a3dde0c95a5f9"
RUN = CAP / "experiments/generated" / PROTOCOL
LEDGER = HERE / "ledgers/revision_20260923"
OBJECTS = ["DLPFC", "MERFISH", "STARmap", "MIBI", "Open-ST", "CODEX", "Zhuang"]
METHODS = ["expression-only", "neighbor-mean", "BANKSY-style", "SpatialLeiden",
           "Tessera", "STAGATE", "SEDR"]
NOT_RUN = ["SpaceFlow", "GraphST", "SpaGCN"]
SEEDS = [1, 2, 3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(np.isfinite(actual) and abs(actual - expected) < 1e-12,
            f"{label}: {actual!r} != {expected!r}")


def check_seed_summary(summary, records, comparator_mean):
    """Reject shuffled/partial seeds and summaries not derived from records."""
    require(summary["status"] == "complete", "Incomplete summary")
    require(summary["seeds"] == SEEDS, "Summary seed identity differs")
    require([r["seed"] for r in records] == SEEDS, "Record seed identity differs")
    require(all(r["status"] == "complete" for r in records), "Incomplete seed record")
    raw = [r["raw_ari"] for r in records]
    require(summary["raw_per_seed"] == raw, "Per-seed summary differs from records")
    close(summary["raw_mean"], np.mean(raw), "Mean ARI")
    close(summary["raw_sd"], np.std(raw, ddof=1), "Seed SD")
    close(summary["delta_ari_vs_expression"], np.mean(raw) - comparator_mean, "Named contrast")
    for key in ["refined_ari", "diagnostic_only_refined_control_ari",
                "diagnostic_only_pooled_coordinate_ari"]:
        if any(key in r for r in records):
            require(all(key in r for r in records), f"Partial diagnostic: {key}")
            close(summary[key + "_mean"], np.mean([r[key] for r in records]), key)


def verify(source_only=False):
    protocol = read(LEDGER / "protocol.json")
    qa = read(LEDGER / "QA.json")
    summary = read(LEDGER / "summary.json")
    require(sha(LEDGER / "protocol.json") == PROTOCOL_SHA, "Frozen protocol changed")
    require(protocol["protocol_id"] == summary["protocol_id"] == PROTOCOL, "Protocol ID")
    require(list(protocol["retained_inputs"]) == OBJECTS, "Retained input set/order")
    require(protocol["methods"] == METHODS and protocol["seeds"] == SEEDS, "Method/seed set")
    require(protocol["unavailable_methods"] == qa["unavailable_methods"] == NOT_RUN, "Missing-method set")
    require(qa["protocol_sha256"] == PROTOCOL_SHA and not qa["scientific_ready"], "QA scope")
    require(qa["completed_method_seeds"] == qa["expected_method_seeds"] == 147, "QA run count")
    require(qa["failed_runs"] == [], "Recorded run failure requires explicit reporting")
    for name in ["protocol.json", "summary.json", "QA.json"]:
        require(sha(LEDGER / name) == sha(RUN / name), f"Bundled/run ledger differs: {name}")
    require([r["dataset"] for r in summary["rows"]] == OBJECTS, "Summary object set/order")
    generated = (LEDGER / "field_results.tex").read_text(encoding="utf-8")
    completed = 0
    for row in summary["rows"]:
        name = row["dataset"]
        folder = RUN / name
        meta = read(folder / "input.json")
        require(all(row[k] == v for k, v in meta.items()), f"Input metadata differs: {name}")
        require(row["graph_cross_field_edges"] == row["labelled_cross_field_edges"] == 0,
                f"Spatial field violation: {name}")
        checked = qa["datasets"][name]
        require(checked["zero_cross_field_edges"] and checked["contiguity_raw_source_reproduced"],
                f"Independent source/graph check: {name}")
        require(checked["completed_method_seeds_rescored"] == 21, f"Incomplete rescore: {name}")
        require(set(row["methods"]) == set(METHODS), f"Mixed/missing method set: {name}")
        for method in METHODS:
            records = [read(folder / f"{method}-seed{s}.json") for s in SEEDS]
            for record, seed in zip(records, SEEDS):
                require((record["dataset"], record["method"], record["seed"]) ==
                        (name, method, seed), "Wrong dataset/method/seed identity")
                require(record["protocol_sha256"] == PROTOCOL_SHA, "Record protocol identity")
                require(record["source_sha256"] == meta["source_sha256"], "Record source identity")
                require(record["predictions_sha256"] == sha(folder / f"{method}-seed{seed}.npz"),
                        f"Prediction identity: {name}/{method}/{seed}")
                if method == "expression-only":
                    require(not record["spatial_refinement"] and "refined_ari" not in record,
                            "Spatially contaminated primary control")
            result = row["methods"][method]
            check_seed_summary(result, records, row["methods"]["expression-only"]["raw_mean"])
            cell = f"${result['raw_mean']:.3f}\\pm{result['raw_sd']:.3f}$"
            require(cell in generated, f"Printed ARI cell differs: {name}/{method}")
            completed += len(records)
        for method in NOT_RUN:
            availability = read(folder / f"{method}-availability.json")
            require(availability["method"] == method and availability["status"] == "not_run"
                    and availability["seeds_not_run"] == SEEDS and availability["reason"],
                    f"Unavailable method not explicit: {name}/{method}")
        input_cells = [name, f"{row['n_cells']:,}", f"{row['n_labelled']:,}",
                       str(row['n_fields']), str(row['n_classes'])]
        input_cells += [f"{row[key]:.3f}" for key in
                        ["contiguity_cell_weighted", "within_field_permutation_expectation", "chance_adjusted"]]
        require(' & '.join(input_cells) + r'\\' in generated, f"Printed input row differs: {name}")
        ari_cells = [f"${row['methods'][m]['raw_mean']:.3f}\\pm{row['methods'][m]['raw_sd']:.3f}$"
                     for m in METHODS]
        require(name + ' & ' + ' & '.join(ari_cells) + r'\\' in generated,
                f"Printed method order/value row differs: {name}")
    require(set(summary["correlations"]) == set(METHODS[1:]), "Incomplete correlation set")
    for method, stats in summary["correlations"].items():
        require(method in METHODS[1:] and stats["n"] == 7, "Wrong association scope")
        outcomes = [r["methods"][method]["delta_ari_vs_expression"] for r in summary["rows"]]
        for key, estimate in [("contiguity_cell_weighted", "rho"), ("chance_adjusted", "chance_adjusted_rho")]:
            close(stats[estimate], spearmanr([r[key] for r in summary["rows"]], outcomes).statistic,
                  f"Summary Spearman: {method}/{key}")
        require(stats["permutation_order_max_abs_difference"] < 1e-12, "Partial rank order dependence")
    figure_source = read(LEDGER / "figure-source.json")
    require(figure_source["renderer_sha256"] == sha(HERE / "scripts/build_field_revision.py"),
            "Figure renderer identity")
    for filename, digest in figure_source["input_sha256"].items():
        require(sha(LEDGER / filename) == digest, f"Figure input identity: {filename}")
    for filename, digest in figure_source["generated_sha256"].items():
        require(sha(HERE / filename) == digest, f"Generated figure/table identity: {filename}")
    audit = read(LEDGER / "estimand-audit.json")
    require(len(audit["generator_rows"]) == 20, "Generator endpoint count")
    lo, hi = audit["midrank_joint_partial_range"]
    require(hi - lo < 1e-12, "Historical midrank diagnostic not invariant")
    text = (HERE / "paper.tex").read_text(encoding="utf-8")
    for token in [r"\input{ledgers/revision_20260923/field_results.tex}",
                  "Not cleared for submission", "No independent contiguity mechanism is identified",
                  "SpaceFlow, GraphST and SpaGCN", "historical", "specified"]:
        require(token in text, f"Missing manuscript scope boundary: {token}")
    for token in ["0.55/0.86/0.80", "only variable moved", "clean decoupling"]:
        require(token not in text, f"Withdrawn claim returned: {token}")
    require(not re.search(r"p=0\.06(?!\d)", text), "Withdrawn independent-binomial p returned")
    if not source_only:
        import fitz
        with fitz.open(HERE / "paper.pdf") as document:
            pdf_text = " ".join(" ".join(p.get_text().split()) for p in document)
            for token in ["Not cleared for submission", "147", "0.486", "0.120", "0.928", *OBJECTS, *NOT_RUN]:
                require(token in pdf_text, f"Current PDF missing: {token}")
            require(document.metadata["title"].startswith("Auditing spatial-prior evaluation"),
                    "Stale PDF title metadata")
    print(f"CURRENT CLAIM CONTRACT PASS: {completed} frozen seed records, 49 summary cells, "
          "21 explicit not-run records; " + ("source only" if source_only else "source and PDF"))
    print("No model refit or raw prediction rescore; historical faces untouched; scientific HOLD remains.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    verify(args.source_only)

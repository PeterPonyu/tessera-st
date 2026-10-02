#!/usr/bin/env python3
"""Export readable SI tables from committed ledgers, without training or reruns."""
from pathlib import Path
import hashlib
import json
from statistics import mean

MAN = Path(__file__).resolve().parents[1]
CAP = MAN.parent
LEDGER = MAN / "ledgers/revision_20260923"
METHODS = ["neighbor-mean", "BANKSY-style", "SpatialLeiden", "Tessera", "STAGATE", "SEDR"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources = {name: json.loads((LEDGER / name).read_text()) for name in
               ["protocol.json", "summary.json", "QA.json", "estimand-audit.json"]}
    protocol, summary, qa, audit = (sources[name] for name in
                                   ["protocol.json", "summary.json", "QA.json", "estimand-audit.json"])
    if qa["failed_runs"] or qa["completed_method_seeds"] != 147 or qa["scientific_ready"]:
        raise ValueError("SI requires the complete, uncleared 147-run field panel")
    if sha(LEDGER / "protocol.json") != qa["protocol_sha256"]:
        raise ValueError("Protocol identity differs from the committed QA")
    if summary["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("Mixed field-panel identities")
    for name, digest in audit["source_sha256"].items():
        if sha(CAP / name) != digest:
            raise ValueError(f"Estimand-audit source changed: {name}")
    rows = summary["rows"]
    if [r["dataset"] for r in rows] != list(protocol["retained_inputs"]):
        raise ValueError("Incomplete or reordered retained objects")
    lines = ["% Generated from committed ledgers; no fitting or new evidence."]
    for row in rows:
        name = row["dataset"]
        check = qa["datasets"][name]
        if (row["graph_cross_field_edges"] != 0 or not check["zero_cross_field_edges"] or
                not check["contiguity_raw_source_reproduced"] or
                check["completed_method_seeds_rescored"] != 21):
            raise ValueError(f"Unverified retained object: {name}")
        field = row["field_key"]
        label = row["label_key"]
        lines.append(f"\\expandafter\\def\\csname siidentity{name}\\endcsname{{"
                     f"{name} & \\path{{{field}}} & \\path{{{label}}} & 0 & 21 \\\\}}")
        base = row["methods"]["expression-only"]["raw_mean"]
        deltas = [row["methods"][method]["raw_mean"] - base for method in METHODS]
        lines.append(f"\\expandafter\\def\\csname sidelta{name}\\endcsname{{"
                     + name + " & " + " & ".join(f"${d:+.3f}$" for d in deltas) + r" \\}")
    rank_values = [audit["legacy_joint_partial_range"], audit["midrank_joint_partial_range"]]
    for name, values in zip(["Ordinal", "Midrank"], rank_values):
        lines.append(f"\\newcommand{{\\Rank{name}Low}}{{{values[0]:.6f}}}")
        lines.append(f"\\newcommand{{\\Rank{name}High}}{{{values[1]:.6f}}}")
    records = sorted(audit["generator_rows"], key=lambda r: (r["substrate"], r["seed"], r["level"]))
    expected = {(sub, seed, float(level)) for sub in ["S1", "S2"]
                for seed in range(1, 6) for level in [0, 1]}
    if len(records) != 20 or {(r["substrate"], r["seed"], r["level"]) for r in records} != expected:
        raise ValueError("Incomplete or duplicated seed-level generator endpoints")
    for sub in ["S1", "S2"]:
        for level in [0, 1]:
            selected = [r for r in records if r["substrate"] == sub and r["level"] == level]
            values = [mean(r[key] for r in selected) for key in
                      ["expression_moran", "small_class_fraction", "residual_sd", "random_same_label"]]
            lines.append(f"\\expandafter\\def\\csname siendpoint{sub}{level}\\endcsname{{"
                         + f"{sub} & {level} & ${values[0]:.6f}$ & {100 * values[1]:.2f} & "
                         + f"{values[2]:.6f} & {values[3]:.6f}" + r" \\}")
    seed_rows = [f"{r['substrate']} & {r['seed']} & {r['level']:.0f} & "
                 f"${r['expression_moran']:.6f}$ & {100*r['small_class_fraction']:.2f} & "
                 f"{r['residual_sd']:.6f} & {r['random_same_label']:.6f}" + r" \\" for r in records]
    lines += [r"\newcommand{\SISeedRows}{%", *seed_rows, "}"]
    output = LEDGER / "supporting_tables.tex"
    output.write_text("\n".join(lines) + "\n")
    record = {"scope": "Readable SI export of committed records; no training or resampling",
              "scientific_ready_attested": False,
              "source_sha256": {name: sha(LEDGER / name) for name in sources},
              "renderer_sha256": sha(Path(__file__)), "generated_sha256": sha(output),
              "field_objects": len(rows), "completed_method_seeds": 147,
              "generator_endpoint_rows": len(records)}
    (LEDGER / "supporting_tables-provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Exported five SI tables from seven objects and 20 generator endpoint records; HOLD retained")


if __name__ == "__main__":
    main()

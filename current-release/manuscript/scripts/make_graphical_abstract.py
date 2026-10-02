#!/usr/bin/env python3
"""Draw the current Tessera protocol-audit graphical abstract, without fitting.

Only the new Tessera_graphical_abstract.* files are written.  Tissue quantities
come from the repaired ledger and actual stored run identities.  Field tiles,
matrix cells and the permutation icon are explicitly schematic, not assay data.
The current paper, architecture, result ledgers and contracts are never edited.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

# The desktop sandbox does not permit a new user-global Matplotlib cache.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tessera-ga-mpl-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ft2font import FT2Font
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


MAN = Path(__file__).resolve().parents[1]
CAPSULE = MAN.parent
LABS = CAPSULE.parent.parent
LEDGER = MAN / "ledgers/revision_20260923"
RUNS = CAPSULE / "experiments/generated/field-identified-fixed-backend-20260923-v1"
BACKUPS = LABS / "docs/revisions/2026-09-30-figure-refinement/backups/Tessera"
NAME = "Tessera_graphical_abstract"
SIZE_IN = (6.5, 4.15)
W, H = (dimension * 72 for dimension in SIZE_IN)
SVG_NS = "http://www.w3.org/2000/svg"
PROTOCOL_ID = "field-identified-fixed-backend-20260923-v1"
EXPECTED_OBJECTS = ["DLPFC", "MERFISH", "STARmap", "MIBI", "Open-ST", "CODEX", "Zhuang"]
EXPECTED_PROGRAMS = ["expression-only", "neighbor-mean", "BANKSY-style", "SpatialLeiden", "Tessera", "STAGATE", "SEDR"]
INK = "#263544"
MUTED = "#4e5d6c"
RULE = "#cdd7de"
BLUE = "#0072b2"
TEAL = "#087f76"
RED = "#aa4535"
AMBER = "#8b662b"
FIELD_COLORS = [BLUE, TEAL, "#7b6292"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(LABS).as_posix()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def font_path(bold: bool = False) -> Path:
    pattern = "Arial:style=Bold" if bold else "Arial:style=Regular"
    result = subprocess.run(["fc-match", "-f", "%{family}\n%{file}\n", pattern],
                            check=True, capture_output=True, text=True).stdout.splitlines()
    if len(result) != 2 or "Arial" not in result[0].split(","):
        raise RuntimeError(f"Arial face unavailable; refusing font fallback: {result}")
    path = Path(result[1])
    face = FT2Font(str(path))
    if face.family_name != "Arial" or (bold and "Bold" not in face.style_name):
        raise RuntimeError(f"Unexpected Arial face: {face.family_name} / {face.style_name}")
    return path


def collect_evidence() -> dict:
    """Verify the plotted counts against the ledger and 147 existing runs."""
    protocol_path = LEDGER / "protocol.json"
    summary_path = LEDGER / "summary.json"
    qa_path = LEDGER / "QA.json"
    historical_audit_path = CAPSULE / "reproducibility/identity-audit.json"
    protocol, summary, qa, historical = map(read_json, [protocol_path, summary_path, qa_path, historical_audit_path])
    if protocol["protocol_id"] != PROTOCOL_ID or summary["protocol_id"] != PROTOCOL_ID:
        raise RuntimeError("Wrong protocol, not the retained field-identified panel")
    if protocol["methods"] != EXPECTED_PROGRAMS or protocol["seeds"] != [1, 2, 3]:
        raise RuntimeError("Program or seed identity changed; graphical abstract requires review")
    rows = summary["rows"]
    if [row["dataset"] for row in rows] != EXPECTED_OBJECTS:
        raise RuntimeError("Retained object identity changed")
    if (qa["completed_method_seeds"] != 147 or qa["expected_method_seeds"] != 147
            or qa["failed_runs"] or qa["protocol_sha256"] != sha(protocol_path)):
        raise RuntimeError("A complete bound 147-run audit ledger is required")

    bound_paths = [protocol_path, summary_path, qa_path, historical_audit_path,
                   MAN / "figs/objects/coordinate-audit.json",
                   LEDGER / "field_results.tex", LEDGER / "estimand-audit.json",
                   MAN / "paper.tex",
                   MAN / "ledgers/science_core_20260918/synth_summaries.json",
                   MAN / "ledgers/science_core_20260918/science_harden_20260918.py",
                   LABS / "docs/revisions/2026-09-30-final-science-and-packaging/SCIENTIFIC-ADJUDICATION.md"]
    run_bindings = []
    dataset_bindings = []
    protocol_digest = sha(protocol_path)
    for row in rows:
        name = row["dataset"]
        if row["graph_cross_field_edges"] != 0 or not qa["datasets"][name]["zero_cross_field_edges"]:
            raise RuntimeError(f"Current graph is not field-disjoint: {name}")
        if (qa["datasets"][name]["n_cells"] != row["n_cells"]
                or qa["datasets"][name]["n_fields"] != row["n_fields"]
                or qa["datasets"][name]["completed_method_seeds_rescored"] != 21):
            raise RuntimeError(f"Ledger / QA input identity mismatch: {name}")
        dataset_bindings.append({
            "object": name, "input_cells": row["n_cells"], "labelled_cells": row["n_labelled"],
            "fields": row["n_fields"], "annotation": row["annotation"],
            "source_path": row["path"], "source_sha256_recorded": row["source_sha256"],
            "raw_hdf5_rehashed_by_this_renderer": False,
            "selected_index_sha256_recorded": row["selected_index_sha256"],
            "graph_cross_field_edges": row["graph_cross_field_edges"],
        })
        for method in protocol["methods"]:
            ledger_method = row["methods"][method]
            if ledger_method["status"] != "complete" or ledger_method["seeds"] != protocol["seeds"]:
                raise RuntimeError(f"Incomplete current result: {name}/{method}")
            for position, seed in enumerate(protocol["seeds"]):
                record_path = RUNS / name / f"{method}-seed{seed}.json"
                predictions_path = record_path.with_suffix(".npz")
                record = read_json(record_path)
                if (record["protocol_id"] != PROTOCOL_ID or record["protocol_sha256"] != protocol_digest
                        or record["dataset"] != name or record["method"] != method or record["seed"] != seed
                        or record["status"] != "complete" or record["n_cells"] != row["n_cells"]
                        or record["source_sha256"] != row["source_sha256"]
                        or abs(record["raw_ari"] - ledger_method["raw_per_seed"][position]) > 1e-12
                        or record["predictions_sha256"] != sha(predictions_path)):
                    raise RuntimeError(f"Saved run / prediction identity mismatch: {record_path}")
                if method == "expression-only" and record["spatial_refinement"]:
                    raise RuntimeError("The true expression-only control cannot be spatially refined")
                bound_paths.extend([record_path, predictions_path])
                run_bindings.append({"record": relative(record_path), "sha256": sha(record_path),
                                     "predictions": relative(predictions_path), "predictions_sha256": sha(predictions_path)})
    if len(run_bindings) != 147:
        raise RuntimeError("The run total is not 147")

    # This is the full selected-cell historical graph diagnostic, not the
    # labelled-neighbour edge denominator in the corrected summary.json.
    historical_rows = {item["name"]: item for item in historical["spatial"]}
    pooled_rates = {}
    for name in ["MIBI", "CODEX"]:
        item = historical_rows[name]
        current = next(row for row in rows if row["dataset"] == name)
        if (item["source_sha256"] != current["source_sha256"] or item["used_n"] != current["n_cells"]
                or item["used_frames"] != current["n_fields"]):
            raise RuntimeError(f"Historical audit population does not match retained input: {name}")
        pooled_rates[name] = {
            "directed_six_neighbour_edges": item["old_edges"],
            "cross_field_edges": item["old_cross_frame_edges"],
            "percent": 100 * item["old_cross_frame_edges"] / item["old_edges"],
            "scope": "historical pooled graph; all selected cells, not labelled-neighbour subset",
        }
    mib = read_json(MAN / "figs/objects/coordinate-audit.json")
    if (mib["directed_six_neighbor_links"] != pooled_rates["MIBI"]["directed_six_neighbour_edges"]
            or mib["cross_field_links"] != pooled_rates["MIBI"]["cross_field_edges"]):
        raise RuntimeError("Historical MIBI coordinate audit disagrees")
    return {
        "protocol_id": PROTOCOL_ID,
        "counts": {"objects": len(rows), "programs": len(protocol["methods"]),
                   "seeds": len(protocol["seeds"]), "stored_runs": len(run_bindings)},
        "seeds": protocol["seeds"], "programs": protocol["methods"],
        "datasets": dataset_bindings, "historical_pooled_cross_field_links": pooled_rates,
        "source_bindings_sha256": {relative(path): sha(path) for path in bound_paths},
        "run_bindings": run_bindings,
        "synthetic_scope": {
            "operation_drawn": "position permutation, preserving expression-label pairs",
            "coupled_change": "expression-position relationships change with label contiguity",
            "scorer": "reference K; max(reference-label ARI of KMeans, tied-covariance GMM), separately per embedding",
            "fallback": "KMeans alone if GMM fails",
            "retained_inference": "specified-intervention total effect conditional on generator and scorer",
            "not_claimed": ["independent continuity mechanism", "label-free deployment", "tissue causal law"],
            "synthetic_effect_size_plotted": False,
        },
    }


def draw(evidence: dict):
    """One page in point coordinates, with the same Arial face across exports."""
    regular, bold = font_path(), font_path(True)
    plt.rcParams.update({
        "font.family": "Arial", "font.weight": "normal", "font.size": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.hashsalt": NAME,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    fig = plt.figure(figsize=SIZE_IN, dpi=144)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_axis_off()
    texts = []

    def text(x, y, value, size=8, color=INK, ha="left", panel=False, key=None):
        if panel and value not in {"A", "B", "C", "D"}:
            raise ValueError("Only panel letters can be bold")
        props = FontProperties(fname=str(bold if panel else regular), family="Arial", size=size,
                               weight="bold" if panel else "normal")
        artist = ax.text(x, y, value, fontproperties=props, color=color, ha=ha, va="center", clip_on=False)
        artist.set_gid(key or f"label-{len(texts):03d}")
        texts.append({"artist": artist, "text": value, "panel_letter": panel, "font_file": str(bold if panel else regular)})
        return artist

    def line(x0, y0, x1, y1, color=RULE, width=.65, dashed=False, key=None):
        artist, = ax.plot([x0, x1], [y0, y1], color=color, linewidth=width,
                          linestyle=(0, (2, 2)) if dashed else "-", solid_capstyle="round", clip_on=False)
        if key:
            artist.set_gid(key)
        return artist

    def rect(x, y, width, height, fill="none", edge=RULE, lw=.65, key=None):
        artist = Rectangle((x, y), width, height, facecolor=fill, edgecolor=edge, linewidth=lw, clip_on=False)
        if key:
            artist.set_gid(key)
        ax.add_patch(artist)
        return artist

    def circle(x, y, radius=1.9, color=BLUE, edge="white", lw=.3, key=None):
        artist = Circle((x, y), radius, facecolor=color, edgecolor=edge, linewidth=lw, clip_on=False)
        if key:
            artist.set_gid(key)
        ax.add_patch(artist)
        return artist

    def arrow(x0, y0, x1, y1, color=MUTED, width=.85, key=None):
        artist = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                 linewidth=width, color=color, shrinkA=0, shrinkB=0, clip_on=False)
        if key:
            artist.set_gid(key)
        ax.add_patch(artist)
        return artist

    def field_tiles(left, top, width, spacing, cross_edges, group):
        points = [(0.18, .27), (.53, .20), (.80, .33), (.29, .64), (.68, .69), (.51, .45)]
        inside = [(0, 1), (1, 2), (0, 3), (1, 5), (2, 4), (3, 5), (4, 5)]
        height = 43
        all_nodes = []
        for field in range(3):
            x = left + field * (width + spacing)
            text(x + width / 2, top - 7, f"field {field + 1}", size=7.6, color=MUTED, ha="center",
                 key=f"{group}-field-{field+1}-label")
            rect(x, top, width, height, fill="#f5f8fa", key=f"{group}-field-{field+1}-frame")
            nodes = [(x + px * width, top + py * height) for px, py in points]
            for first, second in inside:
                line(*nodes[first], *nodes[second], color=FIELD_COLORS[field], width=.55,
                     key=f"{group}-field-{field+1}-edge-{first}-{second}")
            all_nodes.append(nodes)
        if cross_edges:
            # Cross-field arrows are a schematic representation of the error,
            # not positions or edge counts sampled from an assay.
            for first_field, first_node, second_field, second_node in [(0, 2, 1, 0), (0, 4, 1, 3), (1, 2, 2, 0), (1, 4, 2, 3)]:
                line(*all_nodes[first_field][first_node], *all_nodes[second_field][second_node],
                     color=RED, width=.95, dashed=True,
                     key=f"{group}-cross-field-edge-{first_field}-{first_node}-{second_field}-{second_node}")
        for field, nodes in enumerate(all_nodes):
            for number, node in enumerate(nodes):
                circle(*node, color=FIELD_COLORS[field], key=f"{group}-field-{field+1}-node-{number}")

    text(12, 14, "Auditing spatial-prior evaluation", size=12.2, key="ga-title")
    text(12, 31, "Coordinate frames and controls define what is compared", size=8.5, color=MUTED,
         key="ga-subtitle")
    line(12, 40, W - 12, 40)
    # Baselines are identical; each top panel occupies the same visual row.
    text(12, 53, "A", size=11, panel=True, key="panel-A")
    text(29, 53, "Pooled local frames", size=9.2, key="A-title")
    text(164, 53, "B", size=11, panel=True, key="panel-B")
    text(181, 53, "Within-field evaluation", size=9.2, key="B-title")
    text(325, 53, "C", size=11, panel=True, key="panel-C")
    text(342, 53, "Traceable tissue panel", size=9.2, key="C-title")

    field_tiles(14, 81, 36, 9, True, "pooled-schematic")
    text(76, 137, "Cross-field links (schematic)", size=7.8, color=RED, ha="center", key="A-error-label")
    text(14, 157, "Historical pooled 6-NN links", size=7.8, color=MUTED, key="A-historical-denominator")
    rates = evidence["historical_pooled_cross_field_links"]
    text(14, 174, "MIBI-TOF", size=8.5, key="A-MIBI-name")
    text(139, 174, f"{rates['MIBI']['percent']:.2f}%", size=9.5, color=RED, ha="right", key="A-MIBI-value")
    text(14, 190, "CODEX", size=8.5, key="A-CODEX-name")
    text(139, 190, f"{rates['CODEX']['percent']:.2f}%", size=9.5, color=RED, ha="right", key="A-CODEX-value")
    text(14, 210, "Field identity was discarded.", size=7.8, color=MUTED, key="A-footer")
    arrow(144, 102, 159, 102, color=BLUE, key="pooled-to-within-field")

    field_tiles(166, 81, 36, 9, False, "within-field-schematic")
    text(229, 137, "Zero cross-field graph edges", size=7.8, color=TEAL, ha="center", key="B-corrected-label")
    # A shared expression matrix splits into two explicitly different arms.
    rect(168, 154, 25, 36, fill="white", key="expression-matrix-frame")
    values = [[.3, .8, .5], [.6, .2, .9], [.8, .4, .2], [.2, .6, .7]]
    for row, row_values in enumerate(values):
        for column, value in enumerate(row_values):
            tint = (1-value*.42, 1-value*.20, 1-value*.07)
            rect(171+column*6.5, 157+row*7.5, 5, 6, fill=tint, edge="none",
                 key=f"schematic-expression-{row}-{column}")
    text(180.5, 198, "features", size=7.5, color=MUTED, ha="center", key="B-matrix-label")
    line(195, 172, 201, 172, color=MUTED, width=.75)
    line(201, 163, 201, 181, color=MUTED, width=.75)
    arrow(201, 163, 209, 163, color=BLUE, key="expression-to-spatial-arm")
    arrow(201, 181, 209, 181, color=TEAL, key="expression-to-control-arm")
    text(214, 159, "Spatial programs", size=8.4, color=BLUE, key="B-spatial-programs")
    text(214, 170, "within-field inputs", size=7.7, color=MUTED, key="B-spatial-inputs")
    text(214, 185, "PCA30-KMeans", size=8.4, color=TEAL, key="B-control-program")
    text(214, 196, "no coords / refinement", size=7.5, color=MUTED, key="B-control-restriction")
    text(164, 210, "Named contrasts; no winner selection", size=7.6, color=MUTED, key="B-footer")
    arrow(303, 102, 319, 102, color=BLUE, key="within-field-to-audit-panel")
    line(314, 63, 314, 215, width=.5)

    counts = evidence["counts"]
    text(326, 74, f"{counts['objects']} objects × {counts['programs']} programs × {counts['seeds']} seeds",
         size=7.8, key="C-factorisation")
    text(326, 91, f"{counts['stored_runs']} stored runs", size=11.1, color=BLUE, key="C-stored-runs")
    text(326, 111, "Object", size=7.8, color=MUTED, key="C-object-header")
    text(415, 111, "Cells", size=7.8, color=MUTED, ha="right", key="C-cells-header")
    text(454, 111, "Fields", size=7.8, color=MUTED, ha="right", key="C-fields-header")
    line(326, 118, 454, 118, width=.6)
    for index, item in enumerate(evidence["datasets"]):
        y = 127 + index * 11
        text(326, y, item["object"], size=7.8, key=f"C-{item['object']}-name")
        text(415, y, f"{item['input_cells']:,}", size=7.8, ha="right", key=f"C-{item['object']}-cells")
        text(454, y, str(item["fields"]), size=7.8, ha="right", key=f"C-{item['object']}-fields")
    text(326, 210, "Seeds are not tissue replicates.", size=7.5, color=MUTED, key="C-footer")

    # The synthetic branch is deliberately separated from the tissue panel.
    line(12, 223, W - 12, 223)
    text(12, 235, "D", size=11, panel=True, key="panel-D")
    text(29, 235, "Separate synthetic branch", size=9.2, key="D-title")
    for left, scrambled in [(14, False), (74, True)]:
        rect(left, 249, 31, 29, fill="#faf7f0", edge="#dacdac", key=f"D-permutation-frame-{scrambled}")
        ordered = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        permuted = [1, 0, 2, 2, 1, 0, 0, 2, 1]
        for index, color_index in enumerate(permuted if scrambled else ordered):
            circle(left+7+(index%3)*8, 256+(index//3)*8, radius=1.8,
                   color=FIELD_COLORS[color_index], key=f"D-schematic-site-{scrambled}-{index}")
    arrow(49, 263, 69, 263, color=AMBER, key="D-permutation-arrow")
    text(119, 250, "Position permutation", size=8.5, key="D-operation")
    text(119, 262, "expression-label pairs fixed", size=7.8, color=MUTED, key="D-fixed-objects")
    text(119, 274, "expression-position relation changes", size=7.6, color=MUTED, key="D-coupled-change")
    arrow(276, 263, 290, 263, color=AMBER, key="D-total-effect-arrow")
    text(298, 250, "Generator-scoped total effect", size=8.5, color=AMBER, key="D-estimand")
    text(298, 262, "Reference K; max(KMeans, tied-GMM)", size=7.6, key="D-oracle-scorer")
    text(298, 274, "Not an independent contiguity mechanism", size=7.5, color=MUTED, key="D-non-claim")
    text(W/2, 290, "All tiles are schematic; oracle scoring is not label-free; tissue and simulation are not pooled.",
         size=7.5, color=MUTED, ha="center", key="ga-scope-footer")
    return fig, texts, {"regular": regular, "bold_panel_letters": bold}


def inspect_text_layout(fig, texts) -> list[dict]:
    """Mechanical bounding-box check, not a substitute for manual inspection."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for item in texts:
        artist = item["artist"]
        bounds = artist.get_window_extent(renderer)
        rectangle = [bounds.x0*72/fig.dpi, H-bounds.y1*72/fig.dpi,
                     bounds.x1*72/fig.dpi, H-bounds.y0*72/fig.dpi]
        if rectangle[0] < 4 or rectangle[1] < 4 or rectangle[2] > W-4 or rectangle[3] > H-4:
            raise RuntimeError(f"Text outside the fixed figure bounds: {item['text']}: {rectangle}")
        boxes.append({"id": artist.get_gid(), "text": item["text"], "font_family": "Arial",
                      "font_weight": "bold" if item["panel_letter"] else "normal",
                      "font_size_pt": artist.get_fontsize(), "bbox_pt_top_origin": rectangle})
    for index, first in enumerate(boxes):
        a = first["bbox_pt_top_origin"]
        for second in boxes[index+1:]:
            b = second["bbox_pt_top_origin"]
            if min(a[2], b[2]) - max(a[0], b[0]) > .2 and min(a[3], b[3]) - max(a[1], b[1]) > .2:
                raise RuntimeError(f"Overlapping text: {first['id']} / {second['id']}")
    return boxes


def save_svg(fig, path: Path, outlined: bool):
    with matplotlib.rc_context({"svg.fonttype": "path" if outlined else "none"}):
        fig.savefig(path, format="svg", metadata={"Date": None, "Title": "Tessera protocol-audit graphical abstract"})
    root = ET.parse(path).getroot()
    root.set("role", "img")
    root.set("aria-labelledby", "tessera-ga-accessible-title tessera-ga-accessible-description")
    title = ET.Element(f"{{{SVG_NS}}}title", {"id": "tessera-ga-accessible-title"})
    title.text = "Auditing spatial-prior evaluation"
    description = ET.Element(f"{{{SVG_NS}}}desc", {"id": "tessera-ga-accessible-description"})
    description.text = ("Schematic local fields illustrate erroneous pooled cross-field links, repaired within-field inputs, "
                        "a genuinely non-spatial PCA30-KMeans control and 147 existing runs. A separate synthetic "
                        "position permutation estimates a generator-scoped total effect with reference-dependent "
                        "oracle scoring, not an independent contiguity mechanism or a label-free selector.")
    root.insert(0, description)
    root.insert(0, title)
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    text_count = len(root.findall(f".//{{{SVG_NS}}}text"))
    if outlined and text_count:
        raise RuntimeError("Display SVG must be fully outlined")
    if not outlined and text_count < 60:
        raise RuntimeError("Editable SVG must retain text, not a PDF conversion")
    if root.findall(f".//{{{SVG_NS}}}image"):
        raise RuntimeError("Schematic graphical abstract must not embed invented assay images")
    return {"text_elements": text_count, "path_elements": len(root.findall(f".//{{{SVG_NS}}}path")),
            "image_elements": 0, "viewBox": root.attrib["viewBox"]}


CAPTION = """# Suggested graphical-abstract caption (not automatically inserted)

Auditing spatial-prior evaluation. (A) Schematic field tiles illustrate the error of connecting independent local coordinate frames. Historical all-selected-cell six-neighbour graph diagnostics contain 13,208/19,854 cross-field links for MIBI-TOF (66.53%) and 95,840/96,000 for CODEX (99.83%); these denominators are not the labelled-neighbour subset used for corrected contiguity. (B) The repaired protocol retains field identity throughout graph construction, neighbour features, external-method inputs, spatial refinement and metrics. The true expression-only comparator is PCA with 30 components followed by KMeans, receiving neither coordinates nor spatial refinement. (C) The retained panel comprises seven objects, seven programs (including this comparator) and seeds 1, 2 and 3, totalling 147 existing stored runs. Printed cell and field counts are the actual model inputs, not the source-population size, labelled-cell count or independent biological replicate count. Named-method contrasts use raw ARI relative to the expression-only comparator, without backend or refinement winner selection. Embedding methods use reference-conditioned KMeans; SpatialLeiden returns communities and is not guaranteed to match the reference class count. CODEX niche labels are cluster-derived proxies. SpaceFlow, GraphST and SpaGCN remain outside the completed replacement set. (D) A separate specified position permutation preserves each cell's expression-label pair but changes expression-position relationships together with label contiguity. Its total effect is conditional on the chosen generator and a reference-dependent scorer: the larger reference-label ARI from KMeans and a tied-covariance Gaussian mixture, separately for control and spatial embeddings, both using the generated reference class count; KMeans alone is retained if GMM fails. Other label-regeneration interventions also change class balance and expression spatial autocorrelation. Neither operation identifies an independent continuity mechanism. All tiles, nodes and matrix cells are schematic, not measured assay images. Tissue and synthetic estimates are not pooled; the diagram does not validate a label-free deployment selector or remove the scientific HOLD.
"""


def generate(output_dir: Path):
    evidence = collect_evidence()
    protected = [MAN / "paper.tex", MAN / "paper.pdf", MAN / "figs/fig_mechanism_arch.tex",
                 MAN / "figs/fig_mechanism_arch.pdf", CAPSULE / "reproducibility/method-contract.json",
                 LEDGER / "figure-source.json"]
    before = {relative(path): sha(path) for path in protected}
    fig, texts, fonts = draw(evidence)
    # Fail closed on every glyph of both actual Arial faces.
    for item in texts:
        cmap = FT2Font(item["font_file"]).get_charmap()
        missing = {character for character in item["text"] if not character.isspace() and ord(character) not in cmap}
        if missing:
            raise RuntimeError(f"Arial lacks glyphs: {missing}")
    boxes = inspect_text_layout(fig, texts)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.resolve() == (MAN / "figs").resolve():
        old = list(output_dir.glob(f"{NAME}.*"))
        if old:
            backup = BACKUPS / datetime.now().strftime("GA-output-%Y%m%d-%H%M%S-%f")
            backup.mkdir(parents=True, exist_ok=False)
            for path in old:
                if not path.is_file() or path.is_symlink():
                    raise RuntimeError(f"Unsafe existing graphical-abstract output: {path}")
                shutil.copy2(path, backup / path.name)
    native_path = output_dir / f"{NAME}.svg"
    outline_path = output_dir / f"{NAME}.display.svg"
    pdf_path = output_dir / f"{NAME}.pdf"
    png_path = output_dir / f"{NAME}.png"
    caption_path = output_dir / f"{NAME}.caption.md"
    native_info = save_svg(fig, native_path, outlined=False)
    outline_info = save_svg(fig, outline_path, outlined=True)
    fig.savefig(pdf_path, format="pdf", metadata={"CreationDate": None, "ModDate": None,
                                                "Title": "Auditing spatial-prior evaluation",
                                                "Author": "Zeyu Fu", "Subject": "Protocol-audit graphical abstract; schematic"})
    fig.savefig(png_path, format="png", dpi=300)
    plt.close(fig)
    caption_path.write_text(CAPTION, encoding="utf-8")
    after = {relative(path): sha(path) for path in protected}
    if after != before:
        raise RuntimeError("Protected paper / architecture / contract identity changed during GA generation")
    output_paths = [native_path, outline_path, pdf_path, png_path, caption_path]
    provenance = {
        "schema": "tessera-current-graphical-abstract-v1",
        "status": "NEW_GRAPHICAL_ABSTRACT_FOR_AUTHOR_REVIEW_NOT_INTEGRATED_IN_MANUSCRIPT",
        "renderer": relative(Path(__file__)), "renderer_sha256": sha(Path(__file__)),
        "physical_size_inches": list(SIZE_IN), "physical_size_points": [W, H], "png_dpi": 300,
        "content": "protocol audit with a distinct, generator-scoped synthetic branch",
        "schematic_elements": ["field tiles", "nodes and edges", "feature matrix", "position-permutation sites"],
        "real_assay_images_used": False, "scientific_ready_attested": False,
        "fonts": {face: {"path": str(path), "sha256": sha(path)} for face, path in fonts.items()},
        "font_license_boundary": "Local Arial is used for export; font binaries are not distributed. Native SVG requires local Arial; display SVG outlines glyphs.",
        "only_bold_text": ["A", "B", "C", "D"],
        "native_svg": native_info, "display_svg": outline_info,
        "text_layout": {"bounds_check": "PASS", "text_pair_overlap_check": "PASS",
                        "minimum_font_size_pt": min(row["font_size_pt"] for row in boxes), "texts": boxes},
        "evidence": evidence,
        "unchanged_protected_files_sha256": before,
        "output_bindings": {path.name: {"sha256": sha(path), "bytes": path.stat().st_size} for path in output_paths},
        "visual_review": "Pending actual render and manual inspection; layout checks alone are not a visual verdict.",
        "integration_recommendation": "Use as an unnumbered graphical abstract/online submission companion. Main agent may integrate explicitly after caption review; do not replace current architecture or evidence figures. No whole-paper compile or contracts edits performed.",
    }
    provenance_path = output_dir / f"{NAME}.provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "GENERATED_PENDING_VISUAL_REVIEW", "output_directory": str(output_dir.resolve()),
                      "physical_size_inches": list(SIZE_IN), "verified_stored_runs": 147,
                      "native_text_elements": native_info["text_elements"], "outlined_text_elements": outline_info["text_elements"],
                      "minimum_font_size_pt": provenance["text_layout"]["minimum_font_size_pt"],
                      "output_sha256": {path.name: sha(path) for path in output_paths + [provenance_path]}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=MAN / "figs")
    arguments = parser.parse_args()
    generate(arguments.output_dir)

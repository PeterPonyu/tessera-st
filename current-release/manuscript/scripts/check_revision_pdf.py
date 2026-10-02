#!/usr/bin/env python3
"""Machine-only PDF checks and Poppler page renders; never a visual attestation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import fitz

MAN = Path(__file__).resolve().parents[1]
OUT = MAN / "tmp/pdfs/field-revision"
QA = MAN / "attestation/current-pdf-qa.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bounds_report(path):
    pages = []
    errors = []
    with fitz.open(path) as doc:
        for index, page in enumerate(doc):
            words = page.get_text("words")
            text = page.get_text()
            box = page.rect
            outside = [w[4] for w in words if w[0] < -1 or w[1] < -1
                       or w[2] > box.width + 1 or w[3] > box.height + 1]
            if outside:
                errors.append({"page": index + 1, "outside_page_words": outside})
            if not words:
                errors.append({"page": index + 1, "error": "No extractable text"})
            if "\ufffd" in text or "??" in text:
                errors.append({"page": index + 1, "error": "Replacement glyph or unresolved placeholder"})
            sizes = [s["size"] for b in page.get_text("dict")["blocks"] if b["type"] == 0
                     for line in b["lines"] for s in line["spans"]]
            pages.append({"page": index + 1, "words": len(words),
                          "width_pt": box.width, "height_pt": box.height,
                          "outside_page_words": len(outside), "replacement_glyphs": text.count("\ufffd"),
                          "unresolved_placeholders": text.count("??"),
                          "minimum_text_size_pt": min(sizes) if sizes else None})
        return {"path": str(path.relative_to(MAN)), "sha256": sha(path),
                "pages": pages, "errors": errors, "metadata": doc.metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    patterns = [r"Overfull \\[hv]box", r"(?:Reference|Citation).+undefined",
                r"There were undefined", r"LaTeX Error", r"Missing character:",
                r"Rerun to get cross-references right"]
    problems = []
    for stem in ["paper", "SI"]:
        log = (MAN / f"{stem}.log").read_text(encoding="utf-8", errors="replace")
        problems += [f"{stem}: {line}" for line in log.splitlines()
                     if any(re.search(p, line) for p in patterns)]
        if f"Output written on {stem}.pdf" not in log:
            problems.append(f"No successful PDF output for {stem} in native LaTeX log")
    paper = bounds_report(MAN / "paper.pdf")
    supplement = bounds_report(MAN / "SI.pdf")
    source = (MAN / "paper.tex").read_text(encoding="utf-8")
    source += (MAN / "ledgers/revision_20260923/field_results.tex").read_text(encoding="utf-8")
    figures = [bounds_report(MAN / p) for p in sorted(set(
        re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+\.pdf)\}", source)))]
    if paper["errors"] or supplement["errors"] or any(f["errors"] for f in figures):
        problems.append("Text bounding boxes or empty pages require inspection")
    font_report = {"status": "not_checked", "reason": "pdffonts unavailable"}
    if shutil.which("pdffonts"):
        font_text = subprocess.run(["pdffonts", str(MAN / "paper.pdf")], check=True,
                                   capture_output=True, text=True).stdout
        font_rows = [line.split() for line in font_text.splitlines()[2:] if line.strip()]
        # Last columns are emb/sub/uni/object/ID; font names/types may contain spaces.
        unembedded = [row[0] for row in font_rows if row[-5] != "yes"]
        font_report = {"status": "passed" if not unembedded else "failed",
                       "font_entries": len(font_rows), "unembedded": unembedded}
        if unembedded:
            problems.append("Unembedded fonts")
    renders = []
    si_renders = []
    if args.render:
        renderer = shutil.which("pdftoppm")
        if not renderer:
            raise SystemExit("pdftoppm is unavailable; no dependency installed automatically")
        # Isolate shorter rebuilds from trailing pages in an older cache.
        # Earlier render directories remain recoverable.
        render_dir = OUT / paper["sha256"][:16]
        render_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([renderer, "-r", "120", "-png", str(MAN / "paper.pdf"),
                        str(render_dir / "page")], check=True)
        images = sorted(render_dir.glob("page-*.png"))
        if len(images) != len(paper["pages"]):
            raise SystemExit("Rendered page count differs; inspect render directory")
        renders = [{"path": str(p.relative_to(MAN)), "sha256": sha(p)} for p in images]
        si_dir = OUT / supplement["sha256"][:16]
        si_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([renderer, "-r", "120", "-png", str(MAN / "SI.pdf"),
                        str(si_dir / "SI")], check=True)
        si_images = sorted(si_dir.glob("SI-*.png"))
        if len(si_images) != len(supplement["pages"]):
            raise SystemExit("Rendered SI page count differs")
        si_renders = [{"path": str(p.relative_to(MAN)), "sha256": sha(p)} for p in si_images]
        (OUT / "render-manifest.json").write_text(json.dumps({
            "pdf_sha256": paper["sha256"], "renderer": renderer,
            "command": "pdftoppm -r 120 -png paper.pdf tmp/pdfs/field-revision/page",
            "pages": renders, "si_sha256": supplement["sha256"],
            "si_pages": si_renders}, indent=2) + "\n", encoding="utf-8")
    elif (OUT / "render-manifest.json").exists():
        saved = json.loads((OUT / "render-manifest.json").read_text())
        if saved["pdf_sha256"] == paper["sha256"]:
            renders = saved["pages"]
            for row in renders:
                if not (MAN / row["path"]).is_file() or sha(MAN / row["path"]) != row["sha256"]:
                    problems.append("Saved page render missing or changed")
        if saved.get("si_sha256") == supplement["sha256"]:
            si_renders = saved["si_pages"]
            for row in si_renders:
                if not (MAN / row["path"]).is_file() or sha(MAN / row["path"]) != row["sha256"]:
                    problems.append("Saved SI page render missing or changed")
    result = {"scope": "Machine text/bounds/log checks and rasterisation; not visual QA",
              "status": "passed" if not problems else "failed",
              "scientific_ready_attested": False, "visual_review_attested": False,
              "visual_review": "NOT_PERFORMED",
              "visual_review_blocker": "Machine checks do not perform visual review; inspect the separately bound page-review record.",
              "paper": paper, "supporting_information": supplement,
              "figures": figures, "latex_log_sha256": sha(MAN / "paper.log"),
              "si_log_sha256": sha(MAN / "SI.log"),
              "font_embedding": font_report,
              "latex_problems": problems, "rendered_pages": len(renders),
              "rendered_si_pages": len(si_renders)}
    QA.parent.mkdir(parents=True, exist_ok=True)
    QA.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"PDF machine QA: {result['status']}; {len(paper['pages'])} pages; "
          f"{len(figures)} included PDF figures; {len(renders)} rasterised pages; visual review NOT attested")
    if problems:
        raise SystemExit("\n".join(problems))


if __name__ == "__main__":
    main()

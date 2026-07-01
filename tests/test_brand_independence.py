"""Clean-room gate: no baseline brand name may appear in the executable source tree.

Brand names are permitted ONLY in the provenance docs (BASELINE_REFERENCES.md,
ALLOWED_BASELINE_CONTEXTS.md, CLAIM_LEDGER.md). This mirrors the parent program's
leakage-scan discipline so the project can graduate cleanly.
"""

from pathlib import Path

BANNED = [
    "stagate", "graphst", "spaceflow", "sedr", "banksy", "spagcn",
    "inspire", "harvest", "staligner", "deepst",
]

SRC = Path(__file__).resolve().parent.parent / "src"


def test_no_baseline_brand_in_source():
    offenders = []
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for brand in BANNED:
            if brand in text:
                offenders.append(f"{path.name}: {brand}")
    assert not offenders, f"baseline brand leaked into src/: {offenders}"

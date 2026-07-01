#!/usr/bin/env python3
"""Deposit the Tessera-ST code+manuscript archive to Zenodo as a DRAFT, for a citable DOI.

SAFETY / DESIGN:
  * Reads the token ONLY from the environment variable ZENODO_TOKEN (never from disk/logs).
  * Creates a DRAFT deposition and uploads the archive + metadata. It does NOT publish: publishing
    mints a permanent public DOI and is irreversible, so it is gated behind an explicit
    `--publish` flag AND a typed confirmation. The default run leaves a private draft you can review
    and edit (or delete) in the Zenodo web UI.
  * Use the sandbox first: `ZENODO_SANDBOX=1 ... ` targets sandbox.zenodo.org (throwaway DOIs).

USAGE:
  pip install requests          # if needed
  python scripts/make_release_archive.sh   # builds dist/tessera-st-release.tar.gz (or run this script's --build)
  ZENODO_TOKEN=xxxxx python scripts/deposit_zenodo.py            # create/refresh DRAFT (no publish)
  ZENODO_TOKEN=xxxxx ZENODO_SANDBOX=1 python scripts/deposit_zenodo.py   # dry-run on sandbox
  ZENODO_TOKEN=xxxxx python scripts/deposit_zenodo.py --publish  # IRREVERSIBLE: asks for typed confirmation

The token is created at https://zenodo.org/account/settings/applications/tokens/new/
with scopes: deposit:write (and deposit:actions only if you intend to publish).
"""
import argparse
import json
import os
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "dist" / "tessera-st-release.tar.gz"
ZENODO_META = ROOT / ".zenodo.json"

# What goes into the citable archive (code + manuscript + verified artifacts; NOT raw datasets).
INCLUDE = ["experiments", "src", "manuscript", "scripts", "tests",
           "README.md", "DESIGN.md", "CLAIM_LEDGER.md", "BASELINE_REFERENCES.md",
           "ALLOWED_BASELINE_CONTEXTS.md", "pyproject.toml", ".zenodo.json"]
EXCLUDE_SUFFIX = (".pyc", ".log", ".png.tmp")
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".omc", "dist", ".git"}


def build_archive():
    ARCHIVE.parent.mkdir(exist_ok=True)

    def _filter(ti):
        parts = set(Path(ti.name).parts)
        if parts & EXCLUDE_DIRS or ti.name.endswith(EXCLUDE_SUFFIX):
            return None
        return ti

    with tarfile.open(ARCHIVE, "w:gz") as tar:
        for item in INCLUDE:
            p = ROOT / item
            if p.exists():
                tar.add(p, arcname=f"tessera-st/{item}", filter=_filter)
    print(f"built {ARCHIVE} ({ARCHIVE.stat().st_size/1e6:.2f} MB)")
    return ARCHIVE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="(re)build the release archive and exit")
    ap.add_argument("--publish", action="store_true",
                    help="IRREVERSIBLE: publish the draft to mint a permanent public DOI")
    args = ap.parse_args()

    if args.build:
        build_archive()
        return

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("ZENODO_TOKEN not set. Create one at "
                 "https://zenodo.org/account/settings/applications/tokens/new/ "
                 "(scope deposit:write) and re-run: ZENODO_TOKEN=xxxxx python scripts/deposit_zenodo.py")

    try:
        import requests
    except ImportError:
        sys.exit("`pip install requests` first.")

    base = "https://sandbox.zenodo.org" if os.environ.get("ZENODO_SANDBOX") else "https://zenodo.org"
    api = f"{base}/api/deposit/depositions"
    # Token goes in the Authorization header, NOT a URL query param: requests stores the resolved URL
    # (incl. query string) on the response, so a query-param token would leak into any HTTPError message
    # / traceback. A header keeps it out of logged URLs.
    hdr = {"Authorization": f"Bearer {token}"}
    print(f"target: {base}  (set ZENODO_SANDBOX=1 to dry-run on the sandbox)")

    if not ARCHIVE.exists():
        build_archive()
    meta = {"metadata": json.load(open(ZENODO_META))}

    r = requests.post(api, headers=hdr, json={}, timeout=60)
    r.raise_for_status()
    dep = r.json()
    dep_id, bucket = dep["id"], dep["links"]["bucket"]
    print(f"created DRAFT deposition {dep_id} -> {dep['links']['html']}")

    with open(ARCHIVE, "rb") as fh:
        ru = requests.put(f"{bucket}/{ARCHIVE.name}", data=fh, headers=hdr, timeout=600)
    ru.raise_for_status()
    print(f"uploaded {ARCHIVE.name}")

    rm = requests.put(f"{api}/{dep_id}", data=json.dumps(meta),
                      headers={**hdr, "Content-Type": "application/json"}, timeout=60)
    rm.raise_for_status()
    print("metadata set from .zenodo.json")

    if not args.publish:
        print(f"\nDRAFT ready (NOT published). Review/edit/delete at: {dep['links']['html']}")
        print("To mint the permanent DOI, re-run with --publish (after completing affiliation/ORCID).")
        return

    # Irreversible publish — require typed confirmation.
    print("\n*** PUBLISH mints a PERMANENT public DOI and cannot be undone. ***")
    if input("Type 'PUBLISH' to proceed: ").strip() != "PUBLISH":
        sys.exit("aborted — draft left unpublished.")
    rp = requests.post(f"{api}/{dep_id}/actions/publish", headers=hdr, timeout=60)
    rp.raise_for_status()
    print("published. DOI:", rp.json().get("doi"))


if __name__ == "__main__":
    main()

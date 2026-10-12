#!/usr/bin/env python3
"""Hard fetch gate: auto-allow < 10 MiB; larger files need explicit Yes.

Never follows a large body. HEAD + optional Range 0-1023 smoke only.
Large GET is fail-closed unless SC_FETCH_PROXY points at ClashParty 狗狗
(7890/7891/7892), never ClashVerge TAG (7897/7898/7899).

Do not touch GNOME/system proxy. Do not export 7890 into os.environ — desktop
and Cursor agents must keep http_proxy=127.0.0.1:7897.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

MAX_AUTO = 10 * 1024 * 1024
MAX_YES = 1024 * 1024 * 1024  # --yes-large hard cap; >1 GiB needs a separate Yes
SMOKE_BYTES = 1024
UA = "singlecell-genomics-research-fetch-gate/1.0"
CTX = ssl.create_default_context()
TAG_PORTS = frozenset({7897, 7898, 7899})
CHEAP_PORTS = frozenset({7890, 7891, 7892})
DEFAULT_CHEAP_PROXY = "http://127.0.0.1:7890"
BULK_HOST_MARKERS = (
    "nih.gov",
    "nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "10xgenomics.com",
    "xethub.hf.co",
    "cdn-lfs",
)
HF_BULK_PATHS = ("/resolve/", "/blobs/")


def _req(url: str, method: str = "GET", headers: dict | None = None) -> urllib.request.Request:
    h = {"User-Agent": UA}
    if headers:
        h.update(headers)
    r = urllib.request.Request(url, headers=h, method=method)
    return r


def _proxy_port(proxy_url: str) -> int | None:
    parsed = urlparse(proxy_url)
    if parsed.port is not None:
        return parsed.port
    if parsed.scheme in {"http", "https"} and parsed.hostname:
        return 80 if parsed.scheme == "http" else 443
    return None


def _is_bulk_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    if any(marker in host for marker in BULK_HOST_MARKERS):
        return True
    if host.endswith("huggingface.co") or host.endswith(".hf.co") or host == "hf.co":
        return any(seg in path for seg in HF_BULK_PATHS)
    return False


def _requires_cheap_proxy(url: str, yes_large: bool, content_length: int | None) -> bool:
    if yes_large:
        return True
    if content_length is not None and content_length > MAX_AUTO:
        return True
    return _is_bulk_url(url)


def resolve_cheap_proxy() -> tuple[str | None, str | None]:
    """Return (proxy_url, reject_reason). reject_reason set if TAG port was requested."""
    explicit = os.environ.get("SC_FETCH_PROXY", "").strip()
    if explicit:
        port = _proxy_port(explicit)
        if port in TAG_PORTS:
            return None, (
                f"SC_FETCH_PROXY points at ClashVerge TAG port {port}; "
                f"use {DEFAULT_CHEAP_PROXY} (ClashParty 狗狗), never 7897/7898/7899"
            )
        if port not in CHEAP_PORTS:
            return None, (
                f"SC_FETCH_PROXY port {port} is not ClashParty 狗狗 "
                f"(7890/7891/7892); use {DEFAULT_CHEAP_PROXY}"
            )
        return explicit, None
    for key in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        val = os.environ.get(key, "").strip()
        if not val:
            continue
        port = _proxy_port(val)
        if port in CHEAP_PORTS:
            return val, None
    return None, None


def preflight_cheap_proxy(url: str, yes_large: bool, content_length: int | None) -> dict | None:
    """Fail-closed gate for large / bulk GET. None means proceed."""
    if not _requires_cheap_proxy(url, yes_large, content_length):
        return None
    proxy, tag_reason = resolve_cheap_proxy()
    if tag_reason:
        return {
            "ok": False,
            "blocked": True,
            "reason": tag_reason,
        }
    if not proxy:
        return {
            "ok": False,
            "blocked": True,
            "reason": (
                "fail-closed: large/bulk GET requires SC_FETCH_PROXY to ClashParty 狗狗 "
                f"({DEFAULT_CHEAP_PROXY}), not ClashVerge TAG 7897/7898/7899"
            ),
        }
    return None


def _urlopen(req: urllib.request.Request, timeout: int, proxy: str | None = None):
    # Per-request opener only. Never assign os.environ["http_proxy"] = 7890.
    if proxy:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
            urllib.request.HTTPSHandler(context=CTX),
        )
        return opener.open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout, context=CTX)


def head_info(url: str, timeout: int = 20, proxy: str | None = None) -> dict:
    info = {"url": url, "ok": False, "status": None, "content_length": None, "content_type": None, "error": None}
    try:
        with _urlopen(_req(url, "HEAD"), timeout=timeout, proxy=proxy) as resp:
            info["status"] = int(resp.status)
            info["ok"] = 200 <= resp.status < 400
            cl = resp.headers.get("Content-Length")
            info["content_length"] = int(cl) if cl and cl.isdigit() else None
            info["content_type"] = resp.headers.get("Content-Type")
            info["final_url"] = resp.geturl()
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
        # some hosts reject HEAD; try Range GET of 1 KB
        try:
            with _urlopen(
                _req(url, "GET", {"Range": f"bytes=0-{SMOKE_BYTES - 1}"}),
                timeout=timeout,
                proxy=proxy,
            ) as resp:
                raw = resp.read(SMOKE_BYTES)
                info["status"] = int(resp.status)
                info["ok"] = 200 <= resp.status < 400
                info["smoke_bytes"] = len(raw)
                cr = resp.headers.get("Content-Range")
                info["content_range"] = cr
                cl = resp.headers.get("Content-Length")
                if cr and "/" in cr:
                    total = cr.split("/")[-1]
                    if total.isdigit():
                        info["content_length"] = int(total)
                elif cl and cl.isdigit() and int(cl) > SMOKE_BYTES:
                    info["content_length"] = int(cl)
                info["error"] = None
        except Exception as exc2:  # noqa: BLE001
            info["error"] = f"{type(exc2).__name__}: {exc2}"
    return info


def _hash_prefix(path: Path) -> tuple[object, int]:
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 16)
            if not chunk:
                break
            n += len(chunk)
            h.update(chunk)
    return h, n


def _expected_from_response(resp, resume_from: int, head_cl: int | None) -> int | None:
    cr = resp.headers.get("Content-Range")
    if cr and "/" in cr:
        total = cr.split("/")[-1]
        if total.isdigit():
            return int(total)
    cl = resp.headers.get("Content-Length")
    if cl and str(cl).isdigit():
        got = int(cl)
        if int(resp.status) == 206:
            return resume_from + got
        return got
    return head_cl


def fetch(url: str, dest: Path, yes_large: bool = False, timeout: int = 60) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Bulk hosts (NCBI/HF LFS/10x) must not emit even a HEAD on ClashVerge TUN/TAG.
    gate = preflight_cheap_proxy(url, yes_large, None)
    if gate is not None:
        return gate
    proxy, _ = resolve_cheap_proxy()
    meta = head_info(url, timeout=min(timeout, 20), proxy=proxy)
    cl = meta.get("content_length")
    if cl is not None and cl > MAX_YES:
        return {
            "ok": False,
            "blocked": True,
            "reason": f"content_length {cl} > {MAX_YES}; need explicit >1GiB Yes",
            "head": meta,
        }
    if cl is not None and cl > MAX_AUTO and not yes_large:
        return {
            "ok": False,
            "blocked": True,
            "reason": f"content_length {cl} > {MAX_AUTO}; need Yes",
            "head": meta,
        }
    gate = preflight_cheap_proxy(url, yes_large, cl if isinstance(cl, int) else None)
    if gate is not None:
        gate["head"] = meta
        return gate
    tmp = dest.with_suffix(dest.suffix + ".part")
    resume_from = tmp.stat().st_size if tmp.is_file() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else None
    h = hashlib.sha256()
    n = 0
    expected = cl
    try:
        with _urlopen(_req(url, "GET", headers), timeout=timeout, proxy=proxy) as resp:
            status = int(resp.status)
            if resume_from and status == 200:
                resume_from = 0
                tmp.unlink(missing_ok=True)
            expected = _expected_from_response(resp, resume_from, cl)
            if expected is not None and expected > MAX_YES:
                return {
                    "ok": False,
                    "blocked": True,
                    "reason": f"content_length {expected} > {MAX_YES}; need explicit >1GiB Yes",
                    "head": meta,
                }
            if expected is not None and expected > MAX_AUTO and not yes_large:
                return {
                    "ok": False,
                    "blocked": True,
                    "reason": f"content_length {expected} > {MAX_AUTO}; need Yes",
                    "head": meta,
                }
            cap = MAX_YES if yes_large else MAX_AUTO
            if resume_from:
                h, n = _hash_prefix(tmp)
            mode = "ab" if resume_from else "wb"
            with tmp.open(mode) as f:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    n += len(chunk)
                    if cap is not None and n > cap:
                        tmp.unlink(missing_ok=True)
                        return {
                            "ok": False,
                            "blocked": True,
                            "reason": f"stream exceeded {cap} bytes; need Yes",
                            "bytes": n,
                            "head": meta,
                        }
                    f.write(chunk)
                    h.update(chunk)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "blocked": False,
            "error": f"{type(exc).__name__}: {exc}",
            "bytes": n,
            "head": meta,
        }
    if expected is not None and n != expected:
        return {
            "ok": False,
            "truncated": True,
            "reason": f"got {n} bytes, expected {expected}",
            "bytes": n,
            "head": meta,
        }
    tmp.replace(dest)
    sidecar = {
        "url": url,
        "path": str(dest),
        "bytes": n,
        "sha256": h.hexdigest(),
        "download": True,
        "under_10mb": n <= MAX_AUTO,
    }
    dest.with_suffix(dest.suffix + ".sha256.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    sidecar["ok"] = True
    sidecar["head"] = meta
    return sidecar


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("dest", nargs="?", default="-")
    p.add_argument("--yes-large", action="store_true")
    p.add_argument("--smoke-only", action="store_true")
    args = p.parse_args()
    if args.smoke_only:
        gate = preflight_cheap_proxy(args.url, False, None)
        if gate is not None:
            print(json.dumps(gate, indent=2))
            sys.exit(2)
        proxy, _ = resolve_cheap_proxy()
        print(json.dumps(head_info(args.url, proxy=proxy), indent=2))
        return
    if args.dest == "-":
        p.error("dest is required unless --smoke-only")
    rec = fetch(args.url, Path(args.dest), yes_large=args.yes_large)
    print(json.dumps(rec, indent=2))
    if not rec.get("ok"):
        sys.exit(2)


if __name__ == "__main__":
    main()

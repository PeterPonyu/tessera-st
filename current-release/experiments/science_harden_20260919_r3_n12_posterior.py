#!/usr/bin/env python3
"""Tessera R3: recompute n=12 posterior from the SpaceFlow-updated ledger.

Off-print. Does not rebuild the official 15-page PDF. n=11 remains the
headline. Observational and synthetic arms remain non-combinable.
Family-7 Bonferroni p=0.082 remains not significant.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

CAPSULE = Path("/home/zeyufu/Desktop/labs/capsules/tessera-option-hold")
LEDGER = CAPSULE / "manuscript" / "ledgers" / "science_core_20260918_r2"
N12 = LEDGER / "n12_zhuang.json"
POST = LEDGER / "n12_posterior.json"
POST_BAK = LEDGER / "n12_posterior-pre-spaceflow-20260919.json"
LODO = LEDGER / "n12_lodo.json"
OFFICIAL = CAPSULE / "manuscript" / "submission_flat" / "paper.pdf"
EXPECTED_OFFICIAL = "2262380027b5683ce46a8e64463e0f680d444556428689bd5e1c934f2dcdd8cf"
EXPECTED_N12 = "15c321eb344a010aedeac6e91cc453ffae9d0e5a4907c098e315939a07af8db3"
SEED = 20260918
GRID = 4001
N_BOOT = 5000
FAMILY7_P = 0.082
N11_RHO = 0.7243754556493464


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fisher_z_posterior(rho_hat: float, n: int, prior: str) -> dict:
    rho_hat = float(np.clip(rho_hat, -0.999999, 0.999999))
    z_obs = np.arctanh(rho_hat)
    se = 1.0 / np.sqrt(n - 3)
    grid = np.linspace(-0.999, 0.999, GRID)
    z = np.arctanh(grid)
    loglik = -0.5 * ((z_obs - z) / se) ** 2
    if prior == "uniform":
        logprior = np.zeros_like(grid)
    elif prior == "skeptical":
        logprior = -0.5 * (grid / 0.5) ** 2
    else:
        raise ValueError(prior)
    logpost = loglik + logprior
    logpost -= logpost.max()
    post = np.exp(logpost)
    post /= post.sum()
    cdf = np.cumsum(post)

    def q(p):
        return float(grid[int(np.searchsorted(cdf, p).clip(0, len(grid) - 1))])

    return {
        "rho_hat": rho_hat,
        "n": n,
        "prior": prior,
        "likelihood": "Fisher-z Normal(atanh(ρ), 1/sqrt(n-3))",
        "z_obs": float(z_obs),
        "se_z": float(se),
        "posterior_mean": float((grid * post).sum()),
        "posterior_median": q(0.5),
        "credible95": [q(0.025), q(0.975)],
        "P_rho_gt_0": float(post[grid > 0].sum()),
        "P_rho_gt_0.5": float(post[grid > 0.5].sum()),
        "P_rho_gt_0.72": float(post[grid > 0.72].sum()),
    }


def bootstrap_posterior_mass(x, y, rng, b=N_BOOT) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    boots = []
    for _ in range(b):
        i = rng.choice(n, size=n, replace=True)
        if len(np.unique(x[i])) < 3 or len(np.unique(y[i])) < 3:
            continue
        boots.append(float(spearmanr(x[i], y[i]).correlation))
    boots = np.asarray(boots)
    return {
        "method": "Rubin Bayesian bootstrap (multinomial resample), B=%d" % b,
        "n_boot": int(len(boots)),
        "mean": float(boots.mean()),
        "median": float(np.median(boots)),
        "credible95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "P_rho_gt_0": float((boots > 0).mean()),
    }


def main() -> None:
    official_hash = sha256(OFFICIAL)
    if official_hash != EXPECTED_OFFICIAL:
        raise SystemExit(f"official Tessera hash moved: {official_hash}")
    n12_hash = sha256(N12)
    if n12_hash != EXPECTED_N12:
        raise SystemExit(f"n12_zhuang.json hash moved: {n12_hash}")

    if POST.is_file() and not POST_BAK.is_file():
        shutil.copy2(POST, POST_BAK)

    data = json.loads(N12.read_text())
    points = data["points_n12"]
    if len(points) != 12:
        raise SystemExit(f"expected 12 points, got {len(points)}")
    cont = np.array([p["contiguity"] for p in points], float)
    adv = np.array([p["advantage"] for p in points], float)
    names = [p["platform"] for p in points]
    rho, pval = spearmanr(cont, adv)
    rho = float(rho)
    pval = float(pval)
    expected = float(data["n12"]["rho"])
    if abs(rho - expected) > 1e-12:
        raise SystemExit(f"recomputed rho {rho} != ledger {expected}")

    rng = np.random.default_rng(SEED)
    payload = {
        "role": "n=12 posterior after 2026-09-19 SpaceFlow rerun",
        "off_print": True,
        "headline_remains_n11": True,
        "n11_rho_unchanged": N11_RHO,
        "family7_bonferroni_p": FAMILY7_P,
        "family7_significant": False,
        "observational_synthetic_combinable": False,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_ledger": str(N12),
        "source_sha256": n12_hash,
        "pre_rerun_posterior": str(POST_BAK),
        "rho_hat": rho,
        "p_two_sided": pval,
        "n": 12,
        "seed": SEED,
        "grid": GRID,
        "fisher_z_uniform": fisher_z_posterior(rho, 12, "uniform"),
        "fisher_z_skeptical": fisher_z_posterior(rho, 12, "skeptical"),
        "bayes_bootstrap": bootstrap_posterior_mass(cont, adv, rng),
        "ready": "NO",
        "official_pdf_sha256": official_hash,
        "official_pdf_unchanged": True,
    }
    POST.write_text(json.dumps(payload, indent=2) + "\n")

    lodo = []
    for i, name in enumerate(names):
        mask = np.ones(12, dtype=bool)
        mask[i] = False
        r, pv = spearmanr(cont[mask], adv[mask])
        lodo.append(
            {
                "dropped": name,
                "n": 11,
                "rho": float(r),
                "p_two_sided": float(pv),
                "delta_vs_n12": float(r - rho),
            }
        )
    lodo_sorted = sorted(lodo, key=lambda r: r["rho"])
    lodo_payload = {
        "role": "leave-one-dataset-out on the SpaceFlow-updated n=12 points",
        "off_print": True,
        "headline_remains_n11": True,
        "observational_synthetic_combinable": False,
        "n12_rho": rho,
        "n12_p": pval,
        "rows": lodo,
        "min_rho": lodo_sorted[0],
        "max_rho": lodo_sorted[-1],
        "ready": "NO",
    }
    LODO.write_text(json.dumps(lodo_payload, indent=2) + "\n")
    print("n12 rho", rho, "p", pval, flush=True)
    print("uniform P(rho>0)", payload["fisher_z_uniform"]["P_rho_gt_0"], flush=True)
    print("wrote", POST, LODO, flush=True)
    print("official hash unchanged", official_hash[:16], flush=True)


if __name__ == "__main__":
    main()

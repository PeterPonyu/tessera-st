"""Keep repaired benchmark candidates separate from historical oracle results."""
from pathlib import Path
import hashlib
import json
import numpy as np

PROTOCOL_ID = "identity-and-control-audit-candidate-1"
GENERATED = Path(__file__).resolve().parent / "generated" / PROTOCOL_ID


def output_path(name):
    GENERATED.mkdir(parents=True, exist_ok=True)
    return str(GENERATED / name)


def refinement_candidates(method, labels, coords, k=6, *, field_ids=None):
    """The expression-only negative control must never receive spatial labels."""
    if not method:
        raise ValueError("method identity is required before refinement")
    yield "raw", labels
    if method != "floor:nonspatial":
        from tessera_st.eval.refine import refine_labels
        yield "refine", refine_labels(labels, coords, k=k, field_ids=field_ids)


def record_seed(per, method, seed, score):
    if not np.isfinite(score):
        raise ValueError(f"non-finite score: {method}, seed={seed}")
    row = per.setdefault(method, {})
    if seed in row:
        raise ValueError(f"duplicate seed: {method}, seed={seed}")
    row[seed] = float(score)


def complete_seed_summary(per, seeds):
    """Never align list positions after a missing seed or average partial runs."""
    expected = set(seeds)
    if not seeds or len(expected) != len(seeds):
        raise ValueError("a nonempty, unique set of seed identities is required")
    means, stds, missing = {}, {}, {}
    for method, scores in per.items():
        if set(scores) - expected:
            raise ValueError(f"unexpected seed identities for {method}")
        if set(scores) != expected:
            missing[method] = sorted(expected-set(scores))
            continue
        values = [scores[s] for s in seeds]
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite cached seed scores for {method}")
        means[method], stds[method] = float(np.mean(values)), float(np.std(values))
    if "floor:nonspatial" not in means:
        raise ValueError("complete expression-only control required")
    return means, stds, missing


def fingerprint(paths):
    """Cache identity includes content, not a human-assigned row label."""
    base = Path(__file__).resolve().parent
    paths = list(paths) + [base/'protocol_guard.py', base/'coordinate_guard.py']
    paths += [base.parent/'src/tessera_st'/p for p in
              ['spatial.py', 'train.py', 'model/graph.py', 'eval/refine.py']]
    hashes = []
    for value in paths:
        path = Path(value)
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1 << 20), b''):
                digest.update(block)
        hashes.append((path.name, digest.hexdigest()))
    return hashlib.sha256(json.dumps(sorted(hashes)).encode()).hexdigest()


def require_cached_input(row, paths):
    require_candidate_rows([row])
    if row['input_fingerprint'] != fingerprint(paths):
        raise ValueError("cached input or implementation changed; use a fresh candidate output")


def require_candidate_rows(rows):
    for row in rows:
        if row.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("historical/unversioned row cannot enter repaired candidate panel")
        if not row.get("input_fingerprint"):
            raise ValueError("input identity missing from candidate row")
        if not row.get("per_seed"):
            raise ValueError("seed identities missing from candidate row")
        seeds = row.get("seeds")
        if not seeds:
            raise ValueError("expected seed identities missing from candidate row")
        per = {method: {int(seed): value for seed, value in scores.items()}
               for method, scores in row["per_seed"].items()}
        means, _, _ = complete_seed_summary(per, seeds)
        stored = row.get("means", {})
        if set(stored) != set(means) or any(abs(stored[m]-means[m]) > 5.1e-5 for m in means):
            raise ValueError("cached means do not match complete seed identities")


def require_complete_platforms(rows, expected):
    got = [row['platform'] for row in rows]
    if len(set(got)) != len(got) or set(got) != set(expected):
        raise ValueError(f"incomplete/duplicate platform panel: missing={sorted(set(expected)-set(got))}")
    require_candidate_rows(rows)


def preflight_coordinate_frames(platforms):
    """Fail before expensive inference when a legacy external adapter is unsafe."""
    import anndata as ad
    from coordinate_guard import require_single_coordinate_frame
    failures = []
    for name, (path, _) in platforms.items():
        data = None
        try:
            data = ad.read_h5ad(path, backed='r')
            require_single_coordinate_frame(data, name)
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            failures.append(f"{name}: {exc}")
        finally:
            if data is not None:
                data.file.close()
    if failures:
        raise RuntimeError("Method preflight failed; no results written:\n"+'\n'.join(failures))

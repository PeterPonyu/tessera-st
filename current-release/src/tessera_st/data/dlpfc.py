"""DLPFC / spatialLIBD adapter boundary (real-data path).

This is the *only* place real spatial transcriptomics enters the package. It returns the
same plain arrays the synthetic generator does, so the model and eval code never branch on
data origin. Only `anndata` is needed (lazily imported, opt-in via `pip install -e '.[real-data]'`);
preprocessing is numpy+sklearn so we avoid the scanpy/numba/numpy version pins.

Ground truth: the manual cortical-layer annotation shipped with the 12 DLPFC Visium sections
(the standard spatial-domain benchmark). No annotation is invented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SpatialSlide:
    coords: np.ndarray
    expr: np.ndarray
    labels: np.ndarray  # -1 where the section has no manual annotation
    section_id: str
    # Acquisition/frame identity for every row.  A constant vector is valid for a
    # single section; multiple values force field-disjoint spatial graphs.
    field_ids: np.ndarray | None = None
    # marker-prior inputs: per-layer marker z-scores (None if no marker_dict given / no genes).
    # Less-circular than GT-based metrics, but conditioned on the marker list — not ground truth.
    layer_marker_scores: dict[str, np.ndarray] | None = None


def load_h5ad(
    path: str,
    label_key: str = "layer_guess",
    n_top_genes: int = 3000,
    n_pcs: int = 50,
    marker_dict: dict[str, list[str]] | None = None,
) -> SpatialSlide:
    """Load one DLPFC section from an ``.h5ad`` into plain arrays.

    Parameters mirror the conventions every graph spatial-domain method uses (HVG selection
    then PCA), so the input distribution matches the comparison world. The label column is
    read as-is; spots with missing layer calls become ``-1`` and are excluded from metrics.
    """
    import anndata as ad
    from sklearn.decomposition import PCA

    adata = ad.read_h5ad(path)
    obs: Any = adata.obs  # pandas DataFrame at runtime; anndata stubs are imprecise
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    else:
        coords = np.asarray(obs[["array_row", "array_col"]].values, dtype=np.float64)

    # numpy/sklearn preprocessing (no scanpy/numba dependency), matching the field-standard
    # pipeline: CP10k normalise -> log1p -> top-variance HVG -> z-score+clip -> PCA.
    raw_X: Any = adata.X
    to_dense = getattr(raw_X, "toarray", None)
    arr = to_dense() if callable(to_dense) else raw_X
    X = np.asarray(arr, dtype=np.float64)
    lib = X.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    X = np.log1p(X / lib * 1e4)

    # marker-prior view: pull marker genes from the FULL log-norm matrix (markers are often not
    # HVGs) before HVG truncation. This lowers circularity (not from this data's clustering) but is
    # conditioned on the marker list — swap the list per tissue/platform; it is not ground truth.
    layer_marker_scores = None
    if marker_dict is not None:
        from tessera_st.eval.markers import compute_layer_scores

        gene_names = list(map(str, adata.var_names))
        wanted = {g for genes in marker_dict.values() for g in genes}
        present = [g for g in gene_names if g in wanted]
        if present:
            idx = [gene_names.index(g) for g in present]
            layer_marker_scores = compute_layer_scores(X[:, idx], present, marker_dict)

    k = min(n_top_genes, X.shape[1])
    hvg = np.argsort(X.var(axis=0))[::-1][:k]
    X = X[:, hvg]
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    X = np.clip((X - mu) / sd, -10.0, 10.0)
    n_comp = min(n_pcs, X.shape[1] - 1, X.shape[0] - 1)
    expr = PCA(n_components=n_comp, random_state=0).fit_transform(X).astype(np.float32)

    raw_labels = obs[label_key].values if label_key in obs.columns else None
    labels = _encode_labels(raw_labels, n=expr.shape[0])
    field_ids = _infer_field_ids(obs, expr.shape[0])
    section_key = next((key for key in ("sample_id", "section_id", "library_id")
                        if key in obs.columns), None)
    section = str(obs[section_key].iloc[0]) if section_key and adata.n_obs else "unknown"
    return SpatialSlide(coords=coords, expr=expr, labels=labels, section_id=section,
                        field_ids=field_ids,
                        layer_marker_scores=layer_marker_scores)


_FRAME_KEYS = ("library_id", "fov", "FOV", "slide_id", "sample_id", "section_id",
               "sample_Xtile_Ytile")


def _infer_field_ids(obs: Any, n: int) -> np.ndarray | None:
    """Return explicit acquisition-frame ids, or ``None`` when the source has none.

    The adapter never guesses a frame from coordinates.  If a known identity column is
    present, every value is retained and downstream graph construction validates missing
    values before building neighbours.  This keeps pooled local coordinate systems from
    silently becoming one spatial field.
    """
    for key in _FRAME_KEYS:
        if key not in obs.columns:
            continue
        values = np.asarray(obs[key].to_numpy(dtype=object), dtype=object)
        if len(values) != n:
            raise ValueError(f"field identity column {key!r} does not match row count")
        return values
    return None


def _encode_labels(raw, n: int) -> np.ndarray:
    if raw is None:
        return np.full(n, -1, dtype=np.int64)
    out = np.full(n, -1, dtype=np.int64)
    uniq = {v: i for i, v in enumerate(sorted({x for x in raw if _is_valid(x)}))}
    for i, v in enumerate(raw):
        if _is_valid(v):
            out[i] = uniq[v]
    return out


def _is_valid(v) -> bool:
    if v is None:
        return False
    s = str(v).strip().lower()
    return s not in {"", "nan", "none", "na", "unknown"}

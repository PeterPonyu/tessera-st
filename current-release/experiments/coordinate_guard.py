"""Prevent reproduction from silently pooling independent spatial frames."""
def require_single_coordinate_frame(adata, source):
    for key in ('library_id', 'fov', 'FOV', 'slide_id', 'sample_id', 'section_id', 'sample_Xtile_Ytile'):
        if key in adata.obs and (adata.obs[key].isna().any() or adata.obs[key].nunique(dropna=True)>1):
            raise RuntimeError(
                f"SPATIAL COORDINATE HOLD: {source} has multiple {key} values. "
                "This archived algorithm assumes a single coordinate frame. "
                "Use field-disjoint neighbour graphs throughout the method and "
                "refinement paths before rerunning; do not pool local coordinates."
            )

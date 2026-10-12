"""Data adapters. Both return plain arrays so the model never branches on data origin."""

from tessera_st.data.synthetic import SyntheticSlide, make_tessellation

__all__ = ["SyntheticSlide", "make_tessellation"]

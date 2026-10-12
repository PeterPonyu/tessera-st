"""Tessera-ST — boundary-aware, multi-scale spatial-domain detection.

Thesis: a tissue is a tessellation; domains are tiles, boundaries are first-class. Tessera
learns *where to stop smoothing* instead of smoothing the spatial graph uniformly. See
DESIGN.md for the architecture and the four ablatable components.
"""

from tessera_st.config import AblationConfig, ModelConfig, TrainConfig

__version__ = "0.0.0.dev0"
__all__ = ["AblationConfig", "ModelConfig", "TrainConfig"]

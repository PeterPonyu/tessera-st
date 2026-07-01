"""Configuration for Tessera-ST.

The four ablatable components are toggles on :class:`AblationConfig`. Flipping any one
to ``False`` degrades that component to its documented baseline behaviour (see DESIGN.md),
which is exactly what the ablation runner sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class AblationConfig:
    """The paper's spine: four independent switches, one full model.

    Attributes
    ----------
    edge_gating:
        Component 1. Learn per-edge gates ``g_ij`` that attenuate message passing across
        putative boundaries. OFF -> ``g_ij == 1`` everywhere (plain mean aggregation, the
        classic over-smoothing baseline).
    multi_scale:
        Component 2. Fuse per-layer embeddings via attention over depths. OFF -> use the
        last layer only.
    boundary_contrastive:
        Component 3. Add the InfoNCE boundary-contrastive term. OFF -> reconstruction only.
    calibrated_uncertainty:
        Component 4. Temperature + split-conformal calibration of boundary confidence.
        OFF -> raw softmax entropy.
    """

    edge_gating: bool = True
    multi_scale: bool = True
    boundary_contrastive: bool = True
    calibrated_uncertainty: bool = True

    @property
    def name(self) -> str:
        if all(self.as_tuple):
            return "full"
        if not any(self.as_tuple):
            return "backbone"
        off = [k for k, v in self.as_dict().items() if not v]
        if len(off) == 1:
            return f"no_{off[0]}"
        return "+".join(k for k, v in self.as_dict().items() if v) or "backbone"

    @property
    def as_tuple(self) -> tuple[bool, bool, bool, bool]:
        return (
            self.edge_gating,
            self.multi_scale,
            self.boundary_contrastive,
            self.calibrated_uncertainty,
        )

    def as_dict(self) -> dict[str, bool]:
        return {
            "edge_gating": self.edge_gating,
            "multi_scale": self.multi_scale,
            "boundary_contrastive": self.boundary_contrastive,
            "calibrated_uncertainty": self.calibrated_uncertainty,
        }


@dataclass(frozen=True)
class ModelConfig:
    in_dim: int
    hidden_dim: int = 64
    embed_dim: int = 32
    n_layers: int = 3
    gate_hidden: int = 16
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    recon_weight: float = 1.0
    contrastive_weight: float = 0.5
    contrastive_temp: float = 0.2
    n_negatives: int = 10
    seed: int = 17
    device: str = "auto"  # "auto" | "cpu" | "cuda"


def curated_ablation_grid() -> list[AblationConfig]:
    """Full model, four leave-one-out runs, and the bare backbone.

    This is the curated sweep (6 runs) rather than the full 2^4=16 grid: it isolates the
    marginal contribution of each component while keeping the methods table readable.
    """
    full = AblationConfig()
    grid = [full]
    for field in ("edge_gating", "multi_scale", "boundary_contrastive", "calibrated_uncertainty"):
        grid.append(replace(full, **{field: False}))
    grid.append(AblationConfig(False, False, False, False))  # backbone
    return grid

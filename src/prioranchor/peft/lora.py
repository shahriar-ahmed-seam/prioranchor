"""Parameter-efficient Low-Rank Adaptation (LoRA) layers, containers, and weight operators."""

from __future__ import annotations

import math
from typing import Any, Dict, List

from pydantic import BaseModel

from prioranchor.core.models import AdapterType, LoRAConfig


class LoRAWeights(BaseModel):
    """Serialized weights for a single adapted projection layer."""
    layer_name: str
    in_features: int
    out_features: int
    rank: int
    alpha: float
    weight_a: List[List[float]]
    weight_b: List[List[float]]
    merged: bool = False


class LoRALinear:
    """Low-Rank Adaptation linear layer wrapping base weights: y = W_0 x + (alpha/r) B(Ax)."""

    def __init__(
        self,
        name: str,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        use_dora: bool = False,
    ):
        self.name = name
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / float(rank)
        self.dropout = dropout
        self.use_dora = use_dora
        self.merged = False

        # Base weight W_0 initialized (out_features x in_features)
        # Using Xavier uniform-style values for base projection
        std = math.sqrt(2.0 / (in_features + out_features))
        self.base_weight: List[List[float]] = [
            [(math.sin((i + 1) * (j + 1)) * std) for j in range(in_features)]
            for i in range(out_features)
        ]

        # LoRA A matrix (rank x in_features) initialized with Kaiming uniform
        kaiming_std = 1.0 / math.sqrt(in_features)
        self.lora_a: List[List[float]] = [
            [(math.cos((i + 1) * (j + 1)) * kaiming_std) for j in range(in_features)]
            for i in range(rank)
        ]

        # LoRA B matrix (out_features x rank) initialized to zero so delta W is initially 0
        self.lora_b: List[List[float]] = [
            [0.0 for _ in range(rank)]
            for _ in range(out_features)
        ]

    def forward(self, x: List[float]) -> List[float]:
        """Compute forward pass: y = W x (+ delta W x if not merged)."""
        if len(x) != self.in_features:
            msg = f"Input dimension {len(x)} does not match in_features {self.in_features}"
            raise ValueError(msg)

        if self.merged:
            # Base weight already contains LoRA delta
            return [
                sum(w_ij * x_j for w_ij, x_j in zip(row, x))
                for row in self.base_weight
            ]

        # Base projection: y_base = W_0 x
        y_base = [
            sum(w_ij * x_j for w_ij, x_j in zip(row, x))
            for row in self.base_weight
        ]

        # LoRA path: h = A x (length: rank)
        h = [
            sum(a_ij * x_j for a_ij, x_j in zip(row, x))
            for row in self.lora_a
        ]

        # Delta: B h (length: out_features)
        delta = [
            sum(b_ij * h_j for b_ij, h_j in zip(row, h)) * self.scaling
            for row in self.lora_b
        ]

        # Composite output
        return [yb + d for yb, d in zip(y_base, delta)]

    def compute_delta_weight(self) -> List[List[float]]:
        """Compute Delta W = (alpha/r) * B @ A matrix (out_features x in_features)."""
        delta: List[List[float]] = []
        for i in range(self.out_features):
            row: List[float] = []
            for j in range(self.in_features):
                cell_sum = sum(self.lora_b[i][r] * self.lora_a[r][j] for r in range(self.rank))
                row.append(cell_sum * self.scaling)
            delta.append(row)
        return delta

    def merge_weights(self) -> None:
        """Fold LoRA weights directly into the base weight: W_0 += Delta W."""
        if self.merged:
            return

        delta = self.compute_delta_weight()
        for i in range(self.out_features):
            for j in range(self.in_features):
                self.base_weight[i][j] += delta[i][j]
        self.merged = True

    def unmerge_weights(self) -> None:
        """Unfold LoRA weights back out of base weight: W_0 -= Delta W."""
        if not self.merged:
            return

        delta = self.compute_delta_weight()
        for i in range(self.out_features):
            for j in range(self.in_features):
                self.base_weight[i][j] -= delta[i][j]
        self.merged = False

    def export_weights(self) -> LoRAWeights:
        """Serialize LoRA state for safetensors/adapter persistence."""
        return LoRAWeights(
            layer_name=self.name,
            in_features=self.in_features,
            out_features=self.out_features,
            rank=self.rank,
            alpha=self.alpha,
            weight_a=self.lora_a,
            weight_b=self.lora_b,
            merged=self.merged,
        )

    def load_weights(self, weights: LoRAWeights) -> None:
        """Load external LoRA weights."""
        if weights.rank != self.rank:
            raise ValueError(f"Rank mismatch: expected {self.rank}, got {weights.rank}")
        self.lora_a = weights.weight_a
        self.lora_b = weights.weight_b
        self.alpha = weights.alpha
        self.scaling = weights.alpha / float(weights.rank)


class LoRAContainer:
    """Container managing multiple LoRA layers across UNet cross-attention and text encoders."""

    def __init__(self, config: LoRAConfig):
        self.config = config
        self.layers: Dict[str, LoRALinear] = {}

    def inject_layer(
        self,
        name: str,
        in_features: int,
        out_features: int,
    ) -> LoRALinear:
        """Instantiate and register an adapted LoRA linear layer."""
        layer = LoRALinear(
            name=name,
            in_features=in_features,
            out_features=out_features,
            rank=self.config.rank,
            alpha=self.config.alpha,
            dropout=self.config.dropout,
            use_dora=(self.config.adapter_type == AdapterType.DORA),
        )
        self.layers[name] = layer
        return layer

    def get_trainable_param_count(self) -> int:
        """Compute the total number of trainable LoRA parameters."""
        total = 0
        for layer in self.layers.values():
            # A has rank x in_features, B has out_features x rank
            total += (layer.rank * layer.in_features) + (layer.out_features * layer.rank)
        return total

    def merge_all(self) -> None:
        """Fold all active LoRA adapters into their respective base weights."""
        for layer in self.layers.values():
            layer.merge_weights()

    def unmerge_all(self) -> None:
        """Unfold all active LoRA adapters out of base weights."""
        for layer in self.layers.values():
            layer.unmerge_weights()

    def state_dict(self) -> Dict[str, Any]:
        """Export serializable dictionary of all adapters."""
        return {name: layer.export_weights().model_dump() for name, layer in self.layers.items()}

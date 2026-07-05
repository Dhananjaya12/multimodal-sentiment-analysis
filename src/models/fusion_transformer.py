"""
Transformer-based feature fusion for multimodal sentiment analysis.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn


SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]


@dataclass
class ModalityFeature:
    """Container for one modality embedding and optional prediction metadata."""

    name: str
    embedding: torch.Tensor
    confidence: float = 0.0
    sentiment: Optional[str] = None


class CrossModalFusionTransformer(nn.Module):
    """
    Projects modality embeddings into a shared space and fuses them with attention.

    Inputs are modality-specific vectors shaped [batch, feature_dim]. The module
    creates one token per available modality, adds learned modality embeddings,
    runs a Transformer encoder, and classifies from a pooled multimodal token.
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 3,
    ):
        super().__init__()
        self.modalities = list(input_dims.keys())
        self.hidden_dim = hidden_dim

        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for name, dim in input_dims.items()
            }
        )
        self.modality_tokens = nn.ParameterDict(
            {
                name: nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
                for name in input_dims
            }
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tokens: List[torch.Tensor] = []
        present_modalities: List[str] = []

        for name in self.modalities:
            if name not in features or features[name] is None:
                continue
            feature = features[name]
            if feature.dim() == 1:
                feature = feature.unsqueeze(0)
            token = self.projections[name](feature).unsqueeze(1)
            token = token + self.modality_tokens[name]
            tokens.append(token)
            present_modalities.append(name)

        if not tokens:
            raise ValueError("At least one modality feature is required for fusion.")

        sequence = torch.cat(tokens, dim=1)
        encoded = self.encoder(sequence)
        pooled = encoded.mean(dim=1)
        logits = self.classifier(pooled)

        token_norms = encoded.norm(dim=-1)
        attention_scores = torch.softmax(token_norms, dim=1)

        return {
            "logits": logits,
            "fused_embedding": pooled,
            "attention_scores": attention_scores,
            "modalities": present_modalities,
        }


def normalize_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """Flatten and detach an embedding into a CPU float vector."""
    if embedding.dim() > 1:
        embedding = embedding.reshape(embedding.shape[0], -1)
        embedding = embedding.mean(dim=0)
    return embedding.detach().float().cpu()


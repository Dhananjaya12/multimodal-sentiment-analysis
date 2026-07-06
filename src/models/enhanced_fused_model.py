"""
Feature-level multimodal fusion built around a trained Transformer fusion module.

When models/fusion_transformer.pt exists, inference uses that trained checkpoint.
Missing modalities are represented with zero vectors, matching the training and
ablation setup used for CMU-MOSI. If the checkpoint is missing or fails to load,
the app falls back to the older calibrated feature-fusion baseline.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from ..config.settings import FUSION_MODEL_CONFIG, MODELS_DIR
from ..utils.sentiment_mapping import to_three_way_sentiment
from .audio_model import extract_audio_features, predict_audio_sentiment
from .fusion_transformer import (
    CrossModalFusionTransformer,
    ModalityFeature,
    SENTIMENT_LABELS,
)
from .text_model import extract_text_features, predict_text_sentiment
from .vision_model import extract_vision_features, predict_vision_sentiment

logger = logging.getLogger(__name__)


def predict_transformer_fused_sentiment(
    text: Optional[str] = None,
    audio_bytes: Optional[bytes] = None,
    image: Optional[Image.Image] = None,
) -> Tuple[str, float, Dict]:
    """
    Run feature-level fusion over all available modalities.

    Returns:
        Tuple of (sentiment, confidence, metadata)
    """
    modality_features: Dict[str, ModalityFeature] = {}
    modality_scores: Dict[str, Dict[str, float]] = {}

    if text:
        sentiment, confidence = predict_text_sentiment(text)
        feature = extract_text_features(text)
        if feature is not None:
            modality_features["text"] = ModalityFeature(
                "text", feature, confidence, sentiment
            )
            modality_scores["text"] = _score_distribution(sentiment, confidence)

    if audio_bytes:
        sentiment, confidence = predict_audio_sentiment(audio_bytes)
        feature = extract_audio_features(audio_bytes)
        if feature is not None:
            modality_features["audio"] = ModalityFeature(
                "audio", feature, confidence, sentiment
            )
            modality_scores["audio"] = _score_distribution(sentiment, confidence)

    if image:
        sentiment, confidence = predict_vision_sentiment(image, crop_tightness=0.0)
        feature = extract_vision_features(image, crop_tightness=0.0)
        if feature is not None:
            modality_features["vision"] = ModalityFeature(
                "vision", feature, confidence, sentiment
            )
            modality_scores["vision"] = _score_distribution(sentiment, confidence)

    if not modality_features:
        return "No valid modality features", 0.0, {
            "fusion_method": "Transformer Feature Fusion",
            "modalities": [],
            "trained_checkpoint": False,
        }

    checkpoint_path = Path(MODELS_DIR) / FUSION_MODEL_CONFIG["checkpoint_filename"]
    if checkpoint_path.exists():
        try:
            sentiment, confidence, metadata = _predict_with_checkpoint(
                modality_features, checkpoint_path
            )
            metadata["modality_scores"] = modality_scores
            return sentiment, confidence, metadata
        except Exception as exc:
            logger.exception("Fusion checkpoint inference failed: %s", exc)
            fallback_error = str(exc)
    else:
        fallback_error = "Checkpoint not found"

    sentiment, confidence = _calibrated_feature_fusion(modality_scores)
    metadata = {
        "fusion_method": "Transformer-ready Feature Fusion Baseline",
        "trained_checkpoint": False,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_error": fallback_error,
        "modalities": list(modality_features.keys()),
        "modality_scores": modality_scores,
        "attention_summary": _estimate_modality_influence(modality_features),
    }
    return sentiment, confidence, metadata


def _predict_with_checkpoint(
    modality_features: Dict[str, ModalityFeature], checkpoint_path: Path
) -> Tuple[str, float, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    input_dims = checkpoint.get("input_dims") or {
        name: feature.embedding.numel() for name, feature in modality_features.items()
    }
    model = CrossModalFusionTransformer(
        input_dims=input_dims,
        hidden_dim=checkpoint.get("hidden_dim", FUSION_MODEL_CONFIG["hidden_dim"]),
        num_heads=checkpoint.get("num_heads", FUSION_MODEL_CONFIG["num_heads"]),
        num_layers=checkpoint.get("num_layers", FUSION_MODEL_CONFIG["num_layers"]),
        dropout=checkpoint.get("dropout", FUSION_MODEL_CONFIG["dropout"]),
        num_classes=len(SENTIMENT_LABELS),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    features = {}
    for name, expected_dim in input_dims.items():
        if name in modality_features:
            features[name] = _fit_feature_dim(
                modality_features[name].embedding, expected_dim
            ).reshape(1, -1)
        else:
            features[name] = torch.zeros(1, expected_dim, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(features)
        probabilities = F.softmax(outputs["logits"], dim=-1).squeeze(0)
        confidence, predicted = torch.max(probabilities, dim=0)

    sentiment = SENTIMENT_LABELS[predicted.item()]
    attention = {
        modality: float(score)
        for modality, score in zip(
            outputs["modalities"], outputs["attention_scores"].squeeze(0).tolist()
        )
    }
    return sentiment, float(confidence), {
        "fusion_method": "Trained Transformer Cross-Modal Fusion",
        "trained_checkpoint": True,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_metrics": checkpoint.get("metrics", {}),
        "modalities": list(modality_features.keys()),
        "model_modalities": outputs["modalities"],
        "attention_summary": attention,
        "probabilities": {
            SENTIMENT_LABELS[index]: float(value)
            for index, value in enumerate(probabilities.tolist())
        },
    }


def _fit_feature_dim(feature: torch.Tensor, expected_dim: int) -> torch.Tensor:
    feature = feature.detach().cpu().float().reshape(-1)
    if feature.numel() > expected_dim:
        return feature[:expected_dim]
    if feature.numel() < expected_dim:
        return F.pad(feature, (0, expected_dim - feature.numel()))
    return feature


def _score_distribution(sentiment: str, confidence: float) -> Dict[str, float]:
    three_way = to_three_way_sentiment(sentiment)
    confidence = max(0.0, min(float(confidence), 1.0))
    remaining = max(0.0, 1.0 - confidence)
    scores = {
        label: remaining / (len(SENTIMENT_LABELS) - 1) for label in SENTIMENT_LABELS
    }
    scores[three_way] = confidence
    return scores


def _calibrated_feature_fusion(
    modality_scores: Dict[str, Dict[str, float]]
) -> Tuple[str, float]:
    weights = FUSION_MODEL_CONFIG["modality_weights"]
    totals = {label: 0.0 for label in SENTIMENT_LABELS}
    active_weight = 0.0

    for modality, scores in modality_scores.items():
        weight = weights.get(modality, 1.0)
        active_weight += weight
        for label in SENTIMENT_LABELS:
            totals[label] += scores.get(label, 0.0) * weight

    if active_weight:
        totals = {label: value / active_weight for label, value in totals.items()}

    sentiment = max(totals, key=totals.get)
    return sentiment, round(float(totals[sentiment]), 4)


def _estimate_modality_influence(
    modality_features: Dict[str, ModalityFeature]
) -> Dict[str, float]:
    raw_scores = {
        name: max(0.001, feature.confidence)
        for name, feature in modality_features.items()
    }
    total = sum(raw_scores.values())
    return {name: round(score / total, 4) for name, score in raw_scores.items()}
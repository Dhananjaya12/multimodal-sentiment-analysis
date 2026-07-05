"""Build a fusion training dataset from a CSV manifest.

Supported columns:
- label: Negative, Neutral, or Positive
- text or text_path
- audio_path
- image_path or video_path
- split: optional train/valid/test metadata preserved in the output
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import torch

from src.training.feature_extractors import (
    AUDIO_DIM,
    TEXT_DIM,
    VISION_DIM,
    extract_audio_embedding,
    extract_image_embedding,
    extract_text_embedding,
    extract_video_embedding,
)


LABELS = {"negative": 0, "neutral": 1, "positive": 2}
DEFAULT_DIMS = {"text": TEXT_DIM, "audio": AUDIO_DIM, "vision": VISION_DIM}


def build_dataset(manifest: Path, output: Path, root: Optional[Path] = None):
    root = root or manifest.parent
    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8-sig")))
    if not rows:
        raise ValueError("Manifest is empty.")

    raw_features: Dict[str, List[Optional[torch.Tensor]]] = {
        "text": [],
        "audio": [],
        "vision": [],
    }
    labels = []
    metadata = []

    for index, row in enumerate(rows, start=1):
        label = (row.get("label") or "").strip().lower()
        if label not in LABELS:
            raise ValueError(f"Row {index} has invalid label: {row.get('label')}")

        raw_features["text"].append(_extract_text(row, root))
        raw_features["audio"].append(_extract_audio(row, root))
        raw_features["vision"].append(_extract_vision(row, root))
        labels.append(LABELS[label])
        metadata.append(
            {
                "segment_id": row.get("segment_id", ""),
                "video_id": row.get("video_id", ""),
                "split": row.get("split", ""),
                "label": row.get("label", ""),
                "sentiment_score": row.get("sentiment_score", ""),
            }
        )
        print(f"processed_row={index} label={label}")

    feature_tensors = {
        name: _stack_features(values, DEFAULT_DIMS[name])
        for name, values in raw_features.items()
    }
    dataset = {
        "features": feature_tensors,
        "labels": torch.tensor(labels, dtype=torch.long),
        "label_mapping": {"Negative": 0, "Neutral": 1, "Positive": 2},
        "metadata": metadata,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output)
    print(f"saved_dataset={output}")
    for name, tensor in feature_tensors.items():
        print(f"{name}_shape={tuple(tensor.shape)}")


def _extract_text(row, root: Path):
    text = (row.get("text") or "").strip()
    text_path = (row.get("text_path") or "").strip()
    if text_path:
        path = _resolve(root, text_path)
        if path.exists():
            text = path.read_text(encoding="utf-8")
    return extract_text_embedding(text) if text else None


def _extract_audio(row, root: Path):
    audio_path = (row.get("audio_path") or "").strip()
    if not audio_path:
        return None
    path = _resolve(root, audio_path)
    return extract_audio_embedding(path) if path.exists() else None


def _extract_vision(row, root: Path):
    image_path = (row.get("image_path") or "").strip()
    if image_path:
        path = _resolve(root, image_path)
        if path.exists():
            return extract_image_embedding(path)

    video_path = (row.get("video_path") or "").strip()
    if video_path:
        path = _resolve(root, video_path)
        if path.exists():
            return extract_video_embedding(path, max_frames=5)
    return None


def _stack_features(values: List[Optional[torch.Tensor]], fallback_dim: int) -> torch.Tensor:
    dim = next((value.numel() for value in values if value is not None), fallback_dim)
    fixed = []
    for value in values:
        if value is None:
            fixed.append(torch.zeros(dim, dtype=torch.float32))
            continue
        value = value.detach().cpu().float().reshape(-1)
        if value.numel() > dim:
            value = value[:dim]
        elif value.numel() < dim:
            value = torch.nn.functional.pad(value, (0, dim - value.numel()))
        fixed.append(value)
    return torch.stack(fixed).float()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    build_dataset(args.manifest, args.output, args.root)


if __name__ == "__main__":
    main()
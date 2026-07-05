"""Train the Transformer fusion head on pre-extracted multimodal embeddings."""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from src.config.settings import FUSION_MODEL_CONFIG, MODELS_DIR
from src.models.fusion_transformer import CrossModalFusionTransformer


def train_fusion_head(
    dataset_path: Path,
    output_path: Path,
    epochs: int,
    batch_size: int,
    validation_split: float,
    seed: int,
):
    dataset = torch.load(dataset_path, map_location="cpu")
    features = dataset["features"]
    labels = dataset["labels"].long()
    input_dims = {name: tensor.shape[1] for name, tensor in features.items()}

    model = CrossModalFusionTransformer(
        input_dims=input_dims,
        hidden_dim=FUSION_MODEL_CONFIG["hidden_dim"],
        num_heads=FUSION_MODEL_CONFIG["num_heads"],
        num_layers=FUSION_MODEL_CONFIG["num_layers"],
        dropout=FUSION_MODEL_CONFIG["dropout"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=FUSION_MODEL_CONFIG["learning_rate"]
    )

    tensors = [features[name].float() for name in input_dims]
    tensors.append(labels)
    tensor_dataset = TensorDataset(*tensors)
    train_dataset, val_dataset, test_dataset = _make_datasets(
        tensor_dataset,
        dataset.get("metadata", []),
        validation_split,
        seed,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = (
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        if test_dataset is not None
        else None
    )

    best_val_f1 = -1.0
    best_metrics = {}
    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_loader, optimizer, input_dims)
        val_metrics = evaluate(model, val_loader, input_dims)
        print(
            "epoch={epoch} train_loss={loss:.4f} val_loss={val_loss:.4f} "
            "val_accuracy={accuracy:.4f} val_macro_f1={macro_f1:.4f}".format(
                epoch=epoch + 1,
                loss=train_loss,
                **val_metrics,
            )
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_metrics = val_metrics
            _save_checkpoint(model, output_path, input_dims, best_metrics)

    if test_loader is not None:
        checkpoint = torch.load(output_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        best_metrics["test"] = evaluate(model, test_loader, input_dims)

    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    print(f"saved_checkpoint={output_path}")
    print(f"saved_metrics={metrics_path}")


def _run_epoch(model, loader, optimizer, input_dims: Dict[str, int]) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch_features = {
            name: batch[index] for index, name in enumerate(input_dims.keys())
        }
        batch_labels = batch[-1]
        outputs = model(batch_features)
        loss = F.cross_entropy(outputs["logits"], batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
    return total_loss / max(len(loader), 1)


def evaluate(model, loader, input_dims: Dict[str, int]) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            batch_features = {
                name: batch[index] for index, name in enumerate(input_dims.keys())
            }
            batch_labels = batch[-1]
            outputs = model(batch_features)
            loss = F.cross_entropy(outputs["logits"], batch_labels)
            total_loss += float(loss)
            predictions.extend(torch.argmax(outputs["logits"], dim=-1).tolist())
            targets.extend(batch_labels.tolist())

    metrics = _classification_metrics(predictions, targets, num_classes=3)
    metrics["val_loss"] = total_loss / max(len(loader), 1)
    return metrics


def _classification_metrics(predictions, targets, num_classes: int) -> Dict[str, float]:
    correct = sum(int(pred == target) for pred, target in zip(predictions, targets))
    accuracy = correct / max(len(targets), 1)
    f1_scores = []

    for class_id in range(num_classes):
        tp = sum(
            int(pred == class_id and target == class_id)
            for pred, target in zip(predictions, targets)
        )
        fp = sum(
            int(pred == class_id and target != class_id)
            for pred, target in zip(predictions, targets)
        )
        fn = sum(
            int(pred != class_id and target == class_id)
            for pred, target in zip(predictions, targets)
        )
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        f1_scores.append(f1)

    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1_scores) / len(f1_scores),
    }


def _make_datasets(
    dataset: TensorDataset, metadata: list, validation_split: float, seed: int
) -> Tuple[Subset, Subset, Subset]:
    if metadata and any(item.get("split") for item in metadata):
        train_indices = [
            index for index, item in enumerate(metadata) if item.get("split") == "train"
        ]
        val_indices = [
            index
            for index, item in enumerate(metadata)
            if item.get("split") in {"valid", "val", "validation"}
        ]
        test_indices = [
            index for index, item in enumerate(metadata) if item.get("split") == "test"
        ]
        if train_indices and val_indices:
            return (
                Subset(dataset, train_indices),
                Subset(dataset, val_indices),
                Subset(dataset, test_indices) if test_indices else None,
            )

    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    val_size = max(1, int(len(indices) * validation_split))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    if not train_indices:
        raise ValueError("Training dataset is too small after validation split.")
    return Subset(dataset, train_indices), Subset(dataset, val_indices), None


def _save_checkpoint(model, output_path: Path, input_dims: Dict[str, int], metrics):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dims": input_dims,
            "hidden_dim": FUSION_MODEL_CONFIG["hidden_dim"],
            "num_heads": FUSION_MODEL_CONFIG["num_heads"],
            "num_layers": FUSION_MODEL_CONFIG["num_layers"],
            "metrics": metrics,
        },
        output_path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=MODELS_DIR / FUSION_MODEL_CONFIG["checkpoint_filename"],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_fusion_head(
        args.dataset,
        args.output,
        args.epochs,
        args.batch_size,
        args.validation_split,
        args.seed,
    )


if __name__ == "__main__":
    main()
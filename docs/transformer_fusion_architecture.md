# Transformer Feature Fusion Architecture

This project now supports two multimodal inference paths:

1. Decision-level baseline fusion in `src/models/fused_model.py`.
2. Transformer-ready feature-level fusion in `src/models/enhanced_fused_model.py`.

## Baseline

The original application predicts sentiment independently for text, audio, and
vision, then combines the final labels with weighted voting. This is retained as
a baseline for comparison and for future evaluation.

## Enhanced Fusion

The enhanced path extracts feature vectors before final fusion:

- Text: DistilBERT mean-pooled embedding when weights are available, with a
  deterministic TextBlob-aware fallback embedding for offline use.
- Audio: Wav2Vec2 hidden-state embedding from the fine-tuned audio model.
- Vision: Penultimate ResNet-style visual embedding from the fine-tuned vision
  model.

The `CrossModalFusionTransformer` projects available modality embeddings into a
shared hidden space, adds learned modality tokens, applies Transformer
self-attention across modalities, and classifies the fused representation into
Negative, Neutral, or Positive.

## Checkpoint Behavior

At inference time, the app looks for:

```text
models/fusion_transformer.pt
```

If that checkpoint exists, the Transformer classifier output is used. If it does
not exist, the app still extracts modality features and uses a calibrated
feature-fusion baseline so the Streamlit workflow remains usable before
training.

## Training Entry Point

Use `src/training/train_fusion.py` with pre-extracted embeddings:

```bash
python -m src.training.train_fusion --dataset path/to/fusion_dataset.pt
```

The expected dataset format is:

```python
{
    "features": {
        "text": Tensor[num_examples, text_dim],
        "audio": Tensor[num_examples, audio_dim],
        "vision": Tensor[num_examples, vision_dim],
    },
    "labels": Tensor[num_examples],
}
```

Labels are encoded as `0 = Negative`, `1 = Neutral`, and `2 = Positive`.

## Evaluation Plan

Compare the original decision-level baseline and the enhanced feature-fusion
model with Accuracy, F1, Precision, Recall, and inference time. This preserves
the existing Streamlit interface while making the core architecture ready for
true multimodal learning.

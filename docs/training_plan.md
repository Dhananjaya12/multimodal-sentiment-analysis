# Training Plan

The training focus is the fusion model used by the live demo.

## Dataset

Use the Kaggle CMU-MOSI dataset referenced by the notebook. We are not copying the notebook model. We only reuse the useful data setup ideas:

- segmented transcripts
- segmented audio paths
- segmented video paths
- labels from `mosi_data.pkl`
- train/valid/test split metadata

## Step 1: Create CMU-MOSI Manifest

Point `--raw-root` at the Kaggle CMU-MOSI `Raw` folder, and `--label-pkl` at `mosi_data.pkl`.

```bash
python -m src.training.prepare_cmu_mosi_manifest \
  --raw-root /kaggle/input/datasets/mathurinache/cmu-mosi/Raw \
  --label-pkl /kaggle/input/datasets/reganwillis/cmu-mosi/mosi_data.pkl \
  --output data/cmu_mosi_manifest.csv
```

The manifest contains:

```csv
segment_id,video_id,text,audio_path,video_path,sentiment_score,label,split
```

The default label mapping is:

- score > 0.5: `Positive`
- score < -0.5: `Negative`
- otherwise: `Neutral`

You can change this with `--neutral-threshold`.

## Step 2: Build Embedding Dataset

```bash
python -m src.training.build_fusion_dataset \
  --manifest data/cmu_mosi_manifest.csv \
  --output data/cmu_mosi_fusion_dataset.pt
```

This extracts the features our fusion model needs:

- text embedding
- audio embedding
- vision embedding from video frames
- sentiment label
- original split metadata

## Step 3: Train Fusion Model

```bash
python -m src.training.train_fusion \
  --dataset data/cmu_mosi_fusion_dataset.pt \
  --output models/fusion_transformer.pt \
  --epochs 20 \
  --batch-size 16
```

The trainer uses CMU-MOSI's existing `train`, `valid`, and `test` splits when present.

Outputs:

- `models/fusion_transformer.pt`
- `models/fusion_transformer.metrics.json`

## App Integration

Once `models/fusion_transformer.pt` exists, the Streamlit app automatically uses it for enhanced fusion.
# Kaggle Training Cells

Paste these cells into a Kaggle notebook after uploading this project as a Kaggle dataset or cloning it into `/kaggle/working`.

## Cell 1: Install Requirements

```python
PROJECT_DIR = "/kaggle/working/multimodal-sentiment-analysis"

!pip install -q -r {PROJECT_DIR}/requirements-kaggle-training.txt
```

## Cell 2: Configure Paths

Change these paths to match the Kaggle datasets attached to your notebook.

```python
from pathlib import Path
import os

PROJECT_DIR = Path("/kaggle/working/multimodal-sentiment-analysis")
os.chdir(PROJECT_DIR)

CMU_MOSI_RAW = Path("/kaggle/input/datasets/mathurinache/cmu-mosi/Raw")
CMU_MOSI_LABEL_PKL = Path("/kaggle/input/datasets/reganwillis/cmu-mosi/mosi_data.pkl")

DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

MANIFEST_PATH = DATA_DIR / "cmu_mosi_manifest.csv"
FUSION_DATASET_PATH = DATA_DIR / "cmu_mosi_fusion_dataset.pt"
CHECKPOINT_PATH = MODEL_DIR / "fusion_transformer.pt"

print("Project:", PROJECT_DIR)
print("Raw exists:", CMU_MOSI_RAW.exists(), CMU_MOSI_RAW)
print("Labels exist:", CMU_MOSI_LABEL_PKL.exists(), CMU_MOSI_LABEL_PKL)
```

## Cell 3: Create Manifest

```python
!python -m src.training.prepare_cmu_mosi_manifest \
  --raw-root "{CMU_MOSI_RAW}" \
  --label-pkl "{CMU_MOSI_LABEL_PKL}" \
  --output "{MANIFEST_PATH}" \
  --neutral-threshold 0.5
```

## Cell 4: Build Embedding Dataset

This is the slow step. Use GPU runtime.

```python
!python -m src.training.build_fusion_dataset \
  --manifest "{MANIFEST_PATH}" \
  --output "{FUSION_DATASET_PATH}"
```

## Cell 5: Train Fusion Transformer

```python
!python -m src.training.train_fusion \
  --dataset "{FUSION_DATASET_PATH}" \
  --output "{CHECKPOINT_PATH}" \
  --epochs 20 \
  --batch-size 16
```

## Cell 6: Download Outputs

```python
from IPython.display import FileLink, display

display(FileLink(str(CHECKPOINT_PATH)))
display(FileLink(str(CHECKPOINT_PATH.with_suffix(".metrics.json"))))
```

## Notes

- Use Kaggle GPU.
- If internet is disabled, attach cached Hugging Face model files or enable internet for the first feature extraction run.
- The trained app checkpoint is `models/fusion_transformer.pt`.
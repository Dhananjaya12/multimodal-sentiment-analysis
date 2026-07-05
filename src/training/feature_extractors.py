"""Training-only feature extractors with no Streamlit dependency."""

from pathlib import Path
from typing import List, Optional

import cv2
import librosa
import numpy as np
import torch
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer, Wav2Vec2Model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEXT_MODEL_NAME = "bert-base-uncased"
AUDIO_MODEL_NAME = "facebook/wav2vec2-base"
TEXT_DIM = 768
AUDIO_DIM = 768
VISION_DIM = 2048

_text_tokenizer = None
_text_model = None
_audio_processor = None
_audio_model = None
_vision_model = None
_vision_transform = None


def extract_text_embedding(text: str) -> Optional[torch.Tensor]:
    if not text or not text.strip():
        return None
    tokenizer, model = _load_text_model()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().float()


def extract_audio_embedding(audio_path: Path) -> Optional[torch.Tensor]:
    if not audio_path.exists():
        return None
    processor, model = _load_audio_model()
    audio, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    if audio.size == 0:
        return None
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=16000 * 5,
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().float()


def extract_image_embedding(image_path: Path) -> Optional[torch.Tensor]:
    if not image_path.exists():
        return None
    with Image.open(image_path) as image:
        return extract_pil_image_embedding(image.convert("RGB"))


def extract_video_embedding(video_path: Path, max_frames: int = 5) -> Optional[torch.Tensor]:
    if not video_path.exists():
        return None
    frames = _sample_video_frames(video_path, max_frames=max_frames)
    if not frames:
        return None
    embeddings = [extract_pil_image_embedding(frame) for frame in frames]
    embeddings = [embedding for embedding in embeddings if embedding is not None]
    if not embeddings:
        return None
    return torch.stack(embeddings).mean(dim=0).cpu().float()


def extract_pil_image_embedding(image: Image.Image) -> Optional[torch.Tensor]:
    model, transform = _load_vision_model()
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = model(tensor)
    return features.reshape(features.shape[0], -1).squeeze(0).cpu().float()


def _load_text_model():
    global _text_tokenizer, _text_model
    if _text_tokenizer is None or _text_model is None:
        _text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
        _text_model = AutoModel.from_pretrained(TEXT_MODEL_NAME).to(DEVICE)
        _text_model.eval()
    return _text_tokenizer, _text_model


def _load_audio_model():
    global _audio_processor, _audio_model
    if _audio_processor is None or _audio_model is None:
        _audio_processor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL_NAME)
        _audio_model = Wav2Vec2Model.from_pretrained(AUDIO_MODEL_NAME).to(DEVICE)
        _audio_model.eval()
    return _audio_processor, _audio_model


def _load_vision_model():
    global _vision_model, _vision_transform
    if _vision_model is None or _vision_transform is None:
        import torchvision.models as models
        import torchvision.transforms as transforms

        try:
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
        except Exception:
            model = models.resnet50(weights=None)
        _vision_model = torch.nn.Sequential(*list(model.children())[:-1]).to(DEVICE)
        _vision_model.eval()
        _vision_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    return _vision_model, _vision_transform


def _sample_video_frames(video_path: Path, max_frames: int) -> List[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    frame_indices = np.linspace(0, total - 1, max_frames, dtype=int)
    frames = []
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
    cap.release()
    return frames
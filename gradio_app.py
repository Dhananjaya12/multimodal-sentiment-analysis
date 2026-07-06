"""Clean Gradio demo for the trained multimodal fusion model."""

import html
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import gradio as gr
import torch
import torch.nn.functional as F

from src.models.fusion_transformer import CrossModalFusionTransformer
from src.training.feature_extractors import (
    AUDIO_DIM,
    TEXT_DIM,
    VISION_DIM,
    extract_audio_embedding,
    extract_image_embedding,
    extract_text_embedding,
    extract_video_embedding,
)

try:
    from moviepy import VideoFileClip
except Exception:
    VideoFileClip = None

try:
    import speech_recognition as sr
except Exception:
    sr = None


BASE_DIR = Path(__file__).parent
MODEL_PATH = Path(os.environ.get("FUSION_MODEL_PATH", BASE_DIR / "models" / "fusion_transformer.pt"))
LABELS = ["Negative", "Neutral", "Positive"]
COLORS = {"Negative": "#d92d20", "Neutral": "#b7791f", "Positive": "#039855"}


CSS = """
.gradio-container { max-width: 1080px !important; margin: 0 auto !important; background: #f7f8fc !important; }
.header { text-align: center; padding: 26px 12px 20px; }
.header h1 { color: #172033; font-size: 2rem; margin: 0 0 8px; }
.header p { color: #667085; margin: 0; }
.card { background: white !important; border: 1px solid #e5e7eb !important; border-radius: 16px !important; padding: 16px !important; box-shadow: 0 8px 28px rgba(16, 24, 40, 0.06); }
.primary-btn, .secondary-btn { border-radius: 10px !important; min-height: 44px !important; }
.result-card { background: white; border: 1px solid #e5e7eb; border-radius: 16px; padding: 24px; box-shadow: 0 8px 28px rgba(16, 24, 40, 0.06); }
.result-label { font-size: 1.8rem; font-weight: 750; }
.result-meta { color: #667085; margin-top: 4px; }
.confidence-track { height: 9px; margin-top: 20px; background: #eef0f4; border-radius: 999px; overflow: hidden; }
.confidence-fill { height: 100%; border-radius: 999px; }
.transcript-card { background: white; border: 1px solid #e5e7eb; border-radius: 16px; padding: 20px; margin-bottom: 14px; color: #344054; min-height: 100px; box-shadow: 0 8px 28px rgba(16, 24, 40, 0.06); }
.transcript-card h3 { color: #172033; margin: 0 0 10px; }
.placeholder { color: #98a2b3; }
.error-card { color: #b42318; background: #fff4f2; border: 1px solid #fecdca; border-radius: 12px; padding: 16px; }
"""


_model = None
_checkpoint = None


def load_model():
    global _model, _checkpoint
    if _model is not None:
        return _model, _checkpoint
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained fusion checkpoint not found: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    input_dims = checkpoint.get("input_dims", {"text": TEXT_DIM, "audio": AUDIO_DIM, "vision": VISION_DIM})
    model = CrossModalFusionTransformer(
        input_dims=input_dims,
        hidden_dim=checkpoint.get("hidden_dim", 128),
        num_heads=checkpoint.get("num_heads", 4),
        num_layers=checkpoint.get("num_layers", 1),
        dropout=checkpoint.get("dropout", 0.35),
        num_classes=3,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _model = model
    _checkpoint = checkpoint
    return _model, _checkpoint


def media_to_path(value) -> Optional[Path]:
    """Normalize Gradio file outputs across versions/components."""
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        path = Path(value)
        return path if path.exists() else None
    if isinstance(value, dict):
        for key in ("path", "name"):
            if value.get(key):
                path = Path(value[key])
                if path.exists():
                    return path
    if isinstance(value, (tuple, list)):
        for item in value:
            path = media_to_path(item)
            if path is not None:
                return path
    return None


def fit_dim(feature: Optional[torch.Tensor], expected_dim: int) -> torch.Tensor:
    if feature is None:
        return torch.zeros(1, expected_dim, dtype=torch.float32)
    feature = feature.detach().cpu().float().reshape(-1)
    if feature.numel() > expected_dim:
        feature = feature[:expected_dim]
    elif feature.numel() < expected_dim:
        feature = F.pad(feature, (0, expected_dim - feature.numel()))
    return feature.reshape(1, -1)


def extract_audio_from_video(video_path: Path) -> Optional[Path]:
    if not video_path or not video_path.exists():
        return None

    temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp.close()
    audio_path = Path(temp.name)

    if VideoFileClip is not None:
        try:
            video = VideoFileClip(str(video_path))
            if video.audio is not None:
                video.audio.write_audiofile(str(audio_path), fps=16000, logger=None)
                video.close()
                if audio_path.exists() and audio_path.stat().st_size > 0:
                    return audio_path
            video.close()
        except Exception:
            pass

    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio_path),
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0 and audio_path.exists() and audio_path.stat().st_size > 0:
            return audio_path

    try:
        audio_path.unlink(missing_ok=True)
    except Exception:
        pass
    return None


def transcribe_audio(audio_path: Optional[Path]) -> str:
    if audio_path is None or sr is None:
        return ""
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(str(audio_path)) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except Exception:
        return ""


def transcript_html(text: str, notes: list[str]) -> str:
    safe_text = html.escape((text or "").strip())
    if not safe_text:
        safe_text = '<span class="placeholder">No transcript was detected or provided.</span>'
    note_html = "".join(f'<div class="result-meta">{html.escape(note)}</div>' for note in notes)
    return f'<div class="transcript-card"><h3>Transcript / Text</h3><div>{safe_text}</div>{note_html}</div>'


def result_html(label: str, confidence: float, probabilities: dict, modalities: list, elapsed: float) -> str:
    color = COLORS.get(label, "#344054")
    confidence_pct = confidence * 100.0
    probs = " | ".join(f"{name}: {value * 100:.1f}%" for name, value in probabilities.items())
    mods = ", ".join(modalities) if modalities else "none"
    return f"""
    <div class="result-card">
      <div class="result-label" style="color:{color}">{html.escape(label)}</div>
      <div class="result-meta">Confidence: {confidence_pct:.1f}%</div>
      <div class="result-meta">Model: Trained CMU-MOSI Fusion Transformer</div>
      <div class="result-meta">Modalities used: {html.escape(mods)}</div>
      <div class="result-meta">Probabilities: {html.escape(probs)}</div>
      <div class="result-meta">Runtime: {elapsed:.2f}s</div>
      <div class="confidence-track"><div class="confidence-fill" style="width:{confidence_pct}%;background:{color}"></div></div>
    </div>
    """


def analyze(video_value, audio_value, image_value, text, progress=gr.Progress()):
    started = time.perf_counter()
    text = (text or "").strip()
    notes = []

    try:
        model, checkpoint = load_model()
    except Exception as exc:
        return "", f'<div class="error-card">{html.escape(str(exc))}</div>'

    input_dims = checkpoint.get("input_dims", {"text": TEXT_DIM, "audio": AUDIO_DIM, "vision": VISION_DIM})
    modalities = []
    transcript = text
    text_feature = None
    audio_feature = None
    vision_feature = None

    video_path = media_to_path(video_value)
    audio_path = media_to_path(audio_value)
    image_path = media_to_path(image_value)

    if video_path:
        progress(0.15, desc="Extracting video frames")
        try:
            vision_feature = extract_video_embedding(video_path, max_frames=5)
            if vision_feature is not None:
                modalities.append("vision")
            else:
                notes.append("Video was received, but vision features could not be extracted.")
        except Exception as exc:
            notes.append(f"Vision extraction failed: {exc}")

        progress(0.35, desc="Extracting video audio")
        extracted_audio = extract_audio_from_video(video_path)
        if extracted_audio is not None:
            audio_path = extracted_audio
            notes.append("Audio extracted from video.")
        else:
            notes.append("No usable audio track was found in the video.")

    if image_path and vision_feature is None:
        progress(0.35, desc="Extracting image features")
        try:
            vision_feature = extract_image_embedding(image_path)
            if vision_feature is not None:
                modalities.append("vision")
            else:
                notes.append("Image was received, but vision features could not be extracted.")
        except Exception as exc:
            notes.append(f"Image extraction failed: {exc}")

    if audio_path:
        progress(0.55, desc="Extracting audio features")
        try:
            audio_feature = extract_audio_embedding(audio_path)
            if audio_feature is not None:
                modalities.append("audio")
            else:
                notes.append("Audio was received, but audio features could not be extracted.")
        except Exception as exc:
            notes.append(f"Audio extraction failed: {exc}")

        if not transcript:
            transcript = transcribe_audio(audio_path)
            if not transcript:
                notes.append("Transcript was not detected, but audio features were still used.")

        if video_path and audio_path.name.startswith("tmp"):
            try:
                audio_path.unlink(missing_ok=True)
            except Exception:
                pass

    if transcript:
        progress(0.70, desc="Extracting text features")
        try:
            text_feature = extract_text_embedding(transcript)
            if text_feature is not None:
                modalities.append("text")
        except Exception as exc:
            notes.append(f"Text extraction failed: {exc}")

    if not modalities:
        return transcript_html(transcript, notes), '<div class="error-card">No usable modality features could be extracted.</div>'

    features = {
        "text": fit_dim(text_feature, input_dims.get("text", TEXT_DIM)),
        "audio": fit_dim(audio_feature, input_dims.get("audio", AUDIO_DIM)),
        "vision": fit_dim(vision_feature, input_dims.get("vision", VISION_DIM)),
    }

    progress(0.90, desc="Running trained fusion model")
    with torch.no_grad():
        output = model(features)
        probs_tensor = F.softmax(output["logits"], dim=-1).squeeze(0)
        confidence, predicted = torch.max(probs_tensor, dim=0)

    label = LABELS[predicted.item()]
    probabilities = {LABELS[i]: float(probs_tensor[i]) for i in range(len(LABELS))}
    progress(1.0, desc="Complete")
    return transcript_html(transcript, notes), result_html(label, float(confidence), probabilities, sorted(set(modalities)), time.perf_counter() - started)


def clear_all():
    return (
        None,
        None,
        None,
        "",
        '<div class="transcript-card"><h3>Transcript / Text</h3><div class="placeholder">Transcript will appear here.</div></div>',
        '<div class="result-card"><div class="placeholder">Prediction will appear here.</div></div>',
    )


with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"), css=CSS, title="Multimodal Sentiment Analysis") as demo:
    gr.HTML("""
    <div class="header">
      <h1>Multimodal Sentiment Analysis</h1>
      <p>Analyze text, audio, image, uploaded video, or webcam video with the trained fusion model.</p>
    </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=6, elem_classes="card"):
            video_input = gr.Video(label="Video Upload / Webcam", sources=["upload", "webcam"], format=None, include_audio=True, height=300)
            audio_input = gr.Audio(label="Audio Upload / Microphone", sources=["upload", "microphone"], type="filepath")
            image_input = gr.Image(label="Image Upload / Webcam Photo", sources=["upload", "webcam"], type="filepath")
            text_input = gr.Textbox(label="Text", placeholder="Type text here, or leave empty when using video/audio.", lines=3)
            with gr.Row():
                analyze_button = gr.Button("Analyze", variant="primary", elem_classes="primary-btn")
                clear_button = gr.Button("Clear", variant="secondary", elem_classes="secondary-btn")
        with gr.Column(scale=5):
            transcript_output = gr.HTML('<div class="transcript-card"><h3>Transcript / Text</h3><div class="placeholder">Transcript will appear here.</div></div>')
            result_output = gr.HTML('<div class="result-card"><div class="placeholder">Prediction will appear here.</div></div>')

    analyze_button.click(fn=analyze, inputs=[video_input, audio_input, image_input, text_input], outputs=[transcript_output, result_output], concurrency_limit=1, show_progress="full")
    clear_button.click(fn=clear_all, inputs=[], outputs=[video_input, audio_input, image_input, text_input, transcript_output, result_output], queue=False)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860, share=False)
"""Clean Gradio demo for the trained multimodal fusion model."""

import html
import os
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
EMOJIS = {"Negative": "Negative", "Neutral": "Neutral", "Positive": "Positive"}


CSS = """
.gradio-container {
  max-width: 1080px !important;
  margin: 0 auto !important;
  background: #f7f8fc !important;
}
.header { text-align: center; padding: 26px 12px 20px; }
.header h1 { color: #172033; font-size: 2rem; margin: 0 0 8px; }
.header p { color: #667085; margin: 0; }
.card {
  background: white !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 16px !important;
  padding: 16px !important;
  box-shadow: 0 8px 28px rgba(16, 24, 40, 0.06);
}
.primary-btn, .secondary-btn { border-radius: 10px !important; min-height: 44px !important; }
.result-card {
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 8px 28px rgba(16, 24, 40, 0.06);
}
.result-label { font-size: 1.8rem; font-weight: 750; }
.result-meta { color: #667085; margin-top: 4px; }
.confidence-track { height: 9px; margin-top: 20px; background: #eef0f4; border-radius: 999px; overflow: hidden; }
.confidence-fill { height: 100%; border-radius: 999px; }
.transcript-card {
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 20px;
  margin-bottom: 14px;
  color: #344054;
  min-height: 100px;
  box-shadow: 0 8px 28px rgba(16, 24, 40, 0.06);
}
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


def fit_dim(feature: Optional[torch.Tensor], expected_dim: int) -> torch.Tensor:
    if feature is None:
        return torch.zeros(1, expected_dim, dtype=torch.float32)
    feature = feature.detach().cpu().float().reshape(-1)
    if feature.numel() > expected_dim:
        feature = feature[:expected_dim]
    elif feature.numel() < expected_dim:
        feature = F.pad(feature, (0, expected_dim - feature.numel()))
    return feature.reshape(1, -1)


def extract_audio_from_video(video_path: str) -> Optional[Path]:
    if VideoFileClip is None:
        return None
    try:
        video = VideoFileClip(video_path)
        if video.audio is None:
            video.close()
            return None
        temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp.close()
        audio_path = Path(temp.name)
        video.audio.write_audiofile(str(audio_path), logger=None)
        video.close()
        return audio_path
    except Exception:
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


def transcript_html(text: str) -> str:
    safe_text = html.escape((text or "").strip())
    if not safe_text:
        safe_text = '<span class="placeholder">No transcript was detected or provided.</span>'
    return f'<div class="transcript-card"><h3>Transcript / Text</h3><div>{safe_text}</div></div>'


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


def toggle_text_input(media_path):
    if media_path:
        return gr.update(value="", interactive=False, placeholder="Text input is disabled while a video is selected.")
    return gr.update(interactive=True, placeholder="Type text here when no video is selected.")


def analyze(media_path, text, progress=gr.Progress()):
    started = time.perf_counter()
    text = (text or "").strip()

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

    if media_path:
        progress(0.15, desc="Extracting video features")
        vision_feature = extract_video_embedding(Path(media_path), max_frames=5)
        if vision_feature is not None:
            modalities.append("vision")

        progress(0.35, desc="Extracting audio")
        audio_path = extract_audio_from_video(str(media_path))
        if audio_path is not None:
            audio_feature = extract_audio_embedding(audio_path)
            if audio_feature is not None:
                modalities.append("audio")
            transcript = transcribe_audio(audio_path)
            try:
                audio_path.unlink(missing_ok=True)
            except Exception:
                pass

        if transcript:
            progress(0.60, desc="Extracting transcript features")
            text_feature = extract_text_embedding(transcript)
            if text_feature is not None:
                modalities.append("text")
    elif text:
        progress(0.35, desc="Extracting text features")
        text_feature = extract_text_embedding(text)
        if text_feature is not None:
            modalities.append("text")
    else:
        return "", '<div class="error-card">Upload or record a video, or enter text.</div>'

    if not modalities:
        return transcript_html(transcript), '<div class="error-card">No usable model features could be extracted.</div>'

    features = {
        "text": fit_dim(text_feature, input_dims.get("text", TEXT_DIM)),
        "audio": fit_dim(audio_feature, input_dims.get("audio", AUDIO_DIM)),
        "vision": fit_dim(vision_feature, input_dims.get("vision", VISION_DIM)),
    }

    progress(0.85, desc="Running trained fusion model")
    with torch.no_grad():
        output = model(features)
        probs_tensor = F.softmax(output["logits"], dim=-1).squeeze(0)
        confidence, predicted = torch.max(probs_tensor, dim=0)

    label = LABELS[predicted.item()]
    probabilities = {LABELS[i]: float(probs_tensor[i]) for i in range(len(LABELS))}
    progress(1.0, desc="Complete")
    return transcript_html(transcript), result_html(label, float(confidence), probabilities, modalities, time.perf_counter() - started)


def clear_all():
    return (
        None,
        gr.update(value="", interactive=True, placeholder="Type text here when no video is selected."),
        '<div class="transcript-card"><h3>Transcript / Text</h3><div class="placeholder">Transcript will appear here.</div></div>',
        '<div class="result-card"><div class="placeholder">Prediction will appear here.</div></div>',
    )


with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"), css=CSS, title="Multimodal Sentiment Analysis") as demo:
    gr.HTML("""
    <div class="header">
      <h1>Multimodal Sentiment Analysis</h1>
      <p>Upload or record a video, or analyze text with the trained fusion model.</p>
    </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=6, elem_classes="card"):
            media_input = gr.Video(label="Video", sources=["upload", "webcam"], format=None, include_audio=True, height=340)
            text_input = gr.Textbox(label="Text", placeholder="Type text here when no video is selected.", lines=3)
            with gr.Row():
                analyze_button = gr.Button("Analyze", variant="primary", elem_classes="primary-btn")
                clear_button = gr.Button("Clear", variant="secondary", elem_classes="secondary-btn")
        with gr.Column(scale=5):
            transcript_output = gr.HTML('<div class="transcript-card"><h3>Transcript / Text</h3><div class="placeholder">Transcript will appear here.</div></div>')
            result_output = gr.HTML('<div class="result-card"><div class="placeholder">Prediction will appear here.</div></div>')

    media_input.change(fn=toggle_text_input, inputs=[media_input], outputs=[text_input], queue=False)
    analyze_button.click(fn=analyze, inputs=[media_input, text_input], outputs=[transcript_output, result_output], concurrency_limit=1, show_progress="full")
    clear_button.click(fn=clear_all, inputs=[], outputs=[media_input, text_input, transcript_output, result_output], queue=False)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860, share=False)
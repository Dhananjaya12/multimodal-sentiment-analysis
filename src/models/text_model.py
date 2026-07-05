"""
Text sentiment analysis model using TextBlob.
"""

import logging
import hashlib
from typing import Tuple, Optional

import torch
import streamlit as st
from ..config.settings import TEXT_MODEL_CONFIG

logger = logging.getLogger(__name__)


@st.cache_resource
def load_text_encoder():
    """Load a Transformer text encoder when cached weights are available."""
    try:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_CONFIG["model_name"])
        model = AutoModel.from_pretrained(TEXT_MODEL_CONFIG["model_name"])
        model.eval()
        return tokenizer, model
    except Exception as e:
        logger.warning("Transformer text encoder unavailable, using fallback: %s", e)
        return None, None


def predict_text_sentiment(text: str) -> Tuple[str, float]:
    """
    Analyze text sentiment using TextBlob.

    Args:
        text: Input text to analyze

    Returns:
        Tuple of (sentiment, confidence)
    """
    if not text or text.strip() == "":
        return "No text provided", 0.0

    try:
        from textblob import TextBlob

        # Create TextBlob object
        blob = TextBlob(text)

        # Get polarity (-1 to 1, where -1 is very negative, 1 is very positive)
        polarity = blob.sentiment.polarity

        # Get subjectivity (0 to 1, where 0 is very objective, 1 is very subjective)
        subjectivity = blob.sentiment.subjectivity

        # Convert polarity to sentiment categories
        confidence_threshold = TEXT_MODEL_CONFIG["confidence_threshold"]

        if polarity > confidence_threshold:
            sentiment = "Positive"
            confidence = min(0.95, 0.6 + abs(polarity) * 0.3)
        elif polarity < -confidence_threshold:
            sentiment = "Negative"
            confidence = min(0.95, 0.6 + abs(polarity) * 0.3)
        else:
            sentiment = "Neutral"
            confidence = 0.7 - abs(polarity) * 0.2

        # Round confidence to 2 decimal places
        confidence = round(confidence, 2)

        logger.info(
            f"Text sentiment analysis completed: {sentiment} (confidence: {confidence})"
        )
        return sentiment, confidence

    except ImportError:
        logger.error(
            "TextBlob not installed. Please install it with: pip install textblob"
        )
        return "TextBlob not available", 0.0
    except Exception as e:
        logger.error(f"Error in text sentiment analysis: {str(e)}")
        return "Error occurred", 0.0



def extract_text_features(text: str) -> Optional[torch.Tensor]:
    """
    Return a text embedding for feature-level fusion.

    The preferred path is DistilBERT mean pooling. If pretrained weights are
    unavailable, a deterministic lexical fallback keeps the fusion pipeline
    runnable offline.
    """
    if not text or text.strip() == "":
        return None

    tokenizer, model = load_text_encoder()
    if tokenizer is not None and model is not None:
        try:
            inputs = tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = model(**inputs)
                return outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()
        except Exception as e:
            logger.warning("Transformer text feature extraction failed: %s", e)

    return _fallback_text_embedding(text)


def _fallback_text_embedding(text: str) -> torch.Tensor:
    """Create a deterministic 768-dim text feature without external weights."""
    dim = TEXT_MODEL_CONFIG["embedding_dim"]
    embedding = torch.zeros(dim, dtype=torch.float32)
    tokens = text.lower().split()
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        embedding[index] += sign

    try:
        from textblob import TextBlob

        sentiment = TextBlob(text).sentiment
        embedding[0] = float(sentiment.polarity)
        embedding[1] = float(sentiment.subjectivity)
    except Exception:
        pass

    norm = embedding.norm()
    return embedding / norm if norm > 0 else embedding

def get_text_model_info() -> dict:
    """Get information about the text sentiment model."""
    return {
        "model_name": TEXT_MODEL_CONFIG["model_name"],
        "description": "Transformer-ready text feature extraction with TextBlob fallback sentiment",
        "capabilities": [
            "Text sentiment classification (Positive/Negative/Neutral)",
            "Confidence scoring",
            "Real-time analysis",
            "No external API required",
        ],
        "input_format": "Plain text",
        "output_format": "Sentiment label + confidence score",
    }

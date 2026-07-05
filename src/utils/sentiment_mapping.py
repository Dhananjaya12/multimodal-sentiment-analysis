"""
Sentiment mapping utilities for different model outputs.
"""

from typing import Dict
from ..config.settings import SENTIMENT_MAPPINGS


def get_sentiment_mapping(num_classes: int) -> Dict[int, str]:
    """
    Get the sentiment mapping based on number of classes.

    Args:
        num_classes: Number of output classes from the model

    Returns:
        Dictionary mapping class indices to sentiment labels
    """
    return SENTIMENT_MAPPINGS.get(
        num_classes, {i: f"Class_{i}" for i in range(num_classes)}
    )


def get_sentiment_colors() -> Dict[str, str]:
    """
    Get color-coded sentiment display mapping.

    Returns:
        Dictionary mapping sentiment labels to emoji indicators
    """
    return {
        "Positive": "🟢",
        "Negative": "🔴",
        "Neutral": "🟡",
        "Angry": "🔴",
        "Sad": "🔵",
        "Happy": "🟢",
        "Fear": "🟣",
        "Disgust": "🟠",
        "Surprise": "🟡",
    }



def to_three_way_sentiment(sentiment: str) -> str:
    """Normalize emotion labels into Negative, Neutral, or Positive."""
    positive = {"positive", "happy", "surprise", "surprised"}
    negative = {"negative", "angry", "sad", "fear", "fearful", "disgust"}
    normalized = (sentiment or "").strip().lower()

    if normalized in positive:
        return "Positive"
    if normalized in negative:
        return "Negative"
    return "Neutral"

def format_sentiment_result(
    sentiment: str, confidence: float, input_info: str = "", model_name: str = ""
) -> str:
    """
    Format sentiment analysis result for display.

    Args:
        sentiment: Predicted sentiment label
        confidence: Confidence score
        input_info: Information about the input
        model_name: Name of the model used

    Returns:
        Formatted result string
    """
    colors = get_sentiment_colors()
    emoji = colors.get(sentiment, "❓")

    result = f"{emoji} Sentiment: {sentiment}\n"
    result += f"Confidence: {confidence:.2f}\n"

    if input_info:
        result += f"Input: {input_info}\n"

    if model_name:
        result += f"Model: {model_name}\n"

    return result

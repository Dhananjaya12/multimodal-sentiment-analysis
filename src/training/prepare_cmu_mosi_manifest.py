"""Prepare a CMU-MOSI Kaggle manifest for this project.

This keeps only the useful notebook logic:
- parse segmented transcript files
- read sentiment scores from mosi_data.pkl
- align segment_id with transcript/audio/video files
- map continuous MOSI sentiment scores to Negative/Neutral/Positive
"""

import argparse
import csv
import pickle
from pathlib import Path
from typing import Dict, Iterable, List


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def prepare_manifest(
    raw_root: Path,
    label_pkl: Path,
    output: Path,
    neutral_threshold: float,
):
    transcripts = _load_transcripts(raw_root / "Transcript" / "Segmented")
    audio_paths = _index_files(raw_root / "Audio" / "WAV_16000" / "Segmented", (".wav",))
    video_paths = _index_files(raw_root / "Video" / "Segmented", VIDEO_EXTENSIONS)
    labels = _load_labels(label_pkl)

    rows = []
    for item in labels:
        segment_id = item["segment_id"]
        score = item["sentiment_score"]
        rows.append(
            {
                "segment_id": segment_id,
                "video_id": segment_id.rsplit("_", 1)[0],
                "text": transcripts.get(segment_id, ""),
                "audio_path": str(audio_paths.get(segment_id, "")),
                "video_path": str(video_paths.get(segment_id, "")),
                "sentiment_score": score,
                "label": _score_to_label(score, neutral_threshold),
                "split": item["split"],
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "segment_id",
                "video_id",
                "text",
                "audio_path",
                "video_path",
                "sentiment_score",
                "label",
                "split",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    _print_summary(rows, transcripts, audio_paths, video_paths)
    print(f"saved_manifest={output}")


def _load_transcripts(transcript_dir: Path) -> Dict[str, str]:
    transcripts = {}
    for path in sorted(transcript_dir.glob("*.annotprocessed")):
        video_id = path.stem
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        fallback_index = 1
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if "_DELIM_" in line:
                segment_index, text = line.split("_DELIM_", 1)
                segment_index = segment_index.strip()
                text = text.strip()
            else:
                segment_index = str(fallback_index)
                text = line
            transcripts[f"{video_id}_{segment_index}"] = text
            fallback_index += 1
    return transcripts


def _index_files(directory: Path, extensions: Iterable[str]) -> Dict[str, Path]:
    if not directory.exists():
        return {}
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in extensions
    }


def _load_labels(label_pkl: Path) -> List[dict]:
    with label_pkl.open("rb") as file:
        data = pickle.load(file)

    rows = []
    for split in ["train", "valid", "test"]:
        ids = data[split]["id"]
        labels = data[split]["labels"]
        for index in range(len(ids)):
            segment_id = ids[index][0]
            if isinstance(segment_id, bytes):
                segment_id = segment_id.decode("utf-8")
            rows.append(
                {
                    "segment_id": str(segment_id),
                    "sentiment_score": float(labels[index][0]),
                    "split": split,
                }
            )
    return rows


def _score_to_label(score: float, neutral_threshold: float) -> str:
    if score > neutral_threshold:
        return "Positive"
    if score < -neutral_threshold:
        return "Negative"
    return "Neutral"


def _print_summary(rows, transcripts, audio_paths, video_paths):
    labels = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for row in rows:
        labels[row["label"]] += 1
    print(f"rows={len(rows)}")
    print(f"transcripts_indexed={len(transcripts)}")
    print(f"audio_files_indexed={len(audio_paths)}")
    print(f"video_files_indexed={len(video_paths)}")
    print(f"with_text={sum(bool(row['text']) for row in rows)}")
    print(f"with_audio={sum(bool(row['audio_path']) for row in rows)}")
    print(f"with_video={sum(bool(row['video_path']) for row in rows)}")
    print(f"labels={labels}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--label-pkl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--neutral-threshold", type=float, default=0.5)
    args = parser.parse_args()
    prepare_manifest(args.raw_root, args.label_pkl, args.output, args.neutral_threshold)


if __name__ == "__main__":
    main()
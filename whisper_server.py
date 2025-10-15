"""Flask-based local Whisper transcription server.

This server exposes a single endpoint ``/transcribe`` that accepts audio
files, runs them through the local Whisper ``base`` model, stores the
results, and returns a detailed JSON response compatible with the user's
Telegram bot workflow.
"""

from __future__ import annotations
import math
import os
from pathlib import Path
from typing import Dict, List

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

import whisper


# Configuration constants
AUDIO_DIR = Path("audio")
TRANSCRIPTS_DIR = Path("transcripts")
ALLOWED_EXTENSIONS = {".ogg", ".mp3", ".wav"}
DEFAULT_LANGUAGE = "ru"
MODEL_NAME = "base"

# Ensure required directories exist at startup.
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# Load the Whisper model once when the server starts to avoid repeated
# initialization overhead per request.
MODEL = whisper.load_model(MODEL_NAME)

app = Flask(__name__)


def allowed_file(filename: str) -> bool:
    """Return True if the file has an allowed extension."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def compute_confidence(segments: List[Dict]) -> float:
    """Estimate confidence from segment average log probabilities."""
    probabilities = []
    for segment in segments:
        avg_logprob = segment.get("avg_logprob")
        if avg_logprob is None:
            continue
        # Whisper reports log probabilities in natural log; convert to [0, 1].
        probabilities.append(math.exp(avg_logprob))

    if not probabilities:
        return 1.0

    # Clamp confidence to [0, 1] to avoid floating point quirks.
    confidence = sum(probabilities) / len(probabilities)
    return max(0.0, min(1.0, confidence))


def build_response(
    file_name: str,
    segments: List[Dict],
    full_text: str,
    language: str,
) -> Dict:
    """Build the response payload with metadata and segments."""
    transcript_file = TRANSCRIPTS_DIR / (Path(file_name).stem + ".txt")

    duration = 0.0
    if segments:
        duration = float(segments[-1].get("end", 0.0))

    response = {
        "meta": {
            "file_name": file_name,
            "audio_path": str(AUDIO_DIR / file_name).replace(os.sep, "/"),
            "transcript_path": str(transcript_file).replace(os.sep, "/"),
            "language": language,
            "duration": duration,
            "confidence": compute_confidence(segments),
        },
        "segments": [
            {
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", 0.0)),
                "text": segment.get("text", "").strip(),
            }
            for segment in segments
        ],
        "full_text": full_text,
    }
    return response


@app.route("/transcribe", methods=["POST"])
def transcribe() -> tuple:
    """Handle audio transcription requests."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    upload = request.files["file"]
    if upload.filename == "":
        return jsonify({"error": "No file selected for uploading."}), 400

    filename = secure_filename(upload.filename)
    if not allowed_file(filename):
        return (
            jsonify({"error": "Unsupported file format. Use .ogg, .mp3, or .wav."}),
            400,
        )

    audio_path = AUDIO_DIR / filename

    try:
        upload.save(audio_path)
    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": f"Failed to save audio file: {exc}"}), 500

    try:
        # Perform transcription with Whisper.
        result = MODEL.transcribe(str(audio_path), language=DEFAULT_LANGUAGE)
        segments = result.get("segments", [])
        full_text = result.get("text", "").strip()
        language = result.get("language", DEFAULT_LANGUAGE)

        # Persist transcript text using the same base name.
        transcript_file = TRANSCRIPTS_DIR / (Path(filename).stem + ".txt")
        with transcript_file.open("w", encoding="utf-8") as txt_file:
            txt_file.write(full_text)

        response_payload = build_response(filename, segments, full_text, language)
        return jsonify(response_payload), 200

    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"error": f"Transcription failed: {exc}"}), 500

    finally:
        # Whisper may create temporary files in some scenarios; ensure they are removed.
        temp_files = [
            path for path in AUDIO_DIR.glob("*.tmp") if path.is_file()
        ]
        for temp_file in temp_files:
            try:
                temp_file.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000)

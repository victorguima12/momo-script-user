"""
Transcription API clients for OpenAI Whisper and ElevenLabs Speech-to-Text.
"""

import io
import logging
import re
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Ordered list — multi-word patterns first so they match before single-word ones
WORD_TO_SYMBOL = [
    (r"\bponto de exclamação\b", "!"),
    (r"\bponto de interrogação\b", "?"),
    (r"\bponto final\b", "."),
    (r"\breticências\b", "..."),
    (r"\basteriscos?\b", "*"),
    (r"\bvírgula\b", ","),
    (r"\bponto\b", "."),
]


def _apply_replacements(text: str) -> str:
    for pattern, symbol in WORD_TO_SYMBOL:
        # Match optional comma/space before and after the word
        full_pattern = r",?\s*" + pattern + r"\s*,?"
        text = re.sub(full_pattern, symbol, text, flags=re.IGNORECASE)
    # Whisper often adds a period next to asterisks (e.g. ".* " or " *.") — keep only the *
    text = re.sub(r'\.\*', '*', text)
    text = re.sub(r'\*\.', '*', text)
    return text


def transcribe_openai(wav_bytes: bytes, api_key: str, language: str = "pt") -> str:
    """Transcribe audio using OpenAI Whisper API. Returns transcribed text."""
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": ("audio.wav", io.BytesIO(wav_bytes), "audio/wav")}
    data = {"model": "whisper-1", "language": language}

    resp = requests.post(url, headers=headers, files=files, data=data, timeout=60)
    resp.raise_for_status()
    return _apply_replacements(resp.json().get("text", ""))


def transcribe_elevenlabs(wav_bytes: bytes, api_key: str, language: str = "pt") -> str:
    """Transcribe audio using ElevenLabs Speech-to-Text API (Scribe v2)."""
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {"xi-api-key": api_key}
    files = {"file": ("audio.wav", io.BytesIO(wav_bytes), "audio/wav")}
    data = {
        "model_id": "scribe_v2",
        "language_code": language,
        "tag_audio_events": "false",
    }

    resp = requests.post(url, headers=headers, files=files, data=data, timeout=60)
    resp.raise_for_status()
    result = resp.json()
    return _apply_replacements(result.get("text", ""))

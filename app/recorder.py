"""
Audio recording using sounddevice.
"""

import io
import threading
import wave
from typing import List, Optional, Tuple

import numpy as np
import sounddevice as sd


def list_input_devices() -> List[Tuple[int, str]]:
    """Return list of (device_index, device_name) for input devices."""
    devices = sd.query_devices()
    result = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            result.append((i, d["name"]))
    return result


class AudioRecorder:
    """Records audio from a selected input device into a WAV buffer."""

    SAMPLE_RATE = 16000
    CHANNELS = 1
    DTYPE = "int16"

    def __init__(self):
        self._frames: List[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self._recording = False
        self._device_index: Optional[int] = None
        # Split markers: list of (frame_index, separator_char)
        self._markers: List[Tuple[int, str]] = []

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def has_markers(self) -> bool:
        return len(self._markers) > 0

    def set_device(self, device_index: int):
        self._device_index = device_index

    def add_marker(self, separator: str):
        """Record current frame position as a split point with given separator."""
        if self._recording:
            self._markers.append((len(self._frames), separator))

    def start(self):
        if self._recording:
            return
        self._frames.clear()
        self._markers.clear()
        self._recording = True
        self._stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype=self.DTYPE,
            device=self._device_index,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> bytes:
        """Stop recording and return WAV file bytes (single chunk, no markers)."""
        if not self._recording:
            return b""
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._frames:
            return b""

        audio = np.concatenate(self._frames, axis=0)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(audio.tobytes())
        return buf.getvalue()

    def stop_segments(self) -> List[Tuple[bytes, str]]:
        """Stop recording and return audio split at markers.

        Returns list of (wav_bytes, separator_after).
        The last segment has separator_after = "" (end of recording).
        """
        if not self._recording:
            return []
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._frames:
            return []

        # Build split points from markers
        split_indices = [(m[0], m[1]) for m in self._markers]

        # Slice frames into segments
        segments: List[Tuple[bytes, str]] = []
        prev = 0
        for frame_idx, sep in split_indices:
            chunk_frames = self._frames[prev:frame_idx]
            if chunk_frames:
                segments.append((self._frames_to_wav(chunk_frames), sep))
            prev = frame_idx

        # Final segment (after last marker to end)
        chunk_frames = self._frames[prev:]
        if chunk_frames:
            segments.append((self._frames_to_wav(chunk_frames), ""))

        return segments

    def _frames_to_wav(self, frames: List[np.ndarray]) -> bytes:
        """Convert a list of audio frames to WAV bytes."""
        audio = np.concatenate(frames, axis=0)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(audio.tobytes())
        return buf.getvalue()

    def _audio_callback(self, indata, frames, time_info, status):
        if self._recording:
            self._frames.append(indata.copy())

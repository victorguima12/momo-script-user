"""
Transcription Widget - Record audio, transcribe via API, load text into panels.
"""

import logging
from typing import Optional

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QKeySequence

from app.recorder import AudioRecorder, list_input_devices
from app.transcription import transcribe_openai, transcribe_elevenlabs
from app.config import Config
from ui.scale_manager import scale_manager

logger = logging.getLogger(__name__)


class TranscriptionWorker(QThread):
    """Background thread for API transcription."""
    finished = pyqtSignal(str)   # transcribed text
    error = pyqtSignal(str)      # error message
    segment_progress = pyqtSignal(int, int)  # (current, total)

    def __init__(self, wav_bytes: bytes, api: str, api_key: str, language: str,
                 segments=None):
        """
        Args:
            wav_bytes: single WAV buffer (used when no segments)
            segments: list of (wav_bytes, separator_after) from recorder.stop_segments()
        """
        super().__init__()
        self._wav_bytes = wav_bytes
        self._segments = segments  # list of (wav_bytes, sep) or None
        self._api = api
        self._api_key = api_key
        self._language = language

    def _transcribe_one(self, wav: bytes) -> str:
        if self._api == "elevenlabs":
            return transcribe_elevenlabs(wav, self._api_key, self._language)
        return transcribe_openai(wav, self._api_key, self._language)

    def run(self):
        try:
            if self._segments:
                parts = []
                total = len(self._segments)
                for i, (wav, sep) in enumerate(self._segments):
                    self.segment_progress.emit(i + 1, total)
                    text = self._transcribe_one(wav).strip()
                    if text:
                        parts.append(text + sep)
                    elif sep:
                        # Empty transcription but separator exists - still keep it
                        if parts:
                            parts[-1] = parts[-1].rstrip() + sep
                result = "".join(parts)
            else:
                result = self._transcribe_one(self._wav_bytes)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class TranscriptionWidget(QWidget):
    """Widget for recording audio and transcribing it.
    Emits apply_text(str) when user wants to load text into panels."""

    apply_text = pyqtSignal(str)  # emitted with the text to apply

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self.recorder = AudioRecorder()
        self._worker: Optional[TranscriptionWorker] = None

        self._setup_ui()
        self._refresh_devices()

    def _setup_ui(self):
        s = scale_manager
        layout = QVBoxLayout()
        layout.setContentsMargins(s.scale(4), s.scale(4), s.scale(4), s.scale(4))
        layout.setSpacing(s.scale(4))
        self.setLayout(layout)

        # -- Row 1: Mic selector + API selector --
        row1 = QHBoxLayout()
        row1.setSpacing(s.scale(4))

        row1.addWidget(QLabel("Mic:"))
        self.mic_combo = QComboBox()
        self.mic_combo.setMinimumWidth(s.scale(120))
        self.mic_combo.setStyleSheet("QComboBox { background: #404040; color: #fff; }")
        row1.addWidget(self.mic_combo, 1)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedWidth(s.scale(55))
        refresh_btn.clicked.connect(self._refresh_devices)
        row1.addWidget(refresh_btn)

        row1.addSpacing(s.scale(8))

        row1.addWidget(QLabel("API:"))
        self.api_combo = QComboBox()
        self.api_combo.addItems(["OpenAI", "ElevenLabs"])
        saved_api = self.config.get("transcription_api", "openai")
        self.api_combo.setCurrentIndex(1 if saved_api == "elevenlabs" else 0)
        self.api_combo.currentIndexChanged.connect(self._on_api_changed)
        self.api_combo.setStyleSheet("QComboBox { background: #404040; color: #fff; }")
        row1.addWidget(self.api_combo)

        row1.addSpacing(s.scale(4))

        row1.addWidget(QLabel("Lang:"))
        self.lang_edit = QLineEdit()
        self.lang_edit.setText(self.config.get("transcription_language", "pt"))
        self.lang_edit.setFixedWidth(s.scale(35))
        self.lang_edit.setStyleSheet("QLineEdit { background: #404040; color: #fff; }")
        self.lang_edit.textChanged.connect(
            lambda t: self.config.set("transcription_language", t)
        )
        row1.addWidget(self.lang_edit)

        key_btn = QPushButton("API Key")
        key_btn.setFixedWidth(s.scale(55))
        key_btn.clicked.connect(self._set_api_key)
        row1.addWidget(key_btn)

        layout.addLayout(row1)

        # -- Row 2: Record button + status --
        row2 = QHBoxLayout()
        row2.setSpacing(s.scale(4))

        self.record_btn = QPushButton("Record (R)  |  W = .  |  E = *")
        self.record_btn.setCheckable(True)
        self.record_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #d32f2f; color: white;
                font-weight: bold; padding: {s.scale(6)}px {s.scale(14)}px;
                border-radius: {s.scale(4)}px; font-size: {s.scale_font(12)}px;
            }}
            QPushButton:checked {{
                background-color: #ff5252;
                border: 2px solid #ffcdd2;
            }}
            QPushButton:hover {{ background-color: #e53935; }}
        """)
        self.record_btn.clicked.connect(self._toggle_recording)
        row2.addWidget(self.record_btn)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(f"color: #888; font-size: {s.scale_font(11)}px;")
        row2.addWidget(self.status_label, 1)

        layout.addLayout(row2)

        # -- Text loader --
        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText(
            "Transcribed text appears here. Edit before applying.\n"
            "R = record | W = panel break (.) | E = merge (*)"
        )
        self.text_edit.setMinimumHeight(s.scale(60))
        self.text_edit.setMaximumHeight(s.scale(120))
        self.text_edit.setStyleSheet(f"""
            QPlainTextEdit {{
                background-color: #2a2a2a; color: #ffffff;
                border: 1px solid #444; border-radius: {s.scale(3)}px;
                padding: {s.scale(4)}px; font-size: {s.scale_font(12)}px;
            }}
            QPlainTextEdit:focus {{ border: 1px solid #00bcd4; }}
        """)
        layout.addWidget(self.text_edit)

        # -- Apply button --
        self.apply_btn = QPushButton("Apply to Panels")
        self.apply_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #4caf50; color: white;
                font-weight: bold; padding: {s.scale(6)}px {s.scale(14)}px;
                border-radius: {s.scale(4)}px; font-size: {s.scale_font(12)}px;
            }}
            QPushButton:hover {{ background-color: #43a047; }}
        """)
        self.apply_btn.clicked.connect(self._on_apply)
        layout.addWidget(self.apply_btn)

    def _refresh_devices(self):
        self.mic_combo.clear()
        try:
            devices = list_input_devices()
            for idx, name in devices:
                self.mic_combo.addItem(name, idx)
        except Exception as e:
            logger.warning(f"Failed to list audio devices: {e}")
            self.mic_combo.addItem("(no devices found)")

    def _on_api_changed(self):
        api = "elevenlabs" if self.api_combo.currentIndex() == 1 else "openai"
        self.config.set("transcription_api", api)

    def _set_api_key(self):
        api = "elevenlabs" if self.api_combo.currentIndex() == 1 else "openai"
        api_name = "ElevenLabs" if api == "elevenlabs" else "OpenAI"
        current = self.config.get(f"api_keys.{api}", "")

        key, ok = QInputDialog.getText(
            self, f"{api_name} API Key",
            f"Enter your {api_name} API key:",
            QLineEdit.Password,
            current,
        )
        if ok and key.strip():
            self.config.set(f"api_keys.{api}", key.strip())
            self.status_label.setText(f"{api_name} key saved")

    def _toggle_recording(self):
        if self.recorder.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        device_data = self.mic_combo.currentData()
        if device_data is not None:
            self.recorder.set_device(device_data)

        try:
            self.recorder.start()
        except Exception as e:
            self.record_btn.setChecked(False)
            QMessageBox.warning(self, "Recording Error", f"Failed to start recording:\n{e}")
            return

        self.record_btn.setChecked(True)
        self.record_btn.setText("Stop (R)  |  W = .  |  E = *")
        self.status_label.setText("Recording...")
        self.status_label.setStyleSheet(f"color: #ff5252; font-size: {scale_manager.scale_font(11)}px; font-weight: bold;")

    def _stop_recording(self):
        has_markers = self.recorder.has_markers
        if has_markers:
            segments = self.recorder.stop_segments()
        else:
            wav_bytes = self.recorder.stop()
            segments = None

        self._marker_count = 0
        self.record_btn.setChecked(False)
        self.record_btn.setText("Record (R)  |  W = .  |  E = *")
        self.status_label.setStyleSheet(f"color: #888; font-size: {scale_manager.scale_font(11)}px; font-weight: normal;")

        if has_markers and not segments:
            self.status_label.setText("No audio captured")
            return
        if not has_markers and not wav_bytes:
            self.status_label.setText("No audio captured")
            return

        # Get API key
        api = "elevenlabs" if self.api_combo.currentIndex() == 1 else "openai"
        api_key = self.config.get(f"api_keys.{api}", "")
        if not api_key:
            api_name = "ElevenLabs" if api == "elevenlabs" else "OpenAI"
            QMessageBox.warning(
                self, "Missing API Key",
                f"No {api_name} API key configured.\nClick 'API Key' to set one."
            )
            return

        language = self.lang_edit.text().strip() or "pt"

        # Transcribe in background
        if has_markers:
            self.status_label.setText(f"Transcribing {len(segments)} segments...")
            self.record_btn.setEnabled(False)
            self.apply_btn.setEnabled(False)
            self._worker = TranscriptionWorker(b"", api, api_key, language, segments=segments)
        else:
            self.status_label.setText("Transcribing...")
            self.record_btn.setEnabled(False)
            self.apply_btn.setEnabled(False)
            self._worker = TranscriptionWorker(wav_bytes, api, api_key, language)

        self._worker.segment_progress.connect(self._on_segment_progress)
        self._worker.finished.connect(self._on_transcription_done)
        self._worker.error.connect(self._on_transcription_error)
        self._worker.start()

    def _on_transcription_done(self, text: str):
        self.record_btn.setEnabled(True)
        self.apply_btn.setEnabled(True)
        self._worker = None

        if text:
            # Append to existing text
            current = self.text_edit.toPlainText()
            if current and not current.endswith("\n"):
                current += " "
            self.text_edit.setPlainText(current + text)
            self.status_label.setText("Transcription complete")
        else:
            self.status_label.setText("No text returned")

    def _on_transcription_error(self, error: str):
        self.record_btn.setEnabled(True)
        self.apply_btn.setEnabled(True)
        self._worker = None
        self.status_label.setText("Transcription failed")
        QMessageBox.warning(self, "Transcription Error", f"Failed to transcribe:\n{error}")

    def _on_apply(self):
        text = self.text_edit.toPlainText().strip()
        if text:
            self.apply_text.emit(text)

    def _on_segment_progress(self, current: int, total: int):
        self.status_label.setText(f"Transcribing segment {current}/{total}...")

    def handle_key_r(self):
        """Called externally when R is pressed (from ScriptPanel)."""
        self._toggle_recording()

    def insert_separator(self, sep: str):
        """Insert a separator (. or *). If recording, marks the audio; otherwise inserts into text."""
        if self.recorder.is_recording:
            self.recorder.add_marker(sep)
            self._marker_count = getattr(self, '_marker_count', 0) + 1
            label = "panel break" if sep == "." else "merge"
            self.status_label.setText(f"Recording... marker #{self._marker_count} ({label})")
        else:
            cursor = self.text_edit.textCursor()
            cursor.movePosition(cursor.End)
            self.text_edit.setTextCursor(cursor)
            self.text_edit.insertPlainText(sep)

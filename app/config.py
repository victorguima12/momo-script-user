"""
Momo Script - Application Configuration
"""

import json
from pathlib import Path
from typing import Any, Dict


class Config:
    """Persistent configuration manager"""

    def __init__(self):
        self.config_dir = Path.home() / ".momo_script"
        self.config_file = self.config_dir / "config.json"
        self.default_config = {
            "last_download_directory": "",
            "last_load_directory": "",
            "yolo_confidence": 25,
            "yolo_min_size": 40,
            "window_geometry": {
                "width": 1400,
                "height": 900,
                "x": 100,
                "y": 100,
            },
            "api_keys": {
                "openai": "",
                "elevenlabs": "",
                "fish_audio": "",
                "xai": "",
                "google": "",
            },
            "gemini_model": "gemini-3-flash-preview",
            "gemini_expand_pct": 50,
            "grok_model": "grok-4.20-0309-non-reasoning",
            "grok_collection_id": "collection_38b89831-fb0d-4910-abbf-ef36722edb46",
            "transcription_api": "openai",
            "transcription_language": "pt",
        }
        self.config = self._load()

    def _load(self) -> Dict[str, Any]:
        try:
            if self.config_file.exists():
                with open(self.config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                merged = self.default_config.copy()
                merged.update(config)
                return merged
            else:
                self.config_dir.mkdir(exist_ok=True)
                return self.default_config.copy()
        except Exception:
            return self.default_config.copy()

    def save(self):
        try:
            self.config_dir.mkdir(exist_ok=True)
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)
        except Exception:
            pass

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        keys = key.split(".")
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self.save()

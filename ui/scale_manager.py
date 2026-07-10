"""
UI Scale Manager for Momo Script.
Auto-detects screen resolution and provides scaling methods.
"""

from typing import Optional
from PyQt5.QtCore import QObject, pyqtSignal


class ScaleManager(QObject):
    """Singleton to manage UI scaling across the application."""

    scale_changed = pyqtSignal(float)

    _instance: Optional['ScaleManager'] = None
    _initialized = False

    def __new__(cls) -> 'ScaleManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if ScaleManager._initialized:
            return
        super().__init__()
        self._scale_factor = 1.0
        ScaleManager._initialized = True

    @property
    def scale_factor(self) -> float:
        return self._scale_factor

    @scale_factor.setter
    def scale_factor(self, value: float) -> None:
        if value != self._scale_factor:
            self._scale_factor = max(0.5, min(4.0, value))
            self.scale_changed.emit(self._scale_factor)

    def scale(self, value: int) -> int:
        return int(value * self._scale_factor)

    def sf(self, value: int) -> int:
        """Alias for scale() — shorter for stylesheet use."""
        return self.scale(value)

    def scale_font(self, base_size: int) -> int:
        return max(8, int(base_size * self._scale_factor))


# Global instance
scale_manager = ScaleManager()

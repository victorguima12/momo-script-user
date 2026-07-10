"""
Undo/Redo manager for panel text state.
Stores snapshots of all panel texts (up to MAX_STEPS).
"""

import copy
from collections import deque
from typing import Any, Dict, List, Optional


MAX_STEPS = 50


class UndoManager:
    """Manages undo/redo for a dict-based state snapshot."""

    def __init__(self, max_steps: int = MAX_STEPS):
        self._undo_stack: deque = deque(maxlen=max_steps)
        self._redo_stack: List = []

    def push(self, snapshot: Any):
        """Save a new snapshot. Clears the redo stack."""
        self._undo_stack.append(copy.deepcopy(snapshot))
        self._redo_stack.clear()

    def undo(self) -> Optional[Any]:
        """Pop and return the previous snapshot, pushing current to redo.
        Returns None if nothing to undo."""
        if len(self._undo_stack) < 2:
            return None
        # Current state is top of stack — move it to redo
        current = self._undo_stack.pop()
        self._redo_stack.append(current)
        # Return the previous state (now on top)
        return copy.deepcopy(self._undo_stack[-1])

    def redo(self) -> Optional[Any]:
        """Return the next redo snapshot, or None."""
        if not self._redo_stack:
            return None
        snapshot = self._redo_stack.pop()
        self._undo_stack.append(snapshot)
        return copy.deepcopy(snapshot)

    def can_undo(self) -> bool:
        return len(self._undo_stack) >= 2

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def clear(self):
        self._undo_stack.clear()
        self._redo_stack.clear()

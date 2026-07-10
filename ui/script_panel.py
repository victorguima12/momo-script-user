"""
Script Panel - Manhua reader on the left, synchronized text slots on the right.
YOLO detects panels in the image strip; each panel maps to a text slot.
Manual box creation by click-drag; boxes are always sorted top-to-bottom.
"""

import html
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from app.config import Config
from app.edition import IS_USER
if IS_USER:
    # Admin-only modules don't ship in the user edition; the AI-writing UI
    # that would use them is hidden below (search for IS_USER).
    ClaudeBridge = GeminiWriter = GPTEditor = None
    _load_default_ai_prompt = _load_default_gpt_prompt = None
else:
    from app.claude_bridge import ClaudeBridge
    from app.gemini_writer import GeminiWriter, _load_system_prompt as _load_default_ai_prompt
    from app.gpt_editor import GPTEditor, _load_gpt_prompt as _load_default_gpt_prompt
from app.undo import UndoManager
from ui.scale_manager import scale_manager
from yolo.detector import YoloDetector

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


class _ResponsivePadding(QObject):
    """Keeps a QPushButton's text fully visible as the window is squeezed.

    The stylesheet's vertical padding stays constant while the horizontal
    padding shrinks from ``max_hpad`` down to ``min_hpad`` as the widget
    is forced narrower. The button's minimum width is pinned to
    ``text + 2*min_hpad`` so the label never clips — it just ends up with
    the border closer to the text when space is tight.

    Apply to a button after you have set its base stylesheet (WITHOUT the
    ``padding:`` line — this filter owns horizontal padding):

        self.yolo_btn = QPushButton("Run YOLO")
        self.yolo_btn.setStyleSheet("QPushButton { background: #0af; ... }")
        _ResponsivePadding(self.yolo_btn, vpad=8, max_hpad=16, min_hpad=4)
    """

    def __init__(self, btn: QPushButton, vpad: int, max_hpad: int, min_hpad: int):
        super().__init__(btn)
        self._btn = btn
        self._vpad = vpad
        self._max_hpad = max_hpad
        self._min_hpad = min_hpad
        self._base_style = btn.styleSheet()
        self._current = -1
        self._updating = False
        # Pin minimum width = text + minimum padding (text never clips).
        fm = QFontMetrics(btn.font())
        if "font-weight: bold" in self._base_style:
            bold_font = btn.font()
            bold_font.setBold(True)
            fm = QFontMetrics(bold_font)
        self._text_w = fm.horizontalAdvance(btn.text())
        btn.setMinimumWidth(self._text_w + 2 * min_hpad + 4)
        btn.installEventFilter(self)
        # Apply initial full padding.
        self._apply(max_hpad)

    def _apply(self, hpad: int):
        if hpad == self._current or self._updating:
            return
        self._updating = True
        try:
            self._current = hpad
            self._btn.setStyleSheet(
                self._base_style
                + f"\nQPushButton {{ padding: {self._vpad}px {hpad}px; }}"
            )
        finally:
            self._updating = False

    def eventFilter(self, obj, event):
        if (
            obj is self._btn
            and event.type() == QEvent.Resize
            and not self._updating
        ):
            avail = self._btn.width() - self._text_w - 4
            hpad = max(self._min_hpad, min(self._max_hpad, avail // 2))
            self._apply(hpad)
        return False


@dataclass
class ScriptBox:
    """A YOLO-detected or manually created box linked to a text slot."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    image_index: int = 0
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    confidence: float = 1.0
    text: str = ""


# ---------------------------------------------------------------------------
# Image Strip Widget (left panel)
# ---------------------------------------------------------------------------

class ImageStripWidget(QWidget):
    """Renders all manhua images as a vertical strip with YOLO box overlays.
    Supports click-drag to create new boxes and right-click to delete."""

    boxes_changed = pyqtSignal()
    box_deleting = pyqtSignal(str)  # box id — emitted BEFORE removal so text can be rescued
    box_selected = pyqtSignal(str)  # box id
    batch_changed = pyqtSignal()  # emitted when batch is switched

    # Resize handle size
    HANDLE = 8
    # Dynamic batching: close a batch once cumulative original height hits this,
    # with a hard upper cap on image count so ultra-short pages don't balloon.
    BATCH_HEIGHT_TARGET = 250000  # original-image pixels of vertical content per batch
    BATCH_MAX_ITEMS = 200        # cap even if heights are tiny
    BATCH_MIN_ITEMS = 1          # safety floor
    # Drag must overshoot a seam by at least this many display pixels for auto-merge
    SEAM_CROSS_THRESHOLD = 20
    # Mouse within this many px of a seam triggers the hover merge button
    SEAM_HOVER_RADIUS = 14
    MERGE_BTN_SIZE = 22
    # Qt's raster engine silently produces BLACK pixels past row 32767 when
    # scaling or compositing (16-bit coordinate limit). Any image whose
    # display height exceeds MAX_DISPLAY_STRIP is scaled/painted as separate
    # vertical strips that each stay under the limit. Strip boundaries are
    # placed inside content-free bands (uniform color, e.g. panel gutters)
    # found within STRIP_SEAM_WINDOW source px of the ideal split point.
    MAX_DISPLAY_STRIP = 28000    # display px per strip (safe margin under 32767)
    STRIP_MAX_DISPLAY = 31500    # hard per-strip cap after seam drift
    STRIP_SEAM_WINDOW = 4000     # source px search radius for a clean band
    STRIP_SEAM_MIN_BAND = 6      # source px of uniform rows for a safe seam

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

        # Image data (current batch)
        self.image_paths: List[str] = []
        self._source_pixmaps: List[QPixmap] = []  # loaded once from disk
        # Per image: list of (y_offset_within_image, strip_pixmap) scaled for
        # the current display width. Normal images have a single strip; images
        # taller than MAX_DISPLAY_STRIP display px are split (Qt blanks pixels
        # past row 32767 when scaling, so one tall pixmap would render black).
        self.display_pixmaps: List[List[Tuple[int, QPixmap]]] = []
        self.display_heights: List[int] = []  # total display height per image
        self._uniform_rows_cache: Dict[str, Optional[np.ndarray]] = {}
        self.original_sizes: List[Tuple[int, int]] = []  # (w, h) original
        self.image_offsets: List[int] = []  # cumulative Y offset per image
        self.scale_factors: List[float] = []  # display_w / original_w
        self.display_width: int = 600
        self._cached_display_width: int = 0  # width the display pixmaps were built for

        # Batch splitting state
        self.all_image_paths: List[str] = []  # all images in folder
        self.batches: List[List[str]] = []  # split into chunks (variable size)
        self.batch_starts: List[int] = [0]   # global start index for each batch
        self.current_batch_index: int = 0
        # Project root used to compute relative paths for the merged-image
        # cache key. Set by load_images / load_multiple_folders. Empty
        # string means "no project context yet" — _build_merged_image
        # falls back to absolute paths in that case.
        self._image_folder: str = ""

        # Box data (ALL boxes across all batches, using global image_index)
        self.boxes: List[ScriptBox] = []
        self.selected_box_id: Optional[str] = None

        # Interaction state
        self._drag_mode: Optional[str] = None  # "create", "move", "resize"
        self._drag_start: Optional[QPoint] = None
        self._drag_box: Optional[ScriptBox] = None
        self._drag_handle: Optional[str] = None  # which resize handle
        self._drag_origin_rect: Optional[QRect] = None  # original box rect before drag
        self._create_rect: Optional[QRect] = None  # rectangle being drawn

        # Merged-image tracking
        # merge_groups: list of merges, each = list of ABSOLUTE original paths that got
        # stitched into a single cached JPEG. In-memory state only uses abs paths;
        # project save/load serializes to basenames.
        self.merge_groups: List[List[str]] = []

        # Seam UI state
        self._hover_seam: Optional[int] = None  # local idx whose TOP seam is hovered (merges idx-1 and idx)
        self._hover_merge_btn_rect: Optional[QRect] = None
        self._active_cross_seams: List[int] = []  # local seam indices highlighted during create drag

        # Debounce timer for rebuilds
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(50)
        self._rebuild_timer.timeout.connect(self._do_rebuild)

        self.setMinimumWidth(200)

        # Dark background so Qt doesn't flash white on repaint
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(30, 30, 30))
        self.setPalette(pal)

    # -- Batch helpers ---

    @property
    def _batch_offset(self) -> int:
        """Global index of the first image in the current batch."""
        if not self.batch_starts:
            return 0
        idx = max(0, min(self.current_batch_index, len(self.batch_starts) - 1))
        return self.batch_starts[idx]

    def _global_to_batch(self, global_idx: int) -> int:
        """Return the batch index containing the given global image index."""
        if not self.batch_starts:
            return 0
        import bisect
        i = bisect.bisect_right(self.batch_starts, global_idx) - 1
        return max(0, min(i, len(self.batches) - 1))

    @staticmethod
    def _fast_image_height(path: str) -> int:
        """Read image height from header only — no pixel decode. Returns 0 on failure."""
        try:
            reader = QImageReader(path)
            sz = reader.size()
            if sz.isValid():
                return sz.height()
        except Exception:
            pass
        return 0

    def _global_to_local(self, global_idx: int) -> int:
        """Convert global image index to batch-local index."""
        return global_idx - self._batch_offset

    def _local_to_global(self, local_idx: int) -> int:
        """Convert batch-local image index to global index."""
        return local_idx + self._batch_offset

    def current_batch_boxes(self) -> List[ScriptBox]:
        """Return only boxes whose image_index falls in the current batch."""
        start = self._batch_offset
        end = start + len(self.image_paths)
        return [b for b in self.boxes if start <= b.image_index < end]

    # -- Image loading ---

    @staticmethod
    def _collect_images(folder: str) -> list:
        """Collect images from folder. If it contains subfolders with images
        (chapter-per-folder layout), walk them in sorted order."""
        import re

        def _natural_sort_key(s: str):
            return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

        # Check for images directly in the folder
        root_files = sorted(
            (f for f in os.listdir(folder)
             if os.path.isfile(os.path.join(folder, f))
             and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS),
            key=_natural_sort_key,
        )

        # Check for subfolders that contain images (chapter folders)
        subdirs = sorted(
            (d for d in os.listdir(folder)
             if os.path.isdir(os.path.join(folder, d))),
            key=_natural_sort_key,
        )
        chapter_dirs = []
        for d in subdirs:
            sub_path = os.path.join(folder, d)
            has_images = any(
                os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
                for f in os.listdir(sub_path)
                if os.path.isfile(os.path.join(sub_path, f))
            )
            if has_images:
                chapter_dirs.append(sub_path)

        # If subfolders have images, use them (chapter layout)
        if chapter_dirs:
            all_paths = []
            for chap_dir in chapter_dirs:
                chap_files = sorted(
                    (f for f in os.listdir(chap_dir)
                     if os.path.isfile(os.path.join(chap_dir, f))
                     and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS),
                    key=_natural_sort_key,
                )
                for fname in chap_files:
                    all_paths.append(os.path.join(chap_dir, fname))
            return all_paths

        # Otherwise use root-level images
        return [os.path.join(folder, f) for f in root_files]

    def load_images(self, folder: str):
        """Load all images from folder (including chapter subfolders), split into batches, load first batch."""
        self.image_paths.clear()
        self._source_pixmaps.clear()
        self.display_pixmaps.clear()
        self.display_heights.clear()
        self._uniform_rows_cache.clear()
        self.original_sizes.clear()
        self.image_offsets.clear()
        self.scale_factors.clear()
        self.boxes.clear()
        self.all_image_paths.clear()
        self.batches.clear()
        self.merge_groups.clear()
        self.current_batch_index = 0
        self._cached_display_width = 0
        # Remember the project root so _build_merged_image can compute
        # path-stable cache keys (rel-paths instead of absolute).
        self._image_folder = folder

        all_paths = self._collect_images(folder)
        if not all_paths:
            self.setFixedHeight(100)
            self.update()
            return

        self.all_image_paths = all_paths
        self._finalize_batches()

    def load_multiple_folders(self, folders: List[str]):
        """Load images from multiple folders, concatenating them in order."""
        self.image_paths.clear()
        self._source_pixmaps.clear()
        self.display_pixmaps.clear()
        self.display_heights.clear()
        self._uniform_rows_cache.clear()
        self.original_sizes.clear()
        self.image_offsets.clear()
        self.scale_factors.clear()
        self.boxes.clear()
        self.all_image_paths.clear()
        self.batches.clear()
        self.merge_groups.clear()
        self.current_batch_index = 0
        self._cached_display_width = 0
        # For mixed-batch projects the "image_folder" used for relative-path
        # hashing is the parent of all chapter folders. Pick the common
        # ancestor of the supplied folders; falls back to the first folder's
        # parent if there's no shared prefix.
        if folders:
            try:
                self._image_folder = os.path.commonpath(folders)
            except ValueError:
                self._image_folder = os.path.dirname(folders[0])
        else:
            self._image_folder = ""

        all_paths = []
        for folder in folders:
            paths = self._collect_images(folder)
            all_paths.extend(paths)

        if not all_paths:
            self.setFixedHeight(100)
            self.update()
            return

        self.all_image_paths = all_paths
        self._finalize_batches()

    def _finalize_batches(self, initial_batch: int = 0):
        """Split all_image_paths into variable-size batches keyed on cumulative
        original image height, then load the requested batch."""
        all_paths = self.all_image_paths
        self.batches = []
        self.batch_starts = [0]

        if not all_paths:
            self._load_batch_pixmaps(0)
            self.boxes_changed.emit()
            self.batch_changed.emit()
            return

        current: List[str] = []
        current_h = 0
        for path in all_paths:
            h = self._fast_image_height(path)
            # If adding this image would overflow the height target AND we already
            # have at least BATCH_MIN_ITEMS, close the current batch.
            would_overflow = (current_h + h) > self.BATCH_HEIGHT_TARGET
            too_many = len(current) >= self.BATCH_MAX_ITEMS
            if current and (too_many or (would_overflow and len(current) >= self.BATCH_MIN_ITEMS)):
                self.batches.append(current)
                self.batch_starts.append(self.batch_starts[-1] + len(current))
                current = []
                current_h = 0
            current.append(path)
            current_h += h
        if current:
            self.batches.append(current)

        sizes = [len(b) for b in self.batches]
        logger.info(
            f"Split {len(all_paths)} images into {len(self.batches)} height-based batches "
            f"(target {self.BATCH_HEIGHT_TARGET}px, sizes min={min(sizes)} max={max(sizes)} "
            f"avg={sum(sizes)//len(sizes)})"
        )

        # Load requested batch (clamped)
        initial_batch = max(0, min(initial_batch, len(self.batches) - 1))
        self._load_batch_pixmaps(initial_batch)
        self.boxes_changed.emit()
        self.batch_changed.emit()

    def switch_batch(self, batch_index: int):
        """Switch to a different batch, preserving all boxes."""
        if batch_index < 0 or batch_index >= len(self.batches):
            return
        if batch_index == self.current_batch_index:
            return
        self.current_batch_index = batch_index
        self._load_batch_pixmaps(batch_index)
        self.boxes_changed.emit()
        self.batch_changed.emit()

    def _load_batch_pixmaps(self, batch_index: int):
        """Load pixmaps for a specific batch."""
        if batch_index < 0 or batch_index >= len(self.batches):
            return

        self.current_batch_index = batch_index
        batch_paths = self.batches[batch_index]

        self.image_paths.clear()
        self._source_pixmaps.clear()
        self.display_pixmaps.clear()
        self.display_heights.clear()
        self.original_sizes.clear()
        self.image_offsets.clear()
        self.scale_factors.clear()
        self._cached_display_width = 0
        self.selected_box_id = None

        for path in batch_paths:
            pix = QPixmap(path)
            if pix.isNull():
                continue
            self.image_paths.append(path)
            self.original_sizes.append((pix.width(), pix.height()))
            if pix.width() > 2000:
                capped_h = int(pix.height() * 2000 / pix.width())
                if capped_h <= 32767:
                    pix = pix.scaledToWidth(2000, Qt.SmoothTransformation)
                # else keep the full-res source: scaledToWidth would blank
                # every row past 32767 (Qt 16-bit limit); the strip scaler
                # in _do_rebuild handles tall sources correctly.
            self._source_pixmaps.append(pix)

        self._do_rebuild()

    def set_display_width(self, width: int):
        """Schedule a rebuild at new width. Preserves boxes."""
        if width == self.display_width or not self.image_paths:
            return
        self.display_width = width
        self._rebuild_display()

    def _rebuild_display(self):
        """Schedule a debounced rebuild (avoids spam during resize/splitter drag)."""
        self._rebuild_timer.start()

    def _do_rebuild(self):
        """Actually rebuild display pixmaps from cached source pixmaps."""
        if not self._source_pixmaps:
            return
        if self.display_width == self._cached_display_width:
            # Only recompute offsets/scale (pixmaps already valid)
            return

        self.display_pixmaps.clear()
        self.display_heights.clear()
        self.image_offsets.clear()
        self.scale_factors.clear()

        cum_y = 0
        for i, src_pix in enumerate(self._source_pixmaps):
            orig_w, orig_h = self.original_sizes[i]
            scale = self.display_width / orig_w if orig_w > 0 else 1.0
            self.scale_factors.append(scale)
            display_h = int(orig_h * scale)

            # Scale from cached source (fast, no disk I/O). Images whose
            # display height exceeds Qt's safe limit are scaled strip by
            # strip — one .scaled() call on the full pixmap silently turns
            # every row past 32767 black.
            strips: List[Tuple[int, QPixmap]] = []
            split_rows = self._find_split_rows(i, scale)
            if not split_rows:
                pix = src_pix.scaled(
                    self.display_width, display_h,
                    Qt.IgnoreAspectRatio, Qt.FastTransformation
                )
                strips.append((0, pix))
            else:
                bounds = [0] + split_rows + [orig_h]
                # src_pix may have been width-capped at load; map original
                # rows onto the (possibly smaller) source pixmap.
                src_h = src_pix.height()
                for a, b in zip(bounds, bounds[1:]):
                    sa = int(a * src_h / orig_h)
                    sb = int(b * src_h / orig_h)
                    da = int(a * scale)
                    db = display_h if b >= orig_h else int(b * scale)
                    if sb <= sa or db <= da:
                        continue
                    piece = src_pix.copy(0, sa, src_pix.width(), sb - sa)
                    strips.append((da, piece.scaled(
                        self.display_width, db - da,
                        Qt.IgnoreAspectRatio, Qt.FastTransformation
                    )))
            self.display_pixmaps.append(strips)
            self.display_heights.append(display_h)
            self.image_offsets.append(cum_y)
            cum_y += display_h

        self._cached_display_width = self.display_width
        self.setFixedSize(self.display_width, cum_y if cum_y > 0 else 100)
        self.update()

    # -- Tall-image strip splitting ---

    def _uniform_rows(self, path: str) -> Optional[np.ndarray]:
        """Bool mask of content-free rows for the image at `path`, at HALF
        vertical resolution (decoded reduced for speed). True = the whole
        row is a single uniform color (white/black gutter etc.), i.e. safe
        to place a strip seam on. Cached per path; None if decode fails."""
        if path in self._uniform_rows_cache:
            return self._uniform_rows_cache[path]
        mask = None
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_REDUCED_GRAYSCALE_2)
            if img is not None:
                spread = img.max(axis=1).astype(np.int16) - img.min(axis=1)
                mask = spread <= 10  # tolerate JPEG noise
        except Exception:
            logger.exception(f"_uniform_rows: failed to analyze {path}")
        self._uniform_rows_cache[path] = mask
        return mask

    @staticmethod
    def _best_seam_row(mask: np.ndarray, ideal: int, lo: int, hi: int,
                       min_band: int) -> Optional[int]:
        """Center (in FULL-res rows) of the uniform band nearest to `ideal`
        within [lo, hi] full-res rows. `mask` is the half-res uniformity
        mask; a band must span at least min_band full-res rows."""
        h_lo = max(0, lo // 2)
        h_hi = min(len(mask), hi // 2)
        h_ideal = ideal // 2
        min_run = max(2, min_band // 2)
        best = None
        run_start = None
        for i in range(h_lo, h_hi + 1):
            in_run = i < h_hi and mask[i]
            if in_run and run_start is None:
                run_start = i
            elif not in_run and run_start is not None:
                if i - run_start >= min_run:
                    center = (run_start + i - 1) // 2
                    if best is None or abs(center - h_ideal) < abs(best - h_ideal):
                        best = center
                run_start = None
        return best * 2 if best is not None else None

    def _find_split_rows(self, local_idx: int, scale: float) -> List[int]:
        """Source-row split points for image `local_idx` so that every
        scaled strip stays under Qt's blank-past-32767 limit. Splits are
        placed inside content-free bands when one exists near the ideal
        position; otherwise falls back to the ideal row (logged)."""
        orig_w, orig_h = self.original_sizes[local_idx]
        display_h = int(orig_h * scale)
        if display_h <= self.MAX_DISPLAY_STRIP or scale <= 0:
            return []

        n_strips = -(-display_h // self.MAX_DISPLAY_STRIP)  # ceil
        max_src = int(self.STRIP_MAX_DISPLAY / scale)  # hard cap per strip
        mask = self._uniform_rows(self.image_paths[local_idx])

        splits: List[int] = []
        prev = 0
        for k in range(1, n_strips):
            ideal = round(orig_h * k / n_strips)
            # Window around the ideal split, constrained so this strip and
            # all remaining strips can still fit under the hard cap.
            lo = max(prev + 1, ideal - self.STRIP_SEAM_WINDOW,
                     orig_h - (n_strips - k) * max_src)
            hi = min(ideal + self.STRIP_SEAM_WINDOW, prev + max_src)
            row = None
            if mask is not None and lo <= hi:
                row = self._best_seam_row(
                    mask, ideal, lo, hi, self.STRIP_SEAM_MIN_BAND)
                if row is not None:
                    row = max(lo, min(row, hi))
            if row is None:
                row = max(lo, min(ideal, hi))
                logger.warning(
                    f"_find_split_rows: no content-free band near row {ideal} "
                    f"of {self.image_paths[local_idx]}; splitting at {row}"
                )
            splits.append(row)
            prev = row
        logger.info(
            f"_find_split_rows: image {local_idx} "
            f"({orig_w}x{orig_h}, display_h={display_h}) split at {splits}"
        )
        return splits

    # -- Box helpers ---

    def _sort_boxes(self):
        """Sort boxes by their global Y position (top to bottom).
        Sorts ALL boxes across all batches by global image_index then by Y."""
        def sort_key(box: ScriptBox) -> Tuple[int, int]:
            return (box.image_index, box.y)
        self.boxes.sort(key=sort_key)

    def _box_display_rect(self, box: ScriptBox) -> QRect:
        """Convert a box from original image coords to display coords.
        Box.image_index is global; converts to batch-local for display."""
        local_idx = self._global_to_local(box.image_index)
        if local_idx < 0 or local_idx >= len(self.image_offsets):
            return QRect()
        scale = self.scale_factors[local_idx]
        offset_y = self.image_offsets[local_idx]
        return QRect(
            int(box.x * scale),
            int(offset_y + box.y * scale),
            int(box.w * scale),
            int(box.h * scale),
        )

    def _display_to_image_coords(self, pos: QPoint) -> Optional[Tuple[int, int, int]]:
        """Convert display position to (global_image_index, local_x, local_y)."""
        y = pos.y()
        for i in range(len(self.image_offsets)):
            top = self.image_offsets[i]
            h = self.display_heights[i] if i < len(self.display_heights) else 0
            if top <= y < top + h:
                scale = self.scale_factors[i]
                local_x = int(pos.x() / scale) if scale > 0 else 0
                local_y = int((y - top) / scale) if scale > 0 else 0
                return (self._local_to_global(i), local_x, local_y)
        return None

    def _box_at_pos(self, pos: QPoint) -> Optional[ScriptBox]:
        """Find the topmost box at display position."""
        for box in reversed(self.boxes):
            r = self._box_display_rect(box)
            if r.contains(pos):
                return box
        return None

    def _handle_at_pos(self, pos: QPoint, box: ScriptBox) -> Optional[str]:
        """Check if pos is on a resize handle of the box. Returns handle name or None."""
        r = self._box_display_rect(box)
        h = self.HANDLE
        handles = {
            "tl": QRect(r.left() - h//2, r.top() - h//2, h, h),
            "tr": QRect(r.right() - h//2, r.top() - h//2, h, h),
            "bl": QRect(r.left() - h//2, r.bottom() - h//2, h, h),
            "br": QRect(r.right() - h//2, r.bottom() - h//2, h, h),
            "t":  QRect(r.center().x() - h//2, r.top() - h//2, h, h),
            "b":  QRect(r.center().x() - h//2, r.bottom() - h//2, h, h),
            "l":  QRect(r.left() - h//2, r.center().y() - h//2, h, h),
            "r":  QRect(r.right() - h//2, r.center().y() - h//2, h, h),
        }
        for name, rect in handles.items():
            if rect.contains(pos):
                return name
        return None

    # -- Painting ---

    def _visible_range(self) -> Tuple[int, int]:
        """Get the visible Y range from the parent scroll area."""
        parent = self.parent()
        if parent:
            sa = parent.parent()
            if isinstance(sa, QScrollArea):
                vbar = sa.verticalScrollBar()
                top = vbar.value()
                return top, top + sa.viewport().height()
        return 0, self.height()

    def paintEvent(self, event):
        painter = QPainter(self)
        # Always fill background dark first to prevent white flash
        painter.fillRect(event.rect(), QColor(30, 30, 30))
        painter.setRenderHint(QPainter.Antialiasing)

        vis_top, vis_bottom = self._visible_range()
        # Add margin for labels/handles that extend above boxes
        vis_top -= 30
        vis_bottom += 30

        # Draw only visible images (each image = one or more vertical strips)
        for i, strips in enumerate(self.display_pixmaps):
            y = self.image_offsets[i]
            h = self.display_heights[i] if i < len(self.display_heights) else 0
            if y + h < vis_top or y > vis_bottom:
                continue
            for dy, strip in strips:
                if y + dy + strip.height() < vis_top or y + dy > vis_bottom:
                    continue
                painter.drawPixmap(0, y + dy, strip)
            # Seam line at the TOP of image i (between i-1 and i); skip for i == 0
            if i == 0:
                continue
            if i in self._active_cross_seams:
                painter.setPen(QPen(QColor(255, 60, 60), 3, Qt.SolidLine))
            else:
                painter.setPen(QPen(QColor(80, 80, 80), 1, Qt.DashLine))
            painter.drawLine(0, y, self.display_width, y)

        # "Merged" badge on the top-left corner of any merged image in view
        for i, path in enumerate(self.image_paths):
            if not self._is_merged_path(path):
                continue
            y_top = self.image_offsets[i] if i < len(self.image_offsets) else 0
            if y_top > vis_bottom:
                break
            if i + 1 < len(self.image_offsets):
                y_bot = self.image_offsets[i + 1]
            else:
                y_bot = y_top + (self.display_heights[i] if i < len(self.display_heights) else 0)
            if y_bot < vis_top:
                continue
            badge = QRect(6, y_top + 6, 70, 20)
            painter.fillRect(badge, QColor(0, 180, 0, 200))
            painter.setPen(QColor(0, 0, 0))
            font = painter.font()
            font.setPointSize(9)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(badge, Qt.AlignCenter, "MERGED")

        # Draw only visible boxes (current batch only)
        batch_boxes = self.current_batch_boxes()
        # Global zone numbers stay continuous across pages/batches.
        global_zone_num = {b.id: i + 1 for i, b in enumerate(self.boxes)}
        for idx, box in enumerate(batch_boxes):
            r = self._box_display_rect(box)
            if r.bottom() < vis_top or r.top() > vis_bottom:
                continue

            is_selected = (box.id == self.selected_box_id)

            fill_color = QColor(0, 255, 255, 35) if not is_selected else QColor(0, 255, 255, 60)
            painter.fillRect(r, fill_color)

            pen_width = 3 if is_selected else 2
            painter.setPen(QPen(QColor(0, 255, 255), pen_width))
            # NoBrush: the selected box's white handle brush must not leak
            # into this outline (it painted later boxes solid white).
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(r)

            # Zone label
            label = f"Zone {global_zone_num.get(box.id, idx + 1)}"
            font = painter.font()
            font.setPointSize(9)
            font.setBold(True)
            painter.setFont(font)
            fm = QFontMetrics(font)
            tw = fm.horizontalAdvance(label) + 10
            th = fm.height() + 4
            label_rect = QRect(r.left(), r.top() - th - 2, tw, th)
            painter.fillRect(label_rect, QColor(0, 200, 200, 220))
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(label_rect, Qt.AlignCenter, label)

            if is_selected:
                hp = self.HANDLE
                handle_positions = [
                    (r.left(), r.top()), (r.right(), r.top()),
                    (r.left(), r.bottom()), (r.right(), r.bottom()),
                    (r.center().x(), r.top()), (r.center().x(), r.bottom()),
                    (r.left(), r.center().y()), (r.right(), r.center().y()),
                ]
                for hx, hy in handle_positions:
                    painter.setPen(QPen(QColor(0, 0, 0), 1))
                    painter.setBrush(QColor(255, 255, 255))
                    painter.drawRect(hx - hp//2, hy - hp//2, hp, hp)

        # Draw creation rectangle
        if self._create_rect and not self._create_rect.isNull():
            painter.setPen(QPen(QColor(255, 200, 0), 2, Qt.DashLine))
            painter.setBrush(QColor(255, 200, 0, 30))
            painter.drawRect(self._create_rect)

        # Hover merge button near a seam (drawn last so it's on top)
        if self._hover_merge_btn_rect and self._hover_seam is not None and self._drag_mode is None:
            r = self._hover_merge_btn_rect
            painter.setBrush(QColor(0, 180, 0, 230))
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.drawEllipse(r)
            font = painter.font()
            font.setPointSize(12)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(r, Qt.AlignCenter, "⇅")

        # Character-capture rectangle (Story Bible photo grab)
        cap_rect = getattr(self, "_capture_rect", None)
        if cap_rect and not cap_rect.isNull():
            painter.setPen(QPen(QColor(0, 230, 118), 2, Qt.DashLine))
            painter.setBrush(QColor(0, 230, 118, 30))
            painter.drawRect(cap_rect)

        painter.end()

    # -- Character photo capture (Story Bible) ---

    def start_capture(self, callback):
        """One-shot capture mode: the cursor becomes a cross and the next
        left-drag rectangle is cropped from the underlying image (BGR
        ndarray) and passed to callback. Right-click (or a tiny drag)
        cancels with callback(None)."""
        self._capture_cb = callback
        self._capture_origin = None
        self._capture_rect = None
        self.setCursor(Qt.CrossCursor)

    def _finish_capture(self, rect):
        cb = self._capture_cb
        self._capture_cb = None
        self._capture_origin = None
        self._capture_rect = None
        self.unsetCursor()
        self.update()
        crop = self._crop_display_rect(rect) if rect is not None else None
        if cb:
            cb(crop)

    def _crop_display_rect(self, rect: QRect):
        """Crop the display-space rect from the source image under its top
        edge (clamped to that image), returning a BGR ndarray or None."""
        info = self._display_to_image_coords(rect.topLeft())
        if not info:
            return None
        g, x1, y1 = info
        local = self._global_to_local(g)
        if not (0 <= local < len(self.image_paths)):
            return None
        scale = self.scale_factors[local] if local < len(self.scale_factors) else 1.0
        if scale <= 0:
            return None
        x2 = x1 + int(rect.width() / scale)
        y2 = y1 + int(rect.height() / scale)
        path = self.image_paths[local]
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            logger.exception(f"capture: failed to read {path}")
            return None
        if img is None:
            return None
        h, w = img.shape[:2]
        x1, x2 = max(0, min(x1, w - 1)), max(1, min(x2, w))
        y1, y2 = max(0, min(y1, h - 1)), max(1, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None
        return img[y1:y2, x1:x2].copy()

    # -- Mouse interaction ---

    def mousePressEvent(self, event):
        pos = event.pos()

        if getattr(self, "_capture_cb", None):
            if event.button() == Qt.LeftButton:
                self._capture_origin = self._clamp_to_content(pos)
                self._capture_rect = QRect(self._capture_origin, QSize(0, 0))
                self.update()
            else:
                self._finish_capture(None)  # right-click cancels
            return

        if event.button() == Qt.RightButton:
            # Priority 1: delete box under cursor (existing behavior)
            box = self._box_at_pos(pos)
            if box:
                self.box_deleting.emit(box.id)
                self.boxes.remove(box)
                if self.selected_box_id == box.id:
                    self.selected_box_id = None
                self._sort_boxes()
                self.update()
                self.boxes_changed.emit()
                return

            # Priority 2: if clicking on a merged image, offer unmerge
            info = self._display_to_image_coords(pos)
            if info:
                _global, _lx, _ly = info
                local_idx = self._global_to_local(_global)
                if 0 <= local_idx < len(self.image_paths):
                    path = self.image_paths[local_idx]
                    if self._is_merged_path(path):
                        group = self._find_merge_group_for_merged_path(path)
                        if group:
                            menu = QMenu(self)
                            act = menu.addAction(f"Unmerge ({len(group)} images)")
                            chosen = menu.exec_(event.globalPos())
                            if chosen == act:
                                self.unmerge_local(local_idx)
            return

        if event.button() == Qt.LeftButton:
            # Priority: click on the hover merge button → merge those two images
            if self._hover_merge_btn_rect and self._hover_merge_btn_rect.contains(pos):
                if self._hover_seam is not None and self._hover_seam >= 1:
                    self.merge_local_range(self._hover_seam - 1, self._hover_seam)
                self._hover_seam = None
                self._hover_merge_btn_rect = None
                self.update()
                return

            # Check if clicking on selected box's resize handle
            if self.selected_box_id:
                sel_box = next((b for b in self.boxes if b.id == self.selected_box_id), None)
                if sel_box:
                    handle = self._handle_at_pos(pos, sel_box)
                    if handle:
                        self._drag_mode = "resize"
                        self._drag_start = pos
                        self._drag_box = sel_box
                        self._drag_handle = handle
                        self._drag_origin_rect = self._box_display_rect(sel_box)
                        return

            # Check if clicking on any box (select + start move)
            box = self._box_at_pos(pos)
            if box:
                self.selected_box_id = box.id
                self._drag_mode = "move"
                self._drag_start = pos
                self._drag_box = box
                self._drag_origin_rect = self._box_display_rect(box)
                self.update()
                self.box_selected.emit(box.id)
                return

            # Start creating new box
            self.begin_create_at(pos)

    def _clamp_to_content(self, pos: QPoint) -> QPoint:
        """Clamp a (possibly outside) position into the strip content bounds,
        so box creation can start/drag/release beyond the image edges."""
        return QPoint(
            max(0, min(pos.x(), self.width() - 1)),
            max(0, min(pos.y(), self.height() - 1)),
        )

    def begin_create_at(self, pos: QPoint):
        """Start box creation at pos (display coords, clamped into content)."""
        self.selected_box_id = None
        self._drag_mode = "create"
        self._drag_start = self._clamp_to_content(pos)
        self._create_rect = QRect(self._drag_start, QSize(0, 0))
        self.update()

    def drag_create_to(self, pos: QPoint):
        """Extend the in-progress creation rect to pos (clamped)."""
        if self._drag_mode != "create" or self._drag_start is None:
            return
        self._update_create_rect(pos)

    def finish_create_at(self, pos: QPoint):
        """Finish box creation at pos (clamped) and commit the box."""
        if self._drag_mode != "create":
            return
        self._update_create_rect(pos)
        self._finalize_create()
        self._drag_mode = None
        self._drag_start = None
        self.update()

    def _update_create_rect(self, pos: QPoint):
        pos = self._clamp_to_content(pos)
        self._create_rect = QRect(self._drag_start, pos).normalized()
        # Compute which seams the drag is crossing (for red highlight)
        touched = self._local_indices_touched_by_rect(self._create_rect)
        seams: List[int] = []
        if len(touched) >= 2:
            for i in range(len(touched) - 1):
                a = touched[i]
                b = touched[i + 1]
                if b == a + 1:
                    seams.append(b)  # seam at the top of image b (between a and b)
        self._active_cross_seams = seams
        self.update()

    def mouseMoveEvent(self, event):
        pos = event.pos()

        if getattr(self, "_capture_cb", None):
            if getattr(self, "_capture_origin", None) is not None:
                self._capture_rect = QRect(
                    self._capture_origin, self._clamp_to_content(pos)).normalized()
                self.update()
            return

        if self._drag_mode == "create" and self._drag_start:
            self._update_create_rect(pos)

        elif self._drag_mode == "move" and self._drag_box and self._drag_start:
            dx = pos.x() - self._drag_start.x()
            dy = pos.y() - self._drag_start.y()
            orig = self._drag_origin_rect
            new_center = QPoint(orig.center().x() + dx, orig.center().y() + dy)
            # Convert back to image coords (returns global index)
            info = self._display_to_image_coords(new_center)
            if info:
                global_idx, lx, ly = info
                local_idx = self._global_to_local(global_idx)
                scale = self.scale_factors[local_idx]
                new_w = int(orig.width() / scale) if scale > 0 else self._drag_box.w
                new_h = int(orig.height() / scale) if scale > 0 else self._drag_box.h
                # Clamp to image bounds
                orig_w, orig_h = self.original_sizes[local_idx]
                nx = max(0, min(lx - new_w // 2, orig_w - new_w))
                ny = max(0, min(ly - new_h // 2, orig_h - new_h))
                self._drag_box.image_index = global_idx
                self._drag_box.x = nx
                self._drag_box.y = ny
                self._drag_box.w = new_w
                self._drag_box.h = new_h
                self.update()

        elif self._drag_mode == "resize" and self._drag_box and self._drag_start:
            self._apply_resize(pos)
            self.update()

        else:
            # Update cursor based on hover
            if self.selected_box_id:
                sel_box = next((b for b in self.boxes if b.id == self.selected_box_id), None)
                if sel_box:
                    handle = self._handle_at_pos(pos, sel_box)
                    if handle:
                        cursors = {
                            "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
                            "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor,
                            "t": Qt.SizeVerCursor, "b": Qt.SizeVerCursor,
                            "l": Qt.SizeHorCursor, "r": Qt.SizeHorCursor,
                        }
                        self.setCursor(cursors.get(handle, Qt.ArrowCursor))
                        return

            # Seam hover detection (only when not dragging)
            prev_seam = self._hover_seam
            new_seam: Optional[int] = None
            for i in range(1, len(self.image_offsets)):
                seam_y = self.image_offsets[i]
                if abs(pos.y() - seam_y) <= self.SEAM_HOVER_RADIUS:
                    new_seam = i
                    break
            if new_seam != prev_seam:
                self._hover_seam = new_seam
                self._hover_merge_btn_rect = (
                    self._hover_merge_btn_rect_for(new_seam) if new_seam is not None else None
                )
                self.update()

            if self._hover_merge_btn_rect and self._hover_merge_btn_rect.contains(pos):
                self.setCursor(Qt.PointingHandCursor)
                return

            box = self._box_at_pos(pos)
            self.setCursor(Qt.SizeAllCursor if box else Qt.CrossCursor)

    def _finalize_create(self):
        """Commit the in-progress creation rect as a new box (if big enough)."""
        if not self._create_rect:
            return
        rect = self._create_rect.normalized()
        if rect.width() > 10 and rect.height() > 10:
            # If the drag crosses one or more seams with ≥ threshold overshoot
            # on both sides, auto-merge the touched images first.
            touched = self._local_indices_touched_by_rect(rect)
            did_merge = False
            if len(touched) >= 2 and touched == list(range(touched[0], touched[-1] + 1)):
                # Remember display-space rect so we can re-project after merge
                saved_rect = QRect(rect)
                ok = self.merge_local_range(touched[0], touched[-1])
                if ok:
                    did_merge = True
                    # After merge, the single merged image replaces the range;
                    # the display rect position still makes sense because the
                    # merged image now occupies the same vertical extent.
                    # Recompute coords against the new layout.
                    rect = saved_rect

            # Determine image and local coords (returns global index)
            info = self._display_to_image_coords(rect.topLeft())
            if info:
                global_idx, lx, ly = info
                local_idx = self._global_to_local(global_idx)
                if 0 <= local_idx < len(self.scale_factors):
                    scale = self.scale_factors[local_idx]
                    lw = int(rect.width() / scale) if scale > 0 else rect.width()
                    lh = int(rect.height() / scale) if scale > 0 else rect.height()
                    # Clamp to image bounds
                    orig_w, orig_h = self.original_sizes[local_idx]
                    lx = max(0, min(lx, orig_w - 1))
                    ly = max(0, min(ly, orig_h - 1))
                    lw = min(lw, orig_w - lx)
                    lh = min(lh, orig_h - ly)

                    new_box = ScriptBox(
                        image_index=global_idx, x=lx, y=ly, w=lw, h=lh,
                        confidence=1.0,
                    )
                    self.boxes.append(new_box)
                    self._sort_boxes()
                    self.selected_box_id = new_box.id
                    self.boxes_changed.emit()
                    self.box_selected.emit(new_box.id)

        self._create_rect = None
        self._active_cross_seams = []

    def mouseReleaseEvent(self, event):
        if getattr(self, "_capture_cb", None):
            if (event.button() == Qt.LeftButton
                    and getattr(self, "_capture_origin", None) is not None):
                rect = QRect(self._capture_origin,
                             self._clamp_to_content(event.pos())).normalized()
                self._finish_capture(
                    rect if rect.width() > 4 and rect.height() > 4 else None)
            return

        if event.button() == Qt.LeftButton:
            if self._drag_mode == "create" and self._create_rect:
                self._update_create_rect(event.pos())
                self._finalize_create()

            elif self._drag_mode in ("move", "resize"):
                self._sort_boxes()
                self.boxes_changed.emit()

            self._drag_mode = None
            self._drag_start = None
            self._drag_box = None
            self._drag_handle = None
            self._drag_origin_rect = None
            self.update()

    def _apply_resize(self, pos: QPoint):
        """Resize the dragged box based on handle and mouse position."""
        box = self._drag_box
        handle = self._drag_handle
        orig = self._drag_origin_rect
        if not box or not handle or not orig:
            return

        dx = pos.x() - self._drag_start.x()
        dy = pos.y() - self._drag_start.y()

        new_rect = QRect(orig)
        if "l" in handle:
            new_rect.setLeft(orig.left() + dx)
        if "r" in handle:
            new_rect.setRight(orig.right() + dx)
        if "t" in handle:
            new_rect.setTop(orig.top() + dy)
        if "b" in handle:
            new_rect.setBottom(orig.bottom() + dy)

        new_rect = new_rect.normalized()
        if new_rect.width() < 10 or new_rect.height() < 10:
            return

        # Convert back to image coords (returns global index)
        info = self._display_to_image_coords(new_rect.topLeft())
        if info:
            global_idx, lx, ly = info
            local_idx = self._global_to_local(global_idx)
            scale = self.scale_factors[local_idx]
            lw = int(new_rect.width() / scale) if scale > 0 else new_rect.width()
            lh = int(new_rect.height() / scale) if scale > 0 else new_rect.height()
            orig_w, orig_h = self.original_sizes[local_idx]
            lx = max(0, min(lx, orig_w - 1))
            ly = max(0, min(ly, orig_h - 1))
            lw = min(lw, orig_w - lx)
            lh = min(lh, orig_h - ly)
            box.image_index = global_idx
            box.x = lx
            box.y = ly
            box.w = lw
            box.h = lh

    # -- Image merging ---

    @staticmethod
    def _merge_cache_dir() -> str:
        from pathlib import Path
        d = Path.home() / ".momo_script" / "merged_cache"
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    def _merge_cache_path(self, original_paths: List[str],
                          image_folder: Optional[str] = None) -> str:
        """Deterministic cache path for a group of original paths.

        Hashes paths RELATIVE to the project root (`self._image_folder` by
        default; pass `image_folder=""` to force absolute-path hashing for
        ad-hoc/external callers). Rel-path hashing keeps the cache key
        stable when the project is moved on disk and — critically —
        produces the SAME key that `_build_merged_image` uses, so
        `_find_merge_group_for_merged_path` can match against
        `all_image_paths` entries by string equality.

        Earlier this was a classmethod that defaulted to absolute paths,
        which silently broke the round-trip: merged composites in
        `all_image_paths` had rel-path hashes (from `_build_merged_image`),
        but `_find_merge_group_for_merged_path` compared them against
        abs-path hashes, never matched, and `get_state` fell through to
        saving the cache path as a plain entry — corrupting the load.
        """
        import hashlib
        if image_folder is None:
            image_folder = getattr(self, "_image_folder", "") or ""
        if image_folder:
            keys = []
            for p in original_paths:
                try:
                    rel = os.path.relpath(p, image_folder).replace(os.sep, "/")
                except ValueError:
                    rel = os.path.basename(p)
                keys.append(rel)
        else:
            keys = list(original_paths)
        key = "|".join(keys).encode("utf-8", errors="replace")
        h = hashlib.sha1(key).hexdigest()[:16]
        return os.path.join(self._merge_cache_dir(), f"merged_{h}.jpg")

    def _build_merged_image(self, original_paths: List[str],
                            image_folder: Optional[str] = None):
        """Stitch originals into a cached JPEG.
        Returns (merged_path, positions) on success, None on failure.
        positions: list of (start_y, end_y, orig_index) in stitched coords.

        `image_folder` is forwarded to _merge_cache_path so the hash is
        location-independent. When the caller doesn't pass one, falls back
        to the strip's `_image_folder` if set, else to "" (absolute hash).
        """
        from app.image_splitter import stitch_images

        if image_folder is None:
            image_folder = getattr(self, "_image_folder", "") or ""

        out_path = self._merge_cache_path(original_paths, image_folder)

        # Cache hit on the new (rel-path) hash → return immediately.
        if os.path.isfile(out_path):
            stitched_only = stitch_images(original_paths)
            if stitched_only is None:
                return None
            _stitched, positions = stitched_only
            return out_path, positions

        # Migration step: try the legacy (absolute-path) hash. If a cache
        # file is there from a previous version, rename it onto the new
        # path so we don't re-stitch and possibly produce a slightly
        # different composition.
        legacy_path = self._merge_cache_path(original_paths, "")
        if legacy_path != out_path and os.path.isfile(legacy_path):
            try:
                os.replace(legacy_path, out_path)
                logger.info(
                    f"_build_merged_image: migrated legacy cache "
                    f"{os.path.basename(legacy_path)} -> "
                    f"{os.path.basename(out_path)}"
                )
                stitched_only = stitch_images(original_paths)
                if stitched_only is None:
                    return None
                _stitched, positions = stitched_only
                return out_path, positions
            except Exception:
                logger.exception("_build_merged_image: legacy cache migration failed")

        result = stitch_images(original_paths)
        if result is None:
            return None
        stitched, positions = result
        try:
            ok, buf = cv2.imencode(".jpg", stitched, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ok:
                return None
            buf.tofile(out_path)
        except Exception as e:
            logger.error(f"Failed to write merged image to {out_path}: {e}")
            return None
        return out_path, positions

    def _is_merged_path(self, path: str) -> bool:
        """True if this path is one of our cached merged images."""
        return os.path.dirname(path) == self._merge_cache_dir()

    def _find_merge_group_for_merged_path(self, merged_path: str) -> Optional[List[str]]:
        """Given a merged cache path, return the list of originals it came from."""
        for grp in self.merge_groups:
            if self._merge_cache_path(grp) == merged_path:
                return grp
        return None

    def _merge_global_range_dataonly(self, global_start: int, global_end: int) -> bool:
        """Perform a merge on all_image_paths[global_start..global_end] (inclusive)
        WITHOUT touching batches, pixmaps, or signals. Updates all_image_paths,
        merge_groups, and box image_index/coords. Safe to call repeatedly in a
        batched merge loop — caller must _finalize_batches once when done."""
        if global_start < 0 or global_end < 0:
            return False
        if global_end <= global_start:
            return False
        if global_end >= len(self.all_image_paths):
            return False

        current_paths = self.all_image_paths[global_start:global_end + 1]

        # Expand any already-merged entries into their originals for the new merge
        expanded_originals: List[str] = []
        for p in current_paths:
            existing_group = self._find_merge_group_for_merged_path(p) if self._is_merged_path(p) else None
            if existing_group:
                expanded_originals.extend(existing_group)
            else:
                expanded_originals.append(p)

        result = self._build_merged_image(expanded_originals)
        if result is None:
            return False
        merged_path, positions = result

        # Per-current-entry (y_offset, scale) within the new merged image
        per_current_offset_scale: List[Tuple[int, float]] = []
        orig_cursor = 0
        for p in current_paths:
            existing = self._find_merge_group_for_merged_path(p) if self._is_merged_path(p) else None
            n_sub = len(existing) if existing else 1
            sub_start_y = positions[orig_cursor][0]
            sub_end_y = positions[orig_cursor + n_sub - 1][1]
            pix = QPixmap(p)
            if pix.isNull():
                per_current_offset_scale.append((sub_start_y, 1.0))
                orig_cursor += n_sub
                continue
            cur_h = pix.height()
            scale = (sub_end_y - sub_start_y) / cur_h if cur_h > 0 else 1.0
            per_current_offset_scale.append((sub_start_y, scale))
            orig_cursor += n_sub

        # Remap boxes (in place; no widget churn)
        shift = global_end - global_start
        for box in self.boxes:
            if global_start <= box.image_index <= global_end:
                k = box.image_index - global_start
                if k < len(per_current_offset_scale):
                    y_off, scale = per_current_offset_scale[k]
                    box.x = int(box.x * scale)
                    box.y = int(box.y * scale + y_off)
                    box.w = max(1, int(box.w * scale))
                    box.h = max(1, int(box.h * scale))
                box.image_index = global_start
            elif box.image_index > global_end:
                box.image_index -= shift

        # Replace the range in all_image_paths with the single merged path
        self.all_image_paths[global_start:global_end + 1] = [merged_path]

        # Drop obsolete merge_groups whose output got consumed by this merge
        obsolete_merged = [p for p in current_paths if self._is_merged_path(p)]
        if obsolete_merged:
            self.merge_groups = [
                g for g in self.merge_groups
                if self._merge_cache_path(g) not in obsolete_merged
            ]
        self.merge_groups.append(expanded_originals)
        return True

    def merge_local_range(self, local_start: int, local_end: int) -> bool:
        """Merge local images [local_start..local_end] in the current batch into
        one cached image. Interactive path — reloads batches and updates UI."""
        if local_start < 0 or local_end < 0:
            return False
        if local_end <= local_start:
            return False
        if local_end >= len(self.image_paths):
            return False

        global_start = self._local_to_global(local_start)
        global_end = self._local_to_global(local_end)

        try:
            if not self._merge_global_range_dataonly(global_start, global_end):
                return False

            logger.info(f"merge_local_range: data-only merge OK, rebuilding batches")
            self._finalize_batches(initial_batch=0)
            logger.info(f"merge_local_range: batches rebuilt, resolving target batch")
            target_batch = self._global_to_batch(global_start)
            if target_batch != self.current_batch_index:
                logger.info(f"merge_local_range: switching to batch {target_batch}")
                self.switch_batch(target_batch)
            logger.info(f"Merged range at global {global_start}..{global_end}")
            return True
        except Exception:
            logger.exception(
                f"merge_local_range CRASHED at global {global_start}..{global_end}"
            )
            return False

    def unmerge_local(self, local_idx: int) -> bool:
        """If local_idx points to a merged image, expand it back to originals
        and remap boxes on that image back to their source images."""
        if local_idx < 0 or local_idx >= len(self.image_paths):
            return False
        global_idx = self._local_to_global(local_idx)
        merged_path = self.all_image_paths[global_idx]
        if not self._is_merged_path(merged_path):
            return False
        group = self._find_merge_group_for_merged_path(merged_path)
        if not group:
            return False

        # Rebuild positions deterministically by re-stitching
        from app.image_splitter import stitch_images
        result = stitch_images(group)
        if result is None:
            return False
        stitched, positions = result  # positions[k] = (start_y, end_y, orig_index)

        # Dims of each original (for scale)
        orig_dims: List[Tuple[int, int]] = []
        for p in group:
            pix = QPixmap(p)
            orig_dims.append((pix.width(), pix.height()) if not pix.isNull() else (0, 0))

        n = len(group)
        # Remap boxes: any box with image_index == global_idx gets re-assigned to
        # global_idx + k (k = which original it falls into based on y coord)
        new_boxes: List[ScriptBox] = []
        for box in self.boxes:
            if box.image_index == global_idx:
                by_center = box.y + box.h // 2
                k = 0
                for i, (start_y, end_y, _oi) in enumerate(positions):
                    if start_y <= by_center < end_y:
                        k = i
                        break
                else:
                    k = n - 1
                start_y, end_y, _oi = positions[k]
                orig_w, orig_h = orig_dims[k]
                stitched_h = end_y - start_y
                inv_scale = orig_h / stitched_h if stitched_h > 0 else 1.0
                # Translate box into the piece's coord space (subtract offset), then back to original
                local_y = max(0, box.y - start_y)
                box.x = int(box.x * inv_scale)
                box.y = int(local_y * inv_scale)
                box.w = max(1, int(box.w * inv_scale))
                box.h = max(1, int(box.h * inv_scale))
                box.image_index = global_idx + k
            elif box.image_index > global_idx:
                box.image_index += (n - 1)
            new_boxes.append(box)
        self.boxes = new_boxes

        # Replace in all_image_paths
        self.all_image_paths[global_idx:global_idx + 1] = list(group)

        # Drop this group
        self.merge_groups = [g for g in self.merge_groups if g is not group]

        # Rebuild
        self._finalize_batches(initial_batch=0)
        target_batch = self._global_to_batch(global_idx)
        if target_batch != self.current_batch_index:
            self.switch_batch(target_batch)

        # Boxes that straddled a seam in the merged image are now stranded:
        # they got reassigned to whichever original contained their CENTER, but
        # their y/h can extend past that original's bounds (overflow) or start
        # before its top (negative y). Auto re-merge the spanning images so the
        # box lives on a valid canvas instead of leaving cross-image artifacts.
        fixed = self._fix_cross_image_boxes()
        if fixed:
            logger.info(f"unmerge_local: re-merged {fixed} spanning range(s) "
                        f"to keep boxes on valid canvases")

        logger.info(f"Unmerged group of {n} at global {global_idx}")
        return True

    def reapply_merge_groups_on_load(
        self,
        groups_basenames: List[List[str]],
        groups_rel: Optional[List[List[str]]] = None,
        image_folder: str = "",
    ) -> None:
        """Called after project load (once all_image_paths is fresh from folder).
        Re-applies each stored merge group by finding its run in all_image_paths
        and replacing with a cached merged image. Does NOT remap boxes — they're
        loaded AFTER this runs with image_index already in the merged
        coordinate space.

        When `groups_rel` is provided (new save format with relative paths),
        the lookup is unambiguous even across chapter folders that share
        basenames. Old saves (basename-only) keep working but can map to the
        wrong chapter when basenames repeat — those projects need to be
        re-saved once to upgrade to the safe format."""
        self.merge_groups = []

        # Decide which side to match against per-group: rel paths (preferred)
        # when both are present and the lengths agree, basenames otherwise.
        use_rel = bool(
            groups_rel
            and len(groups_rel) == len(groups_basenames)
        )

        def _path_to_rel(p: str) -> str:
            if not image_folder:
                return os.path.basename(p)
            try:
                return os.path.relpath(p, image_folder).replace(os.sep, "/")
            except ValueError:
                return os.path.basename(p)

        all_rel = (
            [_path_to_rel(p) for p in self.all_image_paths]
            if use_rel else None
        )
        all_basenames = [os.path.basename(p) for p in self.all_image_paths]

        for gi, group_names in enumerate(groups_basenames):
            if not group_names:
                continue

            target_rel = (
                [r.replace(os.sep, "/") for r in groups_rel[gi]]
                if use_rel else None
            )
            target_bn = group_names

            run_start = None
            if use_rel:
                # Match by relative path — unique even when basenames clash.
                gn = len(target_rel)
                for i in range(len(all_rel) - gn + 1):
                    if all_rel[i:i + gn] == target_rel:
                        run_start = i
                        break

            if run_start is None:
                # Either no rel paths or rel match failed (e.g. user moved
                # a file). Fall back to basename. Logs a warning when this
                # happens because it can produce wrong matches in chapter
                # projects with duplicate basenames.
                gn = len(target_bn)
                for i in range(len(all_basenames) - gn + 1):
                    if all_basenames[i:i + gn] == target_bn:
                        run_start = i
                        break
                if run_start is not None and use_rel:
                    logger.warning(
                        f"Merge group {gi} matched by basename only "
                        f"(rel-path lookup failed): {target_rel}. May map "
                        f"to wrong chapter if filename repeats."
                    )

            if run_start is None:
                logger.warning(
                    f"Merge group {gi} not found in loaded images: "
                    f"{target_rel if use_rel else target_bn}"
                )
                continue

            run_len = len(target_rel) if use_rel else len(target_bn)
            originals = self.all_image_paths[run_start:run_start + run_len]
            result = self._build_merged_image(originals)
            if result is None:
                logger.warning(f"Failed to rebuild merged image for group {gi}")
                continue
            merged_path, _positions = result
            self.all_image_paths[run_start:run_start + run_len] = [merged_path]
            self.merge_groups.append(list(originals))

            # Keep the per-position views in sync so the next iteration
            # finds the right run (positions shift after each merge).
            if use_rel:
                all_rel[run_start:run_start + run_len] = [_path_to_rel(merged_path)]
            all_basenames[run_start:run_start + run_len] = [os.path.basename(merged_path)]

        # Rebuild batches after all merges applied
        self._finalize_batches()

    def _seam_display_ys(self) -> List[int]:
        """Y positions of each seam line between consecutive images in the current batch.
        seam[i] is the line BETWEEN image i-1 and image i (for i >= 1)."""
        return list(self.image_offsets)

    def _local_indices_touched_by_rect(self, rect: QRect) -> List[int]:
        """Return local image indices whose display row overlaps rect by at least
        SEAM_CROSS_THRESHOLD pixels vertically."""
        if not self.image_offsets or not self.display_pixmaps:
            return []
        touched = []
        rect_top = rect.top()
        rect_bot = rect.bottom()
        for i in range(len(self.image_offsets)):
            img_top = self.image_offsets[i]
            img_h = self.display_heights[i] if i < len(self.display_heights) else 0
            img_bot = img_top + img_h
            overlap = max(0, min(rect_bot, img_bot) - max(rect_top, img_top))
            if overlap >= self.SEAM_CROSS_THRESHOLD:
                touched.append(i)
        return touched

    def _hover_merge_btn_rect_for(self, local_idx: int) -> QRect:
        """Compute the clickable rect for the hover merge button at the TOP seam of local_idx.
        Positioned at the right side of the display."""
        if local_idx <= 0 or local_idx >= len(self.image_offsets):
            return QRect()
        y = self.image_offsets[local_idx]
        size = self.MERGE_BTN_SIZE
        x = self.display_width - size - 10
        return QRect(x, y - size // 2, size, size)

    # -- YOLO integration ---

    def _image_width_at(self, global_idx: int) -> int:
        """Width of the image at a global index. Uses the loaded batch's
        original_sizes when possible, else reads the file header."""
        local = global_idx - self._batch_offset
        if 0 <= local < len(self.original_sizes):
            return self.original_sizes[local][0]
        if 0 <= global_idx < len(self.all_image_paths):
            try:
                reader = QImageReader(self.all_image_paths[global_idx])
                sz = reader.size()
                if sz.isValid():
                    return sz.width()
            except Exception:
                pass
        return 0

    def _stretch_and_merge_rows(self, boxes: List[ScriptBox]) -> List[ScriptBox]:
        """Stretch each detection to the full width of its image, then merge
        detections on the same image whose vertical ranges overlap — two
        panels sitting side by side become a single full-width zone."""
        by_img: Dict[int, List[ScriptBox]] = {}
        for b in boxes:
            by_img.setdefault(b.image_index, []).append(b)

        out: List[ScriptBox] = []
        n_merged = 0
        for img_idx, group in by_img.items():
            width = self._image_width_at(img_idx)
            group.sort(key=lambda b: b.y)
            merged: List[ScriptBox] = []
            for b in group:
                b.x = 0
                if width > 0:
                    b.w = width
                last = merged[-1] if merged else None
                if last is not None and b.y < last.y + last.h:
                    # Same horizontal band as the previous zone — absorb it
                    bottom = max(last.y + last.h, b.y + b.h)
                    last.h = bottom - last.y
                    last.confidence = max(last.confidence, b.confidence)
                    n_merged += 1
                else:
                    merged.append(b)
            out.extend(merged)
        if n_merged:
            logger.info(
                f"YOLO: merged {n_merged} side-by-side detection(s) into "
                f"full-width zones"
            )
        return out

    def run_yolo(self, detector: YoloDetector, min_size: int = 40,
                 auto_fix_cross_image: bool = True):
        """Run YOLO on current batch images, create ScriptBoxes with global indices.

        Stitches all batch images into a single tall strip first, then finds
        natural cut points (gaps between panels) and re-segments at those clean
        boundaries. This prevents panels that span two source images from being
        split by arbitrary file boundaries.

        Coordinates are mapped back to the original image space.

        Args:
            auto_fix_cross_image: when True (default), merges images for
                boxes that span source-image boundaries at the end of this
                call. The YOLO-All loop passes False because that merge
                triggers `_finalize_batches`, which mutates `self.batches`
                mid-iteration and crashes the loop. The caller is expected
                to run `_fix_cross_image_boxes()` once after all batches
                have been processed.
        """
        from app.image_splitter import stitch_split_for_yolo

        if not detector.is_available():
            return 0

        # Remove existing boxes for current batch before re-detecting
        batch_start = self._batch_offset
        batch_end = batch_start + len(self.image_paths)
        self.boxes = [b for b in self.boxes if not (batch_start <= b.image_index < batch_end)]

        # Stitch all batch images → find clean cuts → split into pieces
        pieces = stitch_split_for_yolo(self.image_paths)
        logger.info(f"YOLO: stitched {len(self.image_paths)} images into {len(pieces)} clean pieces")

        new_boxes: List[ScriptBox] = []
        dropped_small = 0
        for piece_img, orig_local_idx, y_offset, scale in pieces:
            global_idx = self._local_to_global(orig_local_idx)
            logger.info(f"YOLO: piece for image {global_idx}, y_offset={y_offset}, "
                        f"shape {piece_img.shape}, scale={scale:.3f}, "
                        f"running detection (conf={detector.confidence_threshold})")
            panels = detector.detect_and_filter(piece_img)
            logger.info(f"YOLO: found {len(panels)} panels in piece")
            for p in panels:
                # Map coordinates back to original image space
                box_w = int((p.x2 - p.x1) * scale)
                box_h = int((p.y2 - p.y1) * scale)
                if box_w < min_size or box_h < min_size:
                    dropped_small += 1
                    continue
                new_boxes.append(ScriptBox(
                    image_index=global_idx,
                    x=int(p.x1 * scale),
                    y=int((p.y1 + y_offset) * scale),
                    w=box_w,
                    h=box_h,
                    confidence=p.confidence,
                ))

        if dropped_small:
            logger.info(f"YOLO: dropped {dropped_small} boxes smaller than {min_size}px")

        # Stretch detections to the full image width and collapse
        # side-by-side panels (overlapping vertical bands) into one zone.
        new_boxes = self._stretch_and_merge_rows(new_boxes)

        logger.info(f"YOLO: total {len(new_boxes)} boxes across {len(self.image_paths)} images (batch {self.current_batch_index + 1})")
        self.boxes.extend(new_boxes)
        self._sort_boxes()

        # Auto-merge images for boxes that cross image boundaries.
        # YOLO's piece-center assignment can put a panel on one image
        # while its y/h extend past that image's real height (or start
        # before it). Detect those and merge the spanning images so the
        # box lives on a single valid canvas. Skipped when the caller is
        # batching multiple `run_yolo` calls (YOLO All) — the merge
        # rebuilds `self.batches`, which would crash the outer loop.
        if auto_fix_cross_image:
            merged_count = self._fix_cross_image_boxes()
            if merged_count:
                logger.info(f"YOLO: auto-merged {merged_count} cross-boundary panel group(s)")

        self.update()
        self.boxes_changed.emit()
        return len(new_boxes)

    def _fix_cross_image_boxes(self, all_batches: bool = False) -> int:
        """Find boxes whose coordinates extend outside their assigned image
        (above its top or below its bottom) and auto-merge the images they
        actually span. Single scan, batched merges, one batch reload.

        Performs ONE scan, collects the union of problem ranges, merges
        overlapping ones, then applies each merge via the data-only helper
        in descending global order so earlier positions stay valid. Finally
        calls _finalize_batches ONCE. This avoids the widget churn that
        blows up the Windows USER handle budget when many merges happen
        back-to-back.

        Args:
            all_batches: when True, scans every box across the whole
                project (not just the current batch). Source-image heights
                are read from disk via `_fast_image_height` (header-only,
                no pixmap load). YOLO All passes True at the end of its
                loop so cross-boundary boxes in non-final batches don't
                stay broken.
        """
        if not all_batches and not self.original_sizes:
            return 0

        problems_global: List[Tuple[int, int]] = []

        if all_batches:
            # Whole-project scan. Indices are global, heights come from
            # `all_image_paths` directly so we don't depend on whichever
            # batch happens to be loaded.
            n_paths = len(self.all_image_paths)
            # Cache heights as we walk to avoid re-reading the same file
            # if multiple boxes share an image.
            height_cache: dict = {}

            def _height_at(idx: int) -> int:
                if idx in height_cache:
                    return height_cache[idx]
                if idx < 0 or idx >= n_paths:
                    height_cache[idx] = 0
                    return 0
                h = self._fast_image_height(self.all_image_paths[idx])
                height_cache[idx] = h
                return h

            for box in self.boxes:
                idx = box.image_index
                if idx < 0 or idx >= n_paths:
                    continue
                orig_h = _height_at(idx)
                if orig_h <= 0:
                    continue

                start_idx = idx
                end_idx = idx
                if box.y + box.h > orig_h + 2:
                    overflow = (box.y + box.h) - orig_h
                    while overflow > 0 and end_idx + 1 < n_paths:
                        end_idx += 1
                        overflow -= _height_at(end_idx) or 0
                if box.y < -2:
                    overflow = -box.y
                    while overflow > 0 and start_idx > 0:
                        start_idx -= 1
                        overflow -= _height_at(start_idx) or 0

                if end_idx > start_idx:
                    problems_global.append((start_idx, end_idx))
        else:
            # Per-batch scan (original behavior). Uses the loaded pixmap
            # heights, which are the same numbers `_fast_image_height`
            # would produce — but they're already in memory so cheaper.
            batch_start_global = self._batch_offset
            for box in self.boxes:
                local = box.image_index - batch_start_global
                if local < 0 or local >= len(self.original_sizes):
                    continue
                _orig_w, orig_h = self.original_sizes[local]

                start_local = local
                end_local = local
                if box.y + box.h > orig_h + 2:
                    overflow = (box.y + box.h) - orig_h
                    while overflow > 0 and end_local + 1 < len(self.original_sizes):
                        end_local += 1
                        overflow -= self.original_sizes[end_local][1]
                if box.y < -2:
                    overflow = -box.y
                    while overflow > 0 and start_local > 0:
                        start_local -= 1
                        overflow -= self.original_sizes[start_local][1]

                if end_local > start_local:
                    problems_global.append((batch_start_global + start_local,
                                            batch_start_global + end_local))

        if not problems_global:
            return 0

        # Merge overlapping/adjacent ranges into a minimal set
        problems_global.sort()
        merged: List[List[int]] = [list(problems_global[0])]
        for s, e in problems_global[1:]:
            last = merged[-1]
            if s <= last[1] + 1:
                last[1] = max(last[1], e)
            else:
                merged.append([s, e])

        # Apply in descending order so earlier (lower-index) merges aren't
        # invalidated by shifting from later merges.
        merged.sort(reverse=True)

        count = 0
        for s, e in merged:
            if self._merge_global_range_dataonly(s, e):
                count += 1

        # One batch reload at the end — cheap, widget-safe, preserves the view
        self._finalize_batches(initial_batch=self.current_batch_index)
        logger.info(
            f"_fix_cross_image_boxes: performed {count} merges in one pass "
            f"(all_batches={all_batches})"
        )
        return count

    def clear_all(self):
        """Clear all images and boxes (for New Project)."""
        self.image_paths.clear()
        self._source_pixmaps.clear()
        self.display_pixmaps.clear()
        self.display_heights.clear()
        self._uniform_rows_cache.clear()
        self.original_sizes.clear()
        self.image_offsets.clear()
        self.scale_factors.clear()
        self.boxes.clear()
        self.all_image_paths.clear()
        self.batches.clear()
        self.batch_starts = [0]
        self.merge_groups.clear()
        self._hover_seam = None
        self._hover_merge_btn_rect = None
        self._active_cross_seams = []
        self.current_batch_index = 0
        self.selected_box_id = None
        self._cached_display_width = 0
        self.setFixedHeight(100)
        self.update()
        self.boxes_changed.emit()
        self.batch_changed.emit()

    def clear_boxes(self):
        self.boxes.clear()
        self.selected_box_id = None
        self.update()
        self.boxes_changed.emit()

    def select_box(self, box_id: str):
        """Select a box and center the viewport on it."""
        self.selected_box_id = box_id
        self.update()
        for box in self.boxes:
            if box.id == box_id:
                r = self._box_display_rect(box)
                parent_scroll = self.parent()
                if isinstance(parent_scroll, QWidget):
                    sa = parent_scroll.parent()
                    if isinstance(sa, QScrollArea):
                        # Center the box vertically in the viewport
                        vp_h = sa.viewport().height()
                        target_y = r.center().y() - vp_h // 2
                        sa.verticalScrollBar().setValue(max(0, target_y))
                break

    def select_next_box(self):
        """Select the next box (by sort order). Wraps to first if at end."""
        batch_boxes = self.current_batch_boxes()
        if not batch_boxes:
            return
        if not self.selected_box_id:
            self.select_box(batch_boxes[0].id)
            self.box_selected.emit(batch_boxes[0].id)
            return
        for i, box in enumerate(batch_boxes):
            if box.id == self.selected_box_id:
                nxt = batch_boxes[(i + 1) % len(batch_boxes)]
                self.select_box(nxt.id)
                self.box_selected.emit(nxt.id)
                return

    def select_prev_box(self):
        """Select the previous box (by sort order). Wraps to last if at start."""
        batch_boxes = self.current_batch_boxes()
        if not batch_boxes:
            return
        if not self.selected_box_id:
            self.select_box(batch_boxes[-1].id)
            self.box_selected.emit(batch_boxes[-1].id)
            return
        for i, box in enumerate(batch_boxes):
            if box.id == self.selected_box_id:
                prev = batch_boxes[(i - 1) % len(batch_boxes)]
                self.select_box(prev.id)
                self.box_selected.emit(prev.id)
                return


# ---------------------------------------------------------------------------
# Text Slot Widget (single slot)
# ---------------------------------------------------------------------------

class TextSlotWidget(QFrame):
    """A single text slot for a ScriptBox. Supports * as panel separator."""
    clicked = pyqtSignal(str)  # box_id
    text_changed = pyqtSignal(str, str)  # box_id, new_text

    def __init__(self, box_id: str, zone_number: int, text: str = "",
                 font_family: str = "Arial", font_size: int = 12,
                 font_color: str = "#ffffff", parent=None):
        super().__init__(parent)
        self.box_id = box_id
        self.zone_number = zone_number
        self._is_active = False
        # Review mode (admin "Load Delivery"): green text + the pre-edit
        # original shown struck-through in red below the editor.
        self._review_green = False
        self._diff_label: Optional[QLabel] = None
        self._cur_family = font_family
        self._cur_size = font_size

        self.setFrameShape(QFrame.StyledPanel)
        self.setObjectName("slotFrame")
        self._update_frame_style(False)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)
        self.setLayout(layout)

        # Header
        header = QHBoxLayout()
        self.zone_label = QLabel(f"Zone {zone_number}")
        self.zone_label.setStyleSheet(f"color: #00e5ff; font-weight: bold; font-size: {scale_manager.scale_font(11)}px; background: transparent;")
        header.addWidget(self.zone_label)

        header.addStretch()
        layout.addLayout(header)

        # Text editor — uses QPlainTextEdit for auto-sizing
        self.text_edit = QPlainTextEdit()
        self.text_edit.setTabChangesFocus(True)  # Tab navigates zones, not inserts whitespace
        self.text_edit.setUndoRedoEnabled(False)
        self.text_edit.setPlainText(text)
        self.text_edit.setMinimumHeight(scale_manager.scale(80))
        self.text_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.text_edit.document().contentsChanged.connect(self._auto_resize)
        self._font_color = font_color
        self._apply_text_style(font_family, font_size, font_color)
        self.text_edit.textChanged.connect(self._on_text_changed)
        layout.addWidget(self.text_edit)

        # Initial auto-resize after layout is ready
        QTimer.singleShot(50, self._auto_resize)
        QTimer.singleShot(200, self._auto_resize)

        self.setCursor(Qt.PointingHandCursor)

    def _apply_text_style(self, family: str, size: int, color: str):
        self._font_color = color
        self._cur_family = family
        self._cur_size = size
        if self._review_green:
            color = "#66bb6a"
        self.text_edit.setStyleSheet(f"""
            QPlainTextEdit {{
                background-color: #2a2a2a;
                color: {color};
                border: 1px solid #444;
                border-radius: 3px;
                padding: 4px;
                font-family: "{family}";
                font-size: {size}px;
            }}
            QPlainTextEdit:focus {{
                border: 1px solid #00bcd4;
            }}
        """)

    def update_font(self, family: str, size: int, color: str):
        self._apply_text_style(family, size, color)

    def _update_frame_style(self, active: bool):
        if active:
            self.setStyleSheet(
                "#slotFrame { background-color: #3a3a3a; border: 2px solid #ffd54f; border-radius: 4px; margin: 2px; }"
            )
        else:
            self.setStyleSheet(
                "#slotFrame { background-color: #333333; border: 1px solid #555; border-radius: 4px; margin: 2px; }"
            )

    def set_active(self, active: bool):
        self._is_active = active
        self._update_frame_style(active)

    def _auto_resize(self):
        """Grow the text edit to fit its content."""
        doc = self.text_edit.document()
        vw = self.text_edit.viewport().width()
        if vw > 50:
            doc.setTextWidth(vw)
        line_count = max(2, doc.lineCount())
        fm = self.text_edit.fontMetrics()
        line_h = fm.lineSpacing()
        h = (line_count * line_h) + 20  # lines + padding
        h = max(scale_manager.scale(80), min(h, scale_manager.scale(800)))
        self.text_edit.setMinimumHeight(h)
        self.text_edit.setMaximumHeight(h)

    def _on_text_changed(self):
        self.text_changed.emit(self.box_id, self.text_edit.toPlainText())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.box_id)
        super().mousePressEvent(event)

    def get_text(self) -> str:
        return self.text_edit.toPlainText()

    def set_zone_number(self, num: int):
        self.zone_number = num
        self.zone_label.setText(f"Zone {num}")

    def set_review_diff(self, original_text: str, size: Optional[int] = None):
        """Mark this slot as writer-edited: current (delivered) text turns
        green, and the pre-edit original appears below it in smaller,
        struck-through red. The red text is display-only — it is NOT part
        of the slot text, so saving/delivering is unaffected."""
        self._review_green = True
        self._review_original = original_text
        self._apply_text_style(self._cur_family, self._cur_size, self._font_color)
        if self._diff_label is None:
            self._diff_label = QLabel()
            self._diff_label.setWordWrap(True)
            self._diff_label.setTextFormat(Qt.RichText)
            self._diff_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._diff_label.setStyleSheet("background: transparent; padding: 2px 4px;")
            self.layout().addWidget(self._diff_label)
        small = size or max(8, self._cur_size - 2)
        esc = html.escape(original_text).replace("\n", "<br/>")
        self._diff_label.setText(
            f'<span style="font-family:\'{self._cur_family}\'; '
            f'font-size:{small}px; color:#e57373;">{esc}</span>')

    def clear_review_diff(self):
        self._review_green = False
        self._apply_text_style(self._cur_family, self._cur_size, self._font_color)
        if self._diff_label is not None:
            self._diff_label.deleteLater()
            self._diff_label = None

    def set_diff_visible(self, visible: bool):
        if self._diff_label is not None:
            self._diff_label.setVisible(visible)


# ---------------------------------------------------------------------------
# Text Slots Panel (right panel)
# ---------------------------------------------------------------------------

class TextSlotsPanel(QWidget):
    """Right panel: font controls + vertically stacked text slots."""
    slot_clicked = pyqtSignal(str)  # box_id
    text_changed = pyqtSignal()  # emitted when any slot's text changes

    def __init__(self, parent=None):
        super().__init__(parent)
        self.slots: Dict[str, TextSlotWidget] = {}
        self._font_family = "Arial"
        self._font_size = 12
        self._font_color = "#ffffff"
        # box_id -> pre-edit original text, set by the admin "Load Delivery"
        # review flow; survives rebuild() so pagination keeps the diffs.
        self.review_diffs: Dict[str, str] = {}

        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.setLayout(outer)

        # Font toolbar
        font_bar = QHBoxLayout()
        font_bar.setContentsMargins(4, 4, 4, 4)
        font_bar.setSpacing(4)

        self.font_combo = QFontComboBox()
        self.font_combo.setCurrentFont(QFont("Arial"))
        self.font_combo.setMaximumWidth(scale_manager.scale(140))
        self.font_combo.setStyleSheet(f"QFontComboBox {{ background: #404040; color: #fff; font-size: {scale_manager.scale_font(11)}px; }}")
        self.font_combo.currentFontChanged.connect(self._on_font_changed)
        font_bar.addWidget(self.font_combo)

        self.size_spin = QSpinBox()
        self.size_spin.setRange(8, 48)
        self.size_spin.setValue(12)
        self.size_spin.setSuffix("px")
        self.size_spin.setFixedWidth(scale_manager.scale(60))
        self.size_spin.valueChanged.connect(self._on_font_changed)
        font_bar.addWidget(self.size_spin)

        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(scale_manager.scale(24), scale_manager.scale(24))
        self.color_btn.setStyleSheet("QPushButton { background-color: #ffffff; border: 1px solid #888; border-radius: 3px; }")
        self.color_btn.setToolTip("Text color")
        self.color_btn.clicked.connect(self._pick_color)
        font_bar.addWidget(self.color_btn)

        # Review-mode eye: toggles the red pre-edit originals under edited
        # zones. Only visible while review diffs are loaded (Load Delivery).
        self.review_eye_btn = QPushButton("\U0001F441")
        self.review_eye_btn.setCheckable(True)
        self.review_eye_btn.setChecked(True)
        self.review_eye_btn.setToolTip(
            "Show/hide the original text (red) under the writer's edits")
        self.review_eye_btn.setStyleSheet(
            "QPushButton { background: #404040; color: #e57373; padding: 2px 8px; }"
            "QPushButton:!checked { color: #777; }")
        self.review_eye_btn.setVisible(False)
        self.review_eye_btn.toggled.connect(self._on_review_eye_toggled)
        font_bar.addWidget(self.review_eye_btn)

        self.review_size_spin = QSpinBox()
        self.review_size_spin.setRange(6, 48)
        self.review_size_spin.setValue(10)
        self.review_size_spin.setSuffix("px")
        self.review_size_spin.setFixedWidth(scale_manager.scale(60))
        self.review_size_spin.setToolTip("Size of the original text (red)")
        self.review_size_spin.setVisible(False)
        self.review_size_spin.valueChanged.connect(self._on_review_size_changed)
        font_bar.addWidget(self.review_size_spin)

        font_bar.addStretch()
        outer.addLayout(font_bar)

        # Slots area
        self._slots_widget = QWidget()
        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(4)
        self._layout.setAlignment(Qt.AlignTop)
        self._slots_widget.setLayout(self._layout)
        outer.addWidget(self._slots_widget, 1)

    def _on_font_changed(self):
        self._font_family = self.font_combo.currentFont().family()
        self._font_size = self.size_spin.value()
        for slot in self.slots.values():
            slot.update_font(self._font_family, self._font_size, self._font_color)

    def _pick_color(self):
        color = QColorDialog.getColor(QColor(self._font_color), self, "Text Color")
        if color.isValid():
            self._font_color = color.name()
            self.color_btn.setStyleSheet(
                f"QPushButton {{ background-color: {self._font_color}; border: 1px solid #888; border-radius: 3px; }}"
            )
            self._on_font_changed()

    def rebuild(self, boxes: List[ScriptBox], zone_numbers: Optional[List[int]] = None,
                force_from_box: bool = False):
        """Rebuild text slots to match current box order.

        By default preserves in-flight slot edits that haven't been flushed
        to `box.text` yet (pagination and box reorder rely on this).

        Pass `force_from_box=True` when box.text has been replaced with new
        text programmatically (Apply translation, Revert, view switch,
        flagged retranslate) and the stale slot contents must be discarded.
        Otherwise rebuild will recreate slots with the old text and the new
        box.text is never shown.

        zone_numbers: optional list of zone numbers (1-based) parallel to `boxes`.
        If omitted, falls back to local batch indexing.
        """
        existing_text: Dict[str, str] = {}
        if not force_from_box:
            for box_id, slot in self.slots.items():
                existing_text[box_id] = slot.get_text()

        while self._layout.count():
            child = self._layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.slots.clear()

        for i, box in enumerate(boxes):
            text = existing_text.get(box.id, box.text)
            zn = zone_numbers[i] if zone_numbers and i < len(zone_numbers) else i + 1
            slot = TextSlotWidget(
                box.id, zn, text,
                self._font_family, self._font_size, self._font_color,
            )
            slot.clicked.connect(self._on_slot_clicked)
            slot.text_changed.connect(self._on_text_changed)
            self.slots[box.id] = slot
            self._layout.addWidget(slot)
            if box.id in self.review_diffs:
                slot.set_review_diff(self.review_diffs[box.id],
                                     size=self.review_size_spin.value())
                slot.set_diff_visible(self.review_eye_btn.isChecked())

        self._layout.addStretch()

    def set_active_slot(self, box_id: Optional[str]):
        for bid, slot in self.slots.items():
            slot.set_active(bid == box_id)
        if box_id and box_id in self.slots:
            slot = self.slots[box_id]
            slot.ensurePolished()
            # Walk up to find the scroll area
            w = self.parent()
            while w and not isinstance(w, QScrollArea):
                w = w.parent()
            if isinstance(w, QScrollArea):
                w.ensureWidgetVisible(slot, 50, 50)

    def _on_slot_clicked(self, box_id: str):
        self.slot_clicked.emit(box_id)

    def _on_text_changed(self, box_id: str, text: str):
        self.text_changed.emit()

    def get_all_text(self) -> Dict[str, str]:
        return {bid: slot.get_text() for bid, slot in self.slots.items()}

    def set_review_diffs(self, diffs: Dict[str, str]):
        """Set (or clear, with {}) the review-mode text diffs and apply
        them to the currently built slots."""
        self.review_diffs = dict(diffs)
        for bid, slot in self.slots.items():
            if bid in self.review_diffs:
                slot.set_review_diff(self.review_diffs[bid],
                                     size=self.review_size_spin.value())
                slot.set_diff_visible(self.review_eye_btn.isChecked())
            else:
                slot.clear_review_diff()
        # The eye + size controls only make sense while reviewing a delivery.
        self.review_eye_btn.setVisible(bool(self.review_diffs))
        self.review_size_spin.setVisible(bool(self.review_diffs))

    def _on_review_eye_toggled(self, checked: bool):
        for slot in self.slots.values():
            slot.set_diff_visible(checked)

    def _on_review_size_changed(self, size: int):
        for bid, slot in self.slots.items():
            if bid in self.review_diffs:
                slot.set_review_diff(self.review_diffs[bid], size=size)


# ---------------------------------------------------------------------------
# Script Panel (main composite widget)
# ---------------------------------------------------------------------------

class ScriptPanel(QWidget):
    """Main script writing panel: image strip (left) + text slots (right)."""

    state_changed = pyqtSignal()

    def __init__(self, config: Optional[Config] = None, parent=None):
        super().__init__(parent)
        self.config = config or Config()
        self.yolo_detector: Optional[YoloDetector] = None
        self._current_folder: Optional[str] = None
        self._mixed_folders: List[str] = []
        self._project_input_tokens = 0
        self._project_output_tokens = 0
        self._project_cost_usd = 0.0
        self._auto_mode = False
        self._auto_retries = 0
        self._auto_max_retries = 3
        self._auto_batch_size = 10
        self._auto_zones_written = 0
        self._auto_skipped_zones: set = set()
        # Captured set of box IDs the auto loop is responsible for, locked
        # in at start. Lets the user switch to other batches and run YOLO
        # there while Auto keeps working on the original batch — even if
        # global zone numbers shift, we resolve box → current zone via id.
        self._auto_box_ids: List[str] = []
        self._auto_skipped_box_ids: set = set()
        # Box IDs in the currently-running batch (set by _auto_next_batch,
        # consumed by _on_ai_error_auto when a batch fails permanently).
        self._auto_current_box_ids: List[str] = []
        self._translate_panel = None
        self._undo = UndoManager(max_steps=25)
        self._undo_debounce = QTimer(self)
        self._undo_debounce.setSingleShot(True)
        self._undo_debounce.setInterval(500)
        self._undo_debounce.timeout.connect(self._push_undo_snapshot)

        self._setup_ui()
        self._init_yolo()
        self._connect_state_signals()

        # Global Tab intercept — catches Tab from ANY widget in the app
        QApplication.instance().installEventFilter(self)

    def set_translate_panel(self, translate_panel):
        """Link the TranslatePanel so the Script-tab language combo can
        flip the displayed box text between stored translations."""
        self._translate_panel = translate_panel

    def _on_ai_lang_changed(self, text: str):
        """Persist the Gemini writer language and flip the Script tab view
        to the matching stored translation (if any)."""
        self.config.set("gemini_language", text)
        if self._translate_panel:
            self._translate_panel.switch_view_language(text)

    def _connect_state_signals(self):
        """Connect internal signals to emit state_changed."""
        self.image_strip.boxes_changed.connect(self.state_changed)
        self.text_slots.text_changed.connect(self.state_changed)
        self.text_slots.font_combo.currentFontChanged.connect(lambda: self.state_changed.emit())
        self.text_slots.size_spin.valueChanged.connect(lambda: self.state_changed.emit())
        self.confidence_spin.valueChanged.connect(lambda: self.state_changed.emit())

        # Batch switching
        self.image_strip.batch_changed.connect(self._update_batch_bar)

        # Word count on text edits
        self.text_slots.text_changed.connect(self._update_word_count)

        # Write back to bridge scripts/ when user edits text
        self.text_slots.text_changed.connect(self._write_zone_to_bridge)

        # Undo: debounced snapshot on text/box changes
        self.image_strip.boxes_changed.connect(self._schedule_undo_snapshot)
        self.text_slots.text_changed.connect(self._schedule_undo_snapshot)

    def _install_undo_filters_on_slots(self):
        """Install event filter on all text slot editors for panel-level undo."""
        for slot in self.text_slots.slots.values():
            slot.text_edit.installEventFilter(self)

    def _schedule_undo_snapshot(self):
        """Debounce undo snapshots to avoid one per keystroke."""
        self._undo_debounce.start()

    def _push_undo_snapshot(self):
        """Save current text state to undo stack."""
        self._undo.push(self._get_text_snapshot())

    def _push_undo_now(self):
        """Immediately push snapshot (for discrete actions like apply-to-panels)."""
        self._undo_debounce.stop()
        self._undo.push(self._get_text_snapshot())

    def _get_text_snapshot(self) -> dict:
        """Capture all box texts + the transcription text loader."""
        texts = self.text_slots.get_all_text()
        for box in self.image_strip.boxes:
            if box.id in texts:
                box.text = texts[box.id]
        box_texts = {box.id: box.text for box in self.image_strip.boxes}
        loader_text = self.transcription_widget.text_edit.toPlainText()
        return {"box_texts": box_texts, "loader_text": loader_text}

    def _restore_text_snapshot(self, snapshot: dict):
        """Restore text state from an undo snapshot."""
        box_texts = snapshot.get("box_texts", {})
        for box in self.image_strip.boxes:
            if box.id in box_texts:
                box.text = box_texts[box.id]
                if box.id in self.text_slots.slots:
                    slot = self.text_slots.slots[box.id]
                    slot.text_edit.blockSignals(True)
                    slot.text_edit.setPlainText(box.text)
                    slot.text_edit.blockSignals(False)
        loader_text = snapshot.get("loader_text", "")
        self.transcription_widget.text_edit.blockSignals(True)
        self.transcription_widget.text_edit.setPlainText(loader_text)
        self.transcription_widget.text_edit.blockSignals(False)

    def _init_yolo(self):
        self.yolo_detector = YoloDetector()
        available = self.yolo_detector.is_available()
        if available:
            self.status_label.setText("YOLO model loaded")
        else:
            self.status_label.setText("YOLO model not found (place panel.pt in project root)")
        self.yolo_btn.setEnabled(available)
        self.yolo_all_btn.setEnabled(available and bool(getattr(self.image_strip, "batches", None)))

    def _setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        self.setLayout(main_layout)

        # -- Toolbar --
        toolbar = QHBoxLayout()
        toolbar.setSpacing(scale_manager.scale(12))

        self.load_btn = QPushButton("Load Folder")
        self.load_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #00bcd4; color: white; border: none;
                font-weight: bold;
                border-radius: {scale_manager.scale(4)}px; font-size: {scale_manager.scale_font(12)}px;
            }}
            QPushButton:hover {{ background-color: #00acc1; }}
        """)
        self.load_btn.clicked.connect(self._load_folder)
        toolbar.addWidget(self.load_btn)

        self.yolo_btn = QPushButton("YOLO")
        self.yolo_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #00e5ff; color: #000; border: none;
                font-weight: bold;
                border-radius: {scale_manager.scale(4)}px; font-size: {scale_manager.scale_font(12)}px;
            }}
            QPushButton:hover {{ background-color: #18ffff; }}
            QPushButton:disabled {{ background-color: #555; color: #888; }}
        """)
        self.yolo_btn.setEnabled(False)
        self.yolo_btn.clicked.connect(self._run_yolo)
        toolbar.addWidget(self.yolo_btn)

        self.yolo_all_btn = QPushButton("YOLO All")
        self.yolo_all_btn.setToolTip(
            "Run YOLO on every batch sequentially. Useful right after loading "
            "a long project so all panels exist before you start writing."
        )
        self.yolo_all_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #00bcd4; color: #000; border: none;
                font-weight: bold;
                border-radius: {scale_manager.scale(4)}px; font-size: {scale_manager.scale_font(12)}px;
            }}
            QPushButton:hover {{ background-color: #00acc1; }}
            QPushButton:disabled {{ background-color: #555; color: #888; }}
        """)
        self.yolo_all_btn.setEnabled(False)
        self.yolo_all_btn.clicked.connect(self._run_yolo_all)
        toolbar.addWidget(self.yolo_all_btn)

        toolbar.addWidget(QLabel("Confidence:"))
        self.confidence_spin = QSpinBox()
        self.confidence_spin.setRange(5, 95)
        self.confidence_spin.setValue(25)
        self.confidence_spin.setSuffix("%")
        self.confidence_spin.setFixedWidth(80)
        toolbar.addWidget(self.confidence_spin)

        toolbar.addWidget(QLabel("Min size:"))
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(0, 500)
        self.min_size_spin.setValue(int(self.config.get("yolo_min_size", 40)))
        self.min_size_spin.setSuffix("px")
        self.min_size_spin.setFixedWidth(80)
        self.min_size_spin.valueChanged.connect(
            lambda v: self.config.set("yolo_min_size", int(v))
        )
        toolbar.addWidget(self.min_size_spin)

        self.clear_btn = QPushButton("Clear Boxes")
        self.clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #d32f2f; color: white; border: none;
                border-radius: {scale_manager.scale(4)}px; font-size: {scale_manager.scale_font(12)}px;
            }}
            QPushButton:hover {{ background-color: #b71c1c; }}
        """)
        self.clear_btn.clicked.connect(self._clear_boxes)
        toolbar.addWidget(self.clear_btn)

        self.export_btn = QPushButton("Export Script")
        self.export_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #4caf50; color: white; border: none;
                font-weight: bold;
                border-radius: {scale_manager.scale(4)}px; font-size: {scale_manager.scale_font(12)}px;
            }}
            QPushButton:hover {{ background-color: #43a047; }}
        """)
        self.export_btn.clicked.connect(self._export_script)
        toolbar.addWidget(self.export_btn)

        self.claude_bridge = ClaudeBridge() if ClaudeBridge else None

        # AI Writer button
        self._gemini_writer = None
        self._gemini_thread = None
        self.ai_write_btn = QPushButton("AI Write")
        self.ai_write_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #e65100; color: white; border: none;
                font-weight: bold;
                border-radius: {scale_manager.scale(4)}px; font-size: {scale_manager.scale_font(12)}px;
            }}
            QPushButton:hover {{ background-color: #bf360c; }}
            QPushButton:checked {{ background-color: #ff6d00; }}
        """)
        self.ai_write_btn.setCheckable(True)
        self.ai_write_btn.clicked.connect(self._toggle_ai_write)
        toolbar.addWidget(self.ai_write_btn)

        self.auto_btn = QPushButton("Auto")
        self.auto_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #6a1b9a; color: white; border: none;
                font-weight: bold;
                border-radius: {scale_manager.scale(4)}px; font-size: {scale_manager.scale_font(12)}px;
            }}
            QPushButton:hover {{ background-color: #8e24aa; }}
            QPushButton:checked {{ background-color: #ab47bc; }}
        """)
        self.auto_btn.setCheckable(True)
        self.auto_btn.clicked.connect(self._toggle_auto_mode)
        toolbar.addWidget(self.auto_btn)

        combo_style = f"""
            QComboBox {{
                background-color: #404040; color: #fff;
                padding: {scale_manager.scale(4)}px {scale_manager.scale(8)}px;
                border-radius: {scale_manager.scale(3)}px; font-size: {scale_manager.scale_font(11)}px;
            }}
        """

        self.ai_model_combo = QComboBox()
        self.ai_model_combo.addItems([
            "GPT-5.6 Terra",
            "3 Flash",
            "3.1 Pro",
            "3.1 Flash-Lite",
        ])
        # Map display names to model IDs
        self._model_id_map = {
            "GPT-5.6 Terra": "gpt-5.6-terra",
            "3 Flash": "gemini-3-flash-preview",
            "3.1 Pro": "gemini-3.1-pro-preview",
            "3.1 Flash-Lite": "gemini-3.1-flash-lite-preview",
        }
        saved_model = self.config.get("gemini_model", "gemini-3-flash-preview")
        # Migrate configs from gpt-5.5 / 5.6-sol (Sol burned too much for
        # this task — Terra is half the price)
        if saved_model in ("gpt-5.5", "gpt-5.6-sol"):
            saved_model = "gpt-5.6-terra"
            self.config.set("gemini_model", saved_model)
        # Find saved model in map and select it
        for display_name, model_id in self._model_id_map.items():
            if model_id == saved_model:
                idx = self.ai_model_combo.findText(display_name)
                if idx >= 0:
                    self.ai_model_combo.setCurrentIndex(idx)
                break
        self.ai_model_combo.setStyleSheet(combo_style)
        self.ai_model_combo.currentTextChanged.connect(
            lambda t: self.config.set("gemini_model", self._model_id_map.get(t, t))
        )
        toolbar.addWidget(self.ai_model_combo)

        self.ai_lang_combo = QComboBox()
        self.ai_lang_combo.addItems(["English", "Portuguese"])
        saved_lang = self.config.get("gemini_language", "English")
        idx = self.ai_lang_combo.findText(saved_lang)
        if idx >= 0:
            self.ai_lang_combo.setCurrentIndex(idx)
        self.ai_lang_combo.setStyleSheet(combo_style)
        self.ai_lang_combo.currentTextChanged.connect(self._on_ai_lang_changed)
        toolbar.addWidget(self.ai_lang_combo)

        self.gemini_key_btn = QPushButton("Keys")
        self.gemini_key_btn.setToolTip(
            "Set the Google (Gemini) and OpenAI (GPT second-pass) API keys."
        )
        self.gemini_key_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #404040; color: #ccc; border: none;
                border-radius: {scale_manager.scale(3)}px; font-size: {scale_manager.scale_font(11)}px;
            }}
            QPushButton:hover {{ background-color: #505050; }}
        """)
        self.gemini_key_btn.clicked.connect(self._edit_api_keys)
        toolbar.addWidget(self.gemini_key_btn)

        self.gemini_prompt_btn = QPushButton("Prompt Gem")
        self.gemini_prompt_btn.setToolTip(
            "Edit the system prompt Gemini uses to write the FIRST DRAFT."
        )
        self.gemini_prompt_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #404040; color: #ccc; border: none;
                border-radius: {scale_manager.scale(3)}px; font-size: {scale_manager.scale_font(11)}px;
            }}
            QPushButton:hover {{ background-color: #505050; }}
        """)
        self.gemini_prompt_btn.clicked.connect(self._edit_ai_prompt)
        toolbar.addWidget(self.gemini_prompt_btn)

        self.gpt_prompt_btn = QPushButton("Prompt GPT")
        self.gpt_prompt_btn.setToolTip(
            "Edit the system prompt GPT uses for the SECOND PASS revision "
            "that runs after each Gemini batch."
        )
        self.gpt_prompt_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #404040; color: #ccc; border: none;
                border-radius: {scale_manager.scale(3)}px; font-size: {scale_manager.scale_font(11)}px;
            }}
            QPushButton:hover {{ background-color: #505050; }}
        """)
        self.gpt_prompt_btn.clicked.connect(self._edit_gpt_prompt)
        toolbar.addWidget(self.gpt_prompt_btn)

        self.gemini_chars_btn = QPushButton("Characters")
        self.gemini_chars_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #404040; color: #ccc; border: none;
                border-radius: {scale_manager.scale(3)}px; font-size: {scale_manager.scale_font(11)}px;
            }}
            QPushButton:hover {{ background-color: #505050; }}
        """)
        self.gemini_chars_btn.clicked.connect(self._edit_characters)
        toolbar.addWidget(self.gemini_chars_btn)

        self.import_part1_btn = QPushButton("Import Part 1")
        self.import_part1_btn.setToolTip(
            "Import a character_sheet bundle exported by momo-rewrite "
            "(characters + reference images + plot summary + Part 1 ending) "
            "to seed the story bible for a Part 2 continuation."
        )
        self.import_part1_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #404040; color: #ccc; border: none;
                border-radius: {scale_manager.scale(3)}px; font-size: {scale_manager.scale_font(11)}px;
            }}
            QPushButton:hover {{ background-color: #505050; }}
        """)
        self.import_part1_btn.clicked.connect(self._import_character_sheet)
        toolbar.addWidget(self.import_part1_btn)

        # GPT-mode toggle: when checked, every Gemini batch is followed by a
        # second-pass GPT revision. When unchecked, Gemini's draft is used
        # as-is (saves money + time when the user trusts Gemini alone).
        self.gpt_mode_check = QCheckBox("GPT Mode?")
        self.gpt_mode_check.setToolTip(
            "When enabled, GPT revises every Gemini batch as a second pass. "
            "When disabled, Gemini's draft is written without a GPT revision."
        )
        self.gpt_mode_check.setChecked(
            bool(self.config.get("ai_gpt_pass_enabled", True))
        )
        self.gpt_mode_check.setStyleSheet(f"""
            QCheckBox {{
                color: #ccc;
                font-size: {scale_manager.scale_font(11)}px;
                padding: 0 {scale_manager.scale(4)}px;
            }}
            QCheckBox::indicator {{
                width: {scale_manager.scale(14)}px;
                height: {scale_manager.scale(14)}px;
            }}
        """)
        self.gpt_mode_check.toggled.connect(self._on_gpt_mode_toggled)
        toolbar.addWidget(self.gpt_mode_check)

        # Gemini loading spinner (hover = red X, click = cancel next calls)
        self._ai_spinner = _SpinnerWidget(scale_manager.scale(20))
        self._ai_spinner.setVisible(False)
        self._ai_spinner.clicked.connect(self._cancel_ai_write)
        toolbar.addWidget(self._ai_spinner)

        # AI cost label (persistent, shows cumulative project cost)
        self._ai_cost_label = QLabel("")
        self._ai_cost_label.setStyleSheet(f"color: #888; font-size: {scale_manager.scale_font(10)}px;")
        toolbar.addWidget(self._ai_cost_label)

        # Responsive horizontal padding: each button keeps its text fully
        # visible at any window width. When there's space, padding is at
        # `max_hpad`. When the window is squeezed, padding shrinks down to
        # `min_hpad`, so the border approaches the text but never clips it.
        # The layout's setSpacing keeps a fixed gap between buttons.
        _ResponsivePadding(
            self.load_btn, vpad=scale_manager.scale(8),
            max_hpad=scale_manager.scale(16), min_hpad=scale_manager.scale(4),
        )
        _ResponsivePadding(
            self.yolo_btn, vpad=scale_manager.scale(8),
            max_hpad=scale_manager.scale(16), min_hpad=scale_manager.scale(4),
        )
        _ResponsivePadding(
            self.yolo_all_btn, vpad=scale_manager.scale(8),
            max_hpad=scale_manager.scale(16), min_hpad=scale_manager.scale(4),
        )
        _ResponsivePadding(
            self.clear_btn, vpad=scale_manager.scale(8),
            max_hpad=scale_manager.scale(12), min_hpad=scale_manager.scale(4),
        )
        _ResponsivePadding(
            self.export_btn, vpad=scale_manager.scale(8),
            max_hpad=scale_manager.scale(12), min_hpad=scale_manager.scale(4),
        )
        _ResponsivePadding(
            self.ai_write_btn, vpad=scale_manager.scale(8),
            max_hpad=scale_manager.scale(12), min_hpad=scale_manager.scale(4),
        )
        _ResponsivePadding(
            self.auto_btn, vpad=scale_manager.scale(8),
            max_hpad=scale_manager.scale(12), min_hpad=scale_manager.scale(4),
        )
        _ResponsivePadding(
            self.gemini_key_btn, vpad=scale_manager.scale(6),
            max_hpad=scale_manager.scale(10), min_hpad=scale_manager.scale(3),
        )
        _ResponsivePadding(
            self.gemini_prompt_btn, vpad=scale_manager.scale(6),
            max_hpad=scale_manager.scale(10), min_hpad=scale_manager.scale(3),
        )
        _ResponsivePadding(
            self.gpt_prompt_btn, vpad=scale_manager.scale(6),
            max_hpad=scale_manager.scale(10), min_hpad=scale_manager.scale(3),
        )
        _ResponsivePadding(
            self.gemini_chars_btn, vpad=scale_manager.scale(6),
            max_hpad=scale_manager.scale(10), min_hpad=scale_manager.scale(3),
        )

        if IS_USER:
            # Writers' edition: no AI writing. Widgets are still created so
            # every attribute exists (no scattered hasattr checks) — they're
            # just never shown. Characters stays visible: it only reads
            # story_bible.json and helps writers keep names straight.
            for w in (self.ai_write_btn, self.auto_btn, self.ai_model_combo,
                      self.ai_lang_combo, self.gemini_key_btn,
                      self.gemini_prompt_btn, self.gpt_prompt_btn,
                      self.import_part1_btn, self.gpt_mode_check,
                      self._ai_cost_label):
                w.setVisible(False)

        toolbar.addStretch()

        # Zoom, box-count, word-count and status have been moved to the batch
        # bar below, so they sit next to the image navigator (where they
        # belong logically).

        main_layout.addLayout(toolbar)

        # -- Batch selector bar (compact paginator) --
        self.batch_bar = QWidget()
        self.batch_bar_layout = QHBoxLayout()
        self.batch_bar_layout.setContentsMargins(4, 2, 4, 2)
        self.batch_bar_layout.setSpacing(scale_manager.scale(10))
        self.batch_bar.setLayout(self.batch_bar_layout)
        self.batch_bar.setStyleSheet(
            "QWidget#batchBar { background-color: #353535; border: 1px solid #555; border-radius: 3px; }"
        )
        self.batch_bar.setObjectName("batchBar")

        self._batch_label = QLabel("Batch:")
        self._batch_label.setStyleSheet("font-weight: bold; border: none; color: #ccc;")
        self.batch_bar_layout.addWidget(self._batch_label)

        btn_style = """
            QPushButton {
                padding: 3px 10px; border-radius: 3px;
                background-color: #404040; color: #ccc; border: none;
                min-width: 28px; font-weight: bold;
            }
            QPushButton:hover { background-color: #505050; }
            QPushButton:disabled { background-color: #2d2d2d; color: #555; }
        """
        self.batch_prev_btn = QPushButton("◀")
        self.batch_prev_btn.setStyleSheet(btn_style)
        self.batch_prev_btn.setToolTip("Previous batch (PgUp)")
        self.batch_prev_btn.clicked.connect(self._batch_prev)
        self.batch_bar_layout.addWidget(self.batch_prev_btn)

        self.batch_combo = QComboBox()
        self.batch_combo.setStyleSheet("""
            QComboBox {
                padding: 3px 10px; border-radius: 3px;
                background-color: #404040; color: #ddd; border: none;
                font-weight: bold; min-width: 160px;
            }
            QComboBox:hover { background-color: #505050; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox QAbstractItemView {
                background-color: #2b2b2b; color: #ddd;
                selection-background-color: #0078d4;
                border: 1px solid #555;
            }
        """)
        self.batch_combo.currentIndexChanged.connect(self._on_batch_combo_changed)
        self.batch_bar_layout.addWidget(self.batch_combo)

        self.batch_next_btn = QPushButton("▶")
        self.batch_next_btn.setStyleSheet(btn_style)
        self.batch_next_btn.setToolTip("Next batch (PgDn)")
        self.batch_next_btn.clicked.connect(self._batch_next)
        self.batch_bar_layout.addWidget(self.batch_next_btn)

        self.batch_count_label = QLabel("")
        self.batch_count_label.setStyleSheet("color: #888; border: none; font-size: 11px;")
        self.batch_bar_layout.addWidget(self.batch_count_label)

        # Visual separator between the batch navigator and the zoom controls.
        self._batch_sep1 = QLabel("│")
        self._batch_sep1.setStyleSheet("color: #555; border: none;")
        self.batch_bar_layout.addWidget(self._batch_sep1)

        # -- Zoom controls (next to the image navigator) --
        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.setFixedSize(scale_manager.scale(28), scale_manager.scale(28))
        self.zoom_out_btn.setStyleSheet(f"QPushButton {{ font-weight: bold; font-size: {scale_manager.scale_font(14)}px; border: none; }}")
        self.zoom_out_btn.clicked.connect(lambda: self._apply_zoom(self._zoom_level / 1.25))
        self.batch_bar_layout.addWidget(self.zoom_out_btn)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setStyleSheet(f"color: #aaa; border: none; font-size: {scale_manager.scale_font(11)}px; min-width: {scale_manager.scale(40)}px;")
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.batch_bar_layout.addWidget(self.zoom_label)

        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedSize(scale_manager.scale(28), scale_manager.scale(28))
        self.zoom_in_btn.setStyleSheet(f"QPushButton {{ font-weight: bold; font-size: {scale_manager.scale_font(14)}px; border: none; }}")
        self.zoom_in_btn.clicked.connect(lambda: self._apply_zoom(self._zoom_level * 1.25))
        self.batch_bar_layout.addWidget(self.zoom_in_btn)

        self.zoom_reset_btn = QPushButton("Fit")
        self.zoom_reset_btn.setFixedSize(scale_manager.scale(36), scale_manager.scale(28))
        self.zoom_reset_btn.setStyleSheet(f"QPushButton {{ font-size: {scale_manager.scale_font(11)}px; border: none; }}")
        self.zoom_reset_btn.setToolTip("Reset zoom to fit width")
        self.zoom_reset_btn.clicked.connect(lambda: self._apply_zoom(1.0))
        self.batch_bar_layout.addWidget(self.zoom_reset_btn)

        # Separator between zoom and box count.
        self._batch_sep2 = QLabel("│")
        self._batch_sep2.setStyleSheet("color: #555; border: none;")
        self.batch_bar_layout.addWidget(self._batch_sep2)

        # -- Box count (live total for current batch) --
        self.box_count_label = QLabel("Boxes: 0")
        self.box_count_label.setStyleSheet("color: #00e5ff; font-weight: bold; border: none;")
        self.batch_bar_layout.addWidget(self.box_count_label)

        # -- Word count --
        self.word_count_label = QLabel("Words: 0")
        self.word_count_label.setStyleSheet("color: #ff9800; font-weight: bold; border: none;")
        self.batch_bar_layout.addWidget(self.word_count_label)

        self.batch_bar_layout.addStretch()

        # -- Status (right-aligned by the stretch above) --
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(f"color: #888; border: none; font-size: {scale_manager.scale_font(11)}px;")
        self.batch_bar_layout.addWidget(self.status_label)

        main_layout.addWidget(self.batch_bar)

        # -- Splitter: image strip (left) | text slots (right) --
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #555;
                width: 3px;
            }
        """)

        # Left: image strip in scroll area
        self._zoom_level = 1.0  # 1.0 = fit to viewport width
        self.image_strip = ImageStripWidget()
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidget(self.image_strip)
        self.image_scroll.setWidgetResizable(False)
        self.image_scroll.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.image_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.image_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.image_scroll.setStyleSheet("QScrollArea { background-color: #1e1e1e; border: 1px solid #555; }")
        self.image_scroll.viewport().installEventFilter(self)
        self.splitter.addWidget(self.image_scroll)

        # Right: transcription widget + text slots in scroll area
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_widget.setLayout(right_layout)

        # Transcription widget (collapsible)
        from ui.transcription_widget import TranscriptionWidget
        self.transcription_widget = TranscriptionWidget(self.config)
        self.transcription_widget.apply_text.connect(self._apply_transcription_text)
        self.transcription_widget.text_edit.setUndoRedoEnabled(False)
        self.transcription_widget.text_edit.installEventFilter(self)
        right_layout.addWidget(self.transcription_widget)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("QFrame { color: #555; }")
        right_layout.addWidget(sep)

        self.text_slots = TextSlotsPanel()
        self.text_scroll = QScrollArea()
        self.text_scroll.setWidget(self.text_slots)
        self.text_scroll.setWidgetResizable(True)
        self.text_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.text_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.text_scroll.setStyleSheet("QScrollArea { background-color: #2b2b2b; border: 1px solid #555; }")
        right_layout.addWidget(self.text_scroll, 1)

        right_widget.setMinimumWidth(scale_manager.scale(250))
        self.splitter.addWidget(right_widget)

        # Default split: 60% left, 40% right
        self.splitter.setSizes([600, 400])
        main_layout.addWidget(self.splitter, 1)

        # -- Connect signals --
        self.image_strip.boxes_changed.connect(self._on_boxes_changed)
        self.image_strip.box_deleting.connect(self._on_box_deleting)
        self.image_strip.box_selected.connect(self._on_box_selected)
        self.text_slots.slot_clicked.connect(self._on_slot_clicked)
        self.splitter.splitterMoved.connect(lambda: QTimer.singleShot(0, self._fit_strip_to_viewport))

    # -- Batch bar ---

    def _update_batch_bar(self):
        """Refresh the batch paginator widgets. The bar itself stays visible
        because it also hosts the zoom and box-count controls."""
        batches = self.image_strip.batches
        has_batches = len(batches) > 1
        # Toggle visibility on the navigation widgets only, not the whole bar.
        self._batch_label.setVisible(has_batches)
        self.batch_prev_btn.setVisible(has_batches)
        self.batch_combo.setVisible(has_batches)
        self.batch_next_btn.setVisible(has_batches)
        self.batch_count_label.setVisible(has_batches)
        self._batch_sep1.setVisible(has_batches)
        if not has_batches:
            return

        batch_starts = self.image_strip.batch_starts
        total_batches = len(batches)
        current = self.image_strip.current_batch_index

        # Rebuild combo without re-triggering the handler
        self.batch_combo.blockSignals(True)
        self.batch_combo.clear()
        for i in range(total_batches):
            start = batch_starts[i] + 1
            end = batch_starts[i] + len(batches[i])
            self.batch_combo.addItem(f"{i + 1}/{total_batches}  ·  images {start}-{end}")
        self.batch_combo.setCurrentIndex(current)
        self.batch_combo.blockSignals(False)

        self.batch_prev_btn.setEnabled(current > 0)
        self.batch_next_btn.setEnabled(current < total_batches - 1)
        self.batch_count_label.setText(f"({len(self.image_strip.image_paths)} images)")

    def _switch_batch(self, batch_index: int):
        """Save current text, switch the image strip batch, update UI."""
        batches = self.image_strip.batches
        if not batches:
            return
        batch_index = max(0, min(batch_index, len(batches) - 1))
        if batch_index == self.image_strip.current_batch_index:
            return

        # Save text from current slots back to boxes before switching
        texts = self.text_slots.get_all_text()
        for box in self.image_strip.boxes:
            if box.id in texts:
                box.text = texts[box.id]

        self.image_strip.switch_batch(batch_index)

        # Update paginator state (signals blocked to avoid recursion)
        self.batch_combo.blockSignals(True)
        self.batch_combo.setCurrentIndex(batch_index)
        self.batch_combo.blockSignals(False)
        total_batches = len(batches)
        self.batch_prev_btn.setEnabled(batch_index > 0)
        self.batch_next_btn.setEnabled(batch_index < total_batches - 1)
        self.batch_count_label.setText(f"({len(self.image_strip.image_paths)} images)")

        self.status_label.setText(
            f"Batch {batch_index + 1}/{total_batches} ({len(self.image_strip.image_paths)} images)"
        )

    def _on_batch_combo_changed(self, index: int):
        if index < 0:
            return
        self._switch_batch(index)

    def _batch_prev(self):
        self._switch_batch(self.image_strip.current_batch_index - 1)

    def _batch_next(self):
        self._switch_batch(self.image_strip.current_batch_index + 1)

    # -- Actions --

    def _load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.load_images_from_folder(folder)

    def load_images_from_folder(self, folder: str):
        """Load images from a folder into the strip."""
        self._current_folder = folder
        self._mixed_folders = []
        self._zoom_level = 1.0
        self.zoom_label.setText("100%")
        display_w = self._base_width()
        self.image_strip.display_width = display_w
        self.image_strip.load_images(folder)
        # Initial undo snapshot
        self._undo.clear()
        QTimer.singleShot(100, self._push_undo_now)

        total = len(self.image_strip.all_image_paths)
        batch_count = len(self.image_strip.batches)
        if batch_count > 1:
            self.status_label.setText(
                f"Loaded {total} images ({batch_count} batches) from {os.path.basename(folder)}"
            )
        else:
            self.status_label.setText(f"Loaded {total} images from {os.path.basename(folder)}")
        if self.yolo_detector and self.yolo_detector.is_available():
            self.yolo_btn.setEnabled(total > 0)
            self.yolo_all_btn.setEnabled(total > 0)

    def mix_batch(self, parent_folder: str):
        """Scan a parent folder for chapter subfolders, merge all images + boxes into one project.

        For each subfolder (sorted naturally), loads images from 'out/' if it exists,
        otherwise from the subfolder itself. If a .mscript exists, imports its boxes
        with adjusted image_index offsets.
        """
        import re
        from app.project import load_project

        # Mix Batch starts a fresh project — zero out the cost accumulators
        # so the new mixed project doesn't inherit token/$$ from whatever
        # project was loaded before this call.
        self.reset_project_costs()

        def _natural_sort_key(s: str):
            return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

        # Find chapter subfolders
        subdirs = sorted(
            (d for d in os.listdir(parent_folder)
             if os.path.isdir(os.path.join(parent_folder, d))),
            key=_natural_sort_key,
        )

        all_folders = []
        all_boxes = []
        merged_translations: dict = {}
        image_offset = 0

        for d in subdirs:
            sub_path = os.path.join(parent_folder, d)

            # Image folder: prefer 'out/' subfolder, fall back to the folder itself
            out_path = os.path.join(sub_path, "out")
            image_folder = out_path if os.path.isdir(out_path) else sub_path

            paths = ImageStripWidget._collect_images(image_folder)
            if not paths:
                continue

            all_folders.append(image_folder)

            # Try to load .mscript for boxes + translations
            mscript = os.path.join(sub_path, "project.mscript")
            if os.path.isfile(mscript):
                try:
                    state = load_project(mscript)
                    for bd in state.get("boxes", []):
                        all_boxes.append(ScriptBox(
                            id=bd["id"],
                            image_index=bd["image_index"] + image_offset,
                            x=bd["x"], y=bd["y"],
                            w=bd["w"], h=bd["h"],
                            confidence=bd.get("confidence", 1.0),
                            text=bd.get("text", ""),
                        ))
                    # Merge per-language {box_id: text} maps across chapters.
                    # Box ids are UUIDs so cross-chapter collisions are not expected.
                    for lang, id_map in (state.get("translations") or {}).items():
                        if isinstance(id_map, dict):
                            merged_translations.setdefault(lang, {}).update(id_map)
                except Exception:
                    pass  # no boxes from this folder, that's fine

            image_offset += len(paths)

        if not all_folders:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Mix Batch", "No subfolders with images found.")
            return None

        # Load all images
        self._current_folder = parent_folder
        self._mixed_folders = all_folders
        self._zoom_level = 1.0
        self.zoom_label.setText("100%")
        display_w = self._base_width()
        self.image_strip.display_width = display_w
        self.image_strip.load_multiple_folders(all_folders)

        # Inject merged boxes
        if all_boxes:
            self.image_strip.boxes = all_boxes
            self.image_strip._sort_boxes()
            self.image_strip.update()

        # Rebuild text slots
        batch_boxes = self.image_strip.current_batch_boxes()
        global_zone_num = {b.id: i + 1 for i, b in enumerate(self.image_strip.boxes)}
        zone_numbers = [global_zone_num[b.id] for b in batch_boxes]
        self.text_slots.rebuild(batch_boxes, zone_numbers=zone_numbers)
        self._install_undo_filters_on_slots()
        self.box_count_label.setText(f"Boxes: {len(self.image_strip.boxes)}")

        self._undo.clear()
        QTimer.singleShot(100, self._push_undo_now)

        total = len(self.image_strip.all_image_paths)
        batch_count = len(self.image_strip.batches)
        self.status_label.setText(
            f"Mixed {len(all_folders)} folders — {total} images, {len(all_boxes)} boxes"
            + (f" ({batch_count} batches)" if batch_count > 1 else "")
        )
        if self.yolo_detector and self.yolo_detector.is_available():
            self.yolo_btn.setEnabled(total > 0)
            self.yolo_all_btn.setEnabled(total > 0)

        self.state_changed.emit()

        return {
            "folders": all_folders,
            "translations": merged_translations,
            "image_count": total,
            "box_count": len(all_boxes),
        }

    def _run_yolo(self):
        if not self.image_strip.image_paths:
            self.status_label.setText("No images loaded")
            return

        conf = self.confidence_spin.value() / 100.0
        self.yolo_detector.confidence_threshold = conf
        min_size = int(self.min_size_spin.value())

        self.yolo_btn.setEnabled(False)
        self.yolo_all_btn.setEnabled(False)
        batch_info = ""
        if len(self.image_strip.batches) > 1:
            batch_info = f" (batch {self.image_strip.current_batch_index + 1})"
        self.status_label.setText(f"Running YOLO detection{batch_info}...")
        QApplication.processEvents()

        count = self.image_strip.run_yolo(self.yolo_detector, min_size=min_size)

        self.yolo_btn.setEnabled(True)
        self.yolo_all_btn.setEnabled(True)
        self.status_label.setText(f"YOLO done: {count} boxes detected{batch_info}")

    def _run_yolo_all(self):
        """Run YOLO on every batch sequentially. Confirms first because on a
        long project this can take a while (each batch loads a fresh strip,
        runs detection, then advances)."""
        if not self.image_strip.batches:
            self.status_label.setText("No batches loaded")
            return

        n_batches = len(self.image_strip.batches)

        # Pre-compute which batches already have boxes so the dialog can
        # tell the user how many will actually run, and so the loop can
        # skip them without losing manually-edited work.
        boxes_per_batch = self._boxes_per_batch_count()
        empty_batches = [i for i, n in enumerate(boxes_per_batch) if n == 0]
        n_to_process = len(empty_batches)
        n_skip = n_batches - n_to_process

        if n_to_process == 0:
            QMessageBox.information(
                self, "Nothing to do",
                f"All {n_batches} batches already have boxes. "
                f"Use the single 'YOLO' button on a batch to re-run "
                f"detection there (it wipes and replaces that batch only)."
            )
            return

        skip_msg = (
            f" ({n_skip} batch(es) already have boxes — skipped to "
            f"preserve manual edits)"
            if n_skip else ""
        )
        reply = QMessageBox.question(
            self, "Run YOLO on all batches?",
            f"Wanna run yolo model on all batches?\n\n"
            f"This will process {n_to_process}/{n_batches} batch(es)"
            f"{skip_msg}.\n\n"
            f"To re-run a batch that already has boxes, use the single "
            f"'YOLO' button on that batch instead.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        conf = self.confidence_spin.value() / 100.0
        self.yolo_detector.confidence_threshold = conf
        min_size = int(self.min_size_spin.value())

        original_batch = self.image_strip.current_batch_index
        self.yolo_btn.setEnabled(False)
        self.yolo_all_btn.setEnabled(False)
        QApplication.processEvents()

        total_boxes = 0
        try:
            for processed_idx, i in enumerate(empty_batches):
                # `self.image_strip.batches` is normally stable during this
                # loop because we pass `auto_fix_cross_image=False` below,
                # but stay defensive in case some other code path mutates
                # it (e.g. user-triggered merge from another widget).
                if i >= len(self.image_strip.batches):
                    logger.warning(
                        f"YOLO All: batches shrank from {n_batches} to "
                        f"{len(self.image_strip.batches)} during run; stopping at batch {i}"
                    )
                    break
                self.image_strip._load_batch_pixmaps(i)
                self.image_strip.batch_changed.emit()
                self.status_label.setText(
                    f"YOLO All: batch {i + 1}/{n_batches} "
                    f"({processed_idx + 1}/{n_to_process} empty)..."
                )
                QApplication.processEvents()
                # auto_fix_cross_image=False so this loop sees a stable
                # `self.batches`. We do one merge pass at the end below.
                count = self.image_strip.run_yolo(
                    self.yolo_detector, min_size=min_size,
                    auto_fix_cross_image=False,
                )
                total_boxes += count
                logger.info(
                    f"YOLO All: batch {i + 1}/{n_batches} -> {count} new boxes"
                )

            # One global cross-boundary merge after every batch was processed.
            # Now batches can be reshaped freely without breaking iteration.
            # `all_batches=True` so cross-boundary boxes from EVERY batch get
            # auto-merged, not just the last batch we happened to load —
            # otherwise YOLO All on a long project would leave overflowing
            # boxes scattered across all the early batches.
            self.status_label.setText("YOLO All: fixing cross-image boxes...")
            QApplication.processEvents()
            try:
                merged_count = self.image_strip._fix_cross_image_boxes(all_batches=True)
                if merged_count:
                    logger.info(
                        f"YOLO All: auto-merged {merged_count} cross-boundary panel group(s)"
                    )
            except Exception:
                logger.exception("YOLO All: cross-image fix failed")
        finally:
            # Restore the batch the user was looking at when they clicked,
            # clamped to whatever the current batch list size is now.
            n_now = len(self.image_strip.batches)
            target = min(max(0, original_batch), max(0, n_now - 1))
            if n_now > 0:
                self.image_strip._load_batch_pixmaps(target)
                self.image_strip.batch_changed.emit()
            self.yolo_btn.setEnabled(True)
            self.yolo_all_btn.setEnabled(True)

        if n_skip:
            self.status_label.setText(
                f"YOLO All done: {total_boxes} new boxes across "
                f"{n_to_process} batch(es) ({n_skip} skipped)"
            )
        else:
            self.status_label.setText(
                f"YOLO All done: {total_boxes} new boxes across "
                f"{n_to_process} batch(es)"
            )

    def _boxes_per_batch_count(self) -> List[int]:
        """Return a list of length len(batches) where entry i is the number
        of boxes whose image_index falls inside batch i. Used by YOLO All
        to skip batches that already have manually-curated work."""
        counts = [0] * len(self.image_strip.batches)
        for n_batch, start in enumerate(self.image_strip.batch_starts):
            try:
                end = self.image_strip.batch_starts[n_batch + 1]
            except IndexError:
                end = start + len(self.image_strip.batches[n_batch])
            for b in self.image_strip.boxes:
                if start <= b.image_index < end:
                    counts[n_batch] += 1
        return counts

    def _clear_boxes(self):
        self.image_strip.clear_boxes()
        self.status_label.setText("All boxes cleared")

    # -- Bridge write-back -----------------------------------------------------

    def _write_zone_to_bridge(self):
        """When user edits zone text in the app, write it back to scripts/ file."""
        if not self._current_folder:
            return
        bridge_dir = os.path.join(self._current_folder, ".claude-bridge")
        scripts_dir = os.path.join(bridge_dir, "scripts")
        if not os.path.isdir(scripts_dir):
            return

        touched_zones: List[int] = []

        # Build box_id -> zone_number map
        for i, box in enumerate(self.image_strip.boxes):
            zone_num = i + 1
            if box.id in self.text_slots.slots:
                text = self.text_slots.slots[box.id].get_text()
                path = os.path.join(scripts_dir, f"zone_{zone_num:03d}.txt")
                # Only write if content actually changed (avoid watcher loop)
                try:
                    existing = open(path, "r", encoding="utf-8").read().strip() if os.path.exists(path) else ""
                except Exception:
                    existing = ""
                if text.strip() != existing:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(text)
                    # Update the bridge's delivered cache so the watcher doesn't re-deliver
                    if self.claude_bridge and self.claude_bridge.active:
                        self.claude_bridge._delivered[f"zone_{zone_num:03d}.txt"] = text.strip()
                    touched_zones.append(zone_num)

        if touched_zones:
            self._update_changes_log(bridge_dir, touched_zones)

    def _update_changes_log(self, bridge_dir: str, zone_nums: List[int]):
        """Maintain <bridge>/changes.json with {zone_NNN: {before, after}} for
        every zone whose user-edited text diverges from the AI baseline in
        scripts/.ai_raw/. Reverted zones are pruned. Used to feed targeted
        prompt-iteration diffs without scanning the whole project."""
        ai_raw_dir = os.path.join(bridge_dir, "scripts", ".ai_raw")
        changes_path = os.path.join(bridge_dir, "changes.json")

        # Load existing log (tolerate corruption — the file is a derived view)
        try:
            with open(changes_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = {}
        zones = data.get("zones") if isinstance(data.get("zones"), dict) else {}

        mutated = False
        for zone_num in zone_nums:
            zone_key = f"zone_{zone_num:03d}"
            baseline_path = os.path.join(ai_raw_dir, f"{zone_key}.txt")
            current_path = os.path.join(bridge_dir, "scripts", f"{zone_key}.txt")
            try:
                before = open(baseline_path, "r", encoding="utf-8").read().strip()
            except OSError:
                # No AI baseline for this zone — nothing meaningful to diff against.
                continue
            try:
                after = open(current_path, "r", encoding="utf-8").read().strip()
            except OSError:
                after = ""

            if after == before:
                if zone_key in zones:
                    zones.pop(zone_key)
                    mutated = True
                continue

            entry = {"before": before, "after": after}
            if zones.get(zone_key) != entry:
                zones[zone_key] = entry
                mutated = True

        if not mutated:
            return

        from datetime import datetime
        payload = {
            "version": 1,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "zones": zones,
        }
        try:
            with open(changes_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except OSError as e:
            logger.warning(f"Failed to write changes.json: {e}")

    # -- Claude Bridge (internal, used by AI Write) ----------------------------

    def _clear_stale_bridge_scripts(self, zone_from: int, zone_to: int):
        """Delete zone_NNN.txt files in the requested range whose box is
        currently EMPTY in the app. GeminiWriter's resume logic skips any
        zone that already has a non-empty script file — so after the user
        deletes/clears zones and asks for a rewrite, stale files from an
        earlier run would silently turn the whole range into a no-op and
        the 'regenerated' text would never appear in the boxes."""
        if not self._current_folder:
            return
        scripts_dir = os.path.join(
            self._current_folder, ".claude-bridge", "scripts")
        if not os.path.isdir(scripts_dir):
            return

        slot_texts = self.text_slots.get_all_text()
        removed = 0
        for i, box in enumerate(self.image_strip.boxes):
            zone_num = i + 1
            if zone_num < zone_from or zone_num > zone_to:
                continue
            text = slot_texts.get(box.id, box.text) or ""
            if text.strip():
                continue
            fname = f"zone_{zone_num:03d}.txt"
            path = os.path.join(scripts_dir, fname)
            if not os.path.exists(path):
                continue
            try:
                os.remove(path)
            except OSError:
                logger.warning(f"AI Write: could not remove stale {fname}")
                continue
            # Forget the old content so the watcher re-delivers the new one
            self.claude_bridge._delivered.pop(fname, None)
            removed += 1
        if removed:
            logger.info(
                f"AI Write: cleared {removed} stale script file(s) in range "
                f"{zone_from}-{zone_to} (their boxes are empty — regenerating)"
            )

    def _on_claude_zone_update(self, box_id: str, zone_num: int, text: str):
        """Called when Claude writes a zone script file."""
        # Update box text
        target = None
        for box in self.image_strip.boxes:
            if box.id == box_id:
                target = box
                break
        if target is None and 1 <= zone_num <= len(self.image_strip.boxes):
            # The manifest's box id can be stale (zones deleted + recreated
            # since the last export). Zone numbering is positional, so fall
            # back to it instead of silently dropping the text.
            target = self.image_strip.boxes[zone_num - 1]
            box_id = target.id
            logger.info(
                f"Claude Bridge: zone {zone_num} had a stale box id — "
                f"delivered by position instead"
            )
        if target is not None:
            target.text = text

        # Update text slot if visible
        if box_id in self.text_slots.slots:
            slot = self.text_slots.slots[box_id]
            slot.text_edit.blockSignals(True)
            slot.text_edit.setPlainText(text)
            slot.text_edit.blockSignals(False)

        self._update_word_count()
        self.status_label.setText(f"Claude wrote Zone {zone_num} ({len(text.split())} words)")

    # -- AI Writer ----------------------------------------------------------

    def _edit_api_keys(self):
        """Single dialog for both Gemini (first-pass writer) and OpenAI
        (second-pass GPT editor) keys. Either field can be left as-is —
        only fields the user actually edits get saved back to config."""
        s = scale_manager
        google_existing = self.config.get("api_keys.google", "")
        openai_existing = self.config.get("api_keys.openai", "")

        dlg = QDialog(self)
        dlg.setWindowTitle("AI API Keys")
        dlg.setMinimumWidth(s.scale(480))
        layout = QVBoxLayout(dlg)
        layout.setSpacing(s.scale(10))
        layout.setContentsMargins(s.scale(14), s.scale(14), s.scale(14), s.scale(14))

        info = QLabel(
            "Gemini writes the first draft. GPT (OpenAI) runs the second-"
            "pass editor on every Gemini batch. Leave a field blank to "
            "skip that step — the OpenAI key is optional."
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: #aaa; font-size: {s.scale_font(11)}px;")
        layout.addWidget(info)

        # Gemini row
        gemini_label = QLabel("Google AI (Gemini) key")
        gemini_label.setStyleSheet(f"font-size: {s.scale_font(12)}px;")
        layout.addWidget(gemini_label)
        gemini_input = QLineEdit()
        gemini_input.setEchoMode(QLineEdit.Password)
        gemini_input.setText(google_existing)
        gemini_input.setPlaceholderText("AIzaSy...")
        gemini_input.setMinimumHeight(s.scale(28))
        layout.addWidget(gemini_input)

        # OpenAI row
        openai_label = QLabel("OpenAI (GPT second-pass) key")
        openai_label.setStyleSheet(f"font-size: {s.scale_font(12)}px;")
        layout.addWidget(openai_label)
        openai_input = QLineEdit()
        openai_input.setEchoMode(QLineEdit.Password)
        openai_input.setText(openai_existing)
        openai_input.setPlaceholderText("sk-...  (leave empty to disable GPT pass)")
        openai_input.setMinimumHeight(s.scale(28))
        layout.addWidget(openai_input)

        # Show/hide toggle so the user can verify what they pasted.
        show_chk = QCheckBox("Show keys")
        show_chk.setStyleSheet(f"font-size: {s.scale_font(11)}px; color: #aaa;")
        def _toggle_show(checked):
            mode = QLineEdit.Normal if checked else QLineEdit.Password
            gemini_input.setEchoMode(mode)
            openai_input.setEchoMode(mode)
        show_chk.toggled.connect(_toggle_show)
        layout.addWidget(show_chk)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumHeight(s.scale(28))
        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        save_btn.setMinimumHeight(s.scale(28))
        save_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #6a1b9a; color: white; font-weight: bold;
                font-size: {s.scale_font(12)}px;
                padding: {s.scale(4)}px {s.scale(14)}px;
                border-radius: {s.scale(3)}px;
            }}
            QPushButton:hover {{ background-color: #8e24aa; }}
        """)
        cancel_btn.clicked.connect(dlg.reject)
        save_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

        if dlg.exec_() != QDialog.Accepted:
            return

        # Persist whichever fields actually changed.
        new_google = gemini_input.text().strip()
        new_openai = openai_input.text().strip()
        changed = []
        if new_google != google_existing:
            self.config.set("api_keys.google", new_google)
            changed.append("Gemini")
        if new_openai != openai_existing:
            self.config.set("api_keys.openai", new_openai)
            changed.append("OpenAI")

        if changed:
            self.status_label.setText(f"Saved: {', '.join(changed)} key(s)")
        else:
            self.status_label.setText("No key changes")

    def _build_gpt_editor(self) -> Optional[GPTEditor]:
        """Construct a GPTEditor for the second-pass revision step, or
        return None if the OpenAI key is missing / GPT pass is disabled.
        Same factory is reused by both AI Write and Auto so the two paths
        get identical second-pass behavior.

        The live checkbox in the toolbar is the source of truth — config
        is just the persisted backing store (set by the toggle handler).
        """
        # Prefer the live UI state when the checkbox exists; fall back to
        # config for tests / headless runs.
        chk = getattr(self, "gpt_mode_check", None)
        enabled = chk.isChecked() if chk is not None else bool(
            self.config.get("ai_gpt_pass_enabled", True)
        )
        if not enabled:
            logger.info("GPT second pass disabled via toolbar checkbox")
            return None
        api_key = self.config.get("api_keys.openai", "")
        if not api_key:
            logger.info("GPT second pass skipped: no OpenAI key in config")
            return None
        model = self.config.get("gpt_editor_model", "gpt-5.6-terra")
        # Migrate configs saved before the Terra switch
        if model in ("gpt-5", "gpt-5.5", "gpt-5.6-sol"):
            model = "gpt-5.6-terra"
            self.config.set("gpt_editor_model", model)
        prompt_override = self.config.get("gpt_system_prompt", "") or None
        return GPTEditor(
            api_key=api_key, model=model,
            system_prompt_override=prompt_override,
        )

    def _on_gpt_mode_toggled(self, checked: bool):
        """Persist the GPT-mode toggle to config so it survives restarts."""
        self.config.set("ai_gpt_pass_enabled", bool(checked))
        self.status_label.setText(
            "GPT second-pass enabled" if checked else
            "GPT second-pass disabled — Gemini draft used as-is"
        )

    def _edit_gpt_prompt(self):
        """Open a dialog to view/edit the GPT second-pass editor prompt.
        Mirrors _edit_ai_prompt but persists into `gpt_system_prompt` and
        resets to `prompts/gpt_editor.txt` instead of the Gemini default."""
        saved = self.config.get("gpt_system_prompt", "") or ""
        current = saved if saved.strip() else _load_default_gpt_prompt()

        dlg = QDialog(self)
        dlg.setWindowTitle("GPT Editor System Prompt")
        dlg.resize(900, 650)
        layout = QVBoxLayout(dlg)

        info = QLabel(
            "This prompt drives the OpenAI second-pass revision that runs "
            "after Gemini writes each batch. Use 'Reset to Default' to "
            "restore the built-in editor prompt."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(info)

        editor = QTextEdit()
        editor.setPlainText(current)
        editor.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #ddd; "
            "font-family: Consolas, monospace; font-size: 12px; }"
        )
        layout.addWidget(editor, 1)

        btn_row = QHBoxLayout()
        reset_btn = QPushButton("Reset to Default")
        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(reset_btn)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

        reset_btn.clicked.connect(
            lambda: editor.setPlainText(_load_default_gpt_prompt())
        )
        cancel_btn.clicked.connect(dlg.reject)
        save_btn.clicked.connect(dlg.accept)

        if dlg.exec_() == QDialog.Accepted:
            text = editor.toPlainText().strip()
            default_text = _load_default_gpt_prompt().strip()
            if not text or text == default_text:
                self.config.set("gpt_system_prompt", "")
                self.status_label.setText("GPT prompt reset to default")
            else:
                self.config.set("gpt_system_prompt", text)
                self.status_label.setText("GPT prompt saved")

    def _edit_ai_prompt(self):
        """Open a dialog to view/edit the system prompt used by AI Write and Auto."""
        saved = self.config.get("gemini_system_prompt", "") or ""
        current = saved if saved.strip() else _load_default_ai_prompt()

        dlg = QDialog(self)
        dlg.setWindowTitle("AI Writer (Gemini) System Prompt")
        dlg.resize(900, 650)
        layout = QVBoxLayout(dlg)

        info = QLabel(
            "This prompt is used by both the AI Write and Auto functions. "
            "Edit it to tweak the writing style. Use 'Reset to Default' to restore the built-in prompt."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(info)

        editor = QTextEdit()
        editor.setPlainText(current)
        editor.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #ddd; "
            "font-family: Consolas, monospace; font-size: 12px; }"
        )
        layout.addWidget(editor, 1)

        btn_row = QHBoxLayout()
        reset_btn = QPushButton("Reset to Default")
        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(reset_btn)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

        def do_reset():
            editor.setPlainText(_load_default_ai_prompt())

        reset_btn.clicked.connect(do_reset)
        cancel_btn.clicked.connect(dlg.reject)
        save_btn.clicked.connect(dlg.accept)

        if dlg.exec_() == QDialog.Accepted:
            text = editor.toPlainText().strip()
            default_text = _load_default_ai_prompt().strip()
            if not text or text == default_text:
                self.config.set("gemini_system_prompt", "")
                self.status_label.setText("AI prompt reset to default")
            else:
                self.config.set("gemini_system_prompt", text)
                self.status_label.setText("AI prompt saved")

    def _edit_characters(self):
        """Open a dialog to view/edit the story bible character entries.

        The bible at <bridge>/story_bible.json drives how the AI identifies
        characters in panels. Add distinguishing notes (cleavage, scar, eyepatch,
        etc.) to disambiguate two characters the AI keeps mixing up.
        """
        if not getattr(self, "_current_folder", None):
            QMessageBox.information(self, "No project loaded",
                                    "Load a folder first — the bible lives in the project's .claude-bridge dir.")
            return

        bridge_dir = os.path.join(self._current_folder, ".claude-bridge")
        bible_path = os.path.join(bridge_dir, "story_bible.json")

        bible = {"characters": {}, "plot_summary": [], "current_state": "", "tone_notes": ""}
        if os.path.exists(bible_path):
            try:
                with open(bible_path, "r", encoding="utf-8") as f:
                    bible = json.load(f)
            except Exception as e:
                QMessageBox.warning(self, "Bible read error", f"Couldn't read story_bible.json:\n{e}")
                return

        characters = bible.get("characters", {}) or {}

        dlg = QDialog(self)
        dlg.setWindowTitle("Story Bible — Characters")
        dlg.resize(950, 600)
        layout = QVBoxLayout(dlg)

        info = QLabel(
            "Edit character entries the AI uses to identify who's in each panel. "
            "When the AI keeps confusing two characters, add a DISTINGUISHING note "
            "to one (e.g. 'Has a huge cleavage', 'scar over left eye', 'always wears red scarf') "
            "so the AI can tell them apart from the visible features."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(info)

        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Photo", "Name", "Description", "Appearance", "First Zone"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        table.setColumnWidth(0, 88)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        table.setStyleSheet(
            "QTableWidget { background-color: #1e1e1e; color: #ddd; gridline-color: #333; }"
            "QHeaderView::section { background-color: #2a2a2a; color: #ccc; padding: 4px; border: 0; }"
            "QTableWidget::item { padding: 4px; }"
        )
        table.setWordWrap(True)
        table.verticalHeader().setDefaultSectionSize(80)

        refs_dir = os.path.join(bridge_dir, "char_refs")

        def _ref_pixmap(info_dict, name):
            """Thumbnail for the character's reference image, if one exists.
            Prefers the bible's ref_image path; falls back to a char_refs/
            file whose name matches the character."""
            import re as _re
            path = None
            rel = info_dict.get("ref_image") if isinstance(info_dict, dict) else None
            if rel:
                cand = os.path.join(bridge_dir, rel)
                if os.path.exists(cand):
                    path = cand
            if path is None and os.path.isdir(refs_dir):
                key = _re.sub(r"[^a-z0-9]", "", str(name).lower())
                if key:
                    for f in sorted(os.listdir(refs_dir)):
                        stem = _re.sub(r"[^a-z0-9]", "", os.path.splitext(f)[0].lower())
                        if key == stem or key in stem:
                            path = os.path.join(refs_dir, f)
                            break
            if not path:
                return None
            pix = QPixmap(path)
            if pix.isNull():
                return None
            return pix.scaled(80, 76, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        def populate_rows(chars):
            table.setRowCount(0)
            for name, info_dict in chars.items():
                if isinstance(info_dict, str):
                    info_dict = {"description": info_dict}
                row = table.rowCount()
                table.insertRow(row)
                pic_item = QTableWidgetItem()
                pic_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                pix = _ref_pixmap(info_dict, name)
                if pix is not None:
                    pic_item.setData(Qt.DecorationRole, pix)
                    pic_item.setToolTip("Click to recapture from the page")
                else:
                    pic_item.setText("+")
                    pic_item.setTextAlignment(Qt.AlignCenter)
                    pic_item.setForeground(QColor("#00e676"))
                    f = pic_item.font()
                    f.setPointSize(18)
                    f.setBold(True)
                    pic_item.setFont(f)
                    pic_item.setToolTip(
                        "Click, then drag a rectangle over the character's "
                        "face on the page to capture it")
                table.setItem(row, 0, pic_item)
                table.setItem(row, 1, QTableWidgetItem(str(name)))
                table.setItem(row, 2, QTableWidgetItem(str(info_dict.get("description", ""))))
                table.setItem(row, 3, QTableWidgetItem(str(info_dict.get("appearance", ""))))
                fz = info_dict.get("first_zone", "")
                table.setItem(row, 4, QTableWidgetItem(str(fz) if fz != "" else ""))

        populate_rows(characters)
        layout.addWidget(table, 1)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Add Character")
        del_btn = QPushButton("Delete Selected")
        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(add_btn)
        btn_row.addWidget(del_btn)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

        def do_add():
            row = table.rowCount()
            table.insertRow(row)
            for col in range(5):
                table.setItem(row, col, QTableWidgetItem(""))
            pic = table.item(row, 0)
            pic.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            pic.setText("+")
            pic.setTextAlignment(Qt.AlignCenter)
            pic.setForeground(QColor("#00e676"))
            f = pic.font()
            f.setPointSize(18)
            f.setBold(True)
            pic.setFont(f)
            pic.setToolTip("Click, then drag a rectangle over the "
                           "character's face on the page to capture it")
            table.editItem(table.item(row, 1))

        def do_delete():
            rows = sorted({i.row() for i in table.selectedIndexes()}, reverse=True)
            for r in rows:
                table.removeRow(r)

        def start_capture_for(row, col):
            """Photo cell clicked: hide the dialog, let the user drag a
            rectangle over the strip, crop it into char_refs/ and show it."""
            if col != 0:
                return
            name_item = table.item(row, 1)
            name = name_item.text().strip() if name_item else ""
            if not name:
                QMessageBox.information(
                    dlg, "Name first",
                    "Give the character a name before capturing a photo.")
                return
            dlg.hide()
            self.status_label.setText(
                f"Drag a rectangle over {name}'s face (right-click cancels)")

            def on_captured(crop):
                try:
                    if crop is not None and crop.size:
                        os.makedirs(refs_dir, exist_ok=True)
                        import re as _re
                        safe = _re.sub(r"[^\w\-]+", "_", name).strip("_") or "char"
                        out = os.path.join(refs_dir, f"{safe}.png")
                        ok, buf = cv2.imencode(".png", crop)
                        if ok:
                            buf.tofile(out)
                            entry = characters.get(name)
                            if isinstance(entry, str):
                                entry = {"description": entry}
                            if not isinstance(entry, dict):
                                entry = {}
                            entry["ref_image"] = f"char_refs/{safe}.png"
                            characters[name] = entry
                            pix = QPixmap(out)
                            if not pix.isNull():
                                item = table.item(row, 0)
                                item.setText("")
                                item.setData(Qt.DecorationRole, pix.scaled(
                                    80, 76, Qt.KeepAspectRatio,
                                    Qt.SmoothTransformation))
                                item.setToolTip("Click to recapture from the page")
                            self.status_label.setText(f"Photo captured for {name}")
                    else:
                        self.status_label.setText("Capture cancelled")
                finally:
                    dlg.show()
                    dlg.raise_()
                    dlg.activateWindow()

            self.image_strip.start_capture(on_captured)

        table.cellClicked.connect(start_capture_for)
        add_btn.clicked.connect(do_add)
        del_btn.clicked.connect(do_delete)
        cancel_btn.clicked.connect(dlg.reject)
        save_btn.clicked.connect(dlg.accept)

        if dlg.exec_() != QDialog.Accepted:
            return

        # Rebuild characters dict from the table. Start each entry from the
        # existing one so keys the table doesn't edit (ref_image, aliases,
        # ...) survive the round-trip instead of being silently dropped.
        new_chars = {}
        for r in range(table.rowCount()):
            name_item = table.item(r, 1)
            name = name_item.text().strip() if name_item else ""
            if not name:
                continue
            old = characters.get(name)
            if isinstance(old, str):
                old = {"description": old}
            entry = dict(old) if isinstance(old, dict) else {}
            desc_item = table.item(r, 2)
            app_item = table.item(r, 3)
            fz_item = table.item(r, 4)
            for key, item in (("description", desc_item), ("appearance", app_item)):
                txt = item.text().strip() if item else ""
                if txt:
                    entry[key] = txt
                else:
                    entry.pop(key, None)
            fz_txt = fz_item.text().strip() if fz_item else ""
            if fz_txt:
                try:
                    entry["first_zone"] = int(fz_txt)
                except ValueError:
                    pass
            else:
                entry.pop("first_zone", None)
            new_chars[name] = entry

        bible["characters"] = new_chars
        os.makedirs(bridge_dir, exist_ok=True)
        try:
            with open(bible_path, "w", encoding="utf-8") as f:
                json.dump(bible, f, indent=2, ensure_ascii=False)
            self.status_label.setText(f"Saved {len(new_chars)} characters to story_bible.json")
        except Exception as e:
            QMessageBox.warning(self, "Save error", f"Couldn't save story_bible.json:\n{e}")

    def _import_character_sheet(self):
        """Import a Part 1 bundle exported by momo-rewrite's Export Characters.

        Seeds <bridge>/story_bible.json with the roster (appearance tags, roles,
        aliases), plot summary and current state; copies reference images to
        <bridge>/char_refs/; and drops part1_ending.txt so the AI writers can
        pick up exactly where Part 1 left off.
        """
        if not getattr(self, "_current_folder", None):
            QMessageBox.information(self, "No project loaded",
                                    "Load a folder first — the bible lives in the project's .claude-bridge dir.")
            return

        folder = QFileDialog.getExistingDirectory(
            self, "Select the character_sheet folder exported by momo-rewrite")
        if not folder:
            return

        sheet_path = os.path.join(folder, "characters.json")
        if not os.path.exists(sheet_path):
            QMessageBox.warning(self, "Invalid bundle",
                                "characters.json not found in that folder.\n"
                                "Select the character_sheet/ folder exported by momo-rewrite.")
            return

        import shutil
        try:
            with open(sheet_path, "r", encoding="utf-8") as f:
                sheet = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, "Read error", f"Couldn't read characters.json:\n{e}")
            return

        story = {}
        story_path = os.path.join(folder, "story.json")
        if os.path.exists(story_path):
            try:
                with open(story_path, "r", encoding="utf-8") as f:
                    story = json.load(f)
            except Exception:
                pass

        bridge_dir = os.path.join(self._current_folder, ".claude-bridge")
        refs_dir = os.path.join(bridge_dir, "char_refs")
        os.makedirs(refs_dir, exist_ok=True)

        bible_path = os.path.join(bridge_dir, "story_bible.json")
        bible = {"characters": {}, "plot_summary": [], "current_state": "", "tone_notes": ""}
        if os.path.exists(bible_path):
            try:
                with open(bible_path, "r", encoding="utf-8") as f:
                    bible = json.load(f)
            except Exception:
                pass

        chars = bible.get("characters", {}) or {}
        n_refs = 0
        for c in sheet.get("characters", []):
            name = (c.get("name") or "").strip()
            if not name:
                continue
            existing = chars.get(name)
            entry = dict(existing) if isinstance(existing, dict) else {}
            if c.get("role") and not entry.get("description"):
                entry["description"] = c["role"]
            if c.get("tags"):
                entry["appearance"] = c["tags"]
            if c.get("aliases"):
                entry["aliases"] = c["aliases"]
            ref_rel = c.get("ref_image")
            if ref_rel:
                src = os.path.join(folder, ref_rel)
                if os.path.exists(src):
                    dst = os.path.join(refs_dir, os.path.basename(src))
                    try:
                        shutil.copy2(src, dst)
                        entry["ref_image"] = f"char_refs/{os.path.basename(src)}"
                        n_refs += 1
                    except Exception:
                        pass
            chars[name] = entry

        bible["characters"] = chars
        if story.get("plot_summary"):
            bible["plot_summary"] = story["plot_summary"]
        if story.get("current_state"):
            bible["current_state"] = story["current_state"]
        if sheet.get("work_title"):
            bible["work_title"] = sheet["work_title"]
        bible["continuation"] = True

        has_ending = False
        ending_src = os.path.join(folder, "part1_ending.txt")
        if os.path.exists(ending_src):
            try:
                shutil.copy2(ending_src, os.path.join(bridge_dir, "part1_ending.txt"))
                has_ending = True
            except Exception:
                pass

        try:
            with open(bible_path, "w", encoding="utf-8") as f:
                json.dump(bible, f, indent=2, ensure_ascii=False)
        except Exception as e:
            QMessageBox.warning(self, "Save error", f"Couldn't save story_bible.json:\n{e}")
            return

        n_chars = len(sheet.get("characters", []))
        self.status_label.setText(
            f"Part 1 imported: {n_chars} characters, {n_refs} reference images")
        QMessageBox.information(
            self, "Part 1 imported",
            f"Imported {n_chars} characters ({n_refs} with reference images).\n"
            f"Plot summary: {len(bible.get('plot_summary') or [])} beats.\n"
            f"Part 1 ending narration: {'yes' if has_ending else 'no'}.\n\n"
            "The AI writer will now treat this project as a Part 2 continuation.")

    def _update_cost_label(self):
        """Update the persistent cost label in the toolbar."""
        total = self._project_input_tokens + self._project_output_tokens
        if total > 0:
            self._ai_cost_label.setText(
                f"{total:,} tok  ~${self._project_cost_usd:.4f}"
            )
        else:
            self._ai_cost_label.setText("")

    def reset_project_costs(self):
        """Zero out the per-project token/cost accumulators. Called when
        starting a fresh project (New Project, Mix Batch) so cost from
        previous projects doesn't bleed across. set_state() loads from the
        saved ai_cost field, so loading an existing project sets correct
        numbers; this helper handles the paths that don't go through
        set_state."""
        self._project_input_tokens = 0
        self._project_output_tokens = 0
        self._project_cost_usd = 0.0
        # Also reset the writer-sync trackers so the next Gemini run
        # starts counting from zero, not from a stale "last synced" value.
        self._last_synced_in = 0
        self._last_synced_out = 0
        self._last_synced_cost = 0.0
        self._update_cost_label()

    def _toggle_ai_write(self):
        """Start or stop the Gemini API writer."""
        if self._gemini_thread and self._gemini_thread.isRunning():
            # Stop
            if self._gemini_writer:
                self._gemini_writer.stop()
            self.ai_write_btn.setChecked(False)
            self.status_label.setText("AI Writer stopped")
            self._ai_spinner.setVisible(False)
            return

        # Validate state
        if not self._current_folder:
            QMessageBox.warning(self, "No Folder", "Load a folder first.")
            self.ai_write_btn.setChecked(False)
            return
        if not self.image_strip.boxes:
            QMessageBox.warning(self, "No Zones", "Run YOLO or create zones first.")
            self.ai_write_btn.setChecked(False)
            return

        # Check API key for the selected writer (OpenAI for gpt-* models,
        # Google for Gemini models)
        selected_model = self.config.get("gemini_model", "gemini-3-flash-preview")
        is_gpt = selected_model.lower().startswith("gpt")
        key_cfg = "api_keys.openai" if is_gpt else "api_keys.google"
        api_key = self.config.get(key_cfg, "")
        if not api_key:
            provider = "OpenAI" if is_gpt else "Google AI"
            key, ok = QInputDialog.getText(
                self, f"{provider} API Key",
                f"Enter your {provider} API key:",
                QLineEdit.Password,
            )
            if not ok or not key.strip():
                self.ai_write_btn.setChecked(False)
                return
            api_key = key.strip()
            self.config.set(key_cfg, api_key)

        # Ask for zone range
        s = scale_manager
        total_zones = len(self.image_strip.boxes)
        dlg = QDialog(self)
        dlg.setWindowTitle("AI Write — Zone Range")
        dlg.setMinimumSize(s.scale(350), s.scale(160))
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(s.scale(12), s.scale(12), s.scale(12), s.scale(12))
        lay.setSpacing(s.scale(10))

        range_lay = QHBoxLayout()
        range_lay.setSpacing(s.scale(6))
        lbl_from = QLabel("From zone:")
        lbl_from.setStyleSheet(f"font-size: {s.scale_font(12)}px;")
        range_lay.addWidget(lbl_from)
        # Find the last zone that already has text (1-indexed)
        last_with_text = 0
        for i, box in enumerate(self.image_strip.boxes):
            if box.text and box.text.strip():
                last_with_text = i + 1
        # Start from the next zone after the last one with text
        smart_from = min(last_with_text + 1, total_zones)
        smart_to = min(smart_from + 9, total_zones)

        from_spin = QSpinBox()
        from_spin.setRange(1, total_zones)
        from_spin.setValue(smart_from)
        from_spin.setMinimumHeight(s.scale(28))
        from_spin.setMinimumWidth(s.scale(70))
        from_spin.setStyleSheet(f"font-size: {s.scale_font(12)}px; padding: {s.scale(3)}px;")
        range_lay.addWidget(from_spin)

        lbl_to = QLabel("  to:")
        lbl_to.setStyleSheet(f"font-size: {s.scale_font(12)}px;")
        range_lay.addWidget(lbl_to)
        to_spin = QSpinBox()
        to_spin.setRange(1, total_zones)
        to_spin.setValue(smart_to)
        to_spin.setMinimumHeight(s.scale(28))
        to_spin.setMinimumWidth(s.scale(70))
        to_spin.setStyleSheet(f"font-size: {s.scale_font(12)}px; padding: {s.scale(3)}px;")
        range_lay.addWidget(to_spin)

        lbl_total = QLabel(f"  / {total_zones}")
        lbl_total.setStyleSheet(f"font-size: {s.scale_font(12)}px; color: #aaa;")
        range_lay.addWidget(lbl_total)
        lay.addLayout(range_lay)

        # Keep to >= from. Keyboard tracking OFF so the constraint applies only
        # when editing finishes — otherwise typing "90" fires valueChanged(9)
        # mid-keystroke and clamps the other spinbox (61 -> 9).
        from_spin.setKeyboardTracking(False)
        to_spin.setKeyboardTracking(False)
        from_spin.valueChanged.connect(lambda v: to_spin.setMinimum(v))
        to_spin.valueChanged.connect(lambda v: from_spin.setMaximum(v))

        btn_lay = QHBoxLayout()
        btn_lay.setSpacing(s.scale(8))
        all_btn = QPushButton("All Zones")
        all_btn.setMinimumHeight(s.scale(30))
        all_btn.setStyleSheet(f"font-size: {s.scale_font(12)}px; padding: {s.scale(6)}px {s.scale(12)}px;")
        all_btn.clicked.connect(lambda: (from_spin.setValue(1), to_spin.setValue(total_zones)))
        btn_lay.addWidget(all_btn)
        btn_lay.addStretch()
        ok_btn = QPushButton("Start")
        ok_btn.setDefault(True)
        ok_btn.setMinimumHeight(s.scale(30))
        ok_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #4caf50; color: white; font-weight: bold;
                font-size: {s.scale_font(12)}px; padding: {s.scale(6)}px {s.scale(16)}px;
                border-radius: {s.scale(3)}px;
            }}
            QPushButton:hover {{ background-color: #43a047; }}
        """)
        ok_btn.clicked.connect(dlg.accept)
        btn_lay.addWidget(ok_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumHeight(s.scale(30))
        cancel_btn.setStyleSheet(f"font-size: {s.scale_font(12)}px; padding: {s.scale(6)}px {s.scale(12)}px;")
        cancel_btn.clicked.connect(dlg.reject)
        btn_lay.addWidget(cancel_btn)
        lay.addLayout(btn_lay)

        if dlg.exec_() != QDialog.Accepted:
            self.ai_write_btn.setChecked(False)
            return

        zone_from = from_spin.value()
        zone_to = to_spin.value()

        bridge_dir = os.path.join(self._current_folder, ".claude-bridge")

        # Start the file watcher so zones appear in real-time when Gemini writes
        if not self.claude_bridge.active:
            from pathlib import Path
            Path(bridge_dir, "scripts").mkdir(parents=True, exist_ok=True)
            self.claude_bridge.start_watching(bridge_dir, self._on_claude_zone_update)

        # Drop stale script files for zones the user cleared/deleted so the
        # writer actually regenerates them instead of skipping ("resume")
        self._clear_stale_bridge_scripts(zone_from, zone_to)

        # Run export + API calls in background thread (no freeze)
        model = self.config.get("gemini_model", "gemini-3-flash-preview")
        expand_pct = self.config.get("gemini_expand_pct", 150) / 100.0
        language = self.ai_lang_combo.currentText()
        prompt_override = self.config.get("gemini_system_prompt", "") or None
        # GeminiWriter handles gpt-* models too (OpenAI backend via
        # app.openai_compat) — same process, different transport.
        self._gemini_writer = GeminiWriter(api_key, model, expand_pct=expand_pct, language=language, system_prompt_override=prompt_override)
        self._last_synced_in = 0
        self._last_synced_out = 0
        self._last_synced_cost = 0.0

        self._gemini_thread = _GeminiThread(
            self._gemini_writer, bridge_dir,
            zone_from=zone_from, zone_to=zone_to,
            claude_bridge=self.claude_bridge,
            boxes=list(self.image_strip.boxes),
            image_paths=list(self.image_strip.all_image_paths),
            image_folder=self._current_folder,
            gpt_editor=self._build_gpt_editor(),
        )
        self._gemini_thread.zone_done.connect(self._on_ai_zone_done)
        self._gemini_thread.batch_done.connect(self._on_ai_batch_done)
        self._gemini_thread.error.connect(self._on_ai_error)
        self._gemini_thread.finished_all.connect(self._on_ai_complete)
        self._ai_cancel_pending = False
        self._gemini_thread.start()

        self.ai_write_btn.setChecked(True)
        self._ai_spinner.setVisible(True)
        self.status_label.setText(f"AI Writer running — zones {zone_from}-{zone_to}...")

    def _on_ai_zone_done(self, zone_num: int, text: str):
        self.status_label.setText(f"Gemini wrote Zone {zone_num} ({len(text.split())} words)")

    def _sync_writer_cost(self):
        """Merge the current writer's token usage into project totals."""
        if not self._gemini_writer:
            return
        # Track delta since last sync to avoid double-counting
        w = self._gemini_writer
        last_in = getattr(self, '_last_synced_in', 0)
        last_out = getattr(self, '_last_synced_out', 0)
        last_cost = getattr(self, '_last_synced_cost', 0.0)

        delta_in = w.total_input_tokens - last_in
        delta_out = w.total_output_tokens - last_out
        delta_cost = w.total_cost_usd - last_cost

        if delta_in > 0 or delta_out > 0:
            self._project_input_tokens += delta_in
            self._project_output_tokens += delta_out
            self._project_cost_usd += delta_cost

        self._last_synced_in = w.total_input_tokens
        self._last_synced_out = w.total_output_tokens
        self._last_synced_cost = w.total_cost_usd
        self._update_cost_label()

    def _on_ai_batch_done(self, batch: int, total: int):
        self._sync_writer_cost()
        self.status_label.setText(f"AI Writer — batch {batch}/{total} complete")

    def _on_ai_error(self, msg: str):
        self._sync_writer_cost()
        if self._auto_mode:
            self._on_ai_error_auto(msg)
            return
        self.status_label.setText(f"AI error: {msg}")
        self._ai_spinner.setVisible(False)
        logger.error(f"GeminiWriter: {msg}")

    def _cancel_ai_write(self):
        """Spinner X clicked: don't send the NEXT API call. The call already
        in flight finishes normally and its zones still land in the boxes
        (the bridge watcher keeps running) — only the remaining batches are
        skipped."""
        if not (self._gemini_thread and self._gemini_thread.isRunning()):
            return
        if getattr(self, "_ai_cancel_pending", False):
            return
        self._ai_cancel_pending = True
        if self._gemini_writer:
            self._gemini_writer.stop()
        if self._auto_mode:
            self._auto_mode = False
            self.auto_btn.setChecked(False)
        self.status_label.setText(
            "AI Writer cancelling — finishing the call in flight, "
            "no further calls..."
        )
        logger.info("AI Write: cancel requested via spinner")
        # generate() returns without emitting finished_all when stopped, so
        # hook the thread's own finished signal for the UI cleanup.
        self._gemini_thread.finished.connect(self._on_ai_cancelled)

    def _on_ai_cancelled(self):
        self._sync_writer_cost()
        self._ai_cancel_pending = False
        self.ai_write_btn.setChecked(False)
        self._ai_spinner.setVisible(False)
        self.status_label.setText(
            "AI Writer cancelled — everything already written was kept")

    def _on_ai_complete(self):
        self._sync_writer_cost()
        self.ai_write_btn.setChecked(False)
        self._ai_spinner.setVisible(False)

        if self._auto_mode:
            self._auto_retries = 0  # reset retries on success
            self._auto_save_project()
            # Schedule next batch (short delay to let UI breathe)
            QTimer.singleShot(500, self._auto_next_batch)
        else:
            self.status_label.setText("AI Writer finished — all zones written")

    def _on_ai_error_auto(self, msg: str):
        """Error handler when in auto mode — retries before skipping."""
        self._sync_writer_cost()
        self._auto_retries += 1
        logger.error(f"Auto mode error (attempt {self._auto_retries}/{self._auto_max_retries}): {msg}")

        if self._auto_retries < self._auto_max_retries:
            wait_sec = self._auto_retries * 5  # 5s, 10s, 15s
            self.status_label.setText(
                f"Auto: error on batch, retrying in {wait_sec}s (attempt {self._auto_retries + 1})..."
            )
            QTimer.singleShot(wait_sec * 1000, self._auto_retry_current)
        else:
            # Skip this batch and move on. Track skips by box id when we
            # have a captured set so the skip survives zone-number shifts;
            # fall back to positional zone numbers in legacy mode.
            skipped_from = self._auto_current_from
            skipped_to = self._auto_current_to
            if self._auto_current_box_ids:
                self._auto_skipped_box_ids.update(self._auto_current_box_ids)
            else:
                self._auto_skipped_zones.update(range(skipped_from, skipped_to + 1))
            logger.warning(
                f"Auto mode: skipping zones {skipped_from}-{skipped_to} "
                f"({len(self._auto_current_box_ids) or (skipped_to - skipped_from + 1)} box(es)) "
                f"after {self._auto_max_retries} failures"
            )
            self.status_label.setText(f"Auto: skipped zones {skipped_from}-{skipped_to}, moving on...")
            self._auto_retries = 0
            QTimer.singleShot(2000, self._auto_next_batch)

    # -- Auto mode -------------------------------------------------------------

    def _toggle_auto_mode(self):
        """Start or stop fully automatic script generation."""
        if self._auto_mode:
            # Stop — let current batch finish, then stop
            self._auto_mode = False
            self.auto_btn.setChecked(False)
            if self._gemini_writer:
                self._gemini_writer.stop()
            self.status_label.setText("Auto mode stopping after current batch...")
            return

        # Validate
        if not self._current_folder:
            QMessageBox.warning(self, "No Folder", "Load a folder first.")
            self.auto_btn.setChecked(False)
            return
        if not self.image_strip.boxes:
            QMessageBox.warning(self, "No Zones", "Run YOLO or create zones first.")
            self.auto_btn.setChecked(False)
            return

        # Check API key
        api_key = self.config.get("api_keys.google", "")
        if not api_key:
            key, ok = QInputDialog.getText(
                self, "Google AI API Key",
                "Enter your Google AI API key:",
                QLineEdit.Password,
            )
            if not ok or not key.strip():
                self.auto_btn.setChecked(False)
                return
            api_key = key.strip()
            self.config.set("api_keys.google", api_key)

        # Zone range dialog
        s = scale_manager
        total_zones = len(self.image_strip.boxes)
        filled = sum(1 for b in self.image_strip.boxes if b.text and b.text.strip())

        # Default the dialog range to the CURRENT BATCH so the user can
        # start Auto on this batch and switch elsewhere afterwards. The
        # actual scope gets locked in via box-id capture after Accept, so
        # mutations on other batches don't disrupt this loop.
        self._auto_skipped_zones = set()
        self._auto_skipped_box_ids = set()
        self._auto_box_ids = []  # cleared until dialog is accepted

        batch_box_ids = self._current_batch_box_ids()
        id_to_zone = {b.id: i + 1 for i, b in enumerate(self.image_strip.boxes)}
        if batch_box_ids:
            batch_zone_nums = [id_to_zone[bid] for bid in batch_box_ids if bid in id_to_zone]
            default_from = min(batch_zone_nums) if batch_zone_nums else 1
            default_to = max(batch_zone_nums) if batch_zone_nums else total_zones
        else:
            default_from = 1
            default_to = total_zones

        # Pre-pick the auto range using batch defaults so the helpers below
        # use the right scope when computing first_empty for the dialog.
        self._auto_range_from = default_from
        self._auto_range_to = default_to

        # Smart defaults: start at first empty zone, cover a reasonable range
        first_empty = self._auto_find_first_empty()
        if first_empty is None:
            QMessageBox.information(
                self, "All Done",
                f"All zones in the current batch ({default_from}-{default_to}) "
                f"already have text. Switch to another batch or use 'All Zones' "
                f"to widen the range."
            )
            self.auto_btn.setChecked(False)
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Auto Mode — Zone Range")
        dlg.setMinimumSize(s.scale(380), s.scale(220))
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(s.scale(12), s.scale(12), s.scale(12), s.scale(12))
        lay.setSpacing(s.scale(10))

        # Info label
        n_batches = len(self.image_strip.batches) if self.image_strip.batches else 1
        if n_batches > 1:
            info_text = (
                f"{filled}/{total_zones} zones have text. Defaulting to current "
                f"batch (zones {default_from}-{default_to}) — Auto will only "
                f"process boxes captured at start, so you can switch batches "
                f"and edit YOLO/text elsewhere while this loop runs.\n\n"
                f"Click 'All Zones' to widen the range."
            )
        else:
            info_text = (
                f"{filled}/{total_zones} zones have text. Auto will process "
                f"empty zones in batches."
            )
        info = QLabel(info_text)
        info.setWordWrap(True)
        info.setStyleSheet(f"color: #aaa; font-size: {s.scale_font(11)}px;")
        lay.addWidget(info)

        # Range row
        range_lay = QHBoxLayout()
        range_lay.setSpacing(s.scale(6))

        lbl_from = QLabel("From zone:")
        lbl_from.setStyleSheet(f"font-size: {s.scale_font(12)}px;")
        range_lay.addWidget(lbl_from)
        auto_from_spin = QSpinBox()
        auto_from_spin.setRange(1, total_zones)
        auto_from_spin.setValue(first_empty)
        auto_from_spin.setMinimumHeight(s.scale(28))
        auto_from_spin.setMinimumWidth(s.scale(70))
        auto_from_spin.setStyleSheet(f"font-size: {s.scale_font(12)}px; padding: {s.scale(3)}px;")
        range_lay.addWidget(auto_from_spin)

        lbl_to = QLabel("  to:")
        lbl_to.setStyleSheet(f"font-size: {s.scale_font(12)}px;")
        range_lay.addWidget(lbl_to)
        auto_to_spin = QSpinBox()
        auto_to_spin.setRange(1, total_zones)
        auto_to_spin.setValue(default_to)
        auto_to_spin.setMinimumHeight(s.scale(28))
        auto_to_spin.setMinimumWidth(s.scale(70))
        auto_to_spin.setStyleSheet(f"font-size: {s.scale_font(12)}px; padding: {s.scale(3)}px;")
        range_lay.addWidget(auto_to_spin)

        lbl_total = QLabel(f"  / {total_zones}")
        lbl_total.setStyleSheet(f"font-size: {s.scale_font(12)}px; color: #aaa;")
        range_lay.addWidget(lbl_total)
        lay.addLayout(range_lay)

        # Keyboard tracking OFF: same mid-keystroke clamping fix as the AI Write dialog
        auto_from_spin.setKeyboardTracking(False)
        auto_to_spin.setKeyboardTracking(False)
        auto_from_spin.valueChanged.connect(lambda v: auto_to_spin.setMinimum(v))
        auto_to_spin.valueChanged.connect(lambda v: auto_from_spin.setMaximum(v))

        # Batch size row
        batch_lay = QHBoxLayout()
        batch_lay.setSpacing(s.scale(6))
        lbl_batch = QLabel("Batch size:")
        lbl_batch.setStyleSheet(f"font-size: {s.scale_font(12)}px;")
        batch_lay.addWidget(lbl_batch)
        batch_spin = QSpinBox()
        batch_spin.setRange(1, 50)
        batch_spin.setValue(self._auto_batch_size)
        batch_spin.setMinimumHeight(s.scale(28))
        batch_spin.setStyleSheet(f"font-size: {s.scale_font(12)}px; padding: {s.scale(3)}px;")
        batch_lay.addWidget(batch_spin)
        batch_lay.addStretch()
        lay.addLayout(batch_lay)

        # Buttons
        btn_lay = QHBoxLayout()
        btn_lay.setSpacing(s.scale(8))
        all_btn = QPushButton("All Zones")
        all_btn.setMinimumHeight(s.scale(30))
        all_btn.setStyleSheet(f"font-size: {s.scale_font(12)}px; padding: {s.scale(6)}px {s.scale(12)}px;")
        all_btn.clicked.connect(lambda: (auto_from_spin.setValue(1), auto_to_spin.setValue(total_zones)))
        btn_lay.addWidget(all_btn)
        btn_lay.addStretch()
        ok_btn = QPushButton("Start Auto")
        ok_btn.setDefault(True)
        ok_btn.setMinimumHeight(s.scale(30))
        ok_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #6a1b9a; color: white; font-weight: bold;
                font-size: {s.scale_font(12)}px; padding: {s.scale(6)}px {s.scale(16)}px;
                border-radius: {s.scale(3)}px;
            }}
            QPushButton:hover {{ background-color: #8e24aa; }}
        """)
        ok_btn.clicked.connect(dlg.accept)
        btn_lay.addWidget(ok_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumHeight(s.scale(30))
        cancel_btn.setStyleSheet(f"font-size: {s.scale_font(12)}px; padding: {s.scale(6)}px {s.scale(12)}px;")
        cancel_btn.clicked.connect(dlg.reject)
        btn_lay.addWidget(cancel_btn)
        lay.addLayout(btn_lay)

        if dlg.exec_() != QDialog.Accepted:
            self.auto_btn.setChecked(False)
            return

        # Start auto mode with selected range
        self._auto_mode = True
        self._auto_retries = 0
        self._auto_zones_written = 0
        self._auto_skipped_zones = set()
        self._auto_skipped_box_ids = set()
        self._auto_current_from = 0
        self._auto_current_to = 0
        self._auto_current_box_ids = []
        self._auto_batch_size = batch_spin.value()
        self._auto_range_from = auto_from_spin.value()
        self._auto_range_to = auto_to_spin.value()
        # Lock in the captured box-id set NOW. From this point on the loop
        # operates on these specific boxes regardless of how the global
        # zone numbering shifts due to YOLO/edits in other batches.
        self._auto_box_ids = [
            b.id for i, b in enumerate(self.image_strip.boxes)
            if self._auto_range_from <= (i + 1) <= self._auto_range_to
        ]
        self.auto_btn.setChecked(True)
        self._ai_spinner.setVisible(True)

        total_in_range = len(self._auto_box_ids)
        logger.info(
            f"Auto mode started: zones {self._auto_range_from}-{self._auto_range_to} "
            f"({total_in_range} captured boxes, batch size {self._auto_batch_size})"
        )
        self._auto_next_batch()

    # -- Auto box-id resolution helpers (used by all auto-loop logic) -------

    def _auto_id_to_zone_map(self) -> Dict[str, int]:
        """Current box.id -> 1-indexed zone number, recomputed every cycle so
        we tolerate boxes being inserted before/after the captured range."""
        return {b.id: i + 1 for i, b in enumerate(self.image_strip.boxes)}

    def _current_batch_box_ids(self) -> List[str]:
        """Box IDs whose image_index falls inside the active batch's image
        range. Returns [] when the project isn't batched."""
        batches = self.image_strip.batches
        if not batches:
            return [b.id for b in self.image_strip.boxes]
        idx = self.image_strip.current_batch_index
        if idx < 0 or idx >= len(batches):
            return [b.id for b in self.image_strip.boxes]
        img_start = sum(len(b) for b in batches[:idx])
        img_end = img_start + len(batches[idx])
        return [
            b.id for b in self.image_strip.boxes
            if img_start <= b.image_index < img_end
        ]

    def _auto_find_first_empty(self) -> Optional[int]:
        """Return the current zone number of the first captured box that
        still has no text. Resolves box.id -> current position so it stays
        correct even if the user adds boxes elsewhere mid-loop."""
        if self._auto_box_ids:
            box_by_id = {b.id: b for b in self.image_strip.boxes}
            id_to_zone = self._auto_id_to_zone_map()
            for box_id in self._auto_box_ids:
                if box_id in self._auto_skipped_box_ids:
                    continue
                box = box_by_id.get(box_id)
                if box is None:
                    continue  # box deleted while loop was running
                if not box.text or not box.text.strip():
                    return id_to_zone.get(box_id)
            return None

        # No captured set yet — used during the initial dialog when we want
        # to suggest a starting zone before Auto is actually running.
        range_from = getattr(self, '_auto_range_from', 1)
        range_to = getattr(self, '_auto_range_to', len(self.image_strip.boxes))
        for i, box in enumerate(self.image_strip.boxes):
            zone_num = i + 1
            if zone_num < range_from or zone_num > range_to:
                continue
            if zone_num in self._auto_skipped_zones:
                continue
            if not box.text or not box.text.strip():
                return zone_num
        return None

    def _auto_count_remaining(self) -> int:
        """Count empty boxes still owned by the auto loop (post box-id capture)."""
        if self._auto_box_ids:
            box_by_id = {b.id: b for b in self.image_strip.boxes}
            n = 0
            for box_id in self._auto_box_ids:
                if box_id in self._auto_skipped_box_ids:
                    continue
                box = box_by_id.get(box_id)
                if box is None:
                    continue
                if not box.text or not box.text.strip():
                    n += 1
            return n

        range_from = getattr(self, '_auto_range_from', 1)
        range_to = getattr(self, '_auto_range_to', len(self.image_strip.boxes))
        return sum(1 for i, b in enumerate(self.image_strip.boxes)
                   if range_from <= (i + 1) <= range_to
                   and (not b.text or not b.text.strip())
                   and (i + 1) not in self._auto_skipped_zones)

    def _auto_next_batch(self):
        """Find the next contiguous run of captured boxes that still need
        text and start a batch. The run is walked through the captured
        box-id list rather than raw zone numbers so it survives shifts
        caused by mutations in other batches."""
        if not self._auto_mode:
            self._auto_finish("Auto mode stopped by user")
            return

        first_empty = self._auto_find_first_empty()
        if first_empty is None:
            self._auto_finish("Auto mode complete — all zones have text!")
            return

        # Walk forward through the captured ids and collect a contiguous run
        # (consecutive current zone numbers) of up to batch_size boxes,
        # starting at the first empty one. If the user inserted boxes in the
        # middle of our captured range the run breaks early and we just
        # process whatever we got — next cycle picks up from the first
        # remaining empty box.
        if self._auto_box_ids:
            id_to_zone = self._auto_id_to_zone_map()
            box_by_id = {b.id: b for b in self.image_strip.boxes}

            try:
                start_idx = next(
                    i for i, bid in enumerate(self._auto_box_ids)
                    if id_to_zone.get(bid) == first_empty
                )
            except StopIteration:
                self._auto_finish("Auto mode: lost track of captured boxes")
                return

            collected_box_ids = [self._auto_box_ids[start_idx]]
            last_zone = first_empty
            for j in range(start_idx + 1, len(self._auto_box_ids)):
                if len(collected_box_ids) >= self._auto_batch_size:
                    break
                bid = self._auto_box_ids[j]
                z = id_to_zone.get(bid)
                if z is None or z != last_zone + 1:
                    break  # deleted box or non-contiguous shift, stop here
                if box_by_id.get(bid) is None:
                    break
                collected_box_ids.append(bid)
                last_zone = z

            zone_from = first_empty
            zone_to = last_zone
            self._auto_current_box_ids = collected_box_ids
        else:
            range_to = getattr(self, '_auto_range_to', len(self.image_strip.boxes))
            zone_from = first_empty
            zone_to = min(first_empty + self._auto_batch_size - 1, range_to)
            self._auto_current_box_ids = []

        self._auto_current_from = zone_from
        self._auto_current_to = zone_to

        remaining = self._auto_count_remaining()
        self.status_label.setText(
            f"Auto: zones {zone_from}-{zone_to} | {remaining} remaining"
        )

        self._auto_start_writer(zone_from, zone_to)

    def _auto_retry_current(self):
        """Retry the current batch after an error. If box positions shifted
        between the failed attempt and the retry (e.g. YOLO added boxes in
        another batch), re-resolve zone_from/zone_to from the captured
        box-ids so we retry the right zones — not stale zone numbers."""
        if not self._auto_mode:
            self._auto_finish("Auto mode stopped by user")
            return

        if self._auto_current_box_ids:
            id_to_zone = self._auto_id_to_zone_map()
            zones = [
                id_to_zone[bid]
                for bid in self._auto_current_box_ids
                if bid in id_to_zone
            ]
            if zones:
                self._auto_current_from = min(zones)
                self._auto_current_to = max(zones)

        self.status_label.setText(
            f"Auto: retrying zones {self._auto_current_from}-{self._auto_current_to} "
            f"(attempt {self._auto_retries + 1})..."
        )
        self._auto_start_writer(self._auto_current_from, self._auto_current_to)

    def _auto_start_writer(self, zone_from: int, zone_to: int):
        """Start a single AI Write batch for auto mode, reusing the same writer/chat session."""
        bridge_dir = os.path.join(self._current_folder, ".claude-bridge")

        if not self.claude_bridge.active:
            from pathlib import Path
            Path(bridge_dir, "scripts").mkdir(parents=True, exist_ok=True)
            self.claude_bridge.start_watching(bridge_dir, self._on_claude_zone_update)

        # Drop stale script files for zones the user cleared/deleted so the
        # writer actually regenerates them instead of skipping ("resume")
        self._clear_stale_bridge_scripts(zone_from, zone_to)

        # Reuse the same writer across auto cycles (chat session persists for context)
        if self._gemini_writer is None:
            model = self.config.get("gemini_model", "gemini-3-flash-preview")
            is_gpt = model.lower().startswith("gpt")
            api_key = self.config.get(
                "api_keys.openai" if is_gpt else "api_keys.google", "")
            expand_pct = self.config.get("gemini_expand_pct", 150) / 100.0
            language = self.ai_lang_combo.currentText()
            prompt_override = self.config.get("gemini_system_prompt", "") or None
            self._gemini_writer = GeminiWriter(api_key, model, expand_pct=expand_pct, language=language, system_prompt_override=prompt_override)
            self._last_synced_in = 0
            self._last_synced_out = 0
            self._last_synced_cost = 0.0

        self._gemini_thread = _GeminiThread(
            self._gemini_writer, bridge_dir,
            zone_from=zone_from, zone_to=zone_to,
            claude_bridge=self.claude_bridge,
            boxes=list(self.image_strip.boxes),
            image_paths=list(self.image_strip.all_image_paths),
            image_folder=self._current_folder,
            gpt_editor=self._build_gpt_editor(),
        )
        self._gemini_thread.zone_done.connect(self._on_ai_zone_done)
        self._gemini_thread.batch_done.connect(self._on_ai_batch_done)
        self._gemini_thread.error.connect(self._on_ai_error)
        self._gemini_thread.finished_all.connect(self._on_ai_complete)
        self._ai_cancel_pending = False
        self._gemini_thread.start()

        self.ai_write_btn.setChecked(True)
        self._ai_spinner.setVisible(True)

    def _auto_save_project(self):
        """Trigger a project save after each auto batch."""
        try:
            self.state_changed.emit()  # marks dirty
            # Find main window and call save
            main_win = self.window()
            if hasattr(main_win, '_do_save') and hasattr(main_win, '_current_project_path'):
                if main_win._current_project_path:
                    main_win._do_save(main_win._current_project_path)
                    logger.info("Auto mode: project saved")
        except Exception as e:
            logger.warning(f"Auto mode: save failed (non-fatal): {e}")

    def _auto_finish(self, message: str):
        """Clean up auto mode state."""
        self._auto_mode = False
        self.auto_btn.setChecked(False)
        self.ai_write_btn.setChecked(False)
        self._ai_spinner.setVisible(False)
        self._sync_writer_cost()
        self._gemini_writer = None  # release for next session
        self._auto_save_project()

        summary = message
        skipped_total = len(self._auto_skipped_zones) + len(self._auto_skipped_box_ids)
        if skipped_total:
            summary += f" (skipped {skipped_total} zone(s) due to errors)"
        self.status_label.setText(summary)
        logger.info(f"Auto mode finished: {summary}")

        # Release the captured set so the next Auto run starts clean.
        self._auto_box_ids = []
        self._auto_current_box_ids = []
        self._auto_skipped_box_ids = set()

    # -- Export ---------------------------------------------------------------

    def _export_script(self):
        """Export zone texts. If a zone's text ends with *, it merges with the next zone.
        Uses \u200B (zero-width space) as an invisible separator between merged texts."""
        texts = self.text_slots.get_all_text()
        boxes = self.image_strip.boxes
        if not boxes:
            QMessageBox.warning(self, "Nothing to Export", "No zones to export.")
            return

        # Build ordered list of (zone_number, text)
        ordered = [(i + 1, texts.get(box.id, box.text).strip()) for i, box in enumerate(boxes)]

        # Merge: if text ends with *, combine with next zone
        # Track which zone numbers are grouped together
        SEPARATOR = "\u200B|\u200B"  # zero-width space + pipe + zero-width space (invisible to TTS)
        merged = []  # list of (zone_label, combined_text)
        carry_text = ""
        carry_zones = []

        for zone_num, t in ordered:
            if t.endswith("*"):
                carry_text += t[:-1].rstrip() + SEPARATOR
                carry_zones.append(zone_num)
            else:
                carry_zones.append(zone_num)
                combined = carry_text + t
                if len(carry_zones) > 1:
                    label = "Panel " + "+".join(str(z) for z in carry_zones)
                else:
                    label = f"Panel {carry_zones[0]}"
                merged.append((label, combined))
                carry_text = ""
                carry_zones = []

        if carry_zones:
            label = "Panel " + "+".join(str(z) for z in carry_zones)
            merged.append((label, carry_text.rstrip(SEPARATOR)))

        # Ask where to save
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Script", "script.txt", "Text Files (*.txt);;All Files (*)"
        )
        if not path:
            return

        with open(path, "w", encoding="utf-8") as f:
            for label, text in merged:
                f.write(f"[{label}]\n")
                f.write(text + "\n\n")

        self._export_plain_script_txt(boxes, texts, "portuguese")

        self.status_label.setText(f"Exported {len(merged)} panels to {os.path.basename(path)}")
        QMessageBox.information(self, "Export Complete", f"Saved {len(merged)} panels to:\n{path}")

    def _export_plain_script_txt(self, boxes, texts, lang_name: str):
        """Dump one-line-per-panel plain text next to the .mscript project file.

        Output path: <project_dir>/script_<lang_name>.txt
        """
        import re as _re
        main_win = self.window()
        project_path = getattr(main_win, "_current_project_path", "") or ""
        if not project_path:
            logger.warning("No project path; skipping plain script txt export.")
            return
        try:
            out_dir = os.path.dirname(os.path.abspath(project_path))
            out_path = os.path.join(out_dir, f"script_{lang_name}.txt")
            lines = []
            for box in boxes:
                text = texts.get(box.id, box.text) or ""
                lines.append(_re.sub(r"\s+", " ", text.strip()))
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            logger.info(f"Plain script exported to {out_path}")
        except Exception:
            logger.exception("Failed to export plain script txt")

    def _on_box_deleting(self, box_id: str):
        """Before a box is removed, sync the latest editor texts to the
        box objects and snapshot for undo. Texts stay glued to their own
        boxes by box.id — deleting box X simply removes X. The boxes that
        come after will renumber their zones, but each one keeps its own
        text. (Earlier behavior shifted texts up one position, which made
        the rest of the script silently misaligned. Removed deliberately.)"""
        self._sync_texts_to_boxes()
        self._push_undo_now()

    def _on_boxes_changed(self):
        """Rebuild text slots when boxes change (current batch only)."""
        # Sync text from slots back to boxes before rebuild
        self._sync_texts_to_boxes()

        # Only show boxes for current batch in text slots
        batch_boxes = self.image_strip.current_batch_boxes()
        global_zone_num = {b.id: i + 1 for i, b in enumerate(self.image_strip.boxes)}
        zone_numbers = [global_zone_num[b.id] for b in batch_boxes]
        self.text_slots.rebuild(batch_boxes, zone_numbers=zone_numbers)
        self._install_undo_filters_on_slots()
        total = len(self.image_strip.boxes)
        batch_count = len(batch_boxes)
        if len(self.image_strip.batches) > 1:
            self.box_count_label.setText(f"Boxes: {batch_count} (total: {total})")
        else:
            self.box_count_label.setText(f"Boxes: {total}")
        self._update_word_count()

    def _sync_texts_to_boxes(self):
        """Copy current slot texts back into box objects."""
        texts = self.text_slots.get_all_text()
        for box in self.image_strip.boxes:
            if box.id in texts:
                box.text = texts[box.id]

    def _update_word_count(self):
        """Count words across ALL boxes (all batches)."""
        self._sync_texts_to_boxes()
        total_words = 0
        for box in self.image_strip.boxes:
            text = box.text.strip()
            if text:
                total_words += len(text.split())
        self.word_count_label.setText(f"Words: {total_words}")

    def _on_box_selected(self, box_id: str):
        """Image strip box was clicked - highlight the text slot."""
        self.text_slots.set_active_slot(box_id)

    def _on_slot_clicked(self, box_id: str):
        """Text slot was clicked - highlight the box in the strip."""
        self.image_strip.select_box(box_id)
        self.text_slots.set_active_slot(box_id)

    # -- Zoom ---

    def _base_width(self) -> int:
        """The viewport width = what zoom 1.0 (fit-width) maps to."""
        vp = self.image_scroll.viewport().width()
        return vp if vp > 50 else 600

    def _apply_zoom(self, new_zoom: float, anchor_viewport_y: Optional[int] = None):
        """Change zoom level, optionally anchored to a viewport Y position."""
        if not self.image_strip.image_paths:
            return

        new_zoom = max(0.05, min(5.0, new_zoom))
        old_zoom = self._zoom_level
        if abs(new_zoom - old_zoom) < 0.001:
            return

        # Remember scroll state for anchor
        vbar = self.image_scroll.verticalScrollBar()
        hbar = self.image_scroll.horizontalScrollBar()
        old_scroll_y = vbar.value()
        old_scroll_x = hbar.value()

        if anchor_viewport_y is None:
            anchor_viewport_y = self.image_scroll.viewport().height() // 2
        anchor_viewport_x = self.image_scroll.viewport().width() // 2

        # The content point under the anchor in old coordinates
        old_content_y = old_scroll_y + anchor_viewport_y
        old_content_x = old_scroll_x + anchor_viewport_x

        # Apply new zoom (immediate, not debounced, for anchor math)
        self._zoom_level = new_zoom
        new_width = int(self._base_width() * new_zoom)
        self.image_strip.display_width = new_width
        self._force_rebuild_now()

        # Scale the content point to new coordinates
        if old_zoom > 0:
            ratio = new_zoom / old_zoom
            new_content_y = int(old_content_y * ratio)
            new_content_x = int(old_content_x * ratio)
        else:
            new_content_y = 0
            new_content_x = 0

        # Set scroll to keep anchor point in the same viewport position
        vbar.setValue(new_content_y - anchor_viewport_y)
        hbar.setValue(new_content_x - anchor_viewport_x)

        self.zoom_label.setText(f"{int(new_zoom * 100)}%")

    def eventFilter(self, obj, event):
        """Single event filter for the panel — Qt keeps only the LAST method
        of a given name per class, so all filter logic must live here:
        Tab/Shift+Tab zone navigation, Ctrl+Wheel zoom, margin box-drawing,
        and Ctrl+Z/Ctrl+Shift+Z from child text editors."""
        # Tab / Shift+Tab → navigate zones
        if event.type() == QEvent.KeyPress and self.isVisible():
            if event.key() == Qt.Key_Tab and not (event.modifiers() & ~Qt.ShiftModifier):
                self._select_zone_relative(1)
                return True
            if event.key() == Qt.Key_Backtab:
                self._select_zone_relative(-1)
                return True

        if obj == self.image_scroll.viewport() and event.type() == QEvent.Wheel:
            if event.modifiers() & Qt.ControlModifier:
                delta = event.angleDelta().y()
                if delta > 0:
                    factor = 1.15
                elif delta < 0:
                    factor = 1 / 1.15
                else:
                    return True

                # Anchor to mouse position in viewport
                mouse_vp_y = event.pos().y()
                self._apply_zoom(self._zoom_level * factor, anchor_viewport_y=mouse_vp_y)
                return True  # consumed

        # Box creation started on the empty margin AROUND the image strip:
        # clicks there never reach the strip widget, so drive its creation
        # handlers from here with positions mapped (and clamped) into it.
        if obj == self.image_scroll.viewport():
            strip = self.image_strip
            if (event.type() == QEvent.MouseButtonPress
                    and event.button() == Qt.LeftButton
                    and strip.image_paths):
                strip_pos = strip.mapFromGlobal(event.globalPos())
                if not strip.rect().contains(strip_pos):
                    strip.begin_create_at(strip_pos)
                    self._margin_create_active = True
                    return True
            elif (event.type() == QEvent.MouseMove
                    and getattr(self, "_margin_create_active", False)):
                strip.drag_create_to(strip.mapFromGlobal(event.globalPos()))
                return True
            elif (event.type() == QEvent.MouseButtonRelease
                    and event.button() == Qt.LeftButton
                    and getattr(self, "_margin_create_active", False)):
                strip.finish_create_at(strip.mapFromGlobal(event.globalPos()))
                self._margin_create_active = False
                return True

        # Intercept Ctrl+Z / Ctrl+Shift+Z from child text editors for panel-level undo
        if isinstance(obj, QPlainTextEdit) and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
                if event.modifiers() & Qt.ShiftModifier:
                    self._do_redo()
                else:
                    self._do_undo()
                return True

        return super().eventFilter(obj, event)

    def _fit_strip_to_viewport(self):
        """Resize the image strip based on current zoom and viewport width."""
        if not self.image_strip.image_paths:
            return
        new_width = int(self._base_width() * self._zoom_level)
        if new_width > 0 and new_width != self.image_strip.display_width:
            self.image_strip.display_width = new_width
            self.image_strip._rebuild_display()  # debounced internally

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_strip_to_viewport()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._fit_strip_to_viewport)

    def _force_rebuild_now(self):
        """Force an immediate (non-debounced) rebuild - used after zoom."""
        if not self.image_strip.image_paths:
            return
        self.image_strip._cached_display_width = 0  # invalidate cache
        self.image_strip._do_rebuild()

    # -- Transcription ---

    def _apply_transcription_text(self, text: str):
        """Split text by '.' and '*' and distribute across panels from current selection."""
        boxes = self.image_strip.current_batch_boxes()
        if not boxes:
            QMessageBox.warning(self, "No Panels", "No panels to apply text to. Run YOLO or create boxes first.")
            return

        # Save undo snapshot before applying
        self._push_undo_now()

        # Find starting index from selected box
        start_idx = 0
        if self.image_strip.selected_box_id:
            for i, box in enumerate(boxes):
                if box.id == self.image_strip.selected_box_id:
                    start_idx = i
                    break

        # Split by '.' and '*' into panel chunks
        # '*' is a separator that stays appended to the current panel's text
        # '.' is a separator that is discarded
        chunks = []
        current = ""
        for ch in text:
            if ch == ".":
                piece = current.strip()
                if piece:
                    chunks.append(piece)
                current = ""
            elif ch == "*":
                piece = current.strip()
                if piece:
                    chunks.append(piece + "*")
                current = ""
            else:
                current += ch
        # Remaining text after last separator
        piece = current.strip()
        if piece:
            chunks.append(piece)

        if not chunks:
            return

        # Apply chunks to panels starting from start_idx
        for i, chunk in enumerate(chunks):
            panel_idx = start_idx + i
            if panel_idx >= len(boxes):
                break
            box = boxes[panel_idx]
            box.text = chunk
            # Update the text slot widget if it exists
            if box.id in self.text_slots.slots:
                slot = self.text_slots.slots[box.id]
                slot.text_edit.blockSignals(True)
                slot.text_edit.setPlainText(chunk)
                slot.text_edit.blockSignals(False)

        applied = min(len(chunks), len(boxes) - start_idx)
        remaining = len(chunks) - applied
        msg = f"Applied {applied} sentences starting from Panel {start_idx + 1}"
        if remaining > 0:
            msg += f" ({remaining} sentences didn't fit)"
        self.status_label.setText(msg)

        # Clear the text loader and push a new snapshot (so undo restores both)
        self.transcription_widget.text_edit.blockSignals(True)
        self.transcription_widget.text_edit.clear()
        self.transcription_widget.text_edit.blockSignals(False)
        self._push_undo_now()

        self.state_changed.emit()

    def _select_zone_relative(self, direction: int):
        """Select next (+1) or previous (-1) zone, updating strip + text slot + focus."""
        batch_boxes = self.image_strip.current_batch_boxes()
        if not batch_boxes:
            return
        current_id = self.image_strip.selected_box_id
        if not current_id:
            target = batch_boxes[0] if direction > 0 else batch_boxes[-1]
        else:
            for i, box in enumerate(batch_boxes):
                if box.id == current_id:
                    target = batch_boxes[(i + direction) % len(batch_boxes)]
                    break
            else:
                target = batch_boxes[0]
        # Select box on image strip + highlight text slot
        self._on_slot_clicked(target.id)
        # Focus the text edit so the user can type immediately
        slot = self.text_slots.slots.get(target.id)
        if slot:
            slot.text_edit.setFocus()
            # Move cursor to end
            cursor = slot.text_edit.textCursor()
            cursor.movePosition(cursor.End)
            slot.text_edit.setTextCursor(cursor)

    def keyPressEvent(self, event):
        """Handle R key for recording, Ctrl+Z/Ctrl+Shift+Z for undo/redo."""
        # Ctrl+Z = undo, Ctrl+Shift+Z = redo (panel-level, not per-text-field)
        if event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            if event.modifiers() & Qt.ShiftModifier:
                self._do_redo()
            else:
                self._do_undo()
            return

        if not event.modifiers():
            focused = QApplication.focusWidget()
            if not isinstance(focused, (QPlainTextEdit, QLineEdit, QTextEdit)):
                if event.key() == Qt.Key_R:
                    self.transcription_widget.handle_key_r()
                    return
                if event.key() == Qt.Key_W:
                    self.transcription_widget.insert_separator(".")
                    return
                if event.key() == Qt.Key_E:
                    self.transcription_widget.insert_separator("*")
                    return
                if event.key() == Qt.Key_PageUp:
                    self._batch_prev()
                    return
                if event.key() == Qt.Key_PageDown:
                    self._batch_next()
                    return
        super().keyPressEvent(event)

    def _do_undo(self):
        snapshot = self._undo.undo()
        if snapshot:
            self._restore_text_snapshot(snapshot)
            self.status_label.setText("Undo")
            self.state_changed.emit()

    def _do_redo(self):
        snapshot = self._undo.redo()
        if snapshot:
            self._restore_text_snapshot(snapshot)
            self.status_label.setText("Redo")
            self.state_changed.emit()

    # -- Project state ---

    def get_state(self) -> dict:
        """Collect all panel state into a dict for project save."""
        # Sync text from slots back to boxes
        texts = self.text_slots.get_all_text()
        for box in self.image_strip.boxes:
            if box.id in texts:
                box.text = texts[box.id]

        image_folder = self._current_folder or ""

        def _rel(p: str) -> str:
            """Path relative to image_folder, with forward-slash separators
            so saves are portable across Windows / macOS / Linux. Falls
            back to basename when the path is on a different drive (Windows
            quirk) or when image_folder is empty."""
            if not image_folder:
                return os.path.basename(p)
            try:
                rel = os.path.relpath(p, image_folder)
            except ValueError:
                return os.path.basename(p)
            return rel.replace(os.sep, "/")

        # Expand merged entries to their raw basenames + relative paths so
        # the saved image_files list mirrors the pristine folder layout.
        # merge_groups is the source of truth for reconstructing the merged
        # view on load. The relative-path field disambiguates chapters that
        # share filenames (e.g. chapter_0001/0007.webp vs
        # chapter_0008/0007.webp) — basename-only saves can't tell them
        # apart, which has corrupted projects in the past.
        image_files: List[str] = []
        image_files_rel: List[str] = []
        for p in self.image_strip.all_image_paths:
            if self.image_strip._is_merged_path(p):
                grp = self.image_strip._find_merge_group_for_merged_path(p)
                if grp:
                    image_files.extend(os.path.basename(x) for x in grp)
                    image_files_rel.extend(_rel(x) for x in grp)
                    continue
            image_files.append(os.path.basename(p))
            image_files_rel.append(_rel(p))

        boxes = []
        for box in self.image_strip.boxes:
            boxes.append({
                "id": box.id,
                "image_index": box.image_index,
                "x": box.x, "y": box.y,
                "w": box.w, "h": box.h,
                "confidence": box.confidence,
                "text": box.text,
            })

        # Serialize merge groups in BOTH basename (legacy) and relative-path
        # form. Loaders that understand the new format use _rel for an
        # unambiguous lookup; older loaders fall back to basenames.
        merge_groups_basenames = [
            [os.path.basename(p) for p in grp]
            for grp in self.image_strip.merge_groups
        ]
        merge_groups_rel = [
            [_rel(p) for p in grp]
            for grp in self.image_strip.merge_groups
        ]

        # Save-time guard: chapter-per-folder layouts have duplicate
        # basenames (chapter_0001/0007.webp vs chapter_0008/0007.webp). If
        # somehow image_files_rel ended up empty for that case (e.g. some
        # future regression on the rel-path logic), refuse to save instead
        # of writing an ambiguous file that would silently corrupt the
        # project on next load.
        from collections import Counter
        bn_counts = Counter(
            os.path.basename(p) for p in self.image_strip.all_image_paths
        )
        has_dup_basenames = any(v > 1 for v in bn_counts.values())
        if has_dup_basenames and not image_files_rel:
            raise RuntimeError(
                "Refusing to save: project has duplicate basenames "
                "(chapter-per-folder layout) but image_files_rel is empty. "
                "This save format would silently corrupt the project on "
                "next load. Please report this — it indicates a regression."
            )

        state = {
            "image_folder": image_folder,
            "image_files": image_files,
            "image_files_rel": image_files_rel,
            "merge_groups": merge_groups_basenames,
            "merge_groups_rel": merge_groups_rel,
            "boxes": boxes,
            "font": {
                "family": self.text_slots._font_family,
                "size": self.text_slots._font_size,
                "color": self.text_slots._font_color,
            },
            "zoom": self._zoom_level,
            "confidence": self.confidence_spin.value(),
            "min_size": self.min_size_spin.value(),
            "splitter_sizes": self.splitter.sizes(),
        }

        # Save multi-folder info if this is a mixed project
        if hasattr(self, '_mixed_folders') and self._mixed_folders:
            state["image_folders"] = self._mixed_folders

        # Save AI cost tracking
        state["ai_cost"] = {
            "input_tokens": self._project_input_tokens,
            "output_tokens": self._project_output_tokens,
            "cost_usd": self._project_cost_usd,
        }

        return state

    def _reorder_images_to_match(self, saved_files: list,
                                 saved_files_rel: Optional[list] = None,
                                 image_folder: str = "") -> None:
        """Reorder all_image_paths to match the saved file order.
        Ensures box image_index values still point to the correct images.

        Drops any `merged_*.jpg` placeholders from saved_files — those don't
        exist in the folder and are reconstructed by `reapply_merge_groups_on_load`.
        Skipping them prevents the old "append extras at the end" fallback from
        scrambling the merged originals.

        When `saved_files_rel` is provided (new save format), matches by
        path relative to image_folder. That's unambiguous even when
        chapter folders share basenames like 0007.webp. Falls back to
        basename matching for old saves without `_rel` data — but that
        path silently breaks projects with duplicate basenames.
        """
        if not saved_files and not saved_files_rel:
            return
        strip = self.image_strip
        if not strip.all_image_paths:
            return

        def _to_rel(p: str) -> str:
            if not image_folder:
                return os.path.basename(p)
            try:
                return os.path.relpath(p, image_folder).replace(os.sep, "/")
            except ValueError:
                return os.path.basename(p)

        # Prefer the relative-path lookup when the save provides one.
        if saved_files_rel:
            saved_rel_raw = [
                f.replace(os.sep, "/") for f in saved_files_rel
                if not os.path.basename(f).startswith("merged_")
            ]
            loaded_by_rel = {_to_rel(p): p for p in strip.all_image_paths}

            reordered = []
            for rel in saved_rel_raw:
                if rel in loaded_by_rel:
                    reordered.append(loaded_by_rel.pop(rel))
            # Append any extras (new images in folder since save).
            for p in strip.all_image_paths:
                rel = _to_rel(p)
                if rel in loaded_by_rel:
                    reordered.append(p)
                    del loaded_by_rel[rel]

            if len(reordered) == len(strip.all_image_paths):
                strip.all_image_paths = reordered
                strip._finalize_batches()
                return
            # If the lengths don't match the rel-path data is stale; fall
            # through to basename matching just in case it produces a
            # workable order.

        # Legacy basename matching. Ambiguous for chapter-per-folder layouts.
        saved_raw = [f for f in saved_files if not f.startswith("merged_")]

        loaded = {}
        for p in strip.all_image_paths:
            loaded[os.path.basename(p)] = p

        # Warn the user when this branch is taken AND duplicates exist,
        # because that combination silently produces broken projects.
        from collections import Counter
        bn_counts = Counter(os.path.basename(p) for p in strip.all_image_paths)
        n_dups = sum(1 for v in bn_counts.values() if v > 1)
        if n_dups:
            logger.warning(
                f"_reorder_images_to_match: legacy save without "
                f"image_files_rel and {n_dups} duplicate basename(s); "
                f"merge groups may map to wrong files. Re-saving the "
                f"project will upgrade it to the unambiguous format."
            )

        reordered = []
        for name in saved_raw:
            if name in loaded:
                reordered.append(loaded.pop(name))
        # Append any extra files not in saved list (new images in folder)
        for p in strip.all_image_paths:
            if os.path.basename(p) in loaded:
                reordered.append(p)
                del loaded[os.path.basename(p)]

        if len(reordered) == len(strip.all_image_paths):
            strip.all_image_paths = reordered
            strip._finalize_batches()

    def set_state(self, state: dict) -> None:
        """Restore panel state from a loaded project dict."""
        # Any project load exits review mode; the Load Delivery flow
        # re-applies its diffs right after restoring the state.
        self.text_slots.set_review_diffs({})
        saved_files = state.get("image_files", [])
        saved_files_rel = state.get("image_files_rel") or None

        # Check for multi-folder (mixed) project
        image_folders = state.get("image_folders", [])
        if image_folders:
            valid_folders = [f for f in image_folders if os.path.isdir(f)]
            if valid_folders:
                self._mixed_folders = valid_folders
                self._current_folder = os.path.dirname(valid_folders[0])
                self._zoom_level = 1.0
                self.zoom_label.setText("100%")
                display_w = self._base_width()
                self.image_strip.display_width = display_w
                self.image_strip.load_multiple_folders(valid_folders)
            else:
                from PyQt5.QtWidgets import QMessageBox, QFileDialog
                QMessageBox.warning(
                    self, "Images Not Found",
                    "The original image folders were not found on this PC.\n\n"
                    "Select the parent folder containing the chapter subfolders.",
                )
                folder = QFileDialog.getExistingDirectory(
                    self, "Select the folder with images on this PC")
                if folder:
                    self._mixed_folders = [folder]
                    self._current_folder = folder
                    self._zoom_level = 1.0
                    self.zoom_label.setText("100%")
                    display_w = self._base_width()
                    self.image_strip.display_width = display_w
                    self.image_strip.load_multiple_folders([folder])
                else:
                    return
        else:
            # Single folder project
            image_folder = state.get("image_folder", "")
            if image_folder and os.path.isdir(image_folder):
                self.load_images_from_folder(image_folder)
            else:
                from PyQt5.QtWidgets import QMessageBox, QFileDialog
                QMessageBox.warning(
                    self, "Images Not Found",
                    f"The original image folder was not found on this PC:\n"
                    f"{image_folder}\n\n"
                    "Select the folder containing the images on this PC.",
                )
                folder = QFileDialog.getExistingDirectory(
                    self, "Select the folder with images on this PC")
                if folder:
                    self.load_images_from_folder(folder)
                else:
                    return

        # Reorder loaded images to match saved file order
        # so that box image_index values still point to the correct images.
        # Pass image_folder and the rel-path list so duplicate basenames
        # can be disambiguated (chapter_0001/0007.webp vs chapter_0008/0007.webp).
        active_image_folder = (
            self._current_folder or state.get("image_folder", "") or ""
        )

        # Detect "legacy save in a project that needs rel-paths" — basename
        # matching is ambiguous when chapter folders share filenames, and
        # has silently corrupted projects in the past. Warn user before
        # proceeding so the corruption isn't silent.
        from collections import Counter
        bn_counts = Counter(
            os.path.basename(p) for p in self.image_strip.all_image_paths
        )
        n_dup_basenames = sum(1 for v in bn_counts.values() if v > 1)
        if n_dup_basenames and not saved_files_rel:
            from PyQt5.QtWidgets import QMessageBox
            reply = QMessageBox.warning(
                self, "Legacy save with ambiguous filenames",
                f"This project has {n_dup_basenames} basename(s) that repeat "
                f"across chapter folders, but the save file is in the legacy "
                f"format (no relative paths).\n\n"
                f"Loading will fall back to first-match-by-basename, which "
                f"can silently put boxes on the wrong chapter's images.\n\n"
                f"To upgrade safely:\n"
                f"  1. Click Cancel below.\n"
                f"  2. Reopen the project (will retry with same data).\n"
                f"  3. Use Clear Boxes + YOLO All to rebuild boxes.\n"
                f"  4. Save — the new save will be in the unambiguous format.\n\n"
                f"Or click OK to load anyway and accept the risk.",
                QMessageBox.Ok | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply != QMessageBox.Ok:
                logger.warning(
                    "Legacy load aborted by user "
                    f"({n_dup_basenames} duplicate basenames present)"
                )
                return

        self._reorder_images_to_match(
            saved_files, saved_files_rel=saved_files_rel,
            image_folder=active_image_folder,
        )

        # Re-apply merge groups BEFORE restoring boxes (box image_index values
        # are saved in the merged-coordinate space). Prefer the relative-path
        # form when available — basenames are ambiguous in chapter projects
        # and have silently corrupted projects in the past.
        merge_groups_rel = state.get("merge_groups_rel")
        merge_groups = state.get("merge_groups", [])
        if merge_groups_rel:
            self.image_strip.reapply_merge_groups_on_load(
                merge_groups, groups_rel=merge_groups_rel,
                image_folder=active_image_folder,
            )
        elif merge_groups:
            self.image_strip.reapply_merge_groups_on_load(merge_groups)

        # Restore boxes
        box_dicts = state.get("boxes", [])
        self.image_strip.boxes.clear()
        for bd in box_dicts:
            self.image_strip.boxes.append(ScriptBox(
                id=bd["id"],
                image_index=bd["image_index"],
                x=bd["x"], y=bd["y"],
                w=bd["w"], h=bd["h"],
                confidence=bd.get("confidence", 1.0),
                text=bd.get("text", ""),
            ))
        self.image_strip._sort_boxes()
        self.image_strip.update()

        # Rebuild text slots with current batch's boxes
        batch_boxes = self.image_strip.current_batch_boxes()
        global_zone_num = {b.id: i + 1 for i, b in enumerate(self.image_strip.boxes)}
        zone_numbers = [global_zone_num[b.id] for b in batch_boxes]
        self.text_slots.rebuild(batch_boxes, zone_numbers=zone_numbers)
        self._install_undo_filters_on_slots()
        self.box_count_label.setText(f"Boxes: {len(self.image_strip.boxes)}")

        # Restore font settings
        font = state.get("font", {})
        if font:
            family = font.get("family", "Arial")
            size = font.get("size", 12)
            color = font.get("color", "#ffffff")
            self.text_slots._font_family = family
            self.text_slots._font_size = size
            self.text_slots._font_color = color
            self.text_slots.font_combo.setCurrentFont(QFont(family))
            self.text_slots.size_spin.setValue(size)
            self.text_slots.color_btn.setStyleSheet(
                f"QPushButton {{ background-color: {color}; border: 1px solid #888; border-radius: 3px; }}"
            )
            self.text_slots._on_font_changed()

        # Restore confidence
        conf = state.get("confidence", 25)
        self.confidence_spin.setValue(conf)

        # Restore min size
        min_size = state.get("min_size", int(self.config.get("yolo_min_size", 40)))
        self.min_size_spin.setValue(min_size)

        # Restore splitter sizes
        sizes = state.get("splitter_sizes")
        if sizes and len(sizes) == 2:
            self.splitter.setSizes(sizes)

        # Restore zoom
        zoom = state.get("zoom", 1.0)
        self._apply_zoom(zoom)

        # Restore AI cost tracking
        ai_cost = state.get("ai_cost", {})
        self._project_input_tokens = ai_cost.get("input_tokens", 0)
        self._project_output_tokens = ai_cost.get("output_tokens", 0)
        self._project_cost_usd = ai_cost.get("cost_usd", 0.0)
        self._update_cost_label()


class _SpinnerWidget(QWidget):
    """Small rotating circle spinner for loading indication.

    Hovering swaps the spinner for a red X and clicking emits `clicked`,
    so the owner can offer cancel-in-place (AI Write uses it to skip all
    remaining API calls while keeping everything already written)."""

    clicked = pyqtSignal()

    def __init__(self, size: int = 20, parent=None):
        super().__init__(parent)
        self._size = size
        self._angle = 0
        self._hover = False
        self.setFixedSize(size, size)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(
            "Cancel — the call in flight still finishes (its zones are "
            "kept), but no further calls are sent")
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._tick)

    def _tick(self):
        self._angle = (self._angle + 15) % 360
        self.update()

    def setVisible(self, visible: bool):
        super().setVisible(visible)
        if visible:
            self._timer.start()
        else:
            self._timer.stop()
            self._hover = False

    def enterEvent(self, event):
        self._hover = True
        self.update()

    def leaveEvent(self, event):
        self._hover = False
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()

    def paintEvent(self, event):
        from PyQt5.QtGui import QPainter, QPen, QColor
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        margin = 2
        rect = self.rect().adjusted(margin, margin, -margin, -margin)

        if self._hover:
            # Red X = click to cancel
            pen = QPen(QColor(220, 60, 60), 3)
            painter.setPen(pen)
            painter.drawEllipse(rect)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            inset = max(4, self._size // 4)
            r = self.rect().adjusted(inset, inset, -inset, -inset)
            painter.drawLine(r.topLeft(), r.bottomRight())
            painter.drawLine(r.topRight(), r.bottomLeft())
            painter.end()
            return

        pen = QPen(QColor(80, 80, 80), 3)
        painter.setPen(pen)
        painter.drawEllipse(rect)

        pen = QPen(QColor(0, 188, 212), 3)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawArc(rect, self._angle * 16, 90 * 16)
        painter.end()


class _GeminiThread(QThread):
    """Background thread for running GeminiWriter (and the optional GPT
    second-pass editor) without blocking the UI."""

    zone_done = pyqtSignal(int, str)      # zone_num, text
    batch_done = pyqtSignal(int, int)     # batch_num, total_batches
    error = pyqtSignal(str)               # error message
    finished_all = pyqtSignal()           # all zones complete

    def __init__(self, writer: GeminiWriter, bridge_dir: str,
                 zone_from: int = 1, zone_to: int = None,
                 claude_bridge=None, boxes=None, image_paths=None, image_folder=None,
                 gpt_editor=None):
        super().__init__()
        self._writer = writer
        self._bridge_dir = bridge_dir
        self._zone_from = zone_from
        self._zone_to = zone_to
        self._claude_bridge = claude_bridge
        self._boxes = boxes
        self._image_paths = image_paths
        self._image_folder = image_folder
        self._gpt_editor = gpt_editor

    def _build_zone_payload(self, batch_zones):
        """Convert GeminiWriter's batch entries into the shape GPTEditor
        wants: {number, text, panel_image_path, expanded_image_path}.

        We re-read the just-written zone_NNN.txt for each zone so the GPT
        pass operates on whatever Gemini ACTUALLY persisted (handles the
        case where the parser dropped a zone)."""
        from pathlib import Path
        bridge = Path(self._bridge_dir)
        scripts_dir = bridge / "scripts"
        debug_dir = bridge / "debug_crops"

        payload = []
        for z in batch_zones:
            num = z.get("number")
            if num is None:
                continue
            zone_file = scripts_dir / f"zone_{num:03d}.txt"
            try:
                draft_text = zone_file.read_text(encoding="utf-8").strip()
            except Exception:
                draft_text = z.get("text", "") or ""
            if not draft_text:
                continue

            panel_rel = z.get("panel_image", "")
            panel_path = bridge / panel_rel if panel_rel else None
            expanded_path = debug_dir / f"zone_{num:03d}_expanded.jpg"

            payload.append({
                "number": num,
                "text": draft_text,
                "panel_image_path": str(panel_path) if panel_path and panel_path.exists() else None,
                "expanded_image_path": str(expanded_path) if expanded_path.exists() else None,
            })
        return payload

    def _on_batch_done(self, batch_num, total_batches, batch_zones):
        """Called by GeminiWriter after each batch of drafts is on disk.
        Runs the GPT second pass synchronously (still off the main thread)
        before letting the writer continue with the next batch."""
        if self._gpt_editor is not None and batch_zones:
            try:
                payload = self._build_zone_payload(batch_zones)
                if payload:
                    from pathlib import Path
                    scripts_dir = Path(self._bridge_dir) / "scripts"
                    replaced = self._gpt_editor.revise_and_write_zones(
                        payload,
                        scripts_dir,
                        on_progress=lambda msg: None,
                        on_zone_done=lambda zn, txt: self.zone_done.emit(zn, txt),
                    )
                    logger.info(
                        f"_GeminiThread: GPT revised {replaced}/{len(payload)} "
                        f"zones in batch {batch_num}/{total_batches}"
                    )
            except Exception as e:
                # GPT failures must not abort Gemini progress — Gemini's
                # draft is already on disk, so we just keep going.
                logger.exception(
                    f"_GeminiThread: GPT pass failed on batch {batch_num}: {e}"
                )
        self.batch_done.emit(batch_num, total_batches)

    def run(self):
        # Export panels in this thread (not main thread) to avoid UI freeze
        if self._claude_bridge and self._boxes:
            self._claude_bridge.export(
                self._boxes, self._image_paths,
                self._image_folder, self._bridge_dir,
            )

        self._writer.generate(
            self._bridge_dir,
            zone_from=self._zone_from,
            zone_to=self._zone_to,
            on_zone_done=lambda zn, txt: self.zone_done.emit(zn, txt),
            on_batch_done=self._on_batch_done,
            on_error=lambda msg: self.error.emit(msg),
            on_complete=lambda: self.finished_all.emit(),
        )

"""
Cut & Stitch Panel - Stitches manhua images vertically and cuts at pure color zones.
Produces properly-sized image segments for YOLO to work with.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from ui.scale_manager import scale_manager

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


class StitchWorker(QThread):
    """Background thread for stitch + cut processing."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(str, int)  # output_dir, segment_count
    error = pyqtSignal(str)

    def __init__(self, image_paths: List[str], output_dir: str,
                 strip_height: int = 100, color_tolerance: int = 15,
                 min_segment_height: int = 2000, target_width: int = 800):
        super().__init__()
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.strip_height = strip_height
        self.color_tolerance = color_tolerance
        self.min_segment_height = min_segment_height
        self.target_width = target_width

    def run(self):
        try:
            self._process()
        except Exception as e:
            logger.error(f"Stitch worker error: {e}", exc_info=True)
            self.error.emit(str(e))

    def _process(self):
        # Step 1: Load and stitch all images vertically
        self.progress.emit(f"Loading {len(self.image_paths)} images...")
        images = []
        for i, path in enumerate(self.image_paths):
            raw = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning(f"Could not load: {path}")
                continue
            images.append(img)
            if (i + 1) % 5 == 0:
                self.progress.emit(f"Loaded {i + 1}/{len(self.image_paths)} images...")

        if not images:
            self.error.emit("No images could be loaded")
            return

        # Normalize all images to same width
        self.progress.emit("Stitching images vertically...")
        target_w = self.target_width
        resized = []
        for img in images:
            h, w = img.shape[:2]
            if w != target_w:
                scale = target_w / w
                new_h = int(h * scale)
                img = cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)
            resized.append(img)

        combined = np.vstack(resized)
        total_h = combined.shape[0]
        logger.info(f"Combined strip: {combined.shape[1]}x{total_h}")
        self.progress.emit(f"Combined strip: {combined.shape[1]}x{total_h} - finding cut points...")

        # Step 2: Find pure color regions for cutting
        cut_points = self._find_cut_points(combined)
        logger.info(f"Found {len(cut_points)} cut points")
        self.progress.emit(f"Found {len(cut_points)} cut points, saving segments...")

        # Step 3: Cut and save segments
        # Prepare output directory
        out_path = Path(self.output_dir)
        if out_path.exists():
            shutil.rmtree(out_path)
        out_path.mkdir(parents=True, exist_ok=True)

        all_cuts = [0] + cut_points + [total_h]
        saved = 0
        for i in range(len(all_cuts) - 1):
            y_start = all_cuts[i]
            y_end = all_cuts[i + 1]
            seg_h = y_end - y_start
            if seg_h < 200:
                continue

            segment = combined[y_start:y_end]
            out_file = out_path / f"segment_{saved + 1:04d}.jpg"
            ok, encoded = cv2.imencode(".jpg", segment, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                encoded.tofile(str(out_file))
                saved += 1
                self.progress.emit(f"Saved segment {saved} ({segment.shape[1]}x{seg_h})")

        logger.info(f"Saved {saved} segments to {self.output_dir}")
        self.finished.emit(self.output_dir, saved)

    def _find_cut_points(self, image: np.ndarray) -> List[int]:
        """Find horizontal pure-color strips suitable for cutting."""
        height, width = image.shape[:2]
        step = max(10, self.strip_height // 2)
        min_search = min(500, height - 100)

        candidates = []
        for y in range(min_search, height - self.strip_height, step):
            strip = image[y:y + self.strip_height]
            pixels = strip.reshape(-1, 3).astype(np.float32)
            std = np.std(pixels, axis=0)
            max_std = float(np.max(std))

            if max_std <= self.color_tolerance:
                confidence = 1.0 - (max_std / 255.0)
                mean_intensity = float(np.mean(pixels))
                if mean_intensity > 240:
                    confidence = min(1.0, confidence + 0.2)
                elif mean_intensity > 200:
                    confidence = min(1.0, confidence + 0.1)
                candidates.append((y + self.strip_height // 2, confidence))

        if not candidates:
            # Fallback: uniform cuts at target segment height
            num_segs = max(1, height // self.min_segment_height)
            return [int(height * i / num_segs) for i in range(1, num_segs)]

        # Merge nearby candidates, keep best confidence
        merged = []
        candidates.sort(key=lambda c: c[0])
        cur_y, cur_conf = candidates[0]
        for y, conf in candidates[1:]:
            if y - cur_y < self.strip_height * 1.5:
                if conf > cur_conf:
                    cur_y, cur_conf = y, conf
            else:
                merged.append((cur_y, cur_conf))
                cur_y, cur_conf = y, conf
        merged.append((cur_y, cur_conf))

        # Filter: ensure minimum distance between cuts
        final = []
        last_cut = 0
        min_dist = max(self.min_segment_height, 600)
        for y, conf in merged:
            if conf < 0.7:
                continue
            if y - last_cut >= min_dist and (height - y) >= 500:
                final.append(y)
                last_cut = y

        return final


class StitchPanel(QWidget):
    """Cut & Stitch panel: stitch images into a strip, then cut at color boundaries."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: Optional[StitchWorker] = None
        self._input_dir: Optional[str] = None
        self._output_dir: Optional[str] = None
        self._setup_ui()

    def _setup_ui(self):
        s = scale_manager
        layout = QHBoxLayout()
        layout.setContentsMargins(s.scale(5), s.scale(5), s.scale(5), s.scale(5))
        self.setLayout(layout)

        # Left: controls
        controls = QWidget()
        controls.setMaximumWidth(s.scale(320))
        controls.setMinimumWidth(s.scale(280))
        cl = QVBoxLayout()
        cl.setContentsMargins(s.scale(5), s.scale(5), s.scale(5), s.scale(5))
        controls.setLayout(cl)

        # Input folder
        in_group = QGroupBox("Input")
        in_group.setStyleSheet("QGroupBox { color: #00bcd4; font-weight: bold; }")
        in_layout = QVBoxLayout()

        self.input_label = QLabel("No folder selected")
        self.input_label.setStyleSheet(f"color: #aaa; font-size: {s.scale_font(11)}px;")
        self.input_label.setWordWrap(True)
        in_layout.addWidget(self.input_label)

        self.input_count_label = QLabel("")
        self.input_count_label.setStyleSheet(f"color: #888; font-size: {s.scale_font(11)}px;")
        in_layout.addWidget(self.input_count_label)

        input_btn = QPushButton("Choose Input Folder")
        input_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #00bcd4; color: white;
                font-weight: bold; padding: {s.scale(8)}px;
                border-radius: {s.scale(4)}px; font-size: {s.scale_font(12)}px;
            }}
            QPushButton:hover {{ background-color: #00acc1; }}
        """)
        input_btn.clicked.connect(self._choose_input)
        in_layout.addWidget(input_btn)

        in_group.setLayout(in_layout)
        cl.addWidget(in_group)

        # Output folder
        out_group = QGroupBox("Output")
        out_group.setStyleSheet("QGroupBox { color: #4caf50; font-weight: bold; }")
        out_layout = QVBoxLayout()

        self.output_label = QLabel("No folder selected")
        self.output_label.setStyleSheet(f"color: #aaa; font-size: {s.scale_font(11)}px;")
        self.output_label.setWordWrap(True)
        out_layout.addWidget(self.output_label)

        output_btn = QPushButton("Choose Output Folder")
        output_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #404040; color: #fff;
                padding: {s.scale(6)}px; border-radius: {s.scale(3)}px;
            }}
            QPushButton:hover {{ background-color: #505050; }}
        """)
        output_btn.clicked.connect(self._choose_output)
        out_layout.addWidget(output_btn)

        out_group.setLayout(out_layout)
        cl.addWidget(out_group)

        # Settings
        settings_group = QGroupBox("Settings")
        settings_group.setStyleSheet("QGroupBox { color: #aaaaaa; font-weight: bold; }")
        sf = QFormLayout()

        self.strip_height_spin = QSpinBox()
        self.strip_height_spin.setRange(20, 500)
        self.strip_height_spin.setValue(100)
        self.strip_height_spin.setSuffix(" px")
        self.strip_height_spin.setToolTip("Height of strips analyzed for color uniformity")
        sf.addRow("Strip Height:", self.strip_height_spin)

        self.tolerance_spin = QSpinBox()
        self.tolerance_spin.setRange(1, 50)
        self.tolerance_spin.setValue(15)
        self.tolerance_spin.setToolTip("Max color std deviation to consider as pure color")
        sf.addRow("Color Tolerance:", self.tolerance_spin)

        self.min_seg_spin = QSpinBox()
        self.min_seg_spin.setRange(500, 20000)
        self.min_seg_spin.setValue(2000)
        self.min_seg_spin.setSuffix(" px")
        self.min_seg_spin.setSingleStep(500)
        self.min_seg_spin.setToolTip("Minimum segment height (won't cut closer than this)")
        sf.addRow("Min Segment H:", self.min_seg_spin)

        self.target_width_spin = QSpinBox()
        self.target_width_spin.setRange(400, 3000)
        self.target_width_spin.setValue(800)
        self.target_width_spin.setSuffix(" px")
        self.target_width_spin.setSingleStep(100)
        self.target_width_spin.setToolTip("All images are resized to this width before stitching")
        sf.addRow("Target Width:", self.target_width_spin)

        settings_group.setLayout(sf)
        cl.addWidget(settings_group)

        # Process button
        self.process_btn = QPushButton("Stitch & Cut")
        self.process_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #ff9800; color: white;
                font-weight: bold; padding: {s.scale(12)}px;
                border-radius: {s.scale(4)}px; font-size: {s.scale_font(14)}px;
            }}
            QPushButton:hover {{ background-color: #f57c00; }}
            QPushButton:disabled {{ background-color: #555; color: #888; }}
        """)
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self._run_stitch)
        cl.addWidget(self.process_btn)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: #3c3c3c; border: 1px solid #555;
                border-radius: {s.scale(3)}px; text-align: center; color: white;
            }}
            QProgressBar::chunk {{ background-color: #ff9800; }}
        """)
        cl.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(f"color: #888; font-size: {s.scale_font(11)}px;")
        self.status_label.setWordWrap(True)
        cl.addWidget(self.status_label)

        cl.addStretch()
        layout.addWidget(controls)

        # Right: preview of output segments
        preview_container = QWidget()
        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_container.setLayout(preview_layout)

        preview_header = QLabel("Output Preview")
        preview_header.setStyleSheet(f"color: #aaa; font-weight: bold; font-size: {s.scale_font(12)}px; padding: {s.scale(4)}px;")
        preview_layout.addWidget(preview_header)

        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.preview_scroll.setStyleSheet("QScrollArea { background-color: #1e1e1e; border: 1px solid #555; }")

        self.preview_widget = QWidget()
        self.preview_vlayout = QVBoxLayout()
        self.preview_vlayout.setContentsMargins(5, 5, 5, 5)
        self.preview_vlayout.setSpacing(4)
        self.preview_vlayout.setAlignment(Qt.AlignTop)
        self.preview_widget.setLayout(self.preview_vlayout)
        self.preview_scroll.setWidget(self.preview_widget)

        preview_layout.addWidget(self.preview_scroll)
        layout.addWidget(preview_container, 1)

    def _choose_input(self):
        d = QFileDialog.getExistingDirectory(self, "Select Input Folder with Images")
        if d:
            self._input_dir = d
            files = self._get_image_files(d)
            display = d if len(d) < 45 else "..." + d[-42:]
            self.input_label.setText(display)
            self.input_count_label.setText(f"{len(files)} images found")
            self._update_process_btn()

    def _choose_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if d:
            self._output_dir = d
            display = d if len(d) < 45 else "..." + d[-42:]
            self.output_label.setText(display)
            self._update_process_btn()

    def _get_image_files(self, folder: str) -> List[str]:
        return sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        )

    def _update_process_btn(self):
        has_input = self._input_dir and len(self._get_image_files(self._input_dir)) > 0
        has_output = self._output_dir is not None
        self.process_btn.setEnabled(bool(has_input and has_output))

    def _run_stitch(self):
        if not self._input_dir or not self._output_dir:
            return

        files = self._get_image_files(self._input_dir)
        if not files:
            self.status_label.setText("No images in input folder")
            return

        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Processing...")

        self._worker = StitchWorker(
            files, self._output_dir,
            strip_height=self.strip_height_spin.value(),
            color_tolerance=self.tolerance_spin.value(),
            min_segment_height=self.min_seg_spin.value(),
            target_width=self.target_width_spin.value(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, msg: str):
        self.status_label.setText(msg)

    def _on_finished(self, output_dir: str, count: int):
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.status_label.setText(f"Done: {count} segments saved to output folder")
        self._load_preview(output_dir)

    def _on_error(self, msg: str):
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.status_label.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Stitch Error", msg)

    def _load_preview(self, folder: str):
        """Show thumbnails of the output segments."""
        # Clear existing
        while self.preview_vlayout.count():
            child = self.preview_vlayout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        files = self._get_image_files(folder)
        vp_w = self.preview_scroll.viewport().width() - 20

        for i, path in enumerate(files):
            pix = QPixmap(path)
            if pix.isNull():
                continue

            # Scale to fit preview width
            if pix.width() > vp_w and vp_w > 50:
                pix = pix.scaledToWidth(vp_w, Qt.SmoothTransformation)

            frame = QFrame()
            frame.setStyleSheet("QFrame { background-color: #2a2a2a; border: 1px solid #444; border-radius: 3px; }")
            fl = QVBoxLayout()
            fl.setContentsMargins(4, 4, 4, 4)
            fl.setSpacing(2)
            frame.setLayout(fl)

            label_text = QLabel(f"Segment {i + 1}  ({pix.width()}x{pix.height()})")
            label_text.setStyleSheet(f"color: #aaa; font-size: {scale_manager.scale_font(10)}px; background: transparent;")
            fl.addWidget(label_text)

            img_label = QLabel()
            img_label.setPixmap(pix)
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setStyleSheet("background: transparent;")
            fl.addWidget(img_label)

            self.preview_vlayout.addWidget(frame)

        self.preview_vlayout.addStretch()

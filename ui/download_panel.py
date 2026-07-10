"""
Download Panel - Extract and download images from web pages.
Opens a real Chrome browser for the user to scroll, then grabs all images on click.
"""

import logging
import os
import re
import hashlib
from typing import List, Optional
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from ui.scale_manager import scale_manager

logger = logging.getLogger(__name__)


class ImageItem:
    """Represents a scraped image with metadata"""
    def __init__(self, url: str, width: int = 0, height: int = 0, size_bytes: int = 0):
        self.url = url
        self.width = width
        self.height = height
        self.size_bytes = size_bytes
        self.thumbnail: Optional[QPixmap] = None
        self.selected = True
        self.data: Optional[bytes] = None
        self.filename = self._guess_filename()

    def _guess_filename(self) -> str:
        parsed = urlparse(self.url)
        name = os.path.basename(parsed.path)
        if not name or '.' not in name:
            h = hashlib.md5(self.url.encode()).hexdigest()[:10]
            name = f"image_{h}.jpg"
        return name

    @property
    def size_str(self) -> str:
        if self.size_bytes > 0:
            if self.size_bytes > 1024 * 1024:
                return f"{self.size_bytes / (1024*1024):.1f} MB"
            return f"{self.size_bytes / 1024:.1f} KB"
        return "?"

    @property
    def resolution_str(self) -> str:
        if self.width > 0 and self.height > 0:
            return f"{self.width}x{self.height}"
        return "?"


class ImageThumbnailWidget(QWidget):
    """Single image thumbnail with checkbox overlay"""
    clicked = pyqtSignal(int)

    def __init__(self, index: int, item: ImageItem, thumb_size: int = 0):
        super().__init__()
        self.index = index
        self.item = item
        self.thumb_size = thumb_size or scale_manager.scale(150)
        self.setFixedSize(self.thumb_size + scale_manager.scale(10),
                          self.thumb_size + scale_manager.scale(40))
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, event):
        s = scale_manager
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.item.selected:
            painter.fillRect(self.rect(), QColor(0, 120, 212, 40))
            painter.setPen(QPen(QColor(0, 180, 255), 2))
            painter.drawRect(self.rect().adjusted(1, 1, -1, -1))
        else:
            painter.fillRect(self.rect(), QColor(60, 60, 60))
            painter.setPen(QPen(QColor(80, 80, 80), 1))
            painter.drawRect(self.rect().adjusted(1, 1, -1, -1))

        pad = s.scale(5)
        thumb_rect = QRect(pad, pad, self.thumb_size, self.thumb_size)
        if self.item.thumbnail:
            scaled = self.item.thumbnail.scaled(
                self.thumb_size, self.thumb_size,
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            x_off = (self.thumb_size - scaled.width()) // 2
            y_off = (self.thumb_size - scaled.height()) // 2
            painter.drawPixmap(pad + x_off, pad + y_off, scaled)
        else:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(thumb_rect, Qt.AlignCenter, "Loading...")

        cb_size = s.scale(18)
        cb_pad = s.scale(8)
        cb_rect = QRect(cb_pad, cb_pad, cb_size, cb_size)
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.setBrush(QColor(40, 40, 40, 200))
        painter.drawRect(cb_rect)
        if self.item.selected:
            painter.setPen(QPen(QColor(0, 200, 80), max(2, s.scale(3))))
            cx, cy = cb_pad + cb_size // 4, cb_pad + cb_size * 2 // 3
            painter.drawLine(cx, cy - cb_size // 6, cx + cb_size // 4, cy)
            painter.drawLine(cx + cb_size // 4, cy, cx + cb_size * 3 // 4, cy - cb_size // 2)

        painter.setPen(QColor(180, 180, 180))
        font = painter.font()
        font.setPointSize(s.scale_font(7))
        painter.setFont(font)
        lh = s.scale(14)
        info = f"{self.item.resolution_str}  {self.item.size_str}"
        painter.drawText(QRect(pad, self.thumb_size + s.scale(7), self.thumb_size, lh), Qt.AlignCenter, info)
        painter.drawText(QRect(pad, self.thumb_size + s.scale(20), self.thumb_size, lh), Qt.AlignCenter, f"#{self.index + 1}")
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.item.selected = not self.item.selected
            self.update()
            self.clicked.emit(self.index)


class GrabWorker(QThread):
    """Background thread that downloads images, preserving page order"""
    progress = pyqtSignal(str)
    image_found = pyqtSignal(int, object)
    finished = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, img_urls: List[str], referer: str,
                 min_width: int = 100, min_height: int = 100):
        super().__init__()
        self.img_urls = img_urls
        self.referer = referer
        self.min_width = min_width
        self.min_height = min_height

    def run(self):
        try:
            import requests
        except ImportError:
            self.error.emit("Missing 'requests' package.\n\npip install requests")
            return

        self.progress.emit(f"Downloading {len(self.img_urls)} images...")
        count = 0
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': self.referer,
        }

        def fetch(idx_url):
            idx, url = idx_url
            try:
                r = requests.get(url, headers=headers, timeout=15)
                r.raise_for_status()
                data = r.content
                if len(data) < 500:
                    return None
                qimg = QImage()
                if not qimg.loadFromData(data):
                    return None
                w, h = qimg.width(), qimg.height()
                if w < self.min_width or h < self.min_height:
                    return None
                item = ImageItem(url, w, h, len(data))
                item.data = data
                item.thumbnail = QPixmap.fromImage(qimg)
                return (idx, item)
            except Exception:
                return None

        results = {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(fetch, (i, url)): i
                       for i, url in enumerate(self.img_urls)}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    idx, item = result
                    results[idx] = item
                    count += 1
                    self.progress.emit(f"Loaded {count} images...")

        for idx in sorted(results.keys()):
            self.image_found.emit(idx, results[idx])

        self.finished.emit(count)


class DownloadPanel(QWidget):
    """Download panel: opens real Chrome, user scrolls, clicks Grab to extract images"""
    images_downloaded = pyqtSignal(str)

    _download_counter = 0

    def __init__(self):
        super().__init__()
        self.items: List[ImageItem] = []
        self.thumb_widgets: List[ImageThumbnailWidget] = []
        self.grab_worker: Optional[GrabWorker] = None
        self.output_dir: Optional[str] = None
        self.driver = None
        self.setup_ui()

    def setup_ui(self):
        s = scale_manager
        layout = QHBoxLayout()
        layout.setContentsMargins(s.scale(5), s.scale(5), s.scale(5), s.scale(5))
        self.setLayout(layout)

        controls = self._create_controls()
        controls.setMaximumWidth(s.scale(280))
        controls.setMinimumWidth(s.scale(240))
        layout.addWidget(controls)

        grid_container = self._create_grid_view()
        layout.addWidget(grid_container, 1)

    def _create_controls(self) -> QWidget:
        s = scale_manager
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(s.scale(5), s.scale(5), s.scale(5), s.scale(5))
        panel.setLayout(layout)

        # URL input
        url_group = QGroupBox("Browser")
        url_group.setStyleSheet("QGroupBox { color: #00bcd4; font-weight: bold; }")
        url_layout = QVBoxLayout()

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Paste URL here...")
        self.url_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: #3c3c3c; color: #ffffff;
                border: 1px solid #555; border-radius: {s.scale(4)}px;
                padding: {s.scale(6)}px; font-size: {s.scale_font(12)}px;
            }}
        """)
        self.url_input.returnPressed.connect(self._open_browser)
        url_layout.addWidget(self.url_input)

        self.open_btn = QPushButton("Open in Chrome")
        self.open_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #00bcd4; color: white;
                font-weight: bold; padding: {s.scale(10)}px;
                border-radius: {s.scale(4)}px; font-size: {s.scale_font(13)}px;
            }}
            QPushButton:hover {{ background-color: #00acc1; }}
            QPushButton:disabled {{ background-color: #555; }}
        """)
        self.open_btn.clicked.connect(self._open_browser)
        url_layout.addWidget(self.open_btn)

        info_label = QLabel(
            "1. Paste URL and click Open\n"
            "2. Scroll to load all images\n"
            "3. Click GRAB below"
        )
        info_label.setStyleSheet(f"color: #888; font-size: {s.scale_font(11)}px;")
        url_layout.addWidget(info_label)

        url_group.setLayout(url_layout)
        layout.addWidget(url_group)

        # Grab button
        self.grab_btn = QPushButton("GRAB IMAGES")
        self.grab_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #ff9800; color: white;
                font-weight: bold; padding: {s.scale(12)}px;
                border-radius: {s.scale(4)}px; font-size: {s.scale_font(14)}px;
            }}
            QPushButton:hover {{ background-color: #f57c00; }}
            QPushButton:disabled {{ background-color: #555; }}
        """)
        self.grab_btn.clicked.connect(self._grab_images)
        self.grab_btn.setEnabled(False)
        layout.addWidget(self.grab_btn)

        # Close browser
        self.close_browser_btn = QPushButton("Close Chrome")
        self.close_browser_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #d32f2f; color: #fff;
                padding: {s.scale(6)}px; border-radius: {s.scale(3)}px;
            }}
            QPushButton:hover {{ background-color: #b71c1c; }}
        """)
        self.close_browser_btn.clicked.connect(self._close_browser)
        self.close_browser_btn.setVisible(False)
        layout.addWidget(self.close_browser_btn)

        # Filters
        filter_group = QGroupBox("Filters")
        filter_group.setStyleSheet("QGroupBox { color: #aaaaaa; font-weight: bold; }")
        filter_layout = QFormLayout()

        self.min_width_spin = QSpinBox()
        self.min_width_spin.setRange(0, 5000)
        self.min_width_spin.setValue(200)
        self.min_width_spin.setSuffix(" px")
        filter_layout.addRow("Min Width:", self.min_width_spin)

        self.min_height_spin = QSpinBox()
        self.min_height_spin.setRange(0, 10000)
        self.min_height_spin.setValue(200)
        self.min_height_spin.setSuffix(" px")
        filter_layout.addRow("Min Height:", self.min_height_spin)

        self.auto_deselect_w_spin = QSpinBox()
        self.auto_deselect_w_spin.setRange(0, 5000)
        self.auto_deselect_w_spin.setValue(500)
        self.auto_deselect_w_spin.setSuffix(" px")
        filter_layout.addRow("Auto-deselect W:", self.auto_deselect_w_spin)

        self.auto_deselect_h_spin = QSpinBox()
        self.auto_deselect_h_spin.setRange(0, 10000)
        self.auto_deselect_h_spin.setValue(3001)
        self.auto_deselect_h_spin.setSuffix(" px")
        filter_layout.addRow("Auto-deselect H:", self.auto_deselect_h_spin)

        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        # Selection
        sel_group = QGroupBox("Selection")
        sel_group.setStyleSheet("QGroupBox { color: #aaaaaa; font-weight: bold; }")
        sel_layout = QVBoxLayout()

        btn_row = QHBoxLayout()
        for text, handler in [("All", lambda: self._set_all_selected(True)),
                              ("None", lambda: self._set_all_selected(False)),
                              ("Invert", self._invert_selection)]:
            btn = QPushButton(text)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: #404040; color: #fff;
                    padding: {s.scale(5)}px {s.scale(10)}px; border-radius: {s.scale(3)}px;
                }}
                QPushButton:hover {{ background-color: #505050; }}
            """)
            btn.clicked.connect(handler)
            btn_row.addWidget(btn)
        sel_layout.addLayout(btn_row)

        self.resort_btn = QPushButton("Re-sort by Filename")
        self.resort_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #7b1fa2; color: #fff;
                font-weight: bold; padding: {s.scale(6)}px;
                border-radius: {s.scale(3)}px;
            }}
            QPushButton:hover {{ background-color: #9c27b0; }}
        """)
        self.resort_btn.setToolTip("Re-sort images by the numeric part of their URL/filename")
        self.resort_btn.clicked.connect(self._resort_images)
        sel_layout.addWidget(self.resort_btn)

        self.selection_label = QLabel("0 / 0 selected")
        self.selection_label.setStyleSheet("color: #aaa;")
        sel_layout.addWidget(self.selection_label)

        sel_group.setLayout(sel_layout)
        layout.addWidget(sel_group)

        # Download
        dl_group = QGroupBox("Download")
        dl_group.setStyleSheet("QGroupBox { color: #4caf50; font-weight: bold; }")
        dl_layout = QVBoxLayout()

        self.output_label = QLabel("No output folder")
        self.output_label.setStyleSheet(f"color: #aaa; font-size: {s.scale_font(11)}px;")
        self.output_label.setWordWrap(True)
        dl_layout.addWidget(self.output_label)

        self.counter_label = QLabel("Next file: 0001")
        self.counter_label.setStyleSheet(f"color: #aaa; font-size: {s.scale_font(11)}px;")
        dl_layout.addWidget(self.counter_label)

        reset_counter_btn = QPushButton("Reset Counter")
        reset_counter_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #404040; color: #fff;
                padding: {s.scale(4)}px; border-radius: {s.scale(3)}px; font-size: {s.scale_font(11)}px;
            }}
            QPushButton:hover {{ background-color: #505050; }}
        """)
        reset_counter_btn.clicked.connect(self._reset_counter)
        dl_layout.addWidget(reset_counter_btn)

        choose_dir_btn = QPushButton("Choose Folder")
        choose_dir_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #404040; color: #fff;
                padding: {s.scale(6)}px; border-radius: {s.scale(3)}px;
            }}
            QPushButton:hover {{ background-color: #505050; }}
        """)
        choose_dir_btn.clicked.connect(self.choose_output_dir)
        dl_layout.addWidget(choose_dir_btn)

        self.download_btn = QPushButton("Download Selected")
        self.download_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #4caf50; color: white;
                font-weight: bold; padding: {s.scale(8)}px;
                border-radius: {s.scale(4)}px; font-size: {s.scale_font(13)}px;
            }}
            QPushButton:hover {{ background-color: #43a047; }}
            QPushButton:disabled {{ background-color: #555; }}
        """)
        self.download_btn.clicked.connect(self.download_selected)
        self.download_btn.setEnabled(False)
        dl_layout.addWidget(self.download_btn)

        self.dl_progress = QProgressBar()
        self.dl_progress.setVisible(False)
        self.dl_progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: #3c3c3c; border: 1px solid #555;
                border-radius: {s.scale(3)}px; text-align: center; color: white;
            }}
            QProgressBar::chunk {{ background-color: #4caf50; }}
        """)
        dl_layout.addWidget(self.dl_progress)

        dl_group.setLayout(dl_layout)
        layout.addWidget(dl_group)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(f"color: #888; font-size: {s.scale_font(11)}px;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()
        return panel

    def _create_grid_view(self) -> QWidget:
        container = QWidget()
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container.setLayout(container_layout)

        self.grid_scroll = QScrollArea()
        self.grid_scroll.setWidgetResizable(True)
        self.grid_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.grid_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.grid_scroll.setStyleSheet("QScrollArea { background-color: #2b2b2b; border: 1px solid #555; }")

        self.grid_widget = QWidget()
        self.grid_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(scale_manager.scale(8))
        self.grid_layout.setContentsMargins(scale_manager.scale(10), scale_manager.scale(10),
                                            scale_manager.scale(10), scale_manager.scale(10))
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.grid_widget.setLayout(self.grid_layout)

        self.grid_scroll.setWidget(self.grid_widget)
        container_layout.addWidget(self.grid_scroll)
        return container

    def _open_browser(self):
        url = self.url_input.text().strip()
        if not url:
            self.status_label.setText("Enter a URL first")
            return
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            self.url_input.setText(url)

        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            from selenium.webdriver.chrome.service import Service as ChromeService
        except ImportError:
            QMessageBox.critical(self, "Missing Selenium",
                "Selenium is required.\n\nInstall with:\n  pip install selenium webdriver-manager")
            return

        self._close_browser()
        self.status_label.setText("Opening Chrome...")
        self.open_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            chrome_options = ChromeOptions()
            chrome_options.add_argument('--window-size=1100,900')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option('excludeSwitches', ['enable-automation', 'enable-logging'])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument('--log-level=3')
            chrome_options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})

            try:
                from webdriver_manager.chrome import ChromeDriverManager
                service = ChromeService(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
            except ImportError:
                self.driver = webdriver.Chrome(options=chrome_options)

            self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'
            })

            self.driver.get(url)
            self.grab_btn.setEnabled(True)
            self.close_browser_btn.setVisible(True)
            self.open_btn.setEnabled(True)
            self.status_label.setText("Chrome opened. Scroll the page, then click GRAB IMAGES.")

        except Exception as e:
            self.open_btn.setEnabled(True)
            self.status_label.setText(f"Failed to open Chrome: {e}")
            QMessageBox.critical(self, "Browser Error", f"Could not open Chrome:\n{e}")

    def _close_browser(self):
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None
        self.grab_btn.setEnabled(False)
        self.close_browser_btn.setVisible(False)

    def _grab_images(self):
        if not self.driver:
            self.status_label.setText("No browser open")
            return

        self.grab_btn.setEnabled(False)
        self.status_label.setText("Extracting image URLs from page...")
        QApplication.processEvents()

        try:
            page_url = self.driver.current_url

            perf_urls = self.driver.execute_script("""
                return performance.getEntriesByType('resource')
                    .filter(function(e) { return e.initiatorType === 'img' || e.initiatorType === 'css' || e.initiatorType === 'other'; })
                    .map(function(e) { return e.name; });
            """) or []

            cdp_urls = []
            try:
                self.driver.execute_cdp_cmd('Performance.getMetrics', {})
                perf_log = self.driver.get_log('performance')
                import json
                for entry in perf_log:
                    try:
                        msg = json.loads(entry['message'])['message']
                        if msg['method'] == 'Network.responseReceived':
                            resp = msg['params']['response']
                            mime = resp.get('mimeType', '')
                            if mime.startswith('image/'):
                                cdp_urls.append(resp['url'])
                    except Exception:
                        pass
            except Exception:
                pass

            dom_urls = self.driver.execute_script("""
                var results = [];
                var seen = {};
                function add(u, y) {
                    if (!u || u.startsWith('data:') || seen[u]) return;
                    seen[u] = true;
                    results.push({url: u, y: y});
                }
                function scan(root) {
                    root.querySelectorAll('img').forEach(function(img) {
                        var rect = img.getBoundingClientRect();
                        var y = rect.top + window.scrollY;
                        if (img.currentSrc) add(img.currentSrc, y);
                        if (img.src) add(img.src, y);
                        for (var i = 0; i < img.attributes.length; i++) {
                            var v = img.attributes[i].value;
                            if (v && v.match && v.match(/^https?:\\/\\//))
                                add(v, y);
                        }
                    });
                    root.querySelectorAll('*').forEach(function(el) {
                        if (el.shadowRoot) scan(el.shadowRoot);
                    });
                }
                scan(document);
                results.sort(function(a, b) { return a.y - b.y; });
                return results.map(function(r) { return r.url; });
            """) or []

            seen_ordered = set(dom_urls)
            extra_urls = []
            for u in (perf_urls + cdp_urls):
                if u not in seen_ordered:
                    seen_ordered.add(u)
                    extra_urls.append(u)

            raw_urls = dom_urls + extra_urls

        except Exception as e:
            self.grab_btn.setEnabled(True)
            self.status_label.setText(f"Grab failed: {e}")
            return

        if not raw_urls:
            self.grab_btn.setEnabled(True)
            self.status_label.setText("No image URLs found. Scroll more and try again.")
            return

        resolved = []
        seen = set()
        for u in raw_urls:
            full = urljoin(page_url, u)
            if full not in seen:
                seen.add(full)
                resolved.append(full)

        self.status_label.setText(f"Found {len(resolved)} URLs, downloading thumbnails...")

        self._clear_grid()
        self.items.clear()
        self.thumb_widgets.clear()

        self.grab_worker = GrabWorker(
            resolved, page_url,
            min_width=self.min_width_spin.value(),
            min_height=self.min_height_spin.value()
        )
        self.grab_worker.progress.connect(self._on_grab_progress)
        self.grab_worker.image_found.connect(self._on_image_found)
        self.grab_worker.finished.connect(self._on_grab_done)
        self.grab_worker.error.connect(self._on_grab_error)
        self.grab_worker.start()

    def _on_grab_progress(self, msg: str):
        self.status_label.setText(msg)

    def _on_image_found(self, original_idx: int, item: ImageItem):
        deselect_w = self.auto_deselect_w_spin.value()
        deselect_h = self.auto_deselect_h_spin.value()
        if item.width < deselect_w or item.height < deselect_h:
            item.selected = False

        self.items.append(item)
        idx = len(self.items) - 1

        thumb = ImageThumbnailWidget(idx, item)
        thumb.clicked.connect(self._update_selection_count)
        self.thumb_widgets.append(thumb)

        self._relayout_grid()
        self._update_selection_count()

    def _get_grid_cols(self) -> int:
        s = scale_manager
        vp_width = self.grid_scroll.viewport().width()
        cell_w = s.scale(150) + s.scale(15)  # thumb_size + spacing
        return max(2, (vp_width - s.scale(20)) // cell_w)

    def _relayout_grid(self):
        cols = self._get_grid_cols()
        for i, thumb in enumerate(self.thumb_widgets):
            row = i // cols
            col = i % cols
            self.grid_layout.addWidget(thumb, row, col)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.thumb_widgets:
            self._relayout_grid()

    def _on_grab_done(self, count: int):
        self.grab_btn.setEnabled(True)
        self.status_label.setText(f"Done: {count} images grabbed. Select and download.")
        self._update_selection_count()

    def _on_grab_error(self, msg: str):
        self.grab_btn.setEnabled(True)
        self.status_label.setText(f"Error: {msg}")

    def _clear_grid(self):
        while self.grid_layout.count():
            child = self.grid_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def _set_all_selected(self, selected: bool):
        for item in self.items:
            item.selected = selected
        for w in self.thumb_widgets:
            w.update()
        self._update_selection_count()

    def _invert_selection(self):
        for item in self.items:
            item.selected = not item.selected
        for w in self.thumb_widgets:
            w.update()
        self._update_selection_count()

    def _resort_images(self):
        """Re-sort grabbed images by the numeric part of their URL/filename."""
        if not self.items:
            return

        def sort_key(item: ImageItem) -> tuple:
            # Extract the filename/path from URL
            parsed = urlparse(item.url)
            path = parsed.path
            # Find all numbers in the path, use the last one as primary sort key
            nums = re.findall(r'\d+', path)
            if nums:
                # Use the last number (most likely the page/image number)
                return (int(nums[-1]), path)
            return (float('inf'), path)

        self.items.sort(key=sort_key)

        # Rebuild thumbnail widgets with new order
        self._clear_grid()
        self.thumb_widgets.clear()
        for idx, item in enumerate(self.items):
            thumb = ImageThumbnailWidget(idx, item)
            thumb.clicked.connect(self._update_selection_count)
            self.thumb_widgets.append(thumb)
        self._relayout_grid()
        self.status_label.setText(f"Re-sorted {len(self.items)} images by filename")

    def _update_selection_count(self, _idx=None):
        selected = sum(1 for i in self.items if i.selected)
        total = len(self.items)
        self.selection_label.setText(f"{selected} / {total} selected")
        self.download_btn.setEnabled(selected > 0 and self.output_dir is not None)

    def choose_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Choose Download Folder")
        if d:
            self.output_dir = d
            display = d if len(d) < 40 else "..." + d[-37:]
            self.output_label.setText(display)
            self._update_selection_count()

    def download_selected(self):
        if not self.output_dir:
            QMessageBox.warning(self, "No Folder", "Choose an output folder first.")
            return

        selected = [i for i in self.items if i.selected]
        if not selected:
            return

        self.download_btn.setEnabled(False)
        self.dl_progress.setVisible(True)
        self.dl_progress.setRange(0, len(selected))

        saved = 0
        for idx, item in enumerate(selected):
            try:
                DownloadPanel._download_counter += 1
                num = DownloadPanel._download_counter
                ext = os.path.splitext(item.filename)[1] or '.jpg'
                out_name = f"{num:04d}{ext}"
                out_path = os.path.join(self.output_dir, out_name)

                if item.data:
                    with open(out_path, 'wb') as f:
                        f.write(item.data)
                    saved += 1
                else:
                    import requests
                    r = requests.get(item.url, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                        'Referer': self.url_input.text(),
                    }, timeout=15)
                    r.raise_for_status()
                    with open(out_path, 'wb') as f:
                        f.write(r.content)
                    saved += 1
            except Exception as e:
                logger.error(f"Failed to download {item.url}: {e}")

            self.dl_progress.setValue(idx + 1)
            QApplication.processEvents()

        self._update_counter_label()
        self.dl_progress.setVisible(False)
        self.download_btn.setEnabled(True)
        self.status_label.setText(f"Downloaded {saved}/{len(selected)} images")

        if saved > 0:
            self.images_downloaded.emit(self.output_dir)

        QMessageBox.information(
            self, "Download Complete",
            f"Saved {saved} images to:\n{self.output_dir}"
        )

    def _reset_counter(self):
        DownloadPanel._download_counter = 0
        self._update_counter_label()
        self.status_label.setText("Counter reset to 0")

    def _update_counter_label(self):
        self.counter_label.setText(f"Next file: {DownloadPanel._download_counter + 1:04d}")

    def closeEvent(self, event):
        self._close_browser()
        super().closeEvent(event)

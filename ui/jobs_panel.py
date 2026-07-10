"""Jobs tab — writer-facing side of the job-distribution system.

Writers see the job board (Firestore `script_jobs`), claim a job, get the
images zip auto-downloaded from Google Drive and the admin's pre-processed
mscript loaded straight into the Script tab, fix boxes/text, then press
Deliver to upload the corrected mscript back to Firestore.

Only the mscript JSON travels back — images never re-upload, so writers
need no Google account. See JOBS_TAB_BRIEF.md for the full design.
"""

import json
import logging
import os
import re
import zipfile
from typing import List, Optional

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from app import jobs_client
from app.gdrive_download import download_public_file, GDriveError, GDriveCancelled
from app.jobs_client import JobsClientError, JobTakenError
from app.project import save_project, load_project
from ui.scale_manager import scale_manager

logger = logging.getLogger(__name__)

# Job downloads live inside the app folder — same pattern in both editions:
# <app root>\Projects\{Title}_{chapters}\ (images.zip, images/, mscripts).
_APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.join(_APP_ROOT, "Projects")
POLL_MS = 30_000

# Row backgrounds by job situation (light backgrounds -> dark text)
COLOR_AVAILABLE = QColor("#e6f7e6")
COLOR_CLAIMED_OTHER = QColor("#fde8e8")
COLOR_CLAIMED_ME = QColor("#e3f2fd")
COLOR_DONE = QColor("#eceff1")
ROW_TEXT = QColor("#1b1b1b")


def _sanitize(name: str) -> str:
    """Filesystem-safe folder fragment."""
    return re.sub(r"[^\w\- ]+", "", str(name)).strip().replace(" ", "_") or "job"


def job_workspace_dir(job: dict) -> str:
    return os.path.join(
        WORKSPACE_ROOT,
        f"{_sanitize(job.get('title', 'job'))}_{_sanitize(job.get('chapters', ''))}",
    )


def _resolve_image_root(extract_dir: str) -> str:
    """If the zip had a single root folder, use it; else the extract dir."""
    try:
        entries = [e for e in os.listdir(extract_dir) if e != "__MACOSX"]
    except OSError:
        return extract_dir
    if len(entries) == 1:
        only = os.path.join(extract_dir, entries[0])
        if os.path.isdir(only):
            return only
    return extract_dir


def _rewrite_state_paths(state: dict, extract_dir: str) -> dict:
    """Point the mscript's image_folder(s) — absolute paths from the ADMIN
    machine — at the local extract folder. image_files_rel then resolves
    the individual files against the zip's internal layout."""
    root = _resolve_image_root(extract_dir)
    state["image_folder"] = root
    folders = state.get("image_folders")
    if folders:
        rewritten = []
        for f in folders:
            cand = os.path.join(root, os.path.basename(os.path.normpath(str(f))))
            if os.path.isdir(cand):
                rewritten.append(cand)
        if rewritten and len(rewritten) == len(folders):
            state["image_folders"] = rewritten
        else:
            # Can't map the multi-folder layout — let the single-folder
            # loader walk chapter subfolders under root instead.
            state.pop("image_folders", None)
    return state


# ================================================================ workers

class _ListWorker(QThread):
    done = pyqtSignal(list)
    error = pyqtSignal(str)

    def run(self):
        try:
            self.done.emit(jobs_client.list_jobs())
        except JobsClientError as e:
            self.error.emit(str(e))
        except Exception as e:  # never let a worker kill the app
            logger.exception("Jobs list failed")
            self.error.emit(f"Unexpected error: {e}")


class _ClaimWorker(QThread):
    """claim -> fetch mscript -> download zip -> extract -> prepare state."""

    progress = pyqtSignal(str, int, int)  # phase text, done, total(-1 = busy)
    ready = pyqtSignal(dict, dict, str)   # state, job, work_mscript_path
    error = pyqtSignal(str)

    def __init__(self, job: dict, writer: str, parent=None):
        super().__init__(parent)
        self._job = job
        self._writer = writer
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            job = jobs_client.claim_job(self._job["_id"], self._writer)
            job.setdefault("title", self._job.get("title", ""))
            job.setdefault("chapters", self._job.get("chapters", ""))

            wdir = job_workspace_dir(job)
            os.makedirs(wdir, exist_ok=True)

            self.progress.emit("Downloading script from server...", 0, -1)
            n_chunks = int(job.get("original_chunks") or
                           self._job.get("original_chunks") or 1)
            state = jobs_client.fetch_original_mscript(
                job["_id"], n_chunks,
                progress_cb=lambda d, t: self.progress.emit(
                    "Downloading script from server...", d, t),
            )

            zip_path = os.path.join(wdir, "images.zip")
            extract_dir = os.path.join(wdir, "images")
            if not os.path.isdir(extract_dir) or not os.listdir(extract_dir):
                gdrive_ref = (job.get("images_gdrive_id") or
                              self._job.get("images_gdrive_id") or "")
                if not os.path.exists(zip_path):
                    self.progress.emit("Downloading images zip...", 0, -1)
                    download_public_file(
                        gdrive_ref, zip_path,
                        progress_cb=lambda d, t: self.progress.emit(
                            "Downloading images zip...", d, t),
                        cancel_cb=lambda: self._cancel,
                    )
                self.progress.emit("Extracting images...", 0, -1)
                with zipfile.ZipFile(zip_path) as zf:
                    members = zf.namelist()
                    for i, member in enumerate(members):
                        if self._cancel:
                            raise GDriveCancelled("Cancelled.")
                        zf.extract(member, extract_dir)
                        if i % 25 == 0:
                            self.progress.emit(
                                "Extracting images...", i + 1, len(members))

            state = _rewrite_state_paths(state, extract_dir)

            # Persist everything the writer needs to reopen / deliver later
            save_project(os.path.join(wdir, "original.mscript"), dict(state))
            work_path = os.path.join(wdir, "work.mscript")
            if not os.path.exists(work_path):
                save_project(work_path, dict(state))
            with open(os.path.join(wdir, "job.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "job_id": job["_id"],
                    "title": job.get("title", ""),
                    "chapters": job.get("chapters", ""),
                    "claimed_by": self._writer,
                    "images_gdrive_id": job.get("images_gdrive_id", ""),
                    "original_chunks": n_chunks,
                }, f, indent=2, ensure_ascii=False)

            self.ready.emit(state, job, work_path)
        except (JobTakenError, JobsClientError, GDriveError, zipfile.BadZipFile,
                OSError) as e:
            self.error.emit(str(e))
        except Exception as e:
            logger.exception("Claim & Download failed")
            self.error.emit(f"Unexpected error: {e}")


class _DeliverWorker(QThread):
    progress = pyqtSignal(str, int, int)
    done = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, job_id: str, writer: str, state: dict,
                 work_path: str, parent=None):
        super().__init__(parent)
        self._job_id = job_id
        self._writer = writer
        self._state = state
        self._work_path = work_path

    def run(self):
        try:
            if self._work_path:
                save_project(self._work_path, self._state)  # local backup first
            self.progress.emit("Uploading corrected script...", 0, -1)
            job = jobs_client.deliver_job(
                self._job_id, self._writer, self._state,
                progress_cb=lambda d, t: self.progress.emit(
                    "Uploading corrected script...", d, t),
            )
            self.done.emit(job)
        except JobsClientError as e:
            self.error.emit(str(e))
        except Exception as e:
            logger.exception("Deliver failed")
            self.error.emit(f"Unexpected error: {e}")


# ================================================================ panel

class JobsPanel(QWidget):
    """The Jobs tab. Talks to MainWindow for state restore/gather."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._main_window = main_window
        self._jobs: List[dict] = []
        self._active_job: Optional[dict] = None   # claimed/opened this session
        self._active_work_path: str = ""
        self._worker: Optional[QThread] = None
        self._settings = QSettings("MomoScript", "JobsPanel")

        self._build_ui()

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(POLL_MS)
        self._poll_timer.timeout.connect(self._refresh_silent)

    # ------------------------------------------------------------- UI

    def _build_ui(self):
        s = scale_manager
        root = QVBoxLayout(self)
        root.setContentsMargins(s.scale(10), s.scale(10), s.scale(10), s.scale(10))
        root.setSpacing(s.scale(8))

        btn_style = f"""
            QPushButton {{
                background-color: #404040; color: #eee; border: none;
                padding: {s.scale(6)}px {s.scale(14)}px;
                border-radius: {s.scale(3)}px; font-size: {s.scale_font(12)}px;
            }}
            QPushButton:hover {{ background-color: #505050; }}
            QPushButton:disabled {{ color: #777; }}
        """

        # -- top row: writer identity + refresh + status
        top = QHBoxLayout()
        top.setSpacing(s.scale(8))
        name_lbl = QLabel("Your name:")
        name_lbl.setStyleSheet(f"color: #ccc; font-size: {s.scale_font(12)}px;")
        top.addWidget(name_lbl)

        self.writer_edit = QLineEdit()
        self.writer_edit.setPlaceholderText("who am I (shown on claimed jobs)")
        self.writer_edit.setText(self._settings.value("writer_name", "", str))
        self.writer_edit.setMaximumWidth(s.scale(220))
        self.writer_edit.setStyleSheet(f"""
            QLineEdit {{
                background-color: #404040; color: #fff; border: 1px solid #555;
                padding: {s.scale(4)}px; border-radius: {s.scale(3)}px;
                font-size: {s.scale_font(12)}px;
            }}
        """)
        self.writer_edit.editingFinished.connect(self._save_writer_name)
        top.addWidget(self.writer_edit)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setStyleSheet(btn_style)
        self.refresh_btn.clicked.connect(self.refresh)
        top.addWidget(self.refresh_btn)

        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet(f"color: #aaa; font-size: {s.scale_font(11)}px;")
        top.addWidget(self.status_lbl, 1)
        root.addLayout(top)

        # -- job table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(
            ["Title", "Chapters", "Status", "Claimed by"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2, 3):
            self.table.horizontalHeader().setSectionResizeMode(
                c, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background-color: #2b2b2b; color: #ddd; gridline-color: #444;
                font-size: {s.scale_font(12)}px;
            }}
            QHeaderView::section {{
                background-color: #353535; color: #ccc;
                padding: {s.scale(5)}px; border: 0;
            }}
        """)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        root.addWidget(self.table, 1)

        # -- progress bar (hidden unless a worker is busy)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setTextVisible(True)
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: #353535; color: #fff; border: 1px solid #555;
                border-radius: {s.scale(3)}px; text-align: center;
                font-size: {s.scale_font(11)}px;
            }}
            QProgressBar::chunk {{ background-color: #00bcd4; }}
        """)
        root.addWidget(self.progress)

        # -- action row
        actions = QHBoxLayout()
        actions.setSpacing(s.scale(8))

        self.claim_btn = QPushButton("Claim && Download")
        self.claim_btn.setStyleSheet(btn_style.replace("#404040", "#2e7d32"))
        self.claim_btn.clicked.connect(self._claim_selected)
        actions.addWidget(self.claim_btn)

        self.open_btn = QPushButton("Open")
        self.open_btn.setStyleSheet(btn_style)
        self.open_btn.clicked.connect(self._open_selected)
        actions.addWidget(self.open_btn)

        self.deliver_btn = QPushButton("Deliver")
        self.deliver_btn.setStyleSheet(btn_style.replace("#404040", "#e65100"))
        self.deliver_btn.clicked.connect(self._deliver_selected)
        actions.addWidget(self.deliver_btn)

        actions.addStretch()
        root.addLayout(actions)

        self._update_buttons()

    # ------------------------------------------------------------- helpers

    def _save_writer_name(self):
        self._settings.setValue("writer_name", self.writer_edit.text().strip())

    def writer_name(self) -> str:
        return self.writer_edit.text().strip()

    def _main(self):
        return self._main_window or self.window()

    def _selected_job(self) -> Optional[dict]:
        rows = {i.row() for i in self.table.selectedIndexes()}
        if len(rows) != 1:
            return None
        row = rows.pop()
        item = self.table.item(row, 0)
        return item.data(Qt.UserRole) if item else None

    def _busy(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _set_status(self, text: str, error: bool = False):
        color = "#ef5350" if error else "#aaa"
        self.status_lbl.setStyleSheet(
            f"color: {color}; font-size: {scale_manager.scale_font(11)}px;")
        self.status_lbl.setText(text)

    def _local_dir_for(self, job: dict) -> Optional[str]:
        """Workspace dir for this job if it was downloaded here."""
        d = job_workspace_dir(job)
        if os.path.isfile(os.path.join(d, "job.json")):
            try:
                with open(os.path.join(d, "job.json"), encoding="utf-8") as f:
                    if json.load(f).get("job_id") == job.get("_id"):
                        return d
            except (OSError, json.JSONDecodeError):
                pass
        return None

    # ------------------------------------------------------------- refresh

    def showEvent(self, event):
        super().showEvent(event)
        self._poll_timer.start()
        if not self._jobs:
            self.refresh()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._poll_timer.stop()

    def refresh(self):
        if self._busy():
            return
        self._set_status("Loading jobs...")
        self._start_list_worker()

    def _refresh_silent(self):
        if not self._busy():
            self._start_list_worker()

    def _start_list_worker(self):
        self._worker = _ListWorker(self)
        self._worker.done.connect(self._on_jobs)
        self._worker.error.connect(
            lambda msg: self._set_status(f"Could not load jobs: {msg}", error=True))
        self._worker.start()

    def _on_jobs(self, jobs: list):
        self._jobs = jobs
        me = self.writer_name()
        self.table.setRowCount(0)
        for job in jobs:
            row = self.table.rowCount()
            self.table.insertRow(row)
            status = str(job.get("status", ""))
            claimed_by = str(job.get("claimed_by", "") or "")
            cells = [
                str(job.get("title", "")),
                str(job.get("chapters", "")),
                status,
                claimed_by,
            ]
            if status == "available":
                bg = COLOR_AVAILABLE
            elif status == "claimed" and me and claimed_by == me:
                bg = COLOR_CLAIMED_ME
            elif status == "claimed":
                bg = COLOR_CLAIMED_OTHER
            else:
                bg = COLOR_DONE
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setBackground(bg)
                item.setForeground(ROW_TEXT)
                if col == 0:
                    item.setData(Qt.UserRole, job)
                self.table.setItem(row, col, item)
        self._set_status(f"{len(jobs)} job(s) — updated just now")
        self._update_buttons()

    def _update_buttons(self):
        job = self._selected_job()
        me = self.writer_name()
        busy = self._busy()
        available = bool(job and job.get("status") == "available")
        mine = bool(job and job.get("claimed_by") == me and me)
        has_local = bool(job and self._local_dir_for(job))
        self.claim_btn.setEnabled(not busy and available)
        self.open_btn.setEnabled(not busy and has_local)
        self.deliver_btn.setEnabled(
            not busy and (
                (mine and job.get("status") in ("claimed", "delivered"))
                or (self._active_job is not None and job is None)
            )
        )

    # ------------------------------------------------------------- claim

    def _claim_selected(self):
        job = self._selected_job()
        if not job or self._busy():
            return
        if not self.writer_name():
            QMessageBox.warning(self, "Name required",
                                "Fill in 'Your name' before claiming a job.")
            self.writer_edit.setFocus()
            return
        self._save_writer_name()

        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self._set_status(f"Claiming '{job.get('title')}'...")

        self._worker = _ClaimWorker(job, self.writer_name(), self)
        self._worker.progress.connect(self._on_progress)
        self._worker.ready.connect(self._on_claim_ready)
        self._worker.error.connect(self._on_action_error)
        self._worker.start()
        self._update_buttons()

    def _on_progress(self, phase: str, done: int, total: int):
        if total <= 0:
            self.progress.setRange(0, 0)
            self.progress.setFormat(phase)
        else:
            self.progress.setRange(0, total)
            self.progress.setValue(done)
            if total > 1024 * 1024:  # bytes → show MB
                self.progress.setFormat(
                    f"{phase}  {done // (1024*1024)} / {total // (1024*1024)} MB")
            else:
                self.progress.setFormat(f"{phase}  {done}/{total}")
        self._set_status(phase)

    def _on_claim_ready(self, state: dict, job: dict, work_path: str):
        self.progress.setVisible(False)
        self._active_job = job
        self._active_work_path = work_path
        self._load_into_editor(state, work_path)
        self._set_status(
            f"Job '{job.get('title')}' loaded — fix it in the Script tab, "
            f"then come back and press Deliver.")
        self.refresh()

    def _on_action_error(self, msg: str):
        self.progress.setVisible(False)
        self._set_status(msg, error=True)
        QMessageBox.warning(self, "Jobs", msg)
        self.refresh()

    def _load_into_editor(self, state: dict, work_path: str):
        mw = self._main()
        try:
            mw._restore_state(state)
        except Exception:
            logger.exception("Failed to restore job state into editor")
            QMessageBox.warning(
                self, "Load problem",
                "The job downloaded fine but loading it into the Script tab "
                "hit an error — check the log. You can retry with Open.")
            return
        # Ctrl+S saves straight into the job workspace
        mw._current_project_path = work_path
        if hasattr(mw, "_update_title"):
            mw._update_title()
        if hasattr(mw, "tab_widget"):
            mw.tab_widget.setCurrentIndex(0)  # Script tab

    # ------------------------------------------------------------- open

    def _open_selected(self):
        job = self._selected_job()
        if not job or self._busy():
            return
        wdir = self._local_dir_for(job)
        if not wdir:
            QMessageBox.information(
                self, "Not downloaded",
                "This job hasn't been downloaded on this PC — use "
                "Claim & Download.")
            return
        path = os.path.join(wdir, "work.mscript")
        if not os.path.exists(path):
            path = os.path.join(wdir, "original.mscript")
        try:
            state = load_project(path)
        except Exception as e:
            QMessageBox.warning(self, "Open failed", f"Couldn't load {path}:\n{e}")
            return
        state = _rewrite_state_paths(state, os.path.join(wdir, "images"))
        self._active_job = job
        self._active_work_path = os.path.join(wdir, "work.mscript")
        self._load_into_editor(state, self._active_work_path)
        self._set_status(f"Reopened '{job.get('title')}' from {wdir}")

    # ------------------------------------------------------------- deliver

    def _deliver_selected(self):
        if self._busy():
            return
        job = self._selected_job() or self._active_job
        if not job:
            return
        me = self.writer_name()
        if not me:
            QMessageBox.warning(self, "Name required",
                                "Fill in 'Your name' before delivering.")
            return
        if job.get("claimed_by") and job.get("claimed_by") != me:
            QMessageBox.warning(
                self, "Not your job",
                f"This job is claimed by '{job.get('claimed_by')}'.")
            return

        reply = QMessageBox.question(
            self, "Deliver job",
            f"Upload your corrected script for:\n\n"
            f"  {job.get('title')}  ({job.get('chapters')})\n\n"
            f"The CURRENT state of the Script tab is what gets sent.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        mw = self._main()
        try:
            state = mw._gather_state()
        except Exception as e:
            logger.exception("gather_state failed")
            QMessageBox.warning(self, "Deliver failed",
                                f"Couldn't read the editor state:\n{e}")
            return

        wdir = self._local_dir_for(job)
        work_path = (self._active_work_path or
                     (os.path.join(wdir, "work.mscript") if wdir else ""))

        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self._set_status("Delivering...")
        self._worker = _DeliverWorker(job["_id"], me, state, work_path, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_delivered)
        self._worker.error.connect(self._on_action_error)
        self._worker.start()
        self._update_buttons()

    def _on_delivered(self, job: dict):
        self.progress.setVisible(False)
        self._set_status(f"Delivered '{job.get('title')}' — thank you!")
        QMessageBox.information(
            self, "Delivered",
            f"'{job.get('title')}' was uploaded successfully.\n"
            f"The admin will review it.")
        self.refresh()

#!/usr/bin/env python3
"""
Momo Script - Manhua Script Writing Tool
Synchronizes manhua images with text using YOLO panel detection.
"""

import os
import sys
import subprocess
import faulthandler
import traceback
import logging
from pathlib import Path

# Diagnostic: dump Python-style stack on C-level crashes to crash.log.
# Fall back to a PID-suffixed filename if the default is locked — this is
# what lets a second app instance launch even if the first one is still
# holding crash.log (or the log got left behind with a stale handle).
def _open_instance_log(base_path: Path):
    try:
        return open(base_path, "a", buffering=1), base_path
    except OSError:
        alt = base_path.with_name(f"{base_path.stem}_{os.getpid()}{base_path.suffix}")
        return open(alt, "a", buffering=1), alt


_crash_log_path = Path(__file__).parent / "crash.log"
_crash_log, _crash_log_path = _open_instance_log(_crash_log_path)
faulthandler.enable(file=_crash_log, all_threads=True)
faulthandler.enable(file=sys.stderr, all_threads=True)


def _log_uncaught(exc_type, exc_value, exc_tb):
    """Catch unhandled Python exceptions (including those inside Qt event
    handlers that PyQt5 normally swallows) and write them to crash.log."""
    import datetime
    header = f"\n===== UNCAUGHT EXCEPTION @ {datetime.datetime.now().isoformat()} =====\n"
    try:
        _crash_log.write(header)
        traceback.print_exception(exc_type, exc_value, exc_tb, file=_crash_log)
        _crash_log.flush()
    except Exception:
        pass
    try:
        sys.stderr.write(header)
        traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
    except Exception:
        pass


sys.excepthook = _log_uncaught


def _log_unraisable(unraisable):
    """Catch 'unraisable' exceptions (e.g. inside Qt event handlers or
    __del__ methods) which Python normally only prints to stderr."""
    import datetime
    header = f"\n===== UNRAISABLE EXCEPTION @ {datetime.datetime.now().isoformat()} =====\n"
    try:
        _crash_log.write(header)
        if unraisable.object is not None:
            _crash_log.write(f"  object: {unraisable.object!r}\n")
        traceback.print_exception(
            unraisable.exc_type, unraisable.exc_value, unraisable.exc_traceback,
            file=_crash_log,
        )
        _crash_log.flush()
    except Exception:
        pass


sys.unraisablehook = _log_unraisable

sys.path.insert(0, str(Path(__file__).parent))

# Pre-import torch BEFORE PyQt5 on Windows. PyQt5 poisons the DLL search path
# in a way that makes a later `import torch` fail with WinError 1114 (failed
# to load c10.dll). The YOLO detector and the Real-ESRGAN upscaler both need
# torch later, so we take the ~2s hit here to keep the real work from failing
# mid-operation. Swallow ImportError so the app still launches if torch is
# missing (features that need it will just stay disabled).
try:
    import torch  # noqa: F401
except ImportError:
    pass

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt

from app.config import Config
from app.bootstrap import ensure_models
from app.updater import check_for_updates, pull_fast_forward
from ui.scale_manager import scale_manager
from ui.main_window import MainWindow


def setup_logging():
    # Try the default log path first; fall back to a PID-suffixed one when
    # another instance has it open. Without the fallback a second instance
    # crashes during FileHandler construction on some Windows setups.
    log_path = Path(__file__).parent / "momo_script.log"
    try:
        file_handler = logging.FileHandler(log_path)
    except OSError:
        log_path = log_path.with_name(f"momo_script_{os.getpid()}.log")
        file_handler = logging.FileHandler(log_path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            file_handler,
            logging.StreamHandler(sys.stdout),
        ],
    )


def build_stylesheet() -> str:
    s = scale_manager
    return f"""
    QMainWindow, QWidget {{
        background-color: #2b2b2b;
        color: #ffffff;
        font-size: {s.scale_font(12)}px;
    }}
    QGroupBox {{
        border: {s.scale(2)}px solid #555555;
        border-radius: {s.scale(5)}px;
        margin-top: {s.scale(2)}px;
        margin-bottom: {s.scale(2)}px;
        padding-top: {s.scale(5)}px;
        font-weight: bold;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: {s.scale(10)}px;
        padding: 0 {s.scale(5)}px;
    }}
    QPushButton {{
        background-color: #404040;
        border: 1px solid #555555;
        border-radius: {s.scale(3)}px;
        padding: {s.scale(5)}px;
        min-height: {s.scale(20)}px;
        font-size: {s.scale_font(12)}px;
    }}
    QPushButton:hover {{ background-color: #505050; }}
    QPushButton:pressed {{ background-color: #353535; }}
    QPushButton:disabled {{ background-color: #2b2b2b; color: #666666; }}
    QLabel {{
        color: #ffffff;
        font-size: {s.scale_font(12)}px;
    }}
    QLineEdit {{
        background-color: #404040;
        border: 1px solid #555555;
        border-radius: {s.scale(3)}px;
        padding: {s.scale(3)}px;
        font-size: {s.scale_font(12)}px;
    }}
    QSpinBox, QDoubleSpinBox {{
        background-color: #404040;
        border: 1px solid #555555;
        border-radius: {s.scale(3)}px;
        padding: {s.scale(3)}px;
        font-size: {s.scale_font(12)}px;
    }}
    QComboBox {{
        font-size: {s.scale_font(12)}px;
        padding: {s.scale(3)}px;
    }}
    QPlainTextEdit, QTextEdit {{
        font-size: {s.scale_font(12)}px;
    }}
    QProgressBar {{
        border: {s.scale(2)}px solid #555555;
        border-radius: {s.scale(5)}px;
        text-align: center;
    }}
    QProgressBar::chunk {{ background-color: #0078d4; border-radius: {s.scale(3)}px; }}
    QStatusBar {{
        background-color: #404040;
        border-top: 1px solid #555555;
        font-size: {s.scale_font(11)}px;
    }}
    QScrollArea {{ background-color: #2b2b2b; }}
    QSplitter::handle {{ background-color: #555; width: {s.scale(3)}px; }}
    QMenuBar {{
        font-size: {s.scale_font(12)}px;
    }}
    QMenu {{
        font-size: {s.scale_font(12)}px;
    }}
    QTabBar::tab {{
        font-size: {s.scale_font(12)}px;
        padding: {s.scale(6)}px {s.scale(14)}px;
    }}
    """


def _restart_app(logger):
    """Relaunch this app so the freshly-pulled files are loaded from disk.

    The current Python process already has the old modules imported in
    memory, so a `git pull` on startup has no effect on the UI that opens
    afterwards — we need a fresh interpreter. subprocess.Popen + sys.exit
    is more reliable on Windows than os.execv, which misbehaves when argv
    paths contain spaces (and ours does: "Área de Trabalho").
    """
    try:
        logger.info("Restarting app to pick up updated files.")
        subprocess.Popen([sys.executable] + sys.argv, close_fds=False)
    except Exception:
        logger.exception("Failed to spawn restart process")
    # Exit immediately so Qt doesn't continue on to MainWindow with the
    # stale in-memory modules. os._exit avoids running atexit hooks that
    # might re-enter Qt teardown after we've started the child.
    os._exit(0)


def _run_update_check(project_dir: str, logger):
    """Fetch from origin and apply updates silently when safe. Never raises.

    - Clean workdir + fast-forward available → pull and restart.
    - Pull fails (divergent history, uncommitted changes, network blip) →
      show a single informational popup telling the user to resolve via
      Claude Code and open the app normally with the local version.
    """
    try:
        result = check_for_updates(project_dir)
    except Exception:
        logger.exception("Update check crashed")
        return

    status = result.get("status")
    logger.info("Update check: %s", status)

    if status in ("not_a_repo", "no_remote", "up_to_date", "local_ahead"):
        return
    if status in ("fetch_failed", "error"):
        logger.warning("Update check skipped: %s", result.get("detail", ""))
        return
    if status != "update_available":
        return

    branch = result["branch"]
    ok, detail = pull_fast_forward(project_dir, branch)
    if ok:
        logger.info("Auto-pulled updates on branch %s; restarting.", branch)
        _restart_app(logger)
        return  # unreachable

    # Pull couldn't apply cleanly (uncommitted edits, diverged history,
    # etc.). Tell the user once and move on — do NOT try to merge or
    # reset anything automatically, that's how uncommitted work gets lost.
    logger.warning("Auto-pull failed on %s (%s); opening with local version.",
                   branch, detail)
    QMessageBox.information(
        None, "Atualização disponível",
        "Foi feito um git pull, mas ele não pôde ser aplicado "
        "automaticamente — provavelmente há um conflito com arquivos "
        "locais seus.\n\n"
        "Abra o Claude Code (ou seu editor preferido) na pasta do "
        "projeto para resolver o merge quando for conveniente.\n\n"
        "O app vai abrir normalmente com a sua versão local.",
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Momo Script")

    # Enable Qt high-DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    app.setApplicationName("Momo Script")
    app.setStyle("Fusion")

    # Check for updates from the GitHub remote before opening the main
    # window. Failures (no network, no remote, not a git repo) are logged
    # and ignored so the app still starts.
    project_dir = str(Path(__file__).parent)
    _run_update_check(project_dir, logger)

    # Ensure large model weights are present. `panel.pt` (~116 MB) is hosted
    # as a GitHub Release asset instead of stored in git, so on first run
    # (or after a ZIP download) the bootstrap downloads it from the release.
    try:
        ensure_models(project_dir)
    except Exception:
        logger.exception("Model bootstrap crashed")

    # Auto-detect scale from monitor DPI, or use saved value
    config = Config()
    saved_scale = config.get("ui_scale_percent")
    if saved_scale is not None:
        scale_manager.scale_factor = saved_scale / 100.0
    else:
        screen = app.primaryScreen()
        if screen:
            dpi = screen.logicalDotsPerInch()
            # 96 DPI = 100% (1.0x), 120 = 125% (1.25x), 144 = 150% (1.5x), 192 = 200% (2.0x)
            sf = round(dpi / 96.0 * 4) / 4  # round to nearest 0.25
            sf = max(1.0, min(4.0, sf))
            scale_manager.scale_factor = sf
            logger.info(f"Auto scale: {screen.size().width()}x{screen.size().height()} @ {dpi:.0f} DPI → {sf}x")

    app.setStyleSheet(build_stylesheet())

    window = MainWindow()
    window.show()

    # Rebuild stylesheet when scale changes
    def on_scale_changed(new_factor):
        app.setStyleSheet(build_stylesheet())

    scale_manager.scale_changed.connect(on_scale_changed)

    logger.info("Application started")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

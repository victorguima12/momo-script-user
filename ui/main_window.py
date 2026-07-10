"""
Momo Script - Main Window with tabbed interface.
"""

import json
import logging
import os

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QKeySequence

from app.config import Config
from app.edition import IS_USER
from app.project import save_project, load_project, autosave_path
from ui.download_panel import DownloadPanel
from ui.jobs_panel import JobsPanel
from ui.scale_manager import scale_manager
from ui.script_panel import ScriptPanel
from ui.stitch_panel import StitchPanel
if IS_USER:
    # Admin-only tabs don't ship in the user edition.
    TranslatePanel = VOPanel = UpscalePanel = FcpxmlPanel = RenderPanel = None
else:
    from ui.translate_panel import TranslatePanel
    from ui.vo_panel import VOPanel
    from ui.upscale_panel import UpscalePanel
    from ui.fcpxml_panel import FcpxmlPanel
    from ui.render_panel import RenderPanel
    from ui.jobs_admin_panel import JobsAdminPanel

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self._current_project_path: str = ""
        self._dirty = False

        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        self._setup_ui()
        self._setup_menu()
        self._connect_signals()
        self._setup_autosave()
        self._update_title()

        # Check for autosave recovery
        QTimer.singleShot(500, self._check_autosave_recovery)

    def _setup_ui(self):
        s = scale_manager

        # Central widget with tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid #555;
                background-color: #2b2b2b;
                top: 0px;
            }}
            QTabBar {{
                background-color: #353535;
            }}
            QTabBar::tab {{
                background-color: #353535;
                color: #ccc;
                padding: {s.scale(6)}px {s.scale(14)}px;
                border: 1px solid #555;
                border-bottom: none;
                border-top-left-radius: {s.scale(3)}px;
                border-top-right-radius: {s.scale(3)}px;
                margin-right: 1px;
                font-size: {s.scale_font(12)}px;
            }}
            QTabBar::tab:selected {{
                background-color: #2b2b2b;
                color: #ffffff;
                font-weight: bold;
            }}
            QTabBar::tab:hover {{
                background-color: #404040;
            }}
        """)
        self.setCentralWidget(self.tab_widget)

        self._build_tabs()

        # Status bar
        self.statusBar().showMessage("Ready")
        self.statusBar().setStyleSheet("QStatusBar { background-color: #353535; color: #aaa; }")

    def _build_tabs(self):
        """Create and add all panels to the tab widget. The user edition
        gets Script / Download / Cut Stitch only; the admin-only panels stay
        None so the guarded references below fall through cleanly."""
        # Tab order: Script > Download > Cut Stitch > (admin tabs)
        self.script_panel = ScriptPanel(config=self.config)
        self.tab_widget.addTab(self.script_panel, "Script")

        self.download_panel = DownloadPanel()
        self.tab_widget.addTab(self.download_panel, "Download")

        self.stitch_panel = StitchPanel()
        self.tab_widget.addTab(self.stitch_panel, "Cut Stitch")

        # Writer-facing job board — present in BOTH editions (the admin
        # uses it to test what writers see; publish/review lives in the
        # separate admin-only Jobs Adm tab).
        self.jobs_panel = JobsPanel(main_window=self)
        self.tab_widget.addTab(self.jobs_panel, "Jobs")

        if IS_USER:
            self.translate_panel = None
            self.vo_panel = None
            self.upscale_panel = None
            self.fcpxml_panel = None
            self.render_panel = None
            self.jobs_admin_panel = None
            return

        self.translate_panel = TranslatePanel(config=self.config)
        self.translate_panel.set_script_panel(self.script_panel)
        self.script_panel.set_translate_panel(self.translate_panel)
        self.tab_widget.addTab(self.translate_panel, "Translate")

        self.vo_panel = VOPanel(config=self.config)
        self.vo_panel.set_script_panel(self.script_panel)
        self.vo_panel.set_translate_panel(self.translate_panel)
        self.tab_widget.addTab(self.vo_panel, "VO")

        self.upscale_panel = UpscalePanel(config=self.config)
        self.tab_widget.addTab(self.upscale_panel, "Upscale")

        self.fcpxml_panel = FcpxmlPanel(config=self.config)
        self.fcpxml_panel.set_vo_panel(self.vo_panel)
        self.fcpxml_panel.set_script_panel(self.script_panel)
        self.tab_widget.addTab(self.fcpxml_panel, "FCPXML")

        self.render_panel = RenderPanel(config=self.config)
        self.tab_widget.addTab(self.render_panel, "Render")

        self.jobs_admin_panel = JobsAdminPanel(main_window=self)
        self.tab_widget.addTab(self.jobs_admin_panel, "Jobs Adm")

    def _setup_menu(self):
        s = scale_manager
        menu_bar = self.menuBar()
        menu_bar.setStyleSheet(f"""
            QMenuBar {{
                background-color: #353535;
                color: #ccc;
                font-size: {s.scale_font(12)}px;
            }}
            QMenuBar::item:selected {{
                background-color: #404040;
            }}
            QMenu {{
                background-color: #353535;
                color: #ccc;
                border: 1px solid #555;
                font-size: {s.scale_font(12)}px;
            }}
            QMenu::item:selected {{
                background-color: #00bcd4;
                color: white;
            }}
        """)

        # -- File menu --
        file_menu = menu_bar.addMenu("&File")

        new_action = file_menu.addAction("&New Project")
        new_action.setShortcut(QKeySequence("Ctrl+N"))
        new_action.triggered.connect(self._new_project)

        open_action = file_menu.addAction("&Open Project...")
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self._open_project)

        file_menu.addSeparator()

        save_action = file_menu.addAction("&Save")
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self._save_project)

        save_as_action = file_menu.addAction("Save &As...")
        save_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_as_action.triggered.connect(self._save_project_as)

        file_menu.addSeparator()

        mix_batch_action = file_menu.addAction("Mi&x Batch...")
        mix_batch_action.triggered.connect(self._mix_batch)

        file_menu.addSeparator()

        reload_action = file_menu.addAction("&Reload App")
        reload_action.setShortcut(QKeySequence("Ctrl+R"))
        reload_action.triggered.connect(self._reload_app)

        # -- UI menu --
        ui_menu = menu_bar.addMenu("&UI")

        size_menu = ui_menu.addMenu("&Size")
        self._size_action_group = QActionGroup(self)
        self._size_action_group.setExclusive(True)

        current_pct = round(scale_manager.scale_factor * 100)
        for pct in range(10, 210, 10):
            action = size_menu.addAction(f"{pct}%")
            action.setCheckable(True)
            if pct == current_pct:
                action.setChecked(True)
            action.triggered.connect(lambda checked, p=pct: self._set_ui_scale(p))
            self._size_action_group.addAction(action)

    def _set_ui_scale(self, percent: int):
        scale_manager.scale_factor = percent / 100.0
        self.config.set("ui_scale_percent", percent)
        # Rebuild tab stylesheet since it's set on the widget directly
        self._setup_ui_tab_style()

    def _setup_ui_tab_style(self):
        s = scale_manager
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid #555;
                background-color: #2b2b2b;
                top: 0px;
            }}
            QTabBar {{
                background-color: #353535;
            }}
            QTabBar::tab {{
                background-color: #353535;
                color: #ccc;
                padding: {s.scale(6)}px {s.scale(14)}px;
                border: 1px solid #555;
                border-bottom: none;
                border-top-left-radius: {s.scale(3)}px;
                border-top-right-radius: {s.scale(3)}px;
                margin-right: 1px;
                font-size: {s.scale_font(12)}px;
            }}
            QTabBar::tab:selected {{
                background-color: #2b2b2b;
                color: #ffffff;
                font-weight: bold;
            }}
            QTabBar::tab:hover {{
                background-color: #404040;
            }}
        """)

    def _connect_signals(self):
        self.script_panel.state_changed.connect(self._mark_dirty)

    def _setup_autosave(self):
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(120_000)  # 2 minutes
        self._autosave_timer.timeout.connect(self._do_autosave)
        self._autosave_timer.start()

    # -- Title ---

    def _update_title(self):
        name = os.path.basename(self._current_project_path) if self._current_project_path else "Untitled"
        dirty_marker = " *" if self._dirty else ""
        self.setWindowTitle(f"{name}{dirty_marker} - Momo Script")

    def _mark_dirty(self):
        if not self._dirty:
            self._dirty = True
            self._update_title()

    def _clear_dirty(self):
        self._dirty = False
        self._update_title()

    # -- Reload ---

    def _reload_app(self):
        """Hot-reload all Python modules and rebuild the UI, preserving project state."""
        import importlib
        import sys

        # Save current state
        state = None
        project_path = self._current_project_path
        try:
            state = self._gather_state()
        except Exception as e:
            logger.warning(f"Could not gather state before reload: {e}")

        # Stop active services
        bridge = getattr(self.script_panel, 'claude_bridge', None)
        if bridge and bridge.active:
            bridge.stop()
        if hasattr(self.script_panel, '_gemini_writer') and self.script_panel._gemini_writer:
            self.script_panel._gemini_writer.stop()

        # Reload all app and ui modules
        modules_to_reload = sorted(
            [name for name in sys.modules if name.startswith(("app.", "ui.", "yolo."))],
            key=lambda x: x.count("."),
        )
        for name in modules_to_reload:
            try:
                importlib.reload(sys.modules[name])
            except Exception as e:
                logger.warning(f"Failed to reload {name}: {e}")

        # Re-import after reload so _build_tabs picks up the fresh classes
        global ScriptPanel, DownloadPanel, StitchPanel, JobsPanel
        global TranslatePanel, VOPanel, UpscalePanel, FcpxmlPanel, RenderPanel
        global JobsAdminPanel
        from ui.script_panel import ScriptPanel
        from ui.download_panel import DownloadPanel
        from ui.stitch_panel import StitchPanel
        from ui.jobs_panel import JobsPanel
        if not IS_USER:
            from ui.translate_panel import TranslatePanel
            from ui.vo_panel import VOPanel
            from ui.upscale_panel import UpscalePanel
            from ui.fcpxml_panel import FcpxmlPanel
            from ui.render_panel import RenderPanel
            from ui.jobs_admin_panel import JobsAdminPanel

        # Tear down old UI
        old_tab = self.tab_widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.setCentralWidget(self.tab_widget)
        self._setup_ui_tab_style()

        self._build_tabs()

        # Reconnect signals
        self.script_panel.state_changed.connect(self._mark_dirty)

        # Restore state
        if state:
            try:
                self._restore_state(state)
            except Exception as e:
                logger.warning(f"Could not restore state after reload: {e}")

        self._current_project_path = project_path
        self._update_title()
        self.statusBar().showMessage("App reloaded")

        # Clean up old widget
        old_tab.deleteLater()

    # -- Project actions ---

    def _gather_state(self) -> dict:
        state = self.script_panel.get_state()
        if self.translate_panel:
            state["translations"] = self.translate_panel.get_translations()
        elif getattr(self, "_passthrough_translations", None):
            # User edition has no Translate tab, but a project saved by the
            # admin build may carry translations — keep them intact so a
            # writer's save doesn't wipe the admin's data.
            state["translations"] = self._passthrough_translations
        return state

    def _restore_state(self, state: dict):
        self.script_panel.set_state(state)
        if not self.translate_panel:
            self._passthrough_translations = state.get("translations") or {}
            return
        self.translate_panel.set_translations(state.get("translations", {}))
        # Apply the current Script-tab view language after restore so boxes
        # get populated from the merged translations dict. Without this step
        # any chapter whose box.text was saved as pt-br but whose English
        # lives in translations["english"] would display stale pt on load,
        # because the combo's setCurrentIndex doesn't fire changed when the
        # target value already matches the default.
        try:
            current_lang = self.script_panel.ai_lang_combo.currentText()
        except AttributeError:
            current_lang = ""
        if current_lang:
            self.translate_panel.switch_view_language(current_lang)

    def _new_project(self):
        if self._dirty:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Do you want to save before creating a new project?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Save:
                self._save_project()
                if self._dirty:  # save was cancelled
                    return
            elif reply == QMessageBox.Cancel:
                return

        self.script_panel.image_strip.clear_all()
        # Zero out cost counters so the new project doesn't inherit the
        # prior project's spending in the toolbar label.
        self.script_panel.reset_project_costs()
        self._current_project_path = ""
        self._clear_dirty()
        self.statusBar().showMessage("New project created")

    def _open_project(self):
        if self._dirty:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Do you want to save before opening another project?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Save:
                self._save_project()
                if self._dirty:
                    return
            elif reply == QMessageBox.Cancel:
                return

        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "Momo Script Files (*.mscript);;All Files (*)"
        )
        if not path:
            return

        self._load_project_file(path)

    def _load_project_file(self, path: str):
        try:
            state = load_project(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load project:\n{e}")
            return

        self._restore_state(state)
        self._current_project_path = path
        if self.vo_panel:
            self.vo_panel.set_project_path(path)
            # If there's a sidecar WAV+JSON next to this .mscript, restore the
            # VO audio + segments so the user doesn't need to re-run TTS.
            try:
                self.vo_panel.load_vo_sidecar(path)
            except Exception:
                logger.exception("Failed to load VO sidecar")
        self._clear_dirty()
        self.statusBar().showMessage(f"Opened {os.path.basename(path)}")

    def _mix_batch(self):
        """Select a parent folder containing chapter subfolders, merge all into one project."""
        if self._dirty:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Do you want to save before mixing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Save:
                self._save_project()
                if self._dirty:
                    return
            elif reply == QMessageBox.Cancel:
                return

        folder = QFileDialog.getExistingDirectory(self, "Select folder with chapter subfolders")
        if not folder:
            return

        result = self.script_panel.mix_batch(folder)
        if not result:
            return

        # Apply merged translations from all chapters so the mixed project
        # has a unified {lang: {box_id: text}} map. Without this, TTS in
        # vo_panel falls back to per-box `text` which can be a mix of
        # languages across chapters.
        merged_translations = result.get("translations") or {}
        if merged_translations and self.translate_panel:
            self.translate_panel.set_translations(merged_translations)
            # Push the current view language onto the boxes so panels whose
            # chapters had English in translations but pt-br in box.text
            # display the English immediately instead of needing a manual
            # combo flip or Apply click.
            try:
                current_lang = self.script_panel.ai_lang_combo.currentText()
            except AttributeError:
                current_lang = ""
            if current_lang:
                self.translate_panel.switch_view_language(current_lang)

        # Auto-save the mixed project as a standalone .mscript at the
        # parent folder so it can be reopened without re-running Mix Batch.
        mixed_path = os.path.join(folder, "mixed.mscript")
        self._current_project_path = mixed_path
        self._mark_dirty()
        self._do_save(mixed_path)

        # Check if subfolders have voiceover data and offer to merge
        import re
        def _natural_sort_key(s):
            return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

        subdirs = sorted(
            (os.path.join(folder, d) for d in os.listdir(folder)
             if os.path.isdir(os.path.join(folder, d))),
            key=lambda p: _natural_sort_key(os.path.basename(p)),
        )

        has_vo = not IS_USER and any(
            os.path.isfile(os.path.join(d, 'voiceover.wav'))
            and os.path.isfile(os.path.join(d, 'voiceover.fcpxml'))
            for d in subdirs
        )

        if has_vo:
            reply = QMessageBox.question(
                self, "Export Combined FCPXML",
                "Voiceover data found in the chapter folders.\n\n"
                "Merge all WAVs + panels into one FCPXML?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._export_mixed_fcpxml(folder, subdirs)
        else:
            self.statusBar().showMessage("Mix Batch loaded")

    def _export_mixed_fcpxml(self, parent_folder: str, chapter_dirs: list):
        """Merge voiceover data from chapter subdirs and export combined FCPXML."""
        from app.voice_over import merge_voiceovers

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Combined FCPXML", os.path.join(parent_folder, "mixed_voiceover"),
            "FCPXML Files (*.fcpxml);;All Files (*)"
        )
        if not path:
            return

        output_dir = os.path.dirname(path)
        project_name = os.path.splitext(os.path.basename(path))[0]

        try:
            result = merge_voiceovers(chapter_dirs, output_dir, project_name)
            if result:
                QMessageBox.information(
                    self, "Export Complete",
                    f"Combined FCPXML exported:\n\n"
                    f"  {result}\n"
                    f"  {os.path.join(output_dir, project_name + '.wav')}\n"
                    f"  {os.path.join(output_dir, project_name + '_panels')}/\n\n"
                    f"Import the .fcpxml into DaVinci Resolve."
                )
                self.statusBar().showMessage(f"Mix Batch exported: {os.path.basename(result)}")
            else:
                QMessageBox.warning(self, "No Voiceover Data", "No voiceover.wav + voiceover.fcpxml pairs found.")
        except Exception as e:
            logger.exception("Mix batch export failed")
            QMessageBox.critical(self, "Export Error", f"Failed to merge voiceovers:\n{e}")

    def _save_project(self):
        if not self._current_project_path:
            self._save_project_as()
            return

        self._do_save(self._current_project_path)

    def _save_project_as(self):
        default_name = self._current_project_path or "project.mscript"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", default_name,
            "Momo Script Files (*.mscript);;All Files (*)"
        )
        if not path:
            return

        self._current_project_path = path
        self._do_save(path)

    def _do_save(self, path: str):
        try:
            state = self._gather_state()
            save_project(path, state)
            self._clear_dirty()
            self.statusBar().showMessage(f"Saved {os.path.basename(path)}")
            # Tell the VO panel where we live so it can drop its sidecar in
            # the right place on the next generation.
            if self.vo_panel:
                self.vo_panel.set_project_path(path)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save project:\n{e}")

    # -- Autosave ---

    def _do_autosave(self):
        if not self._dirty:
            return
        try:
            state = self._gather_state()
            # Remember which real project file this autosave belongs to, so
            # recovery can restore the save path and Ctrl+S writes to the
            # original file instead of silently being Untitled.
            state["_origin_project_path"] = self._current_project_path or ""
            save_project(autosave_path(), state)
            logger.info("Autosaved project")
        except Exception:
            logger.exception("Autosave failed")

    def _check_autosave_recovery(self):
        path = autosave_path()
        if not os.path.exists(path):
            return

        # Peek at the origin path so we can show it in the prompt and restore
        # it after recovery.
        origin_path = ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                origin_path = (json.load(f).get("_origin_project_path") or "").strip()
        except Exception:
            logger.exception("Could not read autosave origin path")

        if origin_path:
            msg = (
                "An autosave file was found from a previous session.\n\n"
                f"Original project: {origin_path}\n\n"
                "Recover it? (Ctrl+S will save back to the original file.)"
            )
        else:
            msg = (
                "An autosave file was found from a previous session.\n\n"
                "Would you like to recover it?"
            )

        reply = QMessageBox.question(
            self, "Recover Autosave", msg,
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._load_project_file(path)
            # Restore the real project path if we know it and it still
            # exists; otherwise leave Untitled so Save As is forced.
            if origin_path and os.path.exists(origin_path):
                self._current_project_path = origin_path
                if self.vo_panel:
                    self.vo_panel.set_project_path(origin_path)
                self._update_title()
            else:
                self._current_project_path = ""
            self._mark_dirty()
        # Remove autosave regardless
        try:
            os.unlink(path)
        except OSError:
            pass

    # -- Close ---

    def closeEvent(self, event):
        if self._dirty:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Do you want to save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Save:
                self._save_project()
                if self._dirty:  # save was cancelled
                    event.ignore()
                    return
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return

        # Clean up autosave on clean exit
        path = autosave_path()
        if os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass

        self.download_panel.closeEvent(event)
        super().closeEvent(event)

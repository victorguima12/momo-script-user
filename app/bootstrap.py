"""Startup bootstrapper: ensures the large model weights are present.

`panel.pt` is the YOLO panel detector (~116 MB). It is hosted on a
public Dropbox share (not in git), so the bootstrap can download it
without any authentication token on first run.

Auto-update logic: at every start we do a HEAD request and compare the
remote Content-Length against the local file size. If they differ, the
remote asset was replaced (retrained model, etc.) and we re-download it.
Same size → assume up to date and skip.

Smaller models (<100 MB) live in git directly as plain binaries and are
not handled here.
"""

import logging
import os
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from PyQt5.QtWidgets import QProgressDialog, QMessageBox
from PyQt5.QtCore import Qt, QCoreApplication

logger = logging.getLogger(__name__)


MODELS = [
    {
        "name": "panel.pt",
        # Dropbox share link with dl=1 to force direct download. If this
        # ever needs to be replaced, keep the query string intact — rlkey
        # / st / dl are all required by Dropbox.
        "url": (
            "https://www.dropbox.com/scl/fi/j2bej0nyryf388rkgoome/panel.pt"
            "?rlkey=p0q2o98yuixtglpn5vg3kcaiy&st=zjzmtp0k&dl=1"
        ),
        "description": "YOLO panel detector",
        "approx_size_mb": 116,
        # Anything smaller than this is treated as "missing" — covers both
        # the file being absent and the occasional leftover LFS pointer
        # (~134 bytes) from older clones.
        "min_size": 1_000_000,
    },
]


def _local_size(path: str) -> int:
    try:
        return os.path.getsize(path) if os.path.exists(path) else 0
    except OSError:
        return 0


def _remote_size(url: str) -> int:
    """HEAD the URL and return the final Content-Length after redirects.

    Returns 0 when the server doesn't advertise a length or the request
    fails; callers should treat 0 as 'unknown' and skip the size check.
    """
    try:
        req = Request(url, method="HEAD", headers={
            "User-Agent": "momo-script-bootstrap",
        })
        with urlopen(req, timeout=15) as r:
            size = int(r.headers.get("Content-Length", 0))
            return size
    except Exception:
        logger.exception("HEAD failed for %s", url)
        return 0


def _needs_download(path: str, min_size: int, remote_size: int) -> bool:
    """Download rule: file is missing, too small (LFS pointer), or the
    remote has a different size (model updated upstream)."""
    local = _local_size(path)
    if local < min_size:
        return True
    if remote_size > 0 and local != remote_size:
        return True
    return False


def _download_with_progress(url: str, dest: str, label: str, parent=None) -> tuple:
    """Download `url` to `dest`, streaming to disk with a Qt progress
    dialog. Returns (ok, detail)."""
    tmp = dest + ".part"
    dialog = None
    try:
        req = Request(url, headers={"User-Agent": "momo-script-bootstrap"})
        with urlopen(req, timeout=30) as response:
            total = int(response.headers.get("Content-Length", 0))
            dialog = QProgressDialog(label, "Cancelar", 0, max(1, total), parent)
            dialog.setWindowTitle("Baixando modelo")
            dialog.setWindowModality(Qt.WindowModal)
            dialog.setMinimumDuration(0)
            dialog.setAutoClose(False)
            dialog.setAutoReset(False)

            downloaded = 0
            chunk_size = 128 * 1024
            with open(tmp, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        dialog.setValue(min(downloaded, total))
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total / (1024 * 1024)
                        dialog.setLabelText(
                            f"{label}\n{mb_done:.1f} / {mb_total:.1f} MB"
                        )
                    else:
                        mb_done = downloaded / (1024 * 1024)
                        dialog.setLabelText(f"{label}\n{mb_done:.1f} MB")
                    QCoreApplication.processEvents()
                    if dialog.wasCanceled():
                        logger.warning("User cancelled download of %s", url)
                        return False, "cancelled"

            dialog.setValue(total if total > 0 else downloaded)

        if os.path.exists(dest):
            os.remove(dest)
        os.rename(tmp, dest)
        return True, ""

    except HTTPError as e:
        logger.exception("HTTP error downloading %s", url)
        return False, f"HTTP {e.code}: {e.reason}"
    except URLError as e:
        logger.exception("Network error downloading %s", url)
        return False, f"erro de rede: {e}"
    except Exception as e:
        logger.exception("Unexpected error downloading %s", url)
        return False, str(e)
    finally:
        if dialog is not None:
            dialog.close()
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def ensure_models(project_dir: str, parent=None) -> list:
    """Check each model and download any that are missing or outdated.

    Returns the list of model names that failed to download (empty = OK)."""
    failures = []

    for m in MODELS:
        path = os.path.join(project_dir, m["name"])
        remote = _remote_size(m["url"])
        if not _needs_download(path, m["min_size"], remote):
            logger.info("Bootstrap: %s up to date (%d bytes)", m["name"], _local_size(path))
            continue

        local = _local_size(path)
        reason = "missing"
        if local >= m["min_size"]:
            reason = f"remote size changed ({local} -> {remote})"

        label = (
            f"Baixando {m['name']} ({m['description']}, "
            f"~{m['approx_size_mb']} MB).\n"
            f"Motivo: {reason}"
        )
        logger.info("Bootstrap: downloading %s (reason=%s)", m["name"], reason)
        ok, detail = _download_with_progress(m["url"], path, label, parent=parent)
        if not ok:
            logger.error("Bootstrap: failed to fetch %s (%s)", m["name"], detail)
            failures.append((m["name"], detail))

    if failures:
        names = ", ".join(n for n, _ in failures)
        details = "\n".join(f"  {n}: {d}" for n, d in failures)
        QMessageBox.warning(
            parent,
            "Modelos não baixados",
            "Não foi possível baixar os seguintes modelos:\n"
            f"{details}\n\n"
            f"Funcionalidades que dependem deles ({names}) podem ficar "
            "indisponíveis até a próxima tentativa. Verifique sua conexão "
            "e reinicie o programa.",
        )

    return [n for n, _ in failures]

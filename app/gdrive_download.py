"""Anonymous download of a public ("anyone with the link") Google Drive file.

No Google account, no API key, no SDK — plain requests. Large files
(>~100MB) hit Google's "can't scan for viruses" interstitial; we handle
both the modern drive.usercontent.google.com form and the legacy
confirm-token cookie flow.
"""

import logging
import os
import re
from html import unescape
from typing import Callable, Optional

import requests

logger = logging.getLogger(__name__)

TIMEOUT = 60
CHUNK = 1024 * 1024  # 1MB read chunks

# progress_cb(bytes_done, total_bytes_or_minus1)
ProgressCb = Optional[Callable[[int, int], None]]
# cancel_cb() -> True to abort
CancelCb = Optional[Callable[[], bool]]


class GDriveError(Exception):
    """Download failure with a user-presentable message."""


class GDriveCancelled(GDriveError):
    """The user cancelled the download."""


def extract_file_id(link_or_id: str) -> str:
    """Accepts a raw file id or any common Drive URL form and returns the id.

    Handles:
      https://drive.google.com/file/d/<id>/view?usp=sharing
      https://drive.google.com/open?id=<id>
      https://drive.google.com/uc?export=download&id=<id>
      https://drive.usercontent.google.com/download?id=<id>&...
      <id>
    """
    s = (link_or_id or "").strip()
    if not s:
        raise GDriveError("Empty Google Drive link/id.")
    if "/" not in s and "?" not in s and "=" not in s:
        return s
    m = re.search(r"/file/d/([\w-]+)", s)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([\w-]+)", s)
    if m:
        return m.group(1)
    raise GDriveError(f"Could not find a Drive file id in: {s[:120]}")


def _parse_interstitial_form(html: str) -> Optional[dict]:
    """Extract action URL + hidden inputs from the virus-scan warning page."""
    m = re.search(r'<form[^>]+action="([^"]+)"[^>]*>(.*?)</form>', html, re.S)
    if not m:
        return None
    action, body = unescape(m.group(1)), m.group(2)
    params = {
        name: unescape(value)
        for name, value in re.findall(
            r'<input[^>]+name="([^"]+)"[^>]*value="([^"]*)"', body)
    }
    return {"action": action, "params": params}


def download_public_file(link_or_id: str, dest_path: str,
                         progress_cb: ProgressCb = None,
                         cancel_cb: CancelCb = None) -> str:
    """Download a public Drive file to dest_path. Returns dest_path.

    Raises GDriveError on failure, GDriveCancelled if cancel_cb fired.
    """
    file_id = extract_file_id(link_or_id)
    session = requests.Session()

    try:
        resp = session.get(
            "https://drive.usercontent.google.com/download",
            params={"id": file_id, "export": "download", "confirm": "t"},
            stream=True, timeout=TIMEOUT,
        )
    except requests.RequestException as e:
        raise GDriveError(f"Network error contacting Google Drive: {e}") from e

    resp = _resolve_interstitial(session, resp, file_id)

    if resp.status_code == 404:
        raise GDriveError(
            "Google Drive says the file does not exist (404). "
            "Check the job's images link."
        )
    if resp.status_code in (401, 403):
        raise GDriveError(
            "Google Drive refused the download (not shared as "
            "'anyone with the link'?)."
        )
    if resp.status_code != 200:
        raise GDriveError(f"Google Drive returned HTTP {resp.status_code}.")

    content_type = (resp.headers.get("Content-Type") or "").lower()
    if "text/html" in content_type:
        raise GDriveError(
            "Google Drive returned a web page instead of the file — the "
            "link may be wrong, quota-limited, or not public."
        )

    total = int(resp.headers.get("Content-Length") or -1)
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    tmp_path = dest_path + ".part"
    done = 0
    try:
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=CHUNK):
                if cancel_cb and cancel_cb():
                    raise GDriveCancelled("Download cancelled.")
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if progress_cb:
                    progress_cb(done, total)
    except requests.RequestException as e:
        raise GDriveError(f"Download interrupted: {e}") from e
    finally:
        if os.path.exists(tmp_path) and (done == 0 or
                                         (total > 0 and done < total)):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    if total > 0 and done < total:
        raise GDriveError(
            f"Download incomplete ({done:,} of {total:,} bytes).")

    os.replace(tmp_path, dest_path)
    logger.info(f"GDrive: downloaded {done:,} bytes -> {dest_path}")
    return dest_path


def _resolve_interstitial(session: requests.Session,
                          resp: requests.Response,
                          file_id: str) -> requests.Response:
    """If Google served the virus-scan warning page, follow its form /
    confirm token and return the real file response."""
    content_type = (resp.headers.get("Content-Type") or "").lower()
    if resp.status_code != 200 or "text/html" not in content_type:
        return resp

    html = resp.text

    # Modern interstitial: a <form> pointing at drive.usercontent
    form = _parse_interstitial_form(html)
    if form:
        try:
            return session.get(form["action"], params=form["params"],
                               stream=True, timeout=TIMEOUT)
        except requests.RequestException as e:
            raise GDriveError(f"Network error on Drive confirm step: {e}") from e

    # Legacy flow: confirm token in a download_warning cookie or the HTML
    token = None
    for name, value in session.cookies.items():
        if name.startswith("download_warning"):
            token = value
            break
    if not token:
        m = re.search(r"confirm=([\w-]+)", html)
        token = m.group(1) if m else None
    if token:
        try:
            return session.get(
                "https://drive.google.com/uc",
                params={"export": "download", "id": file_id, "confirm": token},
                stream=True, timeout=TIMEOUT,
            )
        except requests.RequestException as e:
            raise GDriveError(f"Network error on Drive confirm step: {e}") from e

    return resp

"""Auto-updater: checks the git remote for new commits on startup and
applies them silently when a fast-forward pull is safe.

Called once at app startup from main.py. If a fast-forward isn't
possible (uncommitted edits, diverged history, etc.) the caller just
tells the user to resolve the conflict manually and opens the app with
the local version — we never overwrite or reset the working tree
behind the user's back.
"""

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)


def _startupinfo():
    """Hide the child git process window on Windows."""
    si = None
    if sys.platform == "win32":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE
    return si


def _run_git(args, cwd, timeout=30):
    try:
        return subprocess.run(
            ["git"] + list(args),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            startupinfo=_startupinfo(),
        )
    except Exception:
        logger.exception("git %s failed", " ".join(args))
        return None


def _is_git_repo(path: str) -> bool:
    return os.path.isdir(os.path.join(path, ".git"))


def check_for_updates(project_dir: str) -> dict:
    """Return a dict describing what (if anything) the updater should do.

    Possible `status` values:
      - not_a_repo:        no .git dir; skip silently
      - no_remote:         no origin/<branch> tracked; skip
      - fetch_failed:      no network / auth error; skip with log
      - up_to_date:        local == remote
      - local_ahead:       local has commits that remote does not (user's unpushed work)
      - update_available:  remote has new commits — caller decides what to do
      - error:             unexpected git failure
    """
    if not _is_git_repo(project_dir):
        return {"status": "not_a_repo"}

    r_branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], project_dir)
    if r_branch is None or r_branch.returncode != 0:
        return {"status": "error", "detail": "could not resolve current branch"}
    branch = r_branch.stdout.strip()

    # Short fetch timeout so a flaky network doesn't freeze app startup.
    r_fetch = _run_git(["fetch", "--quiet", "origin"], project_dir, timeout=15)
    if r_fetch is None:
        return {"status": "fetch_failed", "detail": "timeout"}
    if r_fetch.returncode != 0:
        return {"status": "fetch_failed", "detail": (r_fetch.stderr or "").strip()[-300:]}

    r_local = _run_git(["rev-parse", "HEAD"], project_dir)
    r_remote = _run_git(["rev-parse", f"origin/{branch}"], project_dir)
    if r_local is None or r_remote is None:
        return {"status": "error", "detail": "rev-parse failed"}
    if r_remote.returncode != 0:
        return {"status": "no_remote"}

    local_hash = r_local.stdout.strip()
    remote_hash = r_remote.stdout.strip()
    if local_hash == remote_hash:
        return {"status": "up_to_date", "branch": branch}

    # If remote is an ancestor of local, local is ahead (user's unpushed work).
    r_ancestor = _run_git(
        ["merge-base", "--is-ancestor", f"origin/{branch}", "HEAD"], project_dir
    )
    if r_ancestor is not None and r_ancestor.returncode == 0:
        return {"status": "local_ahead", "branch": branch}

    r_status = _run_git(["status", "--porcelain"], project_dir)
    dirty = bool(r_status.stdout.strip()) if r_status else True

    return {
        "status": "update_available",
        "branch": branch,
        "local": local_hash,
        "remote": remote_hash,
        "dirty": dirty,
    }


def pull_fast_forward(project_dir: str, branch: str):
    """Apply updates by fast-forward pull. Fails (and that's OK — caller
    just tells the user) if the workdir is dirty or history has diverged."""
    r = _run_git(["pull", "--ff-only", "origin", branch], project_dir, timeout=300)
    if r is None:
        return False, "timeout"
    if r.returncode != 0:
        return False, (r.stderr or "").strip()[-400:]
    return True, ""

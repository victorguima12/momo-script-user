"""Firestore REST client for the jobs system (writer side).

Talks to the `script_jobs` collection using the public web API key —
no Google SDK, no service account (the user repo is public). Access
control is enforced by Firestore security rules, not by this client.

Data model (see JOBS_TAB_BRIEF.md §4):

    script_jobs/{jobId}:
        title, chapters, status, claimed_by, notes,
        created_at / claimed_at / delivered_at (ISO strings),
        images_gdrive_id, original_chunks, delivered_chunks

    script_jobs/{jobId}/payload/original_0..N   {data: <b64 chunk>}
    script_jobs/{jobId}/payload/delivered_0..N  {data: <b64 chunk>}

mscript payloads are stored as base64(gzip(json)) split into chunks of
at most CHUNK_CHARS characters so no single Firestore doc approaches
the 1MB limit.

Claiming is race-safe: the PATCH carries a `currentDocument.updateTime`
precondition taken from the read, so if two writers race, Firestore
rejects the loser with FAILED_PRECONDITION and we surface JobTakenError.
"""

import base64
import gzip
import json
import logging
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import requests

from app.firebase_config import (
    FIREBASE_WEB_API_KEY,
    FIRESTORE_BASE,
    JOBS_COLLECTION,
)

logger = logging.getLogger(__name__)

TIMEOUT = 30
CHUNK_CHARS = 700_000  # b64 chars per payload doc (~700KB, under the 1MB doc cap)


class JobsClientError(Exception):
    """Network / API failure with a user-presentable message."""


class JobTakenError(JobsClientError):
    """Someone else claimed (or modified) the job first."""


# ---------------------------------------------------------------- payload codec

def encode_mscript_chunks(state: dict) -> List[str]:
    """dict -> base64(gzip(json)) split into <=CHUNK_CHARS strings."""
    raw = json.dumps(state, ensure_ascii=False).encode("utf-8")
    b64 = base64.b64encode(gzip.compress(raw)).decode("ascii")
    return [b64[i:i + CHUNK_CHARS] for i in range(0, len(b64), CHUNK_CHARS)] or [""]


def decode_mscript_chunks(chunks: List[str]) -> dict:
    """Inverse of encode_mscript_chunks."""
    raw = gzip.decompress(base64.b64decode("".join(chunks)))
    return json.loads(raw.decode("utf-8"))


# ---------------------------------------------------------------- Firestore values

def _to_fs(value) -> dict:
    """Python value -> Firestore typed value."""
    if isinstance(value, bool):
        return {"booleanValue": value}
    if isinstance(value, int):
        return {"integerValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": "" if value is None else str(value)}


def _from_fs(value: dict):
    """Firestore typed value -> Python value."""
    if "stringValue" in value:
        return value["stringValue"]
    if "integerValue" in value:
        return int(value["integerValue"])
    if "doubleValue" in value:
        return value["doubleValue"]
    if "booleanValue" in value:
        return value["booleanValue"]
    if "timestampValue" in value:
        return value["timestampValue"]
    if "nullValue" in value:
        return None
    return value  # arrays/maps not used by the jobs schema


def _doc_to_job(doc: dict) -> dict:
    """Firestore document -> flat job dict (+ _id / _update_time)."""
    job = {k: _from_fs(v) for k, v in (doc.get("fields") or {}).items()}
    job["_id"] = doc["name"].rsplit("/", 1)[-1]
    job["_update_time"] = doc.get("updateTime", "")
    return job


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _request(method: str, url: str, **kwargs) -> requests.Response:
    """requests wrapper that turns transport errors into JobsClientError."""
    try:
        resp = requests.request(method, url, timeout=TIMEOUT, **kwargs)
    except requests.RequestException as e:
        raise JobsClientError(f"Network error: {e}") from e
    return resp


def _raise_for(resp: requests.Response, doing: str):
    if resp.status_code in (400, 409, 412):
        body = resp.text or ""
        if "FAILED_PRECONDITION" in body or resp.status_code in (409, 412):
            raise JobTakenError(
                "The job changed on the server (someone else probably got "
                "there first). Refresh the list and try again."
            )
    if resp.status_code != 200:
        raise JobsClientError(
            f"Firestore error while {doing}: HTTP {resp.status_code} "
            f"{resp.text[:300]}"
        )


def _key_params() -> Dict[str, str]:
    return {"key": FIREBASE_WEB_API_KEY}


# ---------------------------------------------------------------- public API

def list_jobs() -> List[dict]:
    """All jobs in the collection, newest first. Handles pagination."""
    jobs: List[dict] = []
    page_token: Optional[str] = None
    url = f"{FIRESTORE_BASE}/{JOBS_COLLECTION}"
    while True:
        params = _key_params()
        params["pageSize"] = "300"
        if page_token:
            params["pageToken"] = page_token
        resp = _request("GET", url, params=params)
        _raise_for(resp, "listing jobs")
        data = resp.json()
        jobs.extend(_doc_to_job(d) for d in data.get("documents", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    jobs.sort(key=lambda j: str(j.get("created_at", "")), reverse=True)
    return jobs


def get_job(job_id: str) -> dict:
    resp = _request("GET", f"{FIRESTORE_BASE}/{JOBS_COLLECTION}/{job_id}",
                    params=_key_params())
    _raise_for(resp, "reading the job")
    return _doc_to_job(resp.json())


def _patch_job(job_id: str, fields: Dict[str, object], update_time: str,
               doing: str) -> dict:
    """PATCH selected fields of a job doc guarded by an updateTime
    precondition. Returns the updated job dict."""
    params: List[tuple] = list(_key_params().items())
    params.append(("currentDocument.updateTime", update_time))
    for name in fields:
        params.append(("updateMask.fieldPaths", name))
    body = {"fields": {k: _to_fs(v) for k, v in fields.items()}}
    resp = _request(
        "PATCH", f"{FIRESTORE_BASE}/{JOBS_COLLECTION}/{job_id}",
        params=params, json=body,
    )
    _raise_for(resp, doing)
    return _doc_to_job(resp.json())


def claim_job(job_id: str, writer_name: str) -> dict:
    """Atomically claim an available job. Raises JobTakenError on a race."""
    job = get_job(job_id)
    if job.get("status") != "available":
        raise JobTakenError(
            f"Job is not available anymore (status: {job.get('status')}, "
            f"claimed by: {job.get('claimed_by') or '—'})."
        )
    return _patch_job(
        job_id,
        {
            "status": "claimed",
            "claimed_by": writer_name,
            "claimed_at": _now_iso(),
        },
        job["_update_time"],
        "claiming the job",
    )


def fetch_original_mscript(job_id: str, n_chunks: int,
                           progress_cb: Optional[Callable[[int, int], None]] = None
                           ) -> dict:
    """Download + decode the admin's original mscript payload."""
    chunks: List[str] = []
    for i in range(int(n_chunks)):
        resp = _request(
            "GET",
            f"{FIRESTORE_BASE}/{JOBS_COLLECTION}/{job_id}/payload/original_{i}",
            params=_key_params(),
        )
        _raise_for(resp, f"downloading script part {i + 1}/{n_chunks}")
        fields = resp.json().get("fields") or {}
        chunks.append(_from_fs(fields.get("data", {"stringValue": ""})))
        if progress_cb:
            progress_cb(i + 1, int(n_chunks))
    return decode_mscript_chunks(chunks)


def deliver_job(job_id: str, writer_name: str, state: dict,
                progress_cb: Optional[Callable[[int, int], None]] = None) -> dict:
    """Upload the writer's corrected mscript and flip the job to delivered.

    Chunk uploads are idempotent (fixed doc names, full overwrite) so a
    failed delivery can simply be retried.
    """
    job = get_job(job_id)
    if job.get("claimed_by") != writer_name:
        raise JobsClientError(
            f"This job is claimed by '{job.get('claimed_by') or '—'}', "
            f"not by you ('{writer_name}') — delivery refused."
        )
    if job.get("status") not in ("claimed", "delivered"):
        raise JobsClientError(
            f"Job status is '{job.get('status')}' — nothing to deliver."
        )

    # Versioned deliveries: each delivery becomes v1, v2, ... — the admin
    # keeps the original ("Main") and every prior version intact. A legacy
    # pre-versioning delivery (chunks at delivered_{i}) counts as v1.
    prev = int(job.get("delivered_versions") or 0)
    if prev == 0 and int(job.get("delivered_chunks") or 0) > 0:
        prev = 1
    version = prev + 1

    chunks = encode_mscript_chunks(state)
    for i, chunk in enumerate(chunks):
        resp = _request(
            "PATCH",
            f"{FIRESTORE_BASE}/{JOBS_COLLECTION}/{job_id}/payload/delivered_v{version}_{i}",
            params=_key_params(),
            json={"fields": {"data": _to_fs(chunk)}},
        )
        _raise_for(resp, f"uploading script part {i + 1}/{len(chunks)}")
        if progress_cb:
            progress_cb(i + 1, len(chunks))

    return _patch_job(
        job_id,
        {
            "status": "delivered",
            "delivered_versions": version,
            f"delivered_v{version}_chunks": len(chunks),
            f"delivered_v{version}_by": writer_name,
            f"delivered_v{version}_at": _now_iso(),
            "delivered_at": _now_iso(),
        },
        job["_update_time"],
        "marking the job delivered",
    )

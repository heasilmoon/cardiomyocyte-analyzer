"""Optional Supabase-backed persistence for analysis results.

Hosted disks (Render, Hugging Face Spaces, ...) are ephemeral: local files
under backend/storage/results/ don't survive a redeploy or restart. This
mirrors each result's files (plot.png, plot.svg, data.csv, summary.json,
panel_*.png/svg) into a Supabase Storage bucket, plus a summary row into a
Postgres table, so results keep existing after that — GET
/results/<id>/<file> in main.py falls back to fetching from Supabase when
the local copy is gone.

Entirely optional: every function here is a no-op unless SUPABASE_URL and
SUPABASE_SERVICE_ROLE_KEY are set (environment variables, or a
backend/.env file — see config.py), so local development needs no
Supabase account. See README for the one-time Supabase project setup
(SQL to create the table, bucket creation).

Every upload reports back what happened (`status()` / the dict returned by
upload_result) so a misconfiguration shows up in the app instead of
failing silently.
"""
from __future__ import annotations

import mimetypes
import os
import time
from pathlib import Path
from typing import Any

BUCKET_NAME = "results"
TABLE_NAME = "analysis_results"

_client = None
_client_checked = False
_client_error: str | None = None

# Last upload outcome, for /api/health and the results page.
_last: dict = {"ok": None, "error": None, "at": None, "result_id": None}


def is_configured() -> bool:
    return bool(os.environ.get("SUPABASE_URL")) and bool(os.environ.get("SUPABASE_SERVICE_ROLE_KEY"))


def _short(exc: BaseException) -> str:
    msg = f"{type(exc).__name__}: {exc}"
    return msg if len(msg) <= 300 else msg[:297] + "..."


def _get_client():
    """Lazily-created, process-wide Supabase client; None if not configured
    or if the client could not be created (error kept in status())."""
    global _client, _client_checked, _client_error
    if _client_checked:
        return _client
    _client_checked = True
    if not is_configured():
        return None
    try:
        from supabase import create_client

        _client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    except Exception as exc:  # bad URL/key, package missing, ...
        _client_error = _short(exc)
        _client = None
    return _client


def status() -> dict:
    """What the app knows about Supabase persistence right now."""
    configured = is_configured()
    hint = None
    if not configured:
        hint = "SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set (environment or backend/.env)"
    return {
        "configured": configured,
        "client_error": _client_error,
        "last_ok": _last["ok"],
        "last_error": _last["error"],
        "last_at": _last["at"],
        "last_result_id": _last["result_id"],
        "bucket": BUCKET_NAME,
        "table": TABLE_NAME,
        "hint": hint,
    }


def upload_result(result_id: str, result_dir: Path, analysis_type: str, summary: Any) -> dict:
    """Best-effort mirror of one result's local files + summary to Supabase.

    Never raises: a Supabase hiccup shouldn't break the analysis response
    the user is already waiting on. The local copy (already written to
    disk before this is called) remains this server process's source of
    truth regardless. Returns {"configured", "ok", "error", "n_files"}.
    """
    if not is_configured():
        return {"configured": False, "ok": None, "error": None, "n_files": 0}
    client = _get_client()
    if client is None:
        _last.update(ok=False, error=_client_error or "client not available", at=time.time(), result_id=result_id)
        return {"configured": True, "ok": False, "error": _last["error"], "n_files": 0}
    n_files = 0
    try:
        names = ["plot.png", "plot.svg", "data.csv", "summary.json"]
        names += sorted(p.name for p in result_dir.glob("panel_*.png")) + sorted(p.name for p in result_dir.glob("panel_*.svg"))
        for filename in names:
            path = result_dir / filename
            if not path.exists():
                continue
            content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            client.storage.from_(BUCKET_NAME).upload(
                f"{result_id}/{filename}",
                path.read_bytes(),
                {"content-type": content_type, "upsert": "true"},
            )
            n_files += 1
        client.table(TABLE_NAME).upsert({"id": result_id, "analysis_type": analysis_type, "summary": summary}).execute()
    except Exception as exc:
        err = _short(exc)
        low = err.lower()
        if "bucket" in low and ("not found" in low or "does not exist" in low):
            err += f" — create a private Storage bucket named '{BUCKET_NAME}' in Supabase"
        elif "relation" in low and "does not exist" in low:
            err += f" — run the README SQL to create the '{TABLE_NAME}' table"
        elif "jwt" in low or "invalid api key" in low or "401" in low or "403" in low:
            err += " — check SUPABASE_SERVICE_ROLE_KEY (service_role key, not the anon key)"
        _last.update(ok=False, error=err, at=time.time(), result_id=result_id)
        return {"configured": True, "ok": False, "error": err, "n_files": n_files}
    _last.update(ok=True, error=None, at=time.time(), result_id=result_id)
    return {"configured": True, "ok": True, "error": None, "n_files": n_files}


def fetch_result_file(result_id: str, filename: str) -> bytes | None:
    """Bytes of a previously-uploaded result file.

    None if Supabase isn't configured, or the file/result doesn't exist
    there (e.g. it was only ever analyzed locally, before this was set up).
    """
    client = _get_client()
    if client is None:
        return None
    try:
        return client.storage.from_(BUCKET_NAME).download(f"{result_id}/{filename}")
    except Exception:
        return None

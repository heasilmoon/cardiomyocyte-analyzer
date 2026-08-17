"""Optional Supabase-backed persistence for analysis results.

Render's disk (like most PaaS free/starter tiers) is ephemeral: local
files under backend/storage/results/ don't survive a redeploy or restart.
This mirrors each result's files (plot.png, data.csv, summary.json) into a
Supabase Storage bucket, plus a summary row into a Postgres table, so
results keep existing after that — GET /results/<id>/<file> in main.py
falls back to fetching from Supabase when the local copy is gone.

Entirely optional: every function here is a no-op unless SUPABASE_URL and
SUPABASE_SERVICE_ROLE_KEY are set (e.g. in Render's Environment tab), so
local development needs no Supabase account. See README for the one-time
Supabase project setup (SQL to create the table, bucket creation).
"""
from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Any

BUCKET_NAME = "results"
TABLE_NAME = "analysis_results"

_client = None
_client_checked = False


def is_configured() -> bool:
    return bool(os.environ.get("SUPABASE_URL")) and bool(os.environ.get("SUPABASE_SERVICE_ROLE_KEY"))


def _get_client():
    """Lazily-created, process-wide Supabase client; None if not configured.

    The `supabase` package is only imported here (not at module load time)
    so that nothing about this module's import breaks a setup that hasn't
    added the dependency's transitive requirements for some reason —
    matches how the rest of this codebase treats optional integrations.
    """
    global _client, _client_checked
    if _client_checked:
        return _client
    _client_checked = True
    if not is_configured():
        return None
    from supabase import create_client

    _client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    return _client


def upload_result(result_id: str, result_dir: Path, analysis_type: str, summary: Any) -> None:
    """Best-effort mirror of one result's local files + summary to Supabase.

    Never raises: a Supabase hiccup shouldn't break the analysis response
    the user is already waiting on. The local copy (already written to
    disk before this is called) remains this server process's source of
    truth regardless of whether this succeeds.
    """
    client = _get_client()
    if client is None:
        return
    try:
        for filename in ("plot.png", "data.csv", "summary.json"):
            path = result_dir / filename
            if not path.exists():
                continue
            content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            client.storage.from_(BUCKET_NAME).upload(
                f"{result_id}/{filename}",
                path.read_bytes(),
                {"content-type": content_type, "upsert": "true"},
            )
        client.table(TABLE_NAME).upsert({"id": result_id, "analysis_type": analysis_type, "summary": summary}).execute()
    except Exception:
        pass


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

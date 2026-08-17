from pathlib import Path

from app.utils import result_storage


def test_is_configured_false_without_env_vars(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    assert result_storage.is_configured() is False


def test_is_configured_true_with_both_env_vars(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "fake-key")
    assert result_storage.is_configured() is True


def test_upload_result_is_a_silent_noop_without_configuration(monkeypatch, tmp_path):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    result_storage._client = None
    result_storage._client_checked = False

    result_dir = Path(tmp_path)
    (result_dir / "summary.json").write_text("{}")
    # Should not raise even though there's no Supabase client at all.
    result_storage.upload_result("some-id", result_dir, "beating", {"n_beats": 3})


def test_fetch_result_file_returns_none_without_configuration(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    result_storage._client = None
    result_storage._client_checked = False

    assert result_storage.fetch_result_file("some-id", "summary.json") is None

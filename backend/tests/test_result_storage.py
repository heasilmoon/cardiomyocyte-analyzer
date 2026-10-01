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


def test_upload_result_reports_status_when_not_configured(monkeypatch, tmp_path):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    result_storage._client = None
    result_storage._client_checked = False
    (tmp_path / "summary.json").write_text("{}")
    st = result_storage.upload_result("id", tmp_path, "beating", {"n_beats": 1})
    assert st == {"configured": False, "ok": None, "error": None, "n_files": 0}
    s = result_storage.status()
    assert s["configured"] is False and "not set" in s["hint"]


def test_upload_result_reports_error_when_client_cannot_be_created(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPABASE_URL", "not-a-url")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "bad")
    result_storage._client = None
    result_storage._client_checked = False
    result_storage._client_error = None
    (tmp_path / "summary.json").write_text("{}")
    st = result_storage.upload_result("id2", tmp_path, "beating", {})
    assert st["configured"] is True and st["ok"] is False and st["error"]
    assert result_storage.status()["last_ok"] is False
    result_storage._client = None
    result_storage._client_checked = False
    result_storage._client_error = None


def test_dotenv_file_is_loaded_without_overriding_existing(monkeypatch, tmp_path):
    from app.config import _load_dotenv

    monkeypatch.delenv("DOTENV_TEST_A", raising=False)
    monkeypatch.setenv("DOTENV_TEST_B", "keep")
    env = tmp_path / ".env"
    env.write_text('# comment\nDOTENV_TEST_A="hello world"\nDOTENV_TEST_B=override\n\nbroken line\n')
    _load_dotenv(env)
    import os

    assert os.environ["DOTENV_TEST_A"] == "hello world"
    assert os.environ["DOTENV_TEST_B"] == "keep"
    monkeypatch.delenv("DOTENV_TEST_A", raising=False)

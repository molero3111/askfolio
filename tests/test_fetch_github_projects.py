import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_github_projects.py"


@pytest.fixture
def fetch():
    spec = importlib.util.spec_from_file_location("fetch_github_projects", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_api_error_exits_without_returning_a_partial_list(fetch, monkeypatch):
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: _Response(403))
    with pytest.raises(SystemExit) as exc:
        fetch.fetch_repos()
    assert exc.value.code == 1


def test_empty_result_keeps_the_existing_file(fetch, monkeypatch, tmp_path):
    out = tmp_path / "projects.json"
    out.write_text("keep me")
    monkeypatch.setattr(fetch, "OUTPUT_PATH", out)
    monkeypatch.setattr(fetch, "fetch_repos", list)
    with pytest.raises(SystemExit) as exc:
        fetch.main()
    assert exc.value.code == 1
    assert out.read_text() == "keep me"

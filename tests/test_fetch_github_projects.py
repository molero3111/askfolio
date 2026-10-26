import sys

import pytest

sys.path.insert(0, "scripts")
import fetch_github_projects as fetch


class _Response:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_api_error_exits_without_returning_a_partial_list(monkeypatch):
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: _Response(403))
    with pytest.raises(SystemExit) as exc:
        fetch.fetch_repos()
    assert exc.value.code == 1

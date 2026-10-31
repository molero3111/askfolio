import logging

import pytest
import requests

from app import config, llm_client
from app.llm_client import LLMCreditExhausted, LLMError, LLMUnavailable, chat

OR_URL = "https://openrouter.ai/api/v1/chat/completions"
LOCAL_URL = "http://localhost:11434/v1/chat/completions"
MSGS = [{"role": "user", "content": "hello"}]


class FakeResponse:
    def __init__(self, status=200, json_data=None, headers=None):
        self.status_code = status
        self._json = json_data
        self.headers = headers or {}

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


def ok(content=" hi ", **extra):
    body = {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.0001},
    }
    body.update(extra)
    return FakeResponse(200, body)


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def cfg(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_URL", OR_URL)
    monkeypatch.setattr(config, "LLM_MODEL", "google/gemma-test:free")
    monkeypatch.setattr(config, "LLM_API_KEY", "sk-test")
    monkeypatch.setattr(config, "LLM_FALLBACK_MODELS", [])
    monkeypatch.setattr(config, "LLM_DATA_COLLECTION", "deny")
    monkeypatch.setattr(config, "LLM_MAX_TOKENS", 800)
    monkeypatch.setattr(config, "LLM_APP_NAME", "askfolio")
    monkeypatch.setattr(config, "LLM_APP_URL", "")


def no_sleep(_):
    pass


def test_success_returns_content_and_usage():
    r = chat(MSGS, session=FakeSession([ok()]), sleep=no_sleep)
    assert r.content == "hi"
    assert r.prompt_tokens == 10
    assert r.completion_tokens == 2
    assert r.cost == 0.0001
    assert r.model == "google/gemma-test:free"


def test_result_model_reports_responding_model():
    s = FakeSession([ok(model="other/fallback:free")])
    assert chat(MSGS, session=s, sleep=no_sleep).model == "other/fallback:free"


def test_openrouter_payload():
    s = FakeSession([ok()])
    chat(MSGS, session=s, sleep=no_sleep)
    call = s.calls[0]
    assert call["json"]["provider"] == {"data_collection": "deny"}
    assert call["json"]["max_tokens"] == 800
    assert call["json"]["messages"] == MSGS
    assert "X-OpenRouter-Title" in call["headers"]
    assert call["headers"]["Authorization"] == "Bearer sk-test"


def test_provider_only_restricts_providers(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER_ONLY", ["google-ai-studio"])
    s = FakeSession([ok()])
    chat(MSGS, session=s, sleep=no_sleep)
    assert s.calls[0]["json"]["provider"] == {
        "data_collection": "deny",
        "only": ["google-ai-studio"],
    }


def test_provider_only_unset_sends_no_restriction(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER_ONLY", [])
    s = FakeSession([ok()])
    chat(MSGS, session=s, sleep=no_sleep)
    assert "only" not in s.calls[0]["json"]["provider"]


def test_provider_only_not_sent_to_other_endpoints(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_URL", LOCAL_URL)
    monkeypatch.setattr(config, "LLM_PROVIDER_ONLY", ["google-ai-studio"])
    s = FakeSession([ok()])
    chat(MSGS, session=s, sleep=no_sleep)
    assert "provider" not in s.calls[0]["json"]


def test_non_openrouter_payload(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_URL", LOCAL_URL)
    s = FakeSession([ok()])
    chat(MSGS, session=s, sleep=no_sleep)
    call = s.calls[0]
    assert "provider" not in call["json"]
    assert "max_tokens" not in call["json"] or call["json"]["max_tokens"] == 800
    assert "X-OpenRouter-Title" not in call["headers"]
    assert "HTTP-Referer" not in call["headers"]
    assert "models" not in call["json"]


def test_referer_header_when_app_url_set(monkeypatch):
    monkeypatch.setattr(config, "LLM_APP_URL", "https://example.org")
    s = FakeSession([ok()])
    chat(MSGS, session=s, sleep=no_sleep)
    assert s.calls[0]["headers"]["HTTP-Referer"] == "https://example.org"


def test_none_content_raises_unavailable():
    with pytest.raises(LLMUnavailable):
        chat(MSGS, session=FakeSession([ok(content=None)]), sleep=no_sleep)


def test_200_error_body_raises_unavailable():
    r = FakeResponse(200, {"error": {"message": "boom", "code": 500}})
    with pytest.raises(LLMUnavailable):
        chat(MSGS, session=FakeSession([r]), sleep=no_sleep)


def test_429_retry_then_success():
    sleeps = []
    s = FakeSession([FakeResponse(429, {}, {"Retry-After": "1"}), ok()])
    assert chat(MSGS, session=s, sleep=sleeps.append).content == "hi"
    assert sleeps == [1]
    assert len(s.calls) == 2


def test_429_twice_raises_unavailable():
    s = FakeSession([FakeResponse(429), FakeResponse(429)])
    with pytest.raises(LLMUnavailable):
        chat(MSGS, session=s, sleep=no_sleep)
    assert len(s.calls) == 2


def test_retry_after_capped_and_default():
    sleeps = []
    s = FakeSession([FakeResponse(429, {}, {"Retry-After": "999"}), ok()])
    chat(MSGS, session=s, sleep=sleeps.append)
    assert sleeps == [10]
    sleeps.clear()
    s = FakeSession([FakeResponse(503), ok()])
    chat(MSGS, session=s, sleep=sleeps.append)
    assert sleeps == [2]


def test_5xx_retry_then_fail():
    s = FakeSession([FakeResponse(503), FakeResponse(503)])
    with pytest.raises(LLMUnavailable):
        chat(MSGS, session=s, sleep=no_sleep)
    assert len(s.calls) == 2


def test_402_no_retry():
    s = FakeSession([FakeResponse(402, {"error": {"message": "credits"}})])
    with pytest.raises(LLMCreditExhausted):
        chat(MSGS, session=s, sleep=no_sleep)
    assert len(s.calls) == 1


def test_timeout_unavailable():
    s = FakeSession([requests.Timeout("slow")])
    with pytest.raises(LLMUnavailable):
        chat(MSGS, session=s, sleep=no_sleep)


def test_connection_error_unavailable():
    s = FakeSession([requests.ConnectionError("down")])
    with pytest.raises(LLMError):
        chat(MSGS, session=s, sleep=no_sleep)


def test_model_override():
    s = FakeSession([ok()])
    chat(MSGS, model="x/y", session=s, sleep=no_sleep)
    assert s.calls[0]["json"]["model"] == "x/y"


def test_fallback_models_present_for_openrouter(monkeypatch):
    monkeypatch.setattr(config, "LLM_FALLBACK_MODELS", ["a/b:free", "c/d:free"])
    s = FakeSession([ok()])
    chat(MSGS, session=s, sleep=no_sleep)
    p = s.calls[0]["json"]
    assert p["models"] == ["google/gemma-test:free", "a/b:free", "c/d:free"]
    assert p["model"] == "google/gemma-test:free"


def test_fallback_models_absent_when_empty():
    s = FakeSession([ok()])
    chat(MSGS, session=s, sleep=no_sleep)
    assert "models" not in s.calls[0]["json"]


def test_fallback_models_absent_for_non_openrouter(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_URL", LOCAL_URL)
    monkeypatch.setattr(config, "LLM_FALLBACK_MODELS", ["a/b:free"])
    s = FakeSession([ok()])
    chat(MSGS, session=s, sleep=no_sleep)
    assert "models" not in s.calls[0]["json"]


def test_model_override_first_in_models(monkeypatch):
    monkeypatch.setattr(config, "LLM_FALLBACK_MODELS", ["a/b:free"])
    s = FakeSession([ok()])
    chat(MSGS, model="x/y", session=s, sleep=no_sleep)
    p = s.calls[0]["json"]
    assert p["models"] == ["x/y", "a/b:free"]
    assert p["model"] == "x/y"


@pytest.mark.parametrize("status", [400, 404])
@pytest.mark.parametrize(
    "msg",
    [
        "No endpoints found matching your data policy (Free model publication)",
        "No endpoints found for google/x:free.",
    ],
)
def test_data_policy_error_no_retry(status, msg, caplog):
    s = FakeSession([FakeResponse(status, {"error": {"message": msg}})])
    with caplog.at_level(logging.ERROR, logger=llm_client.logger.name), pytest.raises(
        LLMUnavailable
    ):
        chat(MSGS, session=s, sleep=no_sleep)
    assert len(s.calls) == 1
    assert (
        "[llm] no provider for google/gemma-test:free matches data_collection=deny"
        in caplog.text
    )


def test_api_key_never_logged(caplog):
    with caplog.at_level(logging.DEBUG):
        chat(MSGS, session=FakeSession([ok()]), sleep=no_sleep)
        with pytest.raises(LLMCreditExhausted):
            chat(MSGS, session=FakeSession([FakeResponse(402)]), sleep=no_sleep)
    assert "sk-test" not in caplog.text


def _with_choice(**choice_extra):
    body = {
        "choices": [{"message": {"content": "partial text"}, **choice_extra}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    return FakeResponse(200, body)


def test_finish_reason_error_is_unavailable_even_with_content():
    with pytest.raises(LLMUnavailable):
        chat(MSGS, session=FakeSession([_with_choice(finish_reason="error")]), sleep=no_sleep)


def test_choice_error_is_unavailable_even_with_content():
    resp = _with_choice(error={"code": 502, "message": "upstream"})
    with pytest.raises(LLMUnavailable):
        chat(MSGS, session=FakeSession([resp]), sleep=no_sleep)


def test_finish_reason_length_warns_but_returns_content(caplog):
    with caplog.at_level(logging.WARNING):
        r = chat(MSGS, session=FakeSession([_with_choice(finish_reason="length")]), sleep=no_sleep)
    assert r.content == "partial text"
    assert "truncated" in caplog.text


def test_empty_reply_after_reasoning_tokens_warns(caplog):
    body = {
        "choices": [{"message": {"content": ""}}],
        "usage": {"completion_tokens": 800, "completion_tokens_details": {"reasoning_tokens": 800}},
    }
    with caplog.at_level(logging.WARNING), pytest.raises(LLMUnavailable):
        chat(MSGS, session=FakeSession([FakeResponse(200, body)]), sleep=no_sleep)
    assert (
        "[llm] reply empty after 800 reasoning tokens; "
        "raise LLM_MAX_TOKENS or pick a non-reasoning model"
    ) in caplog.text

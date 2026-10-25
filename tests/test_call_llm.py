from unittest.mock import MagicMock, patch

from app import graph


def test_call_llm_sends_messages_unchanged():
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
    ]
    resp = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": "  hi  "}}]}
    with patch.object(graph.requests, "post", return_value=resp) as post:
        out = graph._call_llm(messages)
    assert out == "hi"
    assert post.call_args.kwargs["json"]["messages"] == messages

from app.text_format import to_plain_text


def test_strips_bold_markers():
    assert to_plain_text("Skills: **Python** and **Go**") == "Skills: Python and Go"


def test_markdown_link_with_label_becomes_label_and_url():
    assert to_plain_text("[repo](https://github.com/x/repo)") == "repo: https://github.com/x/repo"


def test_markdown_link_whose_label_is_the_url_collapses_to_the_url():
    url = "https://github.com/x/repo"
    assert to_plain_text(f"see [{url}]({url}) now") == f"see {url} now"


def test_plain_text_and_bare_urls_are_untouched():
    text = "• chat_in_django: https://github.com/x/chat_in_django - uses __init__ files"
    assert to_plain_text(text) == text


def test_heading_markers_are_removed():
    assert to_plain_text("## Projects\n- a") == "Projects\n- a"


def test_send_telegram_sends_plain_text(monkeypatch):
    from app import graph

    sent = {}
    monkeypatch.setattr(graph, "send_message", lambda chat_id, text, **kw: sent.update(text=text))
    graph._send_telegram(1, "**Hi** [repo](https://github.com/x/repo)", 2)
    assert sent["text"] == "Hi repo: https://github.com/x/repo"

from app.text_normalize import normalize_text


def test_decodes_nested_entities_and_folds():
    assert normalize_text("&amp;lt;a&amp;gt;") == "<a>"
    assert normalize_text("&#xFF1C;x&#xFF1E;") == "<x>"
    assert normalize_text("a\u200bb\u00adc") == "abc"


def test_plain_text_unchanged():
    assert normalize_text("hello & goodbye") == "hello & goodbye"

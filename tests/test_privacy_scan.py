from scripts.privacy_scan import main, scan


def test_scan_finds_term_case_insensitive(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("ok\nmail JANE.DOE@example.com\n")
    assert scan([f], ["jane.doe@example.com"]) == [(str(f), 2)]


def test_scan_clean(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("nothing here")
    assert scan([f], ["secret"]) == []


def test_scan_skips_binary(tmp_path):
    f = tmp_path / "b.bin"
    f.write_bytes(b"\xff\xfe\x00secret")
    assert scan([f], ["secret"]) == []


def test_main_skips_when_unset(monkeypatch, capsys):
    monkeypatch.delenv("PRIVACY_DENYLIST", raising=False)
    assert main() == 0
    assert "::notice::" in capsys.readouterr().out


def test_main_reports_without_term(monkeypatch, capsys, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x\nJane.Doe@example.com\n")
    monkeypatch.setenv("PRIVACY_DENYLIST", "\n\njane.doe@example.com\n")
    monkeypatch.setattr("scripts.privacy_scan.tracked_files", lambda: [f])
    assert main() == 1
    out = capsys.readouterr().out
    assert f"{f}:2" in out
    assert "jane" not in out.lower()


def _run_main(monkeypatch, tmp_path, secret, content="x\nfoo-term\nbar-term\n"):
    f = tmp_path / "a.txt"
    f.write_text(content)
    monkeypatch.setenv("PRIVACY_DENYLIST", secret)
    monkeypatch.setattr("scripts.privacy_scan.tracked_files", lambda: [f])
    return f


def test_main_comma_separated(monkeypatch, capsys, tmp_path):
    f = _run_main(monkeypatch, tmp_path, "foo-term, bar-term")
    assert main() == 1
    out = capsys.readouterr().out
    assert f"{f}:2" in out and f"{f}:3" in out


def test_main_mixed_commas_and_newlines(monkeypatch, capsys, tmp_path):
    f = _run_main(monkeypatch, tmp_path, "nomatch\r\n foo-term ,\n\n,bar-term,")
    assert main() == 1
    out = capsys.readouterr().out
    assert f"{f}:2" in out and f"{f}:3" in out


def test_main_separators_only_is_error(monkeypatch, capsys, tmp_path):
    _run_main(monkeypatch, tmp_path, " , \n ,, ")
    assert main() == 2
    assert "::error::" in capsys.readouterr().out


def test_main_required_and_empty_is_error(monkeypatch, capsys):
    monkeypatch.delenv("PRIVACY_DENYLIST", raising=False)
    monkeypatch.setenv("REQUIRE_DENYLIST", "true")
    assert main() == 2
    assert "::error::" in capsys.readouterr().out


def test_main_required_with_one_is_error_on_empty_string(monkeypatch, capsys):
    monkeypatch.setenv("PRIVACY_DENYLIST", "")
    monkeypatch.setenv("REQUIRE_DENYLIST", "1")
    assert main() == 2


def test_main_not_required_and_empty_is_notice(monkeypatch, capsys):
    monkeypatch.setenv("PRIVACY_DENYLIST", "")
    monkeypatch.setenv("REQUIRE_DENYLIST", "false")
    assert main() == 0
    assert "::notice::" in capsys.readouterr().out


def test_scan_digits_only_match(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello\ncall 991234567890\n(12) 3456-7890 ext\n1234567890 only\n99 12 345\n9991234\n")
    hits = scan([f], ["+99 12 3456 7890"])
    assert (str(f), 2) in hits
    assert (str(f), 3) in hits  # national part without the country code
    assert (str(f), 1) not in hits
    assert (str(f), 6) not in hits
    assert (str(f), 5) not in hits


def test_scan_digits_only_catches_formatted_lines(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("tel: +99 (12) 3456-7890\n")
    assert scan([f], ["+99 12 3456 7890"]) == [(str(f), 1)]


def test_scan_digits_only_not_for_mostly_text_terms(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("room 1234567\n")
    # term is mostly letters, so digit-only matching must not apply
    assert scan([f], ["abc 1234567 abcdefghij"]) == []


def test_scan_nfkc_and_casefold(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("fullwidth Ｊａｎｅ here\nStraße\n")
    assert scan([f], ["jane"]) == [(str(f), 1)]
    assert scan([f], ["STRASSE"]) == [(str(f), 2)]


def test_scan_reads_undecodable_but_not_binary(tmp_path):
    f = tmp_path / "latin.txt"
    f.write_bytes(b"caf\xe9 secret-term\n")
    assert scan([f], ["secret-term"]) == [(str(f), 1)]


def test_scan_skips_nul_in_first_8kb(tmp_path):
    f = tmp_path / "bin.dat"
    f.write_bytes(b"secret-term\n\x00" + b"a" * 100)
    assert scan([f], ["secret-term"]) == []

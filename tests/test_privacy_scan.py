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

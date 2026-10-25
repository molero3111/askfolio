import os

import pytest

from app import config, offset


@pytest.fixture
def offset_file(tmp_path, monkeypatch):
    path = tmp_path / "state" / "offset.txt"
    monkeypatch.setattr(config, "TELEGRAM_OFFSET_FILE", str(path))
    return path


def test_round_trip(offset_file):
    offset.save_offset(42)
    assert offset.load_offset() == 42


def test_missing_file_returns_none(offset_file):
    assert offset.load_offset() is None


@pytest.mark.parametrize("content", ["garbage", "", "  \n"])
def test_garbage_returns_none(offset_file, content):
    offset_file.parent.mkdir(parents=True)
    offset_file.write_text(content)
    assert offset.load_offset() is None


def test_creates_parent_dir_and_leaves_no_temp(offset_file):
    assert not offset_file.parent.exists()
    offset.save_offset(7)
    assert offset_file.read_text() == "7"
    assert list(offset_file.parent.iterdir()) == [offset_file]


def test_relative_path_resolves_against_project_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TELEGRAM_OFFSET_FILE", "data/o.txt")
    monkeypatch.chdir(tmp_path)
    assert offset.offset_path() == config.PROJECT_ROOT / "data" / "o.txt"


def test_overwrites_existing_value(offset_file):
    offset.save_offset(1)
    offset.save_offset(2)
    assert offset.load_offset() == 2


def test_replace_failure_cleans_temp_and_does_not_raise(offset_file, monkeypatch, caplog):
    offset.save_offset(5)

    def boom(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", boom)
    offset.save_offset(6)
    assert offset.load_offset() == 5
    assert list(offset_file.parent.iterdir()) == [offset_file]
    assert any(r.exc_info for r in caplog.records)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_unwritable_dir_does_not_raise(offset_file):
    offset_file.parent.mkdir(parents=True)
    offset_file.parent.chmod(0o500)
    try:
        offset.save_offset(9)
        assert offset.load_offset() is None
    finally:
        offset_file.parent.chmod(0o700)

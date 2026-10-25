from pathlib import Path

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


def test_relative_path_resolves_against_cwd(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TELEGRAM_OFFSET_FILE", "data/o.txt")
    monkeypatch.chdir(tmp_path)
    assert offset.offset_path() == Path(tmp_path) / "data" / "o.txt"

import os
import stat

import pytest

from app import secure_io


def test_writes_with_private_modes(tmp_path):
    path = tmp_path / "secrets" / "token.json"
    secure_io.atomic_write_secret(path, "s3cret")
    assert path.read_text() == "s3cret"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_overwrites_existing(tmp_path):
    path = tmp_path / "token.json"
    secure_io.atomic_write_secret(path, "old")
    secure_io.atomic_write_secret(path, "new")
    assert path.read_text() == "new"
    assert list(tmp_path.iterdir()) == [path]


def test_failed_replace_keeps_original_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "token.json"
    secure_io.atomic_write_secret(path, "original")

    def boom(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        secure_io.atomic_write_secret(path, "new")
    assert path.read_text() == "original"
    assert list(tmp_path.iterdir()) == [path]

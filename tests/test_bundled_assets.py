"""A data-volume upgrade adds packaged context without replacing observations."""

from cagemetrics.cli import _seed_bundled_assets


def write(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_upgrade_adds_nested_assets_and_preserves_existing_observations(tmp_path):
    source, target = tmp_path / "packaged", tmp_path / "volume"
    assets = {
        "rankings/current.json": b"older bundled observation",
        "rankings/current_media.json": b"new media context",
        "rankings/history/normalized.csv.gz": b"frozen history bytes",
        "rankings/raw/source.html.gz": b"retained source bytes",
        "schedules/upcoming.json": b"bundled schedule",
        "fighter_photos.json": b"bundled portraits",
    }
    for name, content in assets.items():
        write(source, name, content)
    write(source, "processed/private.db", b"outside the asset allowlist")
    current = write(target, "rankings/current.json", b"newer volume observation")
    schedule = write(target, "schedules/upcoming.json", b"newer volume schedule")
    _seed_bundled_assets(source, target)
    assert current.read_bytes() == b"newer volume observation"
    assert schedule.read_bytes() == b"newer volume schedule"
    for name, content in assets.items():
        if name not in ("rankings/current.json", "schedules/upcoming.json"):
            assert (target / name).read_bytes() == content
    assert not (target / "processed").exists()
    _seed_bundled_assets(source, target)
    assert current.read_bytes() == b"newer volume observation"


def test_native_same_path_and_absent_assets_do_not_copy(tmp_path, monkeypatch):
    import shutil

    def unexpected_copy(*args, **kwargs):
        raise AssertionError("Existing native assets must not be copied onto themselves")

    monkeypatch.setattr(shutil, "copy2", unexpected_copy)
    write(tmp_path, "rankings/current.json", b"native observation")
    _seed_bundled_assets(tmp_path, tmp_path)
    _seed_bundled_assets(tmp_path / "missing", tmp_path / "unused")
    assert not (tmp_path / "unused").exists()

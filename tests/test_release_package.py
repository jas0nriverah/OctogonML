"""Release safety uses isolated Git fixtures; never modifies the project repository."""

import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

spec = importlib.util.spec_from_file_location(
    "release_package", Path(__file__).parents[1] / "scripts/package_release.py"
)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE)


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    for name, content in {
        "README.md": "Public source\n",
        "src/package/app.py": "print('safe')\n",
        "data/processed/fighters.csv": "id,name\na,Alpha\n",
        "artifacts/winner/model.joblib": "frozen artifact bytes",
        "data/raw/source.csv": "private download",
        ".secrets/key": "secret",
        ".env": "SECRET=private",
        ".env.example": "KEY=\n",
        "data/local.sqlite": "private db",
        "docs/.env.local": "another secret",
    }.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    git(repo, "add", ".")
    git(
        repo,
        "-c",
        "user.name=Release test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "Fixture",
    )
    return repo


def test_deterministic_tracked_bundle_excludes_even_tracked_secrets(repository, tmp_path):
    first = release.package_release(repository, tmp_path / "first", "v1.2.3")
    (repository / "README.md").write_text("uncommitted change")
    (repository / "src/package/untracked.py").write_text("not reviewed")
    second = release.package_release(repository, tmp_path / "second", "v1.2.3")
    assert Path(first["bundle"]).read_bytes() == Path(second["bundle"]).read_bytes()
    assert first["sha256"] == second["sha256"]
    manifest = release.verify_package(first["bundle"])
    paths = {row["path"] for row in manifest["files"]}
    assert paths == {
        "README.md",
        "src/package/app.py",
        "data/processed/fighters.csv",
        "artifacts/winner/model.joblib",
        ".env.example",
    }
    with tarfile.open(first["bundle"], "r:gz") as archive:
        assert archive.extractfile("octogonml-v1.2.3/README.md").read() == b"Public source\n"
    checksum = Path(first["checksum"]).read_text()
    assert checksum == first["sha256"] + "  octogonml-v1.2.3.tar.gz\n"
    before = Path(first["bundle"]).read_bytes()
    with pytest.raises(FileExistsError):
        release.package_release(repository, tmp_path / "first", "v1.2.3")
    assert Path(first["bundle"]).read_bytes() == before


@pytest.mark.parametrize(
    "ref", ["--all", "../HEAD", "HEAD~1", "refs/heads/a..b", "HEAD^{tree}", "refs//main", "main\nsecret"]
)
def test_ref_rejects_options_expressions_and_unsafe_paths(repository, ref):
    with pytest.raises(ValueError, match="Unsafe"):
        release.resolve_ref(repository, ref)


@pytest.mark.parametrize("version", ["../v1.2.3", "1", "v1.2.3/secret", "1.2.3\n", "$(whoami)"])
def test_invalid_versions_never_write(repository, tmp_path, version):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="Version"):
        release.package_release(repository, output, version)
    assert not output.exists()


def test_rejects_tracked_symlink_in_allowlisted_source(repository, tmp_path):
    (repository / "src/package/escape.py").symlink_to("/etc/passwd")
    git(repository, "add", "src/package/escape.py")
    git(
        repository,
        "-c",
        "user.name=Release test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "Symlink",
    )
    with pytest.raises(ValueError, match="regular"):
        release.package_release(repository, tmp_path / "unsafe", "1.2.3")
    assert not list((tmp_path / "unsafe").glob("*.tar.gz"))


def test_verifier_detects_modified_payload_without_extracting(repository, tmp_path):
    packaged = release.package_release(repository, tmp_path / "good", "1.2.3")
    corrupt = tmp_path / "corrupt.tar.gz"
    with tarfile.open(packaged["bundle"], "r:gz") as source, tarfile.open(corrupt, "w:gz") as target:
        for member in source.getmembers():
            content = source.extractfile(member).read()
            if member.name.endswith("/README.md"):
                content = b"Tampered bytes\n"
                member.size = len(content)
            target.addfile(member, io.BytesIO(content))
    with pytest.raises(ValueError, match="mismatch"):
        release.verify_package(corrupt)
    assert not (tmp_path / "octogonml-1.2.3").exists()


def test_verifier_rejects_manifest_path_traversal(repository, tmp_path):
    packaged = release.package_release(repository, tmp_path / "good", "1.2.3")
    corrupt = tmp_path / "traversal.tar.gz"
    with tarfile.open(packaged["bundle"], "r:gz") as source, tarfile.open(corrupt, "w:gz") as target:
        for member in source.getmembers():
            content = source.extractfile(member).read()
            if member.name.endswith("/MANIFEST.json"):
                manifest = json.loads(content)
                manifest["files"][0]["path"] = "../../secret"
                content = json.dumps(manifest).encode()
                member.size = len(content)
            target.addfile(member, io.BytesIO(content))
    with pytest.raises(ValueError):
        release.verify_package(corrupt)


def test_release_installs_entrypoint_docs_and_bounded_public_evidence(repository, tmp_path):
    public = {
        "start.sh": "#!/bin/sh\nexec python -m package\n",
        "CONTRIBUTING.md": "Development guide\n",
        "SECURITY.md": "Reporting policy\n",
        "compose.yaml": "services: {}\n",
        "data/rankings/raw/" + "a" * 64 + ".html.gz": "public frozen ranking evidence",
        "reports/research_predictions.csv.gz": "compressed public predictions",
        "reports/upcoming_archive/" + "b" * 64 + ".json.gz": "issued public archive",
        "data/fighter_profiles/ufc_coverage.json": "{}",
        "data/fighter_profiles/ufc_image_identity_audit.json": "{}",
    }
    private = {
        "data/rankings/raw/private.html.gz": "raw cache",
        "data/rankings/raw/" + "a" * 63 + ".html.gz": "not canonical",
        "data/schedules/raw/" + "a" * 64 + ".html.gz": "other raw cache",
        "reports/local-download.gz": "local dump",
        "data/fighter_profiles/private.json": "not public audit evidence",
    }
    for name, content in {**public, **private}.items():
        target = repository / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    (repository / "start.sh").chmod(0o755)
    git(repository, "add", ".")
    git(
        repository,
        "-c",
        "user.name=Release test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "Public entrypoints and evidence",
    )
    result = release.package_release(repository, tmp_path / "public", "1.2.3")
    manifest = release.verify_package(result["bundle"])
    paths = {row["path"]: row for row in manifest["files"]}
    assert set(public) <= paths.keys()
    assert not set(private) & paths.keys()
    assert paths["start.sh"]["mode"] == 0o755
    with tarfile.open(result["bundle"], "r:gz") as archive:
        for name, content in public.items():
            assert archive.extractfile("octogonml-1.2.3/" + name).read() == content.encode()


@pytest.mark.parametrize(
    "path",
    [
        "data/rankings/raw/../" + "a" * 64 + ".html.gz",
        "data/rankings/raw/nested/" + "a" * 64 + ".html.gz",
        "data/rankings/raw/" + "A" * 64 + ".html.gz",
        "data/raw/" + "a" * 64 + ".html.gz",
        "../data/rankings/raw/" + "a" * 64 + ".html.gz",
        "data/rankings/raw/" + "a" * 64 + ".html.gz.env",
    ],
)
def test_ranking_source_exception_never_broadens_raw_cache_policy(path):
    assert not release.allowed(path)

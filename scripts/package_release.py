"""Deterministic, tracked-source offline bundle; never reads working-tree payloads.

Only explicit runtime/source paths are shipped. No extraction or model loading is
performed by verification. Git blob identities and per-file SHA256 are recorded.
"""

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile

VERSION = re.compile(r"v?[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*)?\Z")
REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}\Z")
ROOT_FILES = {
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "start.sh",
    "compose.yaml",
    "LICENSE",
    "MODEL_CARD.md",
    "Dockerfile",
    "Makefile",
    "pyproject.toml",
    "requirements-lock.txt",
    "alembic.ini",
    ".env.example",
    ".dockerignore",
    ".gitattributes",
    ".gitignore",
}


def safe_path(name):
    path = PurePosixPath(name)
    return bool(
        name
        and not path.is_absolute()
        and str(path) == name
        and not any(
            part
            in (".", "..", ".git", ".secrets", "__pycache__", ".venv", "raw", ".pytest_cache", ".ruff_cache")
            for part in path.parts
        )
        and "\\" not in name
        and not any(ord(char) < 32 for char in name)
        and not any(part == ".env" or part.startswith(".env.") for part in path.parts)
        and not re.search(r"\.(?:db|sqlite|sqlite3)(?:[-.].*)?$|\.py[co]$", name, re.I)
    )


def allowed(name):
    if name == ".env.example":
        return True
    # Required by offline source verification. No generic raw-cache exception.
    if re.fullmatch(r"data/rankings/raw/[0-9a-f]{64}\.html\.gz", name):
        return True
    if not safe_path(name):
        return False
    path = PurePosixPath(name)
    if name in ROOT_FILES:
        return True
    if path.parts[0] in ("src", "tests", "migrations", "docs", "scripts"):
        return path.suffix in {".py", ".js", ".css", ".html", ".md", ".txt", ".sh", ".json", ".mako", ".svg"}
    if path.parts[0] == ".github":
        return path.suffix in {".yml", ".yaml", ".md"}
    if path.parts[0] == "artifacts":
        return path.suffix in {".joblib", ".json", ".csv"}
    if path.parts[0] == "reports":
        return (
            path.suffix in {".json", ".csv", ".md", ".jpg"}
            or name == "reports/research_predictions.csv.gz"
            or bool(re.fullmatch(r"reports/upcoming_archive/[0-9a-f]{64}\.json\.gz", name))
        )
    if name in (
        "data/ATTRIBUTION.md",
        "data/fighter_photos.json",
        "data/schedules/upcoming.json",
        "data/rankings/current.json",
        "data/rankings/current_media.json",
        "data/fighter_profiles/coverage.json",
        "data/fighter_profiles/ufc_coverage.json",
        "data/fighter_profiles/ufc_image_identity_audit.json",
    ):
        return True
    if name.startswith("data/processed/"):
        return path.suffix in {".csv", ".json"}
    if name.startswith("data/licenses/"):
        return path.suffix in {".md", ".txt"}
    # This frozen archive is required by offline ranking context, unlike raw caches.
    if name.startswith("data/rankings/history/"):
        return path.suffix in {".json", ".gz", ".md"}
    return False


def _git(repo, *arguments):
    return subprocess.check_output(["git", "-C", str(repo), *arguments], stderr=subprocess.PIPE)


def resolve_ref(repo, ref):
    if (
        not REF.fullmatch(ref)
        or any(value in ref for value in ("..", "//", "@{"))
        or ref.endswith(("/", ".", ".lock"))
    ):
        raise ValueError("Unsafe Git ref")
    try:
        commit = _git(repo, "rev-parse", "--verify", "--end-of-options", ref + "^{commit}").decode().strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError("Git ref does not resolve to a commit") from exc
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise ValueError("Invalid resolved Git commit")
    return commit


def _add(archive, name, content, mode, epoch):
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = len(content), mode, epoch
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    archive.addfile(info, io.BytesIO(content))


def package_release(repo, output_dir, version, ref="HEAD"):
    if not VERSION.fullmatch(version):
        raise ValueError("Version must be a safe semantic version, optionally prefixed v")
    repo, output_dir = Path(repo).resolve(), Path(output_dir).resolve()
    commit = resolve_ref(repo, ref)
    epoch = int(_git(repo, "show", "-s", "--format=%ct", commit).decode())
    prefix = "octogonml-" + version
    destination = output_dir / (prefix + ".tar.gz")
    checksum = output_dir / (prefix + ".tar.gz.sha256")
    output_dir.mkdir(parents=True, exist_ok=True)
    if destination.exists() or checksum.exists():
        raise FileExistsError("Release outputs already exist; choose a new output directory")
    entries = []
    for record in _git(repo, "ls-tree", "-rz", "--full-tree", commit).split(b"\0"):
        if not record:
            continue
        header, raw_name = record.split(b"\t", 1)
        name = raw_name.decode("utf-8")
        if not allowed(name):
            continue
        mode, kind, blob = header.decode().split()
        if kind != "blob" or mode not in ("100644", "100755"):
            raise ValueError("Release paths must be regular tracked files, never symlinks or submodules")
        entries.append((name, int(mode, 8) & 0o777, blob))
    entries.sort()
    if not entries:
        raise ValueError("No allowlisted tracked files in release ref")
    manifest = dict(
        format_version=1,
        version=version,
        prefix=prefix,
        source_ref=ref,
        source_commit=commit,
        source_epoch=epoch,
        files=[],
    )
    temporary = None
    created = []
    try:
        with tempfile.NamedTemporaryFile(dir=output_dir, prefix=".release-", delete=False) as target:
            temporary = Path(target.name)
            with gzip.GzipFile(fileobj=target, mode="wb", filename="", mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
                    for name, mode, blob in entries:
                        content = _git(repo, "cat-file", "blob", blob)
                        manifest["files"].append(
                            dict(
                                path=name,
                                size=len(content),
                                sha256=hashlib.sha256(content).hexdigest(),
                                mode=mode,
                                git_blob=blob,
                            )
                        )
                        _add(archive, prefix + "/" + name, content, mode, epoch)
                    _add(
                        archive,
                        prefix + "/MANIFEST.json",
                        (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode(),
                        0o644,
                        epoch,
                    )
        verify_package(temporary)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        # Hard-link creation is atomic and refuses replacement, including a race.
        os.link(temporary, destination)
        created.append(destination)
        with checksum.open("x") as stream:
            created.append(checksum)
            stream.write(f"{digest}  {destination.name}\n")
        return {
            "bundle": str(destination),
            "checksum": str(checksum),
            "sha256": digest,
            "source_commit": commit,
            "files": len(entries),
        }
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_package(path):
    """Verify archive contents without writing or executing any packaged file."""
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or not members or any(not member.isfile() for member in members):
            raise ValueError("Archive must contain unique regular files")
        manifests = [member for member in members if member.name.endswith("/MANIFEST.json")]
        if len(manifests) != 1 or manifests[0].size > 5_000_000:
            raise ValueError("Missing or invalid release manifest")
        manifest = json.load(archive.extractfile(manifests[0]))
        if (
            not isinstance(manifest, dict)
            or not {
                "format_version",
                "version",
                "prefix",
                "source_ref",
                "source_commit",
                "source_epoch",
                "files",
            }
            <= manifest.keys()
        ):
            raise ValueError("Invalid release manifest schema")
        if (
            manifest["format_version"] != 1
            or not isinstance(manifest["files"], list)
            or not re.fullmatch(r"[0-9a-f]{40,64}", str(manifest["source_commit"]))
            or type(manifest["source_epoch"]) is not int
            or manifest["source_epoch"] < 0
        ):
            raise ValueError("Invalid release manifest provenance")
        version, prefix = manifest["version"], manifest["prefix"]
        if (
            not VERSION.fullmatch(version)
            or prefix != "octogonml-" + version
            or manifests[0].name != prefix + "/MANIFEST.json"
        ):
            raise ValueError("Invalid release identity")
        declared = manifest["files"]
        if any(
            not isinstance(item, dict) or not {"path", "size", "sha256", "mode", "git_blob"} <= item.keys()
            for item in declared
        ):
            raise ValueError("Invalid release file manifest")
        paths = [item["path"] for item in declared]
        if len(paths) != len(set(paths)) or set(names) != {prefix + "/" + name for name in paths} | {
            prefix + "/MANIFEST.json"
        }:
            raise ValueError("Archive and manifest differ")
        for item in declared:
            name = item["path"]
            if not allowed(name):
                raise ValueError("Disallowed release path")
            member = archive.getmember(prefix + "/" + name)
            if (
                member.size != item["size"]
                or member.mode != item["mode"]
                or item["mode"] not in (0o644, 0o755)
            ):
                raise ValueError("Release file metadata mismatch")
            content = archive.extractfile(member).read()
            if hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise ValueError("Release file checksum mismatch")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--version")
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_package(args.verify), sort_keys=True))
    elif args.version:
        print(json.dumps(package_release(args.repo, args.output_dir, args.version, args.ref), sort_keys=True))
    else:
        parser.error("--version or --verify is required")


if __name__ == "__main__":
    main()

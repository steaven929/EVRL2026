"""Persist effective parameters and exact source identity for each run."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".toml"}
_EXCLUDED_PARTS = {
    ".git", ".idea", ".venv", "__pycache__", "output", "outputs",
    "resluts2610", "results", "results_v1",
}


def _json_safe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _write_json(path, value):
    path = Path(path)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _git(*args):
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _source_files():
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _SOURCE_SUFFIXES:
            continue
        relative = path.relative_to(PROJECT_ROOT)
        if any(part in _EXCLUDED_PARTS for part in relative.parts):
            continue
        yield relative, path


def build_source_manifest():
    files = {}
    for relative, path in sorted(_source_files(), key=lambda item: item[0].as_posix()):
        files[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "algorithm": "sha256",
        "source_file_count": len(files),
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
    }


def collect_git_version(source_manifest):
    status_text = _git("status", "--porcelain", "--untracked-files=normal") or ""
    status_lines = [line for line in status_text.splitlines() if line.strip()]
    return {
        "commit": _git("rev-parse", "HEAD") or "unknown",
        "branch": _git("branch", "--show-current") or "DETACHED_OR_UNKNOWN",
        "commit_time": _git("show", "-s", "--format=%cI", "HEAD") or "unknown",
        "describe": _git("describe", "--always", "--dirty", "--tags") or "unknown",
        "remote_origin": _git("config", "--get", "remote.origin.url") or "unknown",
        "working_tree_dirty": bool(status_lines),
        "working_tree_change_count": len(status_lines),
        "working_tree_changes": status_lines[:200],
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "source_file_count": source_manifest["source_file_count"],
    }


def write_run_provenance(output_dir, config):
    """Write effective parameters, Git identity and current source hashes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = build_source_manifest()
    effective_config = dict(config)
    effective_config["provenance_created_at"] = datetime.now(timezone.utc).isoformat()
    effective_config["python_executable"] = sys.executable
    effective_config["python_version"] = sys.version
    effective_config["command_line"] = sys.argv

    paths = {
        "effective_config": output_dir / "effective_config.json",
        "git_version": output_dir / "git_version.json",
        "source_manifest": output_dir / "source_manifest.json",
    }
    _write_json(paths["effective_config"], effective_config)
    _write_json(paths["source_manifest"], source_manifest)
    _write_json(paths["git_version"], collect_git_version(source_manifest))
    return paths

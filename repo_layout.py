from __future__ import annotations

import os
from pathlib import Path
import sys


def find_repo_root(start: Path | str | None = None) -> Path:
    current = Path(start) if start is not None else Path(__file__)
    current = current.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "research" / "pipeline").is_dir() and (
            candidate / "models"
        ).is_dir():
            return candidate
    raise RuntimeError(f"unable to locate repo root from: {start or __file__}")


REPO_ROOT = find_repo_root()

DATA_ROOT = REPO_ROOT / "data"
LVIS_RAW_ROOT = DATA_ROOT / "lvis"

THIRD_PARTY_ROOT = REPO_ROOT / "third_party"
SAM3_ROOT = THIRD_PARTY_ROOT / "sam3"
SAM3D_OBJECTS_ROOT = THIRD_PARTY_ROOT / "sam-3d-objects"

RESEARCH_ROOT = REPO_ROOT / "research"
PIPELINE_ROOT = RESEARCH_ROOT / "pipeline"
PIPELINE_OUTPUTS_ROOT = PIPELINE_ROOT / "outputs"
PIPELINE_DATASETS_ROOT = PIPELINE_OUTPUTS_ROOT / "datasets"

DOCS_ROOT = REPO_ROOT / "docs"
PIPELINE_DOCS_ROOT = DOCS_ROOT / "pipeline"

APPS_ROOT = REPO_ROOT / "apps"
WEBSITE_ROOT = APPS_ROOT / "website"
DEMO_UI_ROOT = APPS_ROOT / "demo-ui"
DEMO_DATA_ROOT = APPS_ROOT / "demo-data"

MODELS_ROOT = REPO_ROOT / "models"
DECODER_BASELINE_ROOT = MODELS_ROOT / "affordance_decoder_baseline"
DECODER_DISTILL_ROOT = MODELS_ROOT / "affordance_decoder_distill"

LEGACY_PREFIX_REWRITES = (
    ("pipeline/outputs/", "research/pipeline/outputs/"),
    ("dataset/", "data/"),
    ("outline/", "docs/pipeline/"),
    ("sam-3d-objects/", "third_party/sam-3d-objects/"),
    ("sam3/", "third_party/sam3/"),
    ("website/", "apps/website/"),
    ("demo-ui/", "apps/demo-ui/"),
    ("demo-data/", "apps/demo-data/"),
)


def canonical_repo_relpath(rel_path: str | Path) -> str:
    value = Path(rel_path).as_posix()
    if value.startswith("/"):
        return value
    for legacy_prefix, canonical_prefix in LEGACY_PREFIX_REWRITES:
        if value.startswith(legacy_prefix):
            return canonical_prefix + value[len(legacy_prefix) :]
    return value


def _extract_external_repo_suffix(path: str | Path) -> Path | None:
    value = Path(path).expanduser()
    parts = value.parts
    if not parts:
        return None
    root_name = REPO_ROOT.name
    try:
        root_index = len(parts) - 1 - list(reversed(parts)).index(root_name)
    except ValueError:
        return None
    suffix_parts = parts[root_index + 1 :]
    return Path(*suffix_parts) if suffix_parts else Path()


def _remap_external_repo_path(path: str | Path) -> Path | None:
    suffix = _extract_external_repo_suffix(path)
    if suffix is None:
        return None
    return (REPO_ROOT / suffix).resolve(strict=False)


def resolve_repo_path(rel_path: str | Path) -> Path:
    value = Path(rel_path)
    if value.is_absolute():
        if value.exists():
            return value.resolve(strict=False)
        remapped = _remap_external_repo_path(value)
        if remapped is not None:
            return remapped
        return value.resolve(strict=False)
    return (REPO_ROOT / canonical_repo_relpath(value)).resolve(strict=False)


def repo_relative(path: str | Path) -> str:
    value = Path(path)
    if not value.is_absolute():
        return canonical_repo_relpath(value)
    resolved = value.resolve(strict=False)
    try:
        rel_path = resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        remapped = _remap_external_repo_path(resolved)
        if remapped is None:
            raise
        rel_path = remapped.relative_to(REPO_ROOT).as_posix()
    return canonical_repo_relpath(rel_path)


def lvis_run_root(run_name: str) -> Path:
    return PIPELINE_DATASETS_ROOT / "lvis_real" / run_name


def showcase_run_root(run_name: str) -> Path:
    return PIPELINE_DATASETS_ROOT / "single_image_showcase" / run_name


def _existing_paths(paths: tuple[Path, ...] | list[Path]) -> tuple[Path, ...]:
    results: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            continue
        if not resolved.exists():
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        results.append(resolved)
    return tuple(results)


def _conda_root_candidates() -> tuple[Path, ...]:
    candidates: list[Path] = []

    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        prefix = Path(conda_prefix)
        candidates.append(prefix)
        if prefix.parent.name == "envs":
            candidates.append(prefix.parent.parent)

    conda_exe = os.environ.get("CONDA_EXE", "")
    if conda_exe:
        conda_exe_path = Path(conda_exe)
        if conda_exe_path.parent.name == "bin":
            candidates.append(conda_exe_path.parent.parent)

    python_exe = Path(sys.executable)
    if python_exe.parent.name == "bin":
        prefix = python_exe.parent.parent
        candidates.append(prefix)
        if prefix.parent.name == "envs":
            candidates.append(prefix.parent.parent)

    candidates.extend(
        [
            Path.home() / "miniconda3",
            Path.home() / "anaconda3",
        ]
    )
    return _existing_paths(candidates)


def find_conda_env_prefix(env_name: str) -> Path | None:
    if not env_name:
        return None

    active_prefix = os.environ.get("CONDA_PREFIX", "")
    if active_prefix:
        active_prefix_path = Path(active_prefix).expanduser().resolve()
        if active_prefix_path.name == env_name and (active_prefix_path / "bin" / "python").is_file():
            return active_prefix_path

    for candidate in _conda_root_candidates():
        if candidate.name == env_name and (candidate / "bin" / "python").is_file():
            return candidate
        prefix = candidate / "envs" / env_name
        if (prefix / "bin" / "python").is_file():
            return prefix.resolve()
    return None


def default_conda_env_path(env_name: str, *, fallback: str = "") -> str:
    prefix = find_conda_env_prefix(env_name)
    if prefix is None:
        return fallback
    return str(prefix)


def default_sam3_env_path(*, fallback: str = "") -> str:
    return default_conda_env_path("sam3", fallback=fallback)


def default_sam3d_objects_env_path(*, fallback: str = "") -> str:
    return default_conda_env_path("sam3d-objects", fallback=fallback)

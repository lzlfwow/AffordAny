#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".cff",
    ".css",
    ".html",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".svg",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN_DIRS = {"__pycache__", ".pytest_cache", "artifacts", "checkpoints", "outputs"}
FORBIDDEN_SUFFIXES = {".ckpt", ".pem", ".pth", ".pt", ".safetensors"}
IGNORED_DIRS = {".git", "dist", "node_modules", "playwright-report", "test-results"}
ABSOLUTE_MACHINE_PATHS = re.compile(r"/(?:disk|mnt)/")
EMBEDDED_SECRET = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*['\"][A-Za-z0-9_./+-]{16,}['\"]"
)
MAX_FILE_BYTES = 10 * 1024 * 1024


def iter_release_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if not any(part in IGNORED_DIRS for part in path.relative_to(ROOT).parts)
        and (path.is_file() or path.is_symlink())
    )


def check_release(strict: bool) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    required = [
        ROOT / "README.md",
        ROOT / "requirements.txt",
        ROOT / "repo_layout.py",
        ROOT / "research" / "pipeline",
        ROOT / "models" / "affordance_decoder_vlm_backbone_cosmos2b_v5",
        ROOT / "models" / "affordance_decoder_selftraining_v1",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"missing required path: {path.relative_to(ROOT)}")

    for path in iter_release_files():
        rel = path.relative_to(ROOT)
        if path.is_symlink():
            errors.append(f"symbolic link is not allowed: {rel}")
            continue
        if any(part in FORBIDDEN_DIRS for part in rel.parts):
            errors.append(f"generated directory is not allowed: {rel}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"binary model or secret file is not allowed: {rel}")
        try:
            file_size = path.stat().st_size
        except FileNotFoundError:
            # A concurrent compiler may atomically replace a temporary .pyc.
            continue
        if file_size > MAX_FILE_BYTES:
            errors.append(f"file exceeds 10 MiB: {rel}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"expected UTF-8 text file: {rel}")
            continue
        if ABSOLUTE_MACHINE_PATHS.search(content):
            errors.append(f"machine-specific absolute path found: {rel}")
        if EMBEDDED_SECRET.search(content):
            errors.append(f"possible embedded secret found: {rel}")

    metadata = [ROOT / "LICENSE", ROOT / "CITATION.cff"]
    missing_metadata = [path.name for path in metadata if not path.is_file()]
    if missing_metadata:
        message = "missing publication metadata: " + ", ".join(missing_metadata)
        if strict:
            errors.append(message)
        else:
            warnings.append(message)

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the AffordAny release tree")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also require final license and citation metadata",
    )
    args = parser.parse_args()
    errors, warnings = check_release(strict=args.strict)

    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print(f"Release check failed with {len(errors)} error(s).")
        return 1
    print("Release check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

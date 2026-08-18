from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_cache_config(cache_root: str | Path) -> dict[str, Any]:
    path = Path(cache_root) / "config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def validate_cache_config(
    cache_root: str | Path,
    *,
    vlm_hidden_size: int,
    num_views: int,
    grid_size: int,
) -> dict[str, Any]:
    config = load_cache_config(cache_root)
    if not config:
        return {}

    expected_tokens = num_views * grid_size * grid_size
    mismatches = []
    if int(config.get("hidden_size", vlm_hidden_size)) != int(vlm_hidden_size):
        mismatches.append(
            f"hidden_size={config.get('hidden_size')} != vlm_hidden_size={vlm_hidden_size}"
        )
    if int(config.get("num_views", num_views)) != int(num_views):
        mismatches.append(f"num_views={config.get('num_views')} != num_views={num_views}")
    if int(config.get("total_visual_tokens", expected_tokens)) != int(expected_tokens):
        mismatches.append(
            "total_visual_tokens="
            f"{config.get('total_visual_tokens')} != num_views*grid_size^2={expected_tokens}"
        )
    if mismatches:
        joined = "; ".join(mismatches)
        raise ValueError(f"VLM cache config mismatch in {cache_root}: {joined}")
    return config


def save_json(path: str | Path, payload: Any) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

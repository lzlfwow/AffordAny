from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode

import requests


def _slugify(value: str) -> str:
    return "_".join(value.strip().lower().split()) or "unknown"


def _humanize_slug(value: str) -> str:
    return " ".join(value.strip().replace("_", " ").split()) or "unknown"


def build_base_part_prompt(category_name: str, part_name: str) -> str:
    object_text = _humanize_slug(category_name)
    part_text = _humanize_slug(part_name)
    return f"the {part_text} of the {object_text}"


@dataclass(frozen=True)
class PartPromptConfig:
    model_name: str = "gemini-3.1-pro-preview"
    max_parts: int = 3
    candidate_max_parts: int = 5
    prompt_filename: str = "part_prompts.json"
    consistency_screen_enabled: bool = False
    consistency_num_render_views: int = 6
    base_url_env: str = "GEMINI_BASE_URL"
    api_key_env: str = "GEMINI_API_KEY"
    fallback_api_key_env: str = "GOOGLE_API_KEY"
    endpoint_path_template: str = "/v1beta/models/{model}:generateContent"
    fallback_base_url: str = "https://aiplatform.googleapis.com"
    fallback_endpoint_path_template: str = "/v1/publishers/google/models/{model}:generateContent"
    timeout_seconds: int = 60

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PartPromptItem:
    part_name: str
    prompt: str
    base_prompt: str | None = None

    def to_dict(self) -> dict:
        return {
            "part_name": self.part_name,
            "prompt": self.prompt,
            "base_prompt": self.base_prompt or self.prompt,
        }


@dataclass(frozen=True)
class PartPromptRecord:
    object_id: str
    object_name: str
    model_name: str
    parts: tuple[PartPromptItem, ...]

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "object_name": self.object_name,
            "model_name": self.model_name,
            "parts": [part.to_dict() for part in self.parts],
        }


@dataclass(frozen=True)
class PartPromptExecutionResult:
    object_id: str
    object_name: str
    prompt_path: str
    num_parts: int
    model_name: str

    def to_dict(self) -> dict:
        return asdict(self)


class PartPromptInferenceClient(Protocol):
    def infer(
        self,
        *,
        category_name: str,
        config: "PartPromptConfig",
        object_dir: str | Path | None = None,
    ) -> PartPromptRecord:
        ...


def build_part_prompt_generation_prompt(category_name: str, max_parts: int = 5) -> str:
    return (
        "You are generating part prompts for a 3D interaction dataset. "
        "Focus only on interactive or operable parts. "
        f"Return at most {max_parts} parts for the object category '{category_name}'. "
        "If the category has no stable operable parts, return an empty parts list. "
        "Use English only. "
        "Return JSON only in the form "
        '{"parts":[{"part_name":"...","prompt":"..."}]}. '
        "Each prompt should use the exact pattern "
        f"'{build_base_part_prompt(category_name, 'seat')}'. "
        "For example, use 'the seat of the chair' instead of 'chair seat'."
    )


def build_part_prompt_payload(
    config: PartPromptConfig,
    *,
    category_name: str,
) -> dict[str, Any]:
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": build_part_prompt_generation_prompt(
                            category_name,
                            max_parts=config.candidate_max_parts,
                        )
                    }
                ]
            }
        ],
        "generationConfig": {"responseMimeType": "application/json"},
    }


def build_part_prompt_consistency_prompt(
    category_name: str,
    *,
    candidate_parts: tuple[PartPromptItem, ...],
) -> str:
    candidate_payload = json.dumps(
        [item.to_dict() for item in candidate_parts],
        ensure_ascii=True,
    )
    return (
        "You are verifying part prompts for a 3D interaction dataset. "
        "The images show the specific object instance and its rendered views. "
        "Keep only parts that are visually consistent with this specific instance or clearly supported by the rendered structure. "
        "It is valid to return zero parts if none are reliable for this instance. "
        "Do not keep speculative parts that are common for the category but not visible or not supported here. "
        f"Object category: '{category_name}'. "
        f"Candidate parts: {candidate_payload}. "
        'Return JSON only in the form {"parts":[{"part_name":"...","prompt":"..."}]}.'
    )


def _image_to_base64(image_path: str | Path) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def _guess_mime_type(image_path: str | Path) -> str:
    suffix = Path(image_path).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "image/png"


def collect_consistency_evidence_paths(
    object_dir: str | Path,
    *,
    num_render_views: int,
) -> tuple[str, ...]:
    object_dir = Path(object_dir)
    candidates: list[Path] = []
    for relpath in [
        Path("candidate") / "masked_instance.png",
        Path("source") / "image.png",
    ]:
        path = object_dir / relpath
        if path.is_file():
            candidates.append(path)
    render_dir = object_dir / "render"
    candidates.extend(sorted(render_dir.glob("view_*.png"))[:num_render_views])
    deduped: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        path_str = str(path)
        if path_str in seen:
            continue
        seen.add(path_str)
        deduped.append(path_str)
    return tuple(deduped)


def build_part_prompt_consistency_payload(
    config: PartPromptConfig,
    *,
    category_name: str,
    candidate_parts: tuple[PartPromptItem, ...],
    evidence_image_paths: tuple[str, ...],
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    for image_path in evidence_image_paths:
        parts.append(
            {
                "inline_data": {
                    "mime_type": _guess_mime_type(image_path),
                    "data": _image_to_base64(image_path),
                }
            }
        )
    parts.append(
        {
            "text": build_part_prompt_consistency_prompt(
                category_name,
                candidate_parts=candidate_parts,
            )
        }
    )
    return {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"responseMimeType": "application/json"},
    }


def normalize_part_prompt_items(
    items: list[dict[str, Any]],
    *,
    category_name: str,
    max_parts: int,
) -> tuple[PartPromptItem, ...]:
    normalized: list[PartPromptItem] = []
    seen: set[str] = set()
    for item in items:
        part_name = str(item.get("part_name", "")).strip()
        if not part_name:
            continue
        slug = _slugify(part_name)
        if slug in seen:
            continue
        seen.add(slug)
        base_prompt = build_base_part_prompt(category_name, slug)
        normalized.append(
            PartPromptItem(
                part_name=slug,
                prompt=base_prompt,
                base_prompt=base_prompt,
            )
        )
        if len(normalized) >= max_parts:
            break
    return tuple(normalized)


def canonicalize_part_prompt_items(
    items: tuple[PartPromptItem, ...],
    *,
    category_name: str,
    max_parts: int,
) -> tuple[PartPromptItem, ...]:
    normalized: list[PartPromptItem] = []
    seen: set[str] = set()
    for item in items:
        slug = _slugify(item.part_name)
        if slug in seen:
            continue
        seen.add(slug)
        base_prompt = build_base_part_prompt(category_name, slug)
        normalized.append(
            PartPromptItem(
                part_name=slug,
                prompt=base_prompt,
                base_prompt=base_prompt,
            )
        )
        if len(normalized) >= max_parts:
            break
    return tuple(normalized)


def parse_part_prompt_response(
    response_json: dict[str, Any],
    *,
    category_name: str,
    max_parts: int,
) -> PartPromptRecord:
    parsed = _parse_json_text(_extract_text_from_gemini_response(response_json))
    if isinstance(parsed, list):
        if not parsed or not isinstance(parsed[0], dict):
            raise ValueError("response JSON list must contain an object")
        parsed = parsed[0]
    part_items = parsed.get("parts", [])
    normalized = normalize_part_prompt_items(
        list(part_items),
        category_name=category_name,
        max_parts=max_parts,
    )
    return PartPromptRecord(
        object_id="",
        object_name=category_name,
        model_name="",
        parts=normalized,
    )


class GeminiPartPromptClient:
    def _resolve_endpoint(self, config: PartPromptConfig) -> tuple[str, str, str]:
        base_url = os.environ.get(config.base_url_env, "").rstrip("/")
        api_key = os.environ.get(config.api_key_env, "")
        if base_url and api_key:
            return base_url, api_key, config.endpoint_path_template
        fallback_api_key = os.environ.get(config.fallback_api_key_env, "")
        if fallback_api_key:
            return (
                config.fallback_base_url.rstrip("/"),
                fallback_api_key,
                config.fallback_endpoint_path_template,
            )
        if base_url and not api_key:
            raise RuntimeError(f"missing env var: {config.api_key_env}")
        if api_key and not base_url:
            raise RuntimeError(f"missing env var: {config.base_url_env}")
        raise RuntimeError(
            f"missing env vars: ({config.base_url_env}, {config.api_key_env}) "
            f"or {config.fallback_api_key_env}"
        )

    def build_request(
        self,
        *,
        category_name: str,
        config: PartPromptConfig,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        base_url, api_key, endpoint_template = self._resolve_endpoint(config)
        query = urlencode({"key": api_key})
        endpoint = endpoint_template.format(model=config.model_name)
        url = f"{base_url}{endpoint}?{query}"
        headers = {"Content-Type": "application/json"}
        payload = build_part_prompt_payload(config, category_name=category_name)
        return url, headers, payload

    def build_consistency_request(
        self,
        *,
        category_name: str,
        config: PartPromptConfig,
        candidate_parts: tuple[PartPromptItem, ...],
        evidence_image_paths: tuple[str, ...],
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        base_url, api_key, endpoint_template = self._resolve_endpoint(config)
        query = urlencode({"key": api_key})
        endpoint = endpoint_template.format(model=config.model_name)
        url = f"{base_url}{endpoint}?{query}"
        headers = {"Content-Type": "application/json"}
        payload = build_part_prompt_consistency_payload(
            config,
            category_name=category_name,
            candidate_parts=candidate_parts,
            evidence_image_paths=evidence_image_paths,
        )
        return url, headers, payload

    def infer(
        self,
        *,
        category_name: str,
        config: PartPromptConfig,
        object_dir: str | Path | None = None,
    ) -> PartPromptRecord:
        url, headers, payload = self.build_request(category_name=category_name, config=config)
        response = requests.post(url, headers=headers, json=payload, timeout=config.timeout_seconds)
        response.raise_for_status()
        parsed = parse_part_prompt_response(
            response.json(),
            category_name=category_name,
            max_parts=config.candidate_max_parts,
        )
        candidate_record = PartPromptRecord(
            object_id="",
            object_name=parsed.object_name,
            model_name=config.model_name,
            parts=parsed.parts,
        )
        return candidate_record


def _extract_text_from_gemini_response(response_json: dict[str, Any]) -> str:
    try:
        content = response_json["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unsupported response structure") from exc

    text_chunks = []
    for item in content:
        if isinstance(item, dict) and "text" in item:
            text_chunks.append(item["text"])
    content_text = "".join(text_chunks).strip()
    if not content_text:
        raise ValueError("response content must include text")
    return content_text


def _parse_json_text(content_text: str) -> dict[str, Any]:
    cleaned = content_text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 2:
            cleaned = parts[1]
        cleaned = cleaned.removeprefix("json").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        parsed = None
        for start_char in ("{", "["):
            start_idx = cleaned.find(start_char)
            if start_idx < 0:
                continue
            try:
                parsed, _ = decoder.raw_decode(cleaned[start_idx:])
                break
            except json.JSONDecodeError:
                continue
        if parsed is None:
            raise

    if isinstance(parsed, list):
        if not parsed or not isinstance(parsed[0], dict):
            raise ValueError("response JSON list must contain an object")
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        raise ValueError("response JSON must decode to an object")
    return parsed


def export_part_prompts(
    object_dir: str | Path,
    *,
    object_id: str,
    object_name: str,
    config: PartPromptConfig,
    parts: tuple[PartPromptItem, ...],
) -> str:
    prompt_dir = Path(object_dir) / "prompt"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    canonical_parts = canonicalize_part_prompt_items(
        parts,
        category_name=object_name,
        max_parts=config.candidate_max_parts,
    )
    record = PartPromptRecord(
        object_id=object_id,
        object_name=object_name,
        model_name=config.model_name,
        parts=canonical_parts,
    )
    path = prompt_dir / config.prompt_filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record.to_dict(), handle, indent=2, ensure_ascii=True)
    return str(path)


def run_part_prompt_generation(
    object_dir: str | Path,
    *,
    object_id: str,
    object_name: str,
    config: PartPromptConfig,
    client: PartPromptInferenceClient,
) -> PartPromptExecutionResult:
    record = client.infer(category_name=object_name, config=config, object_dir=object_dir)
    path = export_part_prompts(
        object_dir,
        object_id=object_id,
        object_name=object_name,
        config=config,
        parts=record.parts,
    )
    return PartPromptExecutionResult(
        object_id=object_id,
        object_name=object_name,
        prompt_path=path,
        num_parts=len(record.parts),
        model_name=config.model_name,
    )


def execute_part_prompt_generation(
    object_dir: str | Path,
    *,
    object_id: str,
    object_name: str,
    config: PartPromptConfig | None = None,
    client: PartPromptInferenceClient | None = None,
) -> PartPromptExecutionResult:
    config = config or PartPromptConfig()
    client = client or GeminiPartPromptClient()
    return run_part_prompt_generation(
        object_dir,
        object_id=object_id,
        object_name=object_name,
        config=config,
        client=client,
    )

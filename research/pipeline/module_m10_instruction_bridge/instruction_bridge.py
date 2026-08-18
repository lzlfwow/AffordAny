from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Protocol
from urllib.parse import urlencode

import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_layout import PIPELINE_DATASETS_ROOT, repo_relative, resolve_repo_path


DEFAULT_STAGE2_ROOT = PIPELINE_DATASETS_ROOT / "lvis_real" / "full_dataset_rerun_v3" / "stage2_dataset"


@dataclass(frozen=True)
class InstructionBridgeConfig:
    model_name: str = "gemini-3.1-pro-preview"
    base_url_env: str = "GEMINI_BASE_URL"
    api_key_env: str = "GEMINI_API_KEY"
    fallback_api_key_env: str = "GOOGLE_API_KEY"
    endpoint_path_template: str = "/v1beta/models/{model}:generateContent"
    timeout_seconds: int = 120
    instructions_dirname: str = "instruction"
    instructions_filename: str = "part_instructions.json"
    summary_filename: str = "instruction_bridge_summary.json"
    instructions_per_part: int = 3
    max_instruction_words: int = 14
    require_visualization_dir: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PartInstructionSeed:
    part_sample_id: str
    part_name: str
    prompt: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PartInstructionItem:
    part_sample_id: str
    part_name: str
    prompt: str
    instructions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "part_sample_id": self.part_sample_id,
            "part_name": self.part_name,
            "prompt": self.prompt,
            "instructions": list(self.instructions),
        }


@dataclass(frozen=True)
class InstructionBridgeRequest:
    object_id: str
    object_name: str
    object_dir: str
    part_instances_path: str
    output_path: str
    evidence_image_paths: tuple[str, ...]
    parts: tuple[PartInstructionSeed, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "object_name": self.object_name,
            "object_dir": self.object_dir,
            "part_instances_path": self.part_instances_path,
            "output_path": self.output_path,
            "evidence_image_paths": list(self.evidence_image_paths),
            "parts": [item.to_dict() for item in self.parts],
        }


@dataclass(frozen=True)
class InstructionBridgeRecord:
    object_id: str
    object_name: str
    model_name: str
    instructions_per_part: int
    evidence_image_paths: tuple[str, ...]
    part_instances_path: str
    parts: tuple[PartInstructionItem, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "object_name": self.object_name,
            "model_name": self.model_name,
            "instructions_per_part": self.instructions_per_part,
            "evidence_image_paths": list(self.evidence_image_paths),
            "part_instances_path": self.part_instances_path,
            "parts": [item.to_dict() for item in self.parts],
        }


@dataclass(frozen=True)
class InstructionBridgeExecutionResult:
    object_id: str
    object_name: str
    status: str
    output_path: str
    model_name: str
    num_parts: int
    total_instructions: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InstructionBridgeInferenceClient(Protocol):
    def infer(
        self,
        *,
        request: InstructionBridgeRequest,
        config: InstructionBridgeConfig,
    ) -> InstructionBridgeRecord:
        ...


def _slugify(value: str) -> str:
    return "_".join(value.strip().lower().split()) or "unknown"


def _humanize_slug(value: str) -> str:
    return " ".join(value.strip().replace("_", " ").split()) or "unknown"


def _guess_mime_type(image_path: str | Path) -> str:
    suffix = Path(image_path).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "image/png"


def _image_to_base64(image_path: str | Path) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def _normalize_instruction_text(value: str) -> str:
    text = " ".join(str(value).strip().split())
    text = text.lstrip("-*")
    text = text.removeprefix("1.").removeprefix("2.").removeprefix("3.").strip()
    return text.strip(' \t\n\r\"')


def build_instruction_bridge_prompt(
    object_name: str,
    *,
    parts: tuple[PartInstructionSeed, ...],
    instructions_per_part: int,
    max_instruction_words: int,
) -> str:
    object_text = _humanize_slug(object_name)
    part_lines = "\n".join(
        f"- part_name: {item.part_name}; grounding_prompt: {item.prompt}"
        for item in parts
    )
    return (
        "You are writing English spoken commands for a human-to-robot interaction dataset used to train a text-conditioned affordance decoder. "
        "Downstream use case: a person speaks to a robot, and the robot must infer which object region it should grasp, touch, pull, press, hold, or manipulate. "
        "You are given one masked object image showing the target object instance and a list of target parts that already have valid 3D labels. "
        f"For each listed part, write exactly {instructions_per_part} short natural commands that a person could say to a household robot so that the listed part is the correct contact or manipulation region. "
        "Each command should sound like a real user request, not like dataset metadata or a part definition. "
        "The wording does not need to reuse the exact part name. Prefer realistic user language, including functional descriptions, everyday synonyms, and implicit references, as long as the target part remains the unique best match. It is still acceptable for some commands to mention the part directly when that sounds natural, but do not force exact label wording into every command. "
        "Within the same object, sibling parts must be clearly distinguishable from one another. "
        "Do not produce near-duplicate commands across different listed parts. "
        "Choose verbs, functional cues, and descriptions that make this part more likely than any sibling part on the same object. "
        "If two parts are similar, deliberately emphasize what would disambiguate them in a real interaction request. "
        "When sibling parts are easy to confuse, use mutually exclusive cues instead of overlapping generic wording. "
        "Do not assign the same vague words such as 'top', 'cover', 'open', or 'handle' to multiple sibling parts unless that wording is unavoidable and still clearly disambiguated. "
        "Before writing, compare all listed parts and make sure each part gets its own distinct interaction profile. "
        "Across the 3 commands for one part, try to cover different natural phrasings instead of minor rewrites of the same sentence. "
        "Favor interaction-oriented language such as handing, opening, pressing, pulling, holding, or grabbing when it fits the part. "
        "Do not mention coordinates, camera directions, image positions, or invisible context. "
        "Avoid vague references like 'this thing', 'that area', or commands that target the whole object when the listed part should be selected. "
        f"Keep each instruction concise, ideally no more than {max_instruction_words} words. "
        "Use English only. Return JSON only in the form "
        '{"parts":[{"part_name":"...","instructions":["...","...","..."]}]}. '
        "In the JSON output, use exactly the provided part_name values as identifiers, keep the same part coverage, and do not omit any listed part. This identifier requirement applies only to the JSON keys, not to the wording of the spoken commands themselves. "
        f"Object category: '{object_text}'.\n"
        "Target parts:\n"
        f"{part_lines}"
    )


def build_instruction_bridge_payload(
    config: InstructionBridgeConfig,
    *,
    object_name: str,
    parts: tuple[PartInstructionSeed, ...],
    evidence_image_paths: tuple[str, ...],
) -> dict[str, Any]:
    payload_parts: list[dict[str, Any]] = []
    for image_path in evidence_image_paths:
        payload_parts.append(
            {
                "inline_data": {
                    "mime_type": _guess_mime_type(image_path),
                    "data": _image_to_base64(image_path),
                }
            }
        )
    payload_parts.append(
        {
            "text": build_instruction_bridge_prompt(
                object_name,
                parts=parts,
                instructions_per_part=config.instructions_per_part,
                max_instruction_words=config.max_instruction_words,
            )
        }
    )
    return {
        "contents": [{"parts": payload_parts}],
        "generationConfig": {"responseMimeType": "application/json"},
    }


def _parse_json_text_from_gemini_response(response_json: dict[str, Any]) -> dict[str, Any]:
    try:
        content = response_json["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unsupported response structure") from exc

    text_chunks: list[str] = []
    for item in content:
        if isinstance(item, dict) and "text" in item:
            text_chunks.append(item["text"])
    content_text = "".join(text_chunks).strip()
    if not content_text:
        raise ValueError("response content must include text")

    if content_text.startswith("```"):
        blocks = content_text.split("```")
        if len(blocks) >= 2:
            content_text = blocks[1]
        content_text = content_text.removeprefix("json").strip()

    def _loads_first_json_object(text: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        stripped = text.lstrip()
        parsed_object, _end_index = decoder.raw_decode(stripped)
        if isinstance(parsed_object, list):
            if not parsed_object or not isinstance(parsed_object[0], dict):
                raise ValueError("response JSON list must contain an object")
            parsed_object = parsed_object[0]
        if not isinstance(parsed_object, dict):
            raise ValueError("response JSON must decode to an object")
        return parsed_object

    try:
        parsed = _loads_first_json_object(content_text)
    except json.JSONDecodeError:
        start = content_text.find("{")
        end = content_text.rfind("}")
        if start < 0 or end < 0 or end <= start:
            raise
        parsed = _loads_first_json_object(content_text[start : end + 1])
    return parsed


def _fallback_instructions(
    object_name: str,
    part_name: str,
    prompt: str,
    *,
    count: int,
) -> tuple[str, ...]:
    object_text = _humanize_slug(object_name)
    part_text = _humanize_slug(part_name)
    candidates = [
        f"Grab the {object_text} by the {part_text}.",
        f"Use the {part_text} on the {object_text}.",
        f"Reach for the {part_text} of the {object_text}.",
        f"Hold the {object_text} at the {part_text}.",
        f"Move to the {prompt}.",
    ]
    return tuple(candidates[:count])


def _complete_instruction_set(
    instructions: list[str],
    *,
    object_name: str,
    part_name: str,
    prompt: str,
    instructions_per_part: int,
) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in instructions:
        cleaned = _normalize_instruction_text(item)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
        if len(normalized) >= instructions_per_part:
            return tuple(normalized)
    for fallback in _fallback_instructions(
        object_name,
        part_name,
        prompt,
        count=instructions_per_part,
    ):
        key = fallback.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(fallback)
        if len(normalized) >= instructions_per_part:
            break
    return tuple(normalized[:instructions_per_part])


def parse_instruction_bridge_response(
    response_json: dict[str, Any],
    *,
    request: InstructionBridgeRequest,
    config: InstructionBridgeConfig,
) -> InstructionBridgeRecord:
    parsed = _parse_json_text_from_gemini_response(response_json)
    raw_parts = parsed.get("parts", [])
    if not isinstance(raw_parts, list):
        raise ValueError("response JSON must contain a parts list")

    by_slug: dict[str, dict[str, Any]] = {}
    for item in raw_parts:
        if not isinstance(item, dict):
            continue
        part_name = _slugify(str(item.get("part_name", "")))
        if not part_name or part_name in by_slug:
            continue
        by_slug[part_name] = item

    normalized_parts: list[PartInstructionItem] = []
    for seed in request.parts:
        payload = by_slug.get(_slugify(seed.part_name), {})
        raw_instructions = payload.get("instructions", [])
        if not isinstance(raw_instructions, list):
            raw_instructions = []
        normalized_parts.append(
            PartInstructionItem(
                part_sample_id=seed.part_sample_id,
                part_name=seed.part_name,
                prompt=seed.prompt,
                instructions=_complete_instruction_set(
                    [str(item) for item in raw_instructions],
                    object_name=request.object_name,
                    part_name=seed.part_name,
                    prompt=seed.prompt,
                    instructions_per_part=config.instructions_per_part,
                ),
            )
        )

    return InstructionBridgeRecord(
        object_id=request.object_id,
        object_name=request.object_name,
        model_name=config.model_name,
        instructions_per_part=config.instructions_per_part,
        evidence_image_paths=request.evidence_image_paths,
        part_instances_path=request.part_instances_path,
        parts=tuple(normalized_parts),
    )


def _relative_to_object(object_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(object_dir.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _safe_repo_relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return repo_relative(resolved)
    except ValueError:
        return str(resolved)


def collect_instruction_bridge_evidence_paths(
    object_dir: str | Path,
    *,
    config: InstructionBridgeConfig,
) -> tuple[str, ...]:
    object_dir = Path(object_dir)
    candidate_path = object_dir / "candidate" / "masked_instance.png"
    if not candidate_path.is_file():
        return tuple()
    return (str(candidate_path.resolve()),)


def _build_default_part_prompt(object_name: str, part_name: str) -> str:
    return f"the {_humanize_slug(part_name)} of the {_humanize_slug(object_name)}"


def object_has_nonempty_part_instances(object_dir: str | Path) -> bool:
    part_instances_path = Path(object_dir) / "package" / "part_instances.json"
    if not part_instances_path.is_file():
        return False
    try:
        payload = json.loads(part_instances_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return isinstance(payload, list) and any(isinstance(item, dict) for item in payload)


def build_instruction_bridge_request(
    object_dir: str | Path,
    *,
    config: InstructionBridgeConfig | None = None,
) -> InstructionBridgeRequest:
    config = config or InstructionBridgeConfig()
    object_dir = Path(object_dir).resolve()
    source_meta_path = object_dir / "source" / "source_meta.json"
    part_instances_path = object_dir / "package" / "part_instances.json"
    if not source_meta_path.is_file():
        raise FileNotFoundError(f"missing required input: {source_meta_path}")
    if not part_instances_path.is_file():
        raise FileNotFoundError(f"missing required input: {part_instances_path}")

    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
    part_instances = json.loads(part_instances_path.read_text(encoding="utf-8"))
    object_name = str(source_meta["category_name"])
    parts: list[PartInstructionSeed] = []
    for item in part_instances:
        if not isinstance(item, dict):
            continue
        part_name = _slugify(str(item.get("part_name", "")))
        if not part_name:
            continue
        part_sample_id = str(item.get("part_sample_id") or f"{object_name}__{object_dir.name}__{part_name}")
        prompt = str(item.get("prompt") or _build_default_part_prompt(object_name, part_name))
        parts.append(
            PartInstructionSeed(
                part_sample_id=part_sample_id,
                part_name=part_name,
                prompt=prompt,
            )
        )

    evidence_image_paths = collect_instruction_bridge_evidence_paths(object_dir, config=config)
    if not evidence_image_paths:
        raise FileNotFoundError(
            f"missing required input: {object_dir / 'candidate' / 'masked_instance.png'}"
        )
    output_path = object_dir / config.instructions_dirname / config.instructions_filename
    return InstructionBridgeRequest(
        object_id=object_dir.name,
        object_name=object_name,
        object_dir=str(object_dir),
        part_instances_path=_relative_to_object(object_dir, part_instances_path),
        output_path=str(output_path),
        evidence_image_paths=tuple(
            _relative_to_object(object_dir, Path(path)) for path in evidence_image_paths
        ),
        parts=tuple(parts),
    )


class GeminiInstructionBridgeClient:
    def get_base_url(self, config: InstructionBridgeConfig) -> str:
        value = os.environ.get(config.base_url_env, "").rstrip("/")
        if not value:
            raise RuntimeError(f"missing env var: {config.base_url_env}")
        return value

    def get_api_key(self, config: InstructionBridgeConfig) -> str:
        value = os.environ.get(config.api_key_env, "")
        if value:
            return value
        fallback = os.environ.get(config.fallback_api_key_env, "")
        if fallback:
            return fallback
        raise RuntimeError(
            f"missing env vars: {config.api_key_env} or {config.fallback_api_key_env}"
        )

    def build_request(
        self,
        *,
        request: InstructionBridgeRequest,
        config: InstructionBridgeConfig,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        query = urlencode({"key": self.get_api_key(config)})
        url = f"{self.get_base_url(config)}{config.endpoint_path_template.format(model=config.model_name)}?{query}"
        headers = {"Content-Type": "application/json"}
        object_dir = Path(request.object_dir)
        evidence_abs = tuple(str((object_dir / path).resolve()) for path in request.evidence_image_paths)
        payload = build_instruction_bridge_payload(
            config,
            object_name=request.object_name,
            parts=request.parts,
            evidence_image_paths=evidence_abs,
        )
        return url, headers, payload

    def infer(
        self,
        *,
        request: InstructionBridgeRequest,
        config: InstructionBridgeConfig,
    ) -> InstructionBridgeRecord:
        url, headers, payload = self.build_request(request=request, config=config)
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=config.timeout_seconds,
        )
        response.raise_for_status()
        return parse_instruction_bridge_response(
            response.json(),
            request=request,
            config=config,
        )


def export_instruction_bridge_record(
    request: InstructionBridgeRequest,
    *,
    config: InstructionBridgeConfig,
    record: InstructionBridgeRecord,
) -> str:
    output_path = Path(request.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(record.to_dict(), handle, indent=2, ensure_ascii=True)
    return str(output_path)


def run_instruction_bridge_generation(
    object_dir: str | Path,
    *,
    config: InstructionBridgeConfig,
    client: InstructionBridgeInferenceClient,
    overwrite: bool = False,
) -> InstructionBridgeExecutionResult:
    request = build_instruction_bridge_request(object_dir, config=config)
    if not request.parts:
        return InstructionBridgeExecutionResult(
            object_id=request.object_id,
            object_name=request.object_name,
            status="skipped_no_parts",
            output_path=request.output_path,
            model_name=config.model_name,
            num_parts=0,
            total_instructions=0,
        )
    output_path = Path(request.output_path)
    if output_path.is_file() and not overwrite:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        num_parts = len(payload.get("parts", []))
        total_instructions = sum(len(item.get("instructions", [])) for item in payload.get("parts", []))
        return InstructionBridgeExecutionResult(
            object_id=request.object_id,
            object_name=request.object_name,
            status="skipped",
            output_path=str(output_path),
            model_name=str(payload.get("model_name", config.model_name)),
            num_parts=num_parts,
            total_instructions=total_instructions,
        )

    record = client.infer(request=request, config=config)
    written_path = export_instruction_bridge_record(request, config=config, record=record)
    return InstructionBridgeExecutionResult(
        object_id=request.object_id,
        object_name=request.object_name,
        status="completed",
        output_path=written_path,
        model_name=config.model_name,
        num_parts=len(record.parts),
        total_instructions=sum(len(item.instructions) for item in record.parts),
    )


def execute_instruction_bridge_generation(
    object_dir: str | Path,
    *,
    config: InstructionBridgeConfig | None = None,
    client: InstructionBridgeInferenceClient | None = None,
    overwrite: bool = False,
) -> InstructionBridgeExecutionResult:
    config = config or InstructionBridgeConfig()
    client = client or GeminiInstructionBridgeClient()
    return run_instruction_bridge_generation(
        object_dir,
        config=config,
        client=client,
        overwrite=overwrite,
    )


def object_has_visualization(object_dir: str | Path) -> bool:
    return (Path(object_dir) / "visualization").is_dir()


def list_stage2_object_dirs(stage2_root: str | Path) -> tuple[Path, ...]:
    stage2_root = Path(stage2_root).resolve()
    dataset_index_candidates = [
        stage2_root / "dataset_index.json",
        stage2_root / "aggregated_stage2_dataset_for_baseline" / "dataset_index.json",
    ]
    for dataset_index_path in dataset_index_candidates:
        if not dataset_index_path.is_file():
            continue
        payload = json.loads(dataset_index_path.read_text(encoding="utf-8"))
        object_dirs: list[Path] = []
        for item in payload.get("objects", []):
            if not isinstance(item, dict):
                continue
            object_dir_value = item.get("object_dir")
            if not object_dir_value:
                continue
            object_dir = resolve_repo_path(object_dir_value)
            if object_has_visualization(object_dir) and object_has_nonempty_part_instances(object_dir):
                object_dirs.append(object_dir)
        if object_dirs:
            return tuple(sorted(set(object_dirs)))

    direct_objects_root = stage2_root / "objects"
    if direct_objects_root.is_dir():
        direct_objects = [
            path
            for path in direct_objects_root.glob("*")
            if path.is_dir() and object_has_visualization(path) and object_has_nonempty_part_instances(path)
        ]
        if direct_objects:
            return tuple(sorted(direct_objects))

    aggregate_objects_root = stage2_root / "aggregated_stage2_dataset_for_baseline" / "objects"
    if aggregate_objects_root.is_dir():
        aggregate_objects = [
            path
            for path in aggregate_objects_root.glob("*")
            if path.is_dir() and object_has_visualization(path) and object_has_nonempty_part_instances(path)
        ]
        if aggregate_objects:
            return tuple(sorted(aggregate_objects))

    worker_objects = [
        path
        for objects_root in stage2_root.glob("workers/*/stage2_dataset/objects")
        for path in objects_root.glob("*")
        if path.is_dir() and object_has_visualization(path) and object_has_nonempty_part_instances(path)
    ]
    return tuple(sorted(worker_objects))


def execute_instruction_bridge_dataset(
    stage2_root: str | Path,
    *,
    config: InstructionBridgeConfig | None = None,
    client: InstructionBridgeInferenceClient | None = None,
    object_ids: tuple[str, ...] = (),
    start_index: int = 0,
    limit: int | None = None,
    overwrite: bool = False,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    config = config or InstructionBridgeConfig()
    client = client or GeminiInstructionBridgeClient()
    stage2_root = Path(stage2_root).resolve()
    object_dirs = list(list_stage2_object_dirs(stage2_root))
    if object_ids:
        wanted = set(object_ids)
        object_dirs = [path for path in object_dirs if path.name in wanted]
    if start_index > 0:
        object_dirs = object_dirs[start_index:]
    if limit is not None:
        object_dirs = object_dirs[:limit]

    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for object_dir in object_dirs:
        try:
            result = execute_instruction_bridge_generation(
                object_dir,
                config=config,
                client=client,
                overwrite=overwrite,
            )
        except Exception as exc:  # noqa: BLE001
            failed.append({"object_id": object_dir.name, "error": str(exc)})
            continue
        target = skipped if result.status == "skipped" else completed
        target.append(result.to_dict())

    summary = {
        "stage": "m10_instruction_bridge",
        "stage2_root": _safe_repo_relative(stage2_root),
        "config": config.to_dict(),
        "num_requested_objects": len(object_dirs),
        "num_completed": len(completed),
        "num_skipped": len(skipped),
        "num_failed": len(failed),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
    }
    resolved_summary_path = Path(summary_path) if summary_path is not None else stage2_root / config.summary_filename
    resolved_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    summary["summary_path"] = str(resolved_summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate colloquial part-level instructions for stage2 dataset objects.")
    parser.add_argument("--stage2-root", type=Path, default=DEFAULT_STAGE2_ROOT)
    parser.add_argument("--object-id", action="append", default=[])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--model-name", default=InstructionBridgeConfig.model_name)
    parser.add_argument("--instructions-per-part", type=int, default=InstructionBridgeConfig.instructions_per_part)
    parser.add_argument("--timeout-seconds", type=int, default=InstructionBridgeConfig.timeout_seconds)
    parser.add_argument("--max-instruction-words", type=int, default=InstructionBridgeConfig.max_instruction_words)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = InstructionBridgeConfig(
        model_name=args.model_name,
        timeout_seconds=args.timeout_seconds,
        instructions_per_part=args.instructions_per_part,
        max_instruction_words=args.max_instruction_words,
    )
    summary = execute_instruction_bridge_dataset(
        args.stage2_root,
        config=config,
        object_ids=tuple(args.object_id),
        start_index=args.start_index,
        limit=args.limit,
        overwrite=args.overwrite,
        summary_path=args.summary_path,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

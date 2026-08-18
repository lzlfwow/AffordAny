from __future__ import annotations

from dataclasses import asdict, dataclass
import base64
import json
import os
from pathlib import Path
import time
from typing import Any, Protocol
from urllib.parse import urlencode

import numpy as np
from PIL import Image
import requests


@dataclass(frozen=True)
class GeminiFilterConfig:
    model_name: str = "gemini-3.1-flash-lite-preview"
    base_url_env: str = "GEMINI_BASE_URL"
    api_key_env: str = "GEMINI_API_KEY"
    endpoint_path_template: str = "/v1beta/models/{model}:generateContent"
    timeout_seconds: int = 60
    max_retries: int = 5
    retry_backoff_seconds: float = 2.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GeminiCandidateDecision:
    recognizable_from_mask_only: bool
    clear_not_blurry: bool
    severe_crop: bool
    interactive_object: bool
    passed: bool
    short_reason: str

    def to_dict(self) -> dict:
        return {
            "recognizable_from_mask_only": self.recognizable_from_mask_only,
            "clear_not_blurry": self.clear_not_blurry,
            "severe_crop": self.severe_crop,
            "interactive_object": self.interactive_object,
            "pass": self.passed,
            "short_reason": self.short_reason,
        }


@dataclass(frozen=True)
class GeminiCandidateRecord:
    object_id: str
    category_name: str
    model_name: str
    passed: bool
    reasons: tuple[str, ...]
    decision: dict[str, Any]
    masked_view_path: str
    original_image_path: str

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "category_name": self.category_name,
            "model_name": self.model_name,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "decision": self.decision,
            "masked_view_path": self.masked_view_path,
            "original_image_path": self.original_image_path,
        }


class GeminiInferenceClient(Protocol):
    def infer(
        self,
        *,
        category_name: str,
        masked_image_path: str | Path,
        original_image_path: str | Path,
    ) -> GeminiCandidateDecision:
        ...


def create_masked_instance_view(
    image_path: str | Path,
    mask_path: str | Path,
    output_path: str | Path,
) -> str:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    image_np = np.asarray(image, dtype=np.uint8)
    mask_np = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    masked = image_np.copy()
    masked[mask_np == 0] = 0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(masked).save(output_path)
    return str(output_path)


def build_gemini_candidate_prompt(category_name: str) -> str:
    return (
        "You are filtering object instances for a 3D interaction dataset. "
        "The masked object image is the primary evidence, while the original image is weak context only. "
        "Judge whether the instance is recognizable from the masked region only, "
        "clear not blurry, not severely cropped, and belongs to an interactive object category. "
        "Return JSON only with keys: recognizable_from_mask_only, clear_not_blurry, "
        "severe_crop, interactive_object, pass, short_reason. "
        f"Target category hint: {category_name}."
    )


def _image_to_base64(image_path: str | Path) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def _guess_mime_type(image_path: str | Path) -> str:
    suffix = Path(image_path).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "image/png"


def build_gemini_payload(
    config: GeminiFilterConfig,
    *,
    category_name: str,
    masked_image_path: str | Path,
    original_image_path: str | Path,
) -> dict[str, Any]:
    prompt = build_gemini_candidate_prompt(category_name)
    return {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": _guess_mime_type(masked_image_path),
                            "data": _image_to_base64(masked_image_path),
                        }
                    },
                    {
                        "inline_data": {
                            "mime_type": _guess_mime_type(original_image_path),
                            "data": _image_to_base64(original_image_path),
                        }
                    },
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {"responseMimeType": "application/json"},
    }


def decision_to_reasons(decision: GeminiCandidateDecision) -> tuple[str, ...]:
    reasons: list[str] = []
    if not decision.recognizable_from_mask_only:
        reasons.append("not_recognizable_from_mask_only")
    if not decision.clear_not_blurry:
        reasons.append("not_clear")
    if decision.severe_crop:
        reasons.append("severe_crop")
    if not decision.interactive_object:
        reasons.append("non_interactive_object")
    if not decision.passed and not reasons:
        reasons.append("model_rejected")
    return tuple(reasons)


def _extract_text_from_gemini_response(response_json: dict[str, Any]) -> str:
    try:
        content = response_json["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("unsupported response structure") from exc

    text_chunks = []
    for item in content:
        if isinstance(item, dict) and "text" in item:
            text_chunks.append(item["text"])
    content_text = "".join(text_chunks)
    if not content_text:
        raise ValueError("response content must include text")
    return content_text.strip()


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


def parse_gemini_candidate_decision(response_json: dict[str, Any]) -> GeminiCandidateDecision:
    parsed = _parse_json_text(_extract_text_from_gemini_response(response_json))

    return GeminiCandidateDecision(
        recognizable_from_mask_only=bool(parsed["recognizable_from_mask_only"]),
        clear_not_blurry=bool(parsed["clear_not_blurry"]),
        severe_crop=bool(parsed["severe_crop"]),
        interactive_object=bool(parsed["interactive_object"]),
        passed=bool(parsed["pass"]),
        short_reason=str(parsed["short_reason"]),
    )


class GeminiCandidateFilterClient:
    def __init__(self, config: GeminiFilterConfig):
        self.config = config

    def get_base_url(self) -> str:
        value = os.environ.get(self.config.base_url_env, "").rstrip("/")
        if not value:
            raise RuntimeError(f"missing env var: {self.config.base_url_env}")
        return value

    def get_api_key(self) -> str:
        value = os.environ.get(self.config.api_key_env, "")
        if not value:
            raise RuntimeError(f"missing env var: {self.config.api_key_env}")
        return value

    def build_request(
        self,
        *,
        category_name: str,
        masked_image_path: str | Path,
        original_image_path: str | Path,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        api_key = self.get_api_key()
        endpoint = self.config.endpoint_path_template.format(model=self.config.model_name)
        query = urlencode({"key": api_key})
        url = f"{self.get_base_url()}{endpoint}?{query}"
        headers = {"Content-Type": "application/json"}
        payload = build_gemini_payload(
            self.config,
            category_name=category_name,
            masked_image_path=masked_image_path,
            original_image_path=original_image_path,
        )
        return url, headers, payload

    def infer(
        self,
        *,
        category_name: str,
        masked_image_path: str | Path,
        original_image_path: str | Path,
    ) -> GeminiCandidateDecision:
        url, headers, payload = self.build_request(
            category_name=category_name,
            masked_image_path=masked_image_path,
            original_image_path=original_image_path,
        )
        last_exc: Exception | None = None
        for attempt_idx in range(1, self.config.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
                response.raise_for_status()
                return parse_gemini_candidate_decision(response.json())
            except (
                requests.RequestException,
                json.JSONDecodeError,
                ValueError,
                KeyError,
            ) as exc:
                last_exc = exc
                if attempt_idx >= self.config.max_retries:
                    break
                time.sleep(self.config.retry_backoff_seconds * attempt_idx)
        assert last_exc is not None
        raise last_exc


def run_gemini_candidate_filter(
    records,
    *,
    export_root: str | Path,
    client: GeminiInferenceClient,
    config: GeminiFilterConfig,
) -> dict[str, Any]:
    export_root = Path(export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    candidate_records: list[GeminiCandidateRecord] = []
    for record in records:
        object_dir = export_root / record.object_id
        source_dir = object_dir / "source"
        candidate_dir = object_dir / "candidate"
        candidate_dir.mkdir(parents=True, exist_ok=True)

        original_image_path = source_dir / "image.png"
        mask_path = source_dir / "instance_mask.png"
        if not original_image_path.is_file():
            raise FileNotFoundError(f"missing source image: {original_image_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"missing source mask: {mask_path}")

        masked_view_path = candidate_dir / "masked_instance.png"
        create_masked_instance_view(original_image_path, mask_path, masked_view_path)

        decision = client.infer(
            category_name=record.category_name,
            masked_image_path=masked_view_path,
            original_image_path=original_image_path,
        )
        candidate_record = GeminiCandidateRecord(
            object_id=record.object_id,
            category_name=record.category_name,
            model_name=config.model_name,
            passed=decision.passed,
            reasons=decision_to_reasons(decision),
            decision=decision.to_dict(),
            masked_view_path=f"{record.object_id}/candidate/masked_instance.png",
            original_image_path=f"{record.object_id}/source/image.png",
        )
        candidate_meta_path = candidate_dir / "candidate_meta.json"
        with candidate_meta_path.open("w", encoding="utf-8") as handle:
            json.dump(candidate_record.to_dict(), handle, indent=2, ensure_ascii=True)
        candidate_records.append(candidate_record)

    summary_payload = {
        "model_name": config.model_name,
        "num_records": len(candidate_records),
        "num_passed": sum(1 for item in candidate_records if item.passed),
        "records": [item.to_dict() for item in candidate_records],
    }
    summary_path = export_root / "candidate_records.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=True)

    return {
        "export_root": str(export_root),
        "summary_path": str(summary_path),
        "num_records": summary_payload["num_records"],
        "num_passed": summary_payload["num_passed"],
    }

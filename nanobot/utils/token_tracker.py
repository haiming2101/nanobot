"""Token tracking utilities for monitoring model token usage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from nanobot.utils.helpers import ensure_dir, get_data_path

TOKEN_USAGE_FILENAME = "token_usage.json"


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _sanitize_token_count(value: int | float | str | None) -> int:
    """Convert a token count to a non-negative integer."""
    if value is None:
        return 0

    try:
        count = int(value)
    except (TypeError, ValueError):
        return 0

    return max(0, count)


def _normalize_usage(usage: Mapping[str, int | float | str | None] | None) -> dict[str, int]:
    """Normalize provider usage payload into standard token fields."""
    raw = usage or {}
    prompt_tokens = _sanitize_token_count(raw.get("prompt_tokens"))
    completion_tokens = _sanitize_token_count(raw.get("completion_tokens"))

    total_tokens_raw = raw.get("total_tokens")
    total_tokens = _sanitize_token_count(total_tokens_raw)
    if total_tokens == 0 and total_tokens_raw in (None, 0, "0"):
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _default_usage_payload() -> dict[str, object]:
    """Build a default usage document shape."""
    return {
        "updated_at": _utc_now_iso(),
        "total": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "models": {},
    }


def _load_usage(path: Path) -> dict[str, object]:
    """Load token usage document, returning defaults on missing/invalid data."""
    if not path.exists():
        return _default_usage_payload()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _default_usage_payload()

    if not isinstance(data, dict):
        return _default_usage_payload()

    models = data.get("models")
    total = data.get("total")
    if not isinstance(models, dict) or not isinstance(total, dict):
        return _default_usage_payload()

    return data


def track_model_token_usage(
    model: str,
    usage: Mapping[str, int | float | str | None] | None,
    storage_path: Path | None = None,
) -> dict[str, int]:
    """Track cumulative token usage for a model.

    Args:
        model: Model identifier (for example ``gpt-4o-mini``).
        usage: Token usage payload with standard usage keys.
        storage_path: Optional custom file path for persisted usage.

    Returns:
        Updated cumulative usage for the given model.
    """
    if not model:
        raise ValueError("model must be a non-empty string")

    normalized = _normalize_usage(usage)
    target_path = storage_path or (get_data_path() / TOKEN_USAGE_FILENAME)
    ensure_dir(target_path.parent)

    usage_doc = _load_usage(target_path)
    models = usage_doc.setdefault("models", {})
    total = usage_doc.setdefault("total", {})

    model_usage = models.setdefault(model, {})

    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        current_model_value = _sanitize_token_count(model_usage.get(field))
        current_total_value = _sanitize_token_count(total.get(field))
        increment = normalized[field]

        model_usage[field] = current_model_value + increment
        total[field] = current_total_value + increment

    usage_doc["updated_at"] = _utc_now_iso()

    tmp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
    tmp_path.write_text(json.dumps(usage_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(target_path)

    return {
        "prompt_tokens": int(model_usage["prompt_tokens"]),
        "completion_tokens": int(model_usage["completion_tokens"]),
        "total_tokens": int(model_usage["total_tokens"]),
    }


def get_tracked_token_usage(storage_path: Path | None = None) -> dict[str, object]:
    """Get persisted token usage data for all models."""
    target_path = storage_path or (get_data_path() / TOKEN_USAGE_FILENAME)
    return _load_usage(target_path)

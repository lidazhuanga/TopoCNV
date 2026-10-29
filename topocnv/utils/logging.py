"""Console and optional JSON Lines logging for TopoCNV."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Union

PathLike = Union[str, Path]


def _json_default(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


class JSONLHandler(logging.Handler):
    """Write one structured, UTF-8 JSON object per log record."""

    def __init__(self, path: PathLike) -> None:
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        """Serialize a record with optional ``structured`` fields."""
        try:
            payload = {
                "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            structured = getattr(record, "structured", None)
            if isinstance(structured, Mapping):
                payload.update(structured)
            if record.exc_info:
                payload["exception"] = self.formatException(record.exc_info)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")
        except Exception:
            self.handleError(record)

    @staticmethod
    def formatException(exc_info: Any) -> str:
        """Format exception information using the standard formatter."""
        return logging.Formatter().formatException(exc_info)


def configure_logging(
    name: str = "topocnv",
    *,
    level: Union[int, str] = logging.INFO,
    jsonl_path: Optional[PathLike] = None,
    console: bool = True,
    force: bool = False,
) -> logging.Logger:
    """Configure an idempotent package logger.

    Console output uses a concise timestamped format. When ``jsonl_path`` is
    supplied, the same events are additionally persisted as structured JSONL.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    if console and not any(getattr(handler, "_topocnv_console", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
        handler._topocnv_console = True  # type: ignore[attr-defined]
        logger.addHandler(handler)

    if jsonl_path is not None:
        resolved = str(Path(jsonl_path).resolve())
        if not any(getattr(handler, "_topocnv_json_path", None) == resolved for handler in logger.handlers):
            json_handler = JSONLHandler(resolved)
            json_handler.setLevel(level)
            json_handler._topocnv_json_path = resolved  # type: ignore[attr-defined]
            logger.addHandler(json_handler)
    return logger


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit a human-readable event with structured fields for JSONL output."""
    logger.log(level, event, extra={"structured": {"event": event, **fields}})


def log_metrics(
    logger: logging.Logger,
    metrics: Mapping[str, Any],
    *,
    step: Optional[int] = None,
    prefix: str = "",
    level: int = logging.INFO,
) -> None:
    """Log a metric mapping in console and machine-readable forms."""
    normalized = {f"{prefix}{key}": _json_default(value) for key, value in metrics.items()}
    parts = [f"{key}={value:.6g}" if isinstance(value, (int, float)) else f"{key}={value}" for key, value in normalized.items()]
    if step is not None:
        parts.insert(0, f"step={step}")
    structured = {"event": "metrics", "metrics": normalized}
    if step is not None:
        structured["step"] = int(step)
    logger.log(level, " | ".join(parts), extra={"structured": structured})


__all__ = ["JSONLHandler", "configure_logging", "log_event", "log_metrics"]

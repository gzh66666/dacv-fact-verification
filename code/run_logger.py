
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def sha256_text(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunLogger:

    def __init__(self, run_dir: str | Path, *, run_id: Optional[str] = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or self.run_dir.name
        self.api_log = self.run_dir / "02_API调用日志.jsonl"
        self.search_log = self.run_dir / "03_Serper查询日志.jsonl"
        self.events_log = self.run_dir / "04_逐样本索引.jsonl"
        self.runtime_log = self.run_dir / "01_运行日志.log"
        for path in (self.api_log, self.search_log, self.events_log, self.runtime_log):
            path.touch(exist_ok=True)

    def write_preflight(self, config: Dict[str, Any]) -> None:
        path = self.run_dir / "00_实验前配置.json"
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {"run_id": self.run_id, "recorded_at_utc": _utc_now()}
            stage = config.get("stage", "unspecified")
            payload.setdefault("stage_configs", {})[stage] = _jsonable(config)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return
        payload = {
            "run_id": self.run_id,
            "recorded_at_utc": _utc_now(),
            "python": sys.version,
            "platform": platform.platform(),
            "config": _jsonable(config),
            "stage_configs": {config.get("stage", "unspecified"): _jsonable(config)},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def append_jsonl(self, path: Path, record: Dict[str, Any]) -> None:
        record = {"run_id": self.run_id, "recorded_at_utc": _utc_now(), **_jsonable(record)}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_text(self, message: str) -> None:
        with self.runtime_log.open("a", encoding="utf-8") as f:
            f.write(f"{_utc_now()} {message}\n")

    @staticmethod
    def _usage_from_result(result: Any) -> Dict[str, Any]:
        cost = getattr(result, "cost", None) or {}
        usage = cost.get("usage_excluding_cached_inference") or cost.get("usage_including_cached_inference") or {}
        if isinstance(usage, dict) and usage and not any(isinstance(v, dict) for v in usage.values()):
            usage_by_model = {str(usage.get("model", "unknown")): usage}
        else:
            usage_by_model = usage if isinstance(usage, dict) else {}
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        models = []
        for model, values in usage_by_model.items():
            if not isinstance(values, dict):
                continue
            models.append(str(model))
            for key in totals:
                try:
                    totals[key] += int(values.get(key, 0) or 0)
                except (TypeError, ValueError):
                    pass
        return {
            **totals,
            "models": models,
            "usage_excluding_cached_inference": _jsonable(cost.get("usage_excluding_cached_inference")),
            "usage_including_cached_inference": _jsonable(cost.get("usage_including_cached_inference")),
            "cost": _jsonable(cost),
        }

    def log_llm_call(
        self,
        *,
        result: Any,
        input_text: str,
        output_text: str,
        context: Optional[Dict[str, Any]] = None,
        start_time: float,
        end_time: float,
        status: str = "ok",
        parse_ok: Optional[bool] = None,
        retry_count: int = 0,
    ) -> None:
        usage = self._usage_from_result(result)
        self.append_jsonl(
            self.api_log,
            {
                **(context or {}),
                "status": status,
                "start_epoch": start_time,
                "end_epoch": end_time,
                "latency_sec": round(end_time - start_time, 6),
                "retry_count": retry_count,
                "input_chars": len(input_text),
                "output_chars": len(output_text),
                "input_sha256": sha256_text(input_text),
                "output_sha256": sha256_text(output_text),
                "parse_ok": parse_ok,
                "request_id": None,
                "system_fingerprint": None,
                "usage": usage,
            },
        )

    def log_search(self, record: Dict[str, Any]) -> None:
        self.append_jsonl(self.search_log, record)

    def log_sample(self, record: Dict[str, Any]) -> None:
        self.append_jsonl(self.events_log, record)

    def write_metrics(self, metrics: Dict[str, Any]) -> None:
        path = self.run_dir / "05_指标结果.json"
        payload = {"run_id": self.run_id, "recorded_at_utc": _utc_now(), **_jsonable(metrics)}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_status(self, status: Dict[str, Any]) -> None:
        path = self.run_dir / "06_运行状态.json"
        payload = {
            "run_id": self.run_id,
            "updated_at_utc": _utc_now(),
            **_jsonable(status),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

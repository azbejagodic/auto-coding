"""Append-only operational events for Auto-Coding v1."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_REASON_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,63}\Z")
_STATUSES = {"started", "succeeded", "failed", "stopped"}


class OperationsLog:
    """Write structured events to a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8"):
            pass

    def append(
        self,
        *,
        run_id: str,
        stage: str,
        status: str,
        batch_id: str | None = None,
        task_id: str | None = None,
        reason_code: str | None = None,
    ) -> None:
        """Append one event, rejecting free-form values and unknown fields."""
        for name, value in (
            ("run_id", run_id),
            ("stage", stage),
            ("batch_id", batch_id),
            ("task_id", task_id),
        ):
            if value is not None and (
                not isinstance(value, str) or not _IDENTIFIER.fullmatch(value)
            ):
                raise ValueError(f"{name} must be a short identifier")
        if not run_id or not stage:
            raise ValueError("run_id and stage are required")
        if not isinstance(status, str) or status not in _STATUSES:
            raise ValueError("invalid status")
        if reason_code is not None and (
            not isinstance(reason_code, str) or not _REASON_CODE.fullmatch(reason_code)
        ):
            raise ValueError("reason_code must be an uppercase identifier")

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "stage": stage,
            "status": status,
        }
        if batch_id is not None:
            event["batch_id"] = batch_id
        if task_id is not None:
            event["task_id"] = task_id
        if reason_code is not None:
            event["reason_code"] = reason_code

        with self.path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(event, separators=(",", ":")) + "\n")
            log_file.flush()
            os.fsync(log_file.fileno())

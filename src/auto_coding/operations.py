"""Append-only operational events for Auto-Coding v1."""

from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import json
import math
import os
from pathlib import Path
import re
import sys


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_REASON_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,63}\Z")
_STATUSES = {"started", "succeeded", "failed", "stopped"}


@dataclass(frozen=True)
class TimeoutDiagnostic:
    """A confirmed controller timeout, without command arguments or output.

    Elapsed time includes invocation/cleanup when observed; None is unavailable.
    Context (batch, task and earlier publications) comes from the controller.
    """

    operation: str
    configured_seconds: float
    elapsed_seconds: float | None = None
    classification: str = "TIMEOUT"

    def __post_init__(self):
        if self.classification != "TIMEOUT" or self.operation not in {
            "implementation", "tests", "review", "git_preflight", "git_staging",
            "git_commit", "git_push", "git_verification",
        }:
            raise ValueError("invalid timeout operation")
        for value, positive in ((self.configured_seconds, True), (self.elapsed_seconds, False)):
            if value is None and not positive:
                continue
            try:
                valid = (type(value) in (int, float) and math.isfinite(value)
                         and (value > 0 if positive else value >= 0))
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError("invalid timeout duration")

    def record(self):
        return asdict(self)


def show_timeout(diagnostic, batch_id=None, task_id=None, previously_published=()):
    """Emit only validated identifiers and fixed diagnostic text, immediately."""
    def safe_id(value):
        return value if isinstance(value, str) and _IDENTIFIER.fullmatch(value) else "unavailable"

    labels = {"implementation": "Codex A implementation", "tests": "Approved terminal tests",
              "review": "Codex B review", "git_preflight": "Git preflight",
              "git_staging": "Git staging", "git_commit": "Git commit",
              "git_push": "Git push", "git_verification": "Git verification"}
    seconds = diagnostic.configured_seconds
    if type(seconds) is float and seconds.is_integer():
        seconds = int(seconds)
    lines = ["AUTO-CODING STOPPED - TIMEOUT", f"Batch: {safe_id(batch_id)}", f"Task: {safe_id(task_id)}",
             f"Operation: {labels[diagnostic.operation]}",
             f"Configured limit: {seconds} seconds",
             "Reason: Operation exceeded its configured time limit.",
             "Batch execution: STOPPED; later tasks: NOT STARTED",
             f"Previously published tasks: {len(previously_published)}"]
    if diagnostic.elapsed_seconds is not None:
        lines.append(f"Elapsed duration (including cleanup): {diagnostic.elapsed_seconds:g} seconds")
    lines.extend("Published task: " + safe_id(task_id) for task_id in previously_published)
    try:
        print("\n".join(lines), file=sys.stderr, flush=True)
    except (OSError, ValueError):
        # A closed terminal must not prevent durable stopping/diagnostic writes.
        pass


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

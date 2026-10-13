"""Fingerprint and store explicitly approved v1 task plans."""

import hashlib
import json
import os
from pathlib import Path
import re

from auto_coding.plans import PlanValidationError, validate_plan


class PlanIntegrityError(ValueError):
    """An approved fingerprint or frozen plan failed an integrity check."""


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _canonical_bytes(plan: dict) -> bytes:
    return json.dumps(
        plan, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _check_approval(approved_sha256: str, actual_sha256: str) -> None:
    if not isinstance(approved_sha256, str) or not _SHA256.fullmatch(approved_sha256):
        raise PlanIntegrityError("approved_sha256 must be a lowercase SHA-256 digest")
    if approved_sha256 != actual_sha256:
        raise PlanIntegrityError("approved_sha256 does not match the plan")


def plan_fingerprint(data: object) -> str:
    """Validate a plan and return its canonical SHA-256 fingerprint."""
    plan = validate_plan(data)
    return hashlib.sha256(_canonical_bytes(plan)).hexdigest()


def freeze_approved_plan(data: object, approved_sha256: str, destination: str | Path) -> None:
    """Create a frozen plan only when its supplied approval matches."""
    plan = validate_plan(data)
    canonical = _canonical_bytes(plan)
    _check_approval(approved_sha256, hashlib.sha256(canonical).hexdigest())

    try:
        with Path(destination).open("xb") as frozen_file:
            frozen_file.write(canonical)
            frozen_file.flush()
            os.fsync(frozen_file.fileno())
    except FileExistsError as exc:
        raise PlanIntegrityError("frozen plan destination already exists") from exc


def load_frozen_plan(destination: str | Path, approved_sha256: str) -> dict:
    """Load a plan only when its bytes match a supplied approved fingerprint."""
    if not isinstance(approved_sha256, str) or not _SHA256.fullmatch(approved_sha256):
        raise PlanIntegrityError("approved_sha256 must be a lowercase SHA-256 digest")
    try:
        frozen_bytes = Path(destination).read_bytes()
    except FileNotFoundError as exc:
        raise PlanIntegrityError("frozen plan is missing") from exc
    _check_approval(approved_sha256, hashlib.sha256(frozen_bytes).hexdigest())

    try:
        data = json.loads(frozen_bytes.decode("utf-8"))
        return validate_plan(data)
    except (UnicodeDecodeError, json.JSONDecodeError, PlanValidationError) as exc:
        raise PlanIntegrityError("frozen plan is invalid") from exc

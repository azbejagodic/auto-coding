"""Extract one validated Codex session ID from bounded JSONL output."""

import json
import uuid


MAX_JSONL_BYTES = 8 * 1024 * 1024


def session_from_jsonl(output: bytes) -> tuple[str | None, str | None]:
    if len(output) > MAX_JSONL_BYTES:
        return None, "OUTPUT_TOO_LARGE"
    if output and not output.endswith(b"\n"):
        return None, "SESSION_EVIDENCE_MALFORMED"

    session_id = None
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, "SESSION_EVIDENCE_MALFORMED"
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            return None, "SESSION_EVIDENCE_MALFORMED"
        if event["type"] != "thread.started":
            continue
        value = event.get("thread_id")
        if not isinstance(value, str):
            return None, "SESSION_ID_INVALID"
        try:
            parsed = uuid.UUID(value)
        except ValueError:
            return None, "SESSION_ID_INVALID"
        if parsed.int == 0 or str(parsed) != value.lower():
            return None, "SESSION_ID_INVALID"
        if session_id is not None:
            return None, "SESSION_ID_AMBIGUOUS"
        session_id = value

    if session_id is None:
        return None, "SESSION_ID_MISSING"
    return session_id, None

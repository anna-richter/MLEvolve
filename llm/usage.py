"""Run-wide LLM token counter. Written to a JSON file after every recorded call."""

import json
import logging
import threading

logger = logging.getLogger("MLEvolve")

_lock = threading.Lock()
_totals = {"total_input_tokens": 0, "total_output_tokens": 0, "total_tokens": 0, "calls": 0}
_path = None


def set_path(path) -> None:
    global _path
    _path = path


def add(in_tok: int, out_tok: int) -> None:
    with _lock:
        _totals["total_input_tokens"] += in_tok
        _totals["total_output_tokens"] += out_tok
        _totals["total_tokens"] += in_tok + out_tok
        _totals["calls"] += 1
        if _path is not None:
            with open(_path, "w") as f:
                json.dump({"total": _totals}, f, indent=2)


def add_gemini(usage_metadata) -> tuple[int, int]:
    """Record a Gemini usage_metadata. Output includes thinking tokens (total - prompt)."""
    if usage_metadata is None:
        logger.warning("No usage_metadata in Gemini response, tokens not counted")
        return 0, 0
    in_tok = usage_metadata.prompt_token_count or 0
    total = usage_metadata.total_token_count
    if total:
        out_tok = total - in_tok
    else:
        out_tok = (usage_metadata.candidates_token_count or 0) + (usage_metadata.thoughts_token_count or 0)
    add(in_tok, out_tok)
    return in_tok, out_tok

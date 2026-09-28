"""Token counting with the Laya tokenizer when cached locally, else a char heuristic.

The same budget is applied to every backend so the comparison is fair.
"""
from __future__ import annotations

import json
from functools import lru_cache

CHARS_PER_TOKEN = 3.2  # measured on state JSON with the ModernBERT tokenizer (see tests)


@lru_cache(maxsize=1)
def _tokenizer():
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        path = hf_hub_download("convaiinnovations/laya", "tokenizer.json",
                               subfolder="typed-decisions/tokenizer", local_files_only=True)
        return Tokenizer.from_file(path)
    except Exception:
        return None


def to_text(obj) -> str:
    return obj if isinstance(obj, str) else json.dumps(obj, separators=(",", ":"))


def count_tokens(obj) -> int:
    text = to_text(obj)
    tok = _tokenizer()
    if tok is not None:
        return len(tok.encode(text).ids)
    return int(len(text) / CHARS_PER_TOKEN) + 1


def tokenizer_name() -> str:
    return "laya-typed-decisions" if _tokenizer() is not None else f"chars/{CHARS_PER_TOKEN}"

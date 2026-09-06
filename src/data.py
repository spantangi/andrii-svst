"""Stage 1 eval set: 1,000 sequences, stratified into three buckets.

Stratification is not cosmetic. The paper's claim is not just "99.2% align" --
it is that failures concentrate in chat-formatted text with smart quotes, box
drawing and emoji. You can only check the second half of that claim if the
buckets are labelled, and it is the second half that distinguishes "expected
tokenizer-pair behaviour" from "the expansion loop is broken".

  prose  -- FineWeb. Plain English. Failures here mean a bug.
  chat   -- chat-templated multi-turn text.
  stress -- smart quotes, box drawing, emoji, CJK, zero-width joiners.
"""

from __future__ import annotations

import random

SMART = ["“quoted”", "‘single’", "em—dash", "ellipsis…",
         "non‑breaking‑hyphen", "café naïve résumé"]
BOX = ["┌───┐", "│ cell │", "└───┘",
       "╚══╝", "├┤ ┬┴"]
EMOJI = ["\U0001f600", "\U0001f469‍\U0001f4bb", "\U0001f1fa\U0001f1f8",
         "✅ done", "\U0001f3f3️‍\U0001f308", "family \U0001f468‍\U0001f469‍\U0001f467"]
CJK = ["你好世界", "こんにちは", "안녕하세요",
       "漢字とかな", "繁體字"]
MISC = ["​ zero-width ​", "﻿BOM-lead", "tab\there", "á combining accent",
        "\U0001d400\U0001d401 math bold", "رياض RTL mix"]


def build_stress(n: int, seed: int = 0) -> list[str]:
    """Synthetic stress bucket: the character classes the paper says fail."""
    rng = random.Random(seed)
    pools = [SMART, BOX, EMOJI, CJK, MISC]
    out = []
    for _ in range(n):
        parts = []
        for _ in range(rng.randint(3, 8)):
            parts.append(rng.choice(rng.choice(pools)))
            if rng.random() < 0.5:
                parts.append("some ordinary words between them")
        out.append(" ".join(parts))
    return out


def build_chat(n: int, seed: int = 0) -> list[str]:
    """Chat-formatted text.

    NOTE: lmsys/lmsys-chat-1m is gated on the Hub. HuggingFaceH4/ultrachat_200k
    is the ungated substitute; it is the same shape (multi-turn user/assistant)
    and serves the plumbing check. Swap back once LMSYS access is granted.
    """
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    out = []
    for row in ds:
        turns = row.get("messages") or []
        if len(turns) < 2:
            continue
        text = "\n".join(f"<|{t['role']}|>\n{t['content']}" for t in turns[:4])
        out.append(text)
        if len(out) >= n:
            break
    return out


def build_prose(n: int, max_chars: int = 4000) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
    out = []
    for row in ds:
        t = (row.get("text") or "").strip()
        if len(t) < 200:
            continue
        out.append(t[:max_chars])
        if len(out) >= n:
            break
    return out


def build_eval_set(n_prose: int, n_chat: int, n_stress: int, seed: int = 0):
    """Returns [(bucket, text), ...]."""
    items = [("prose", t) for t in build_prose(n_prose)]
    items += [("chat", t) for t in build_chat(n_chat)]
    items += [("stress", t) for t in build_stress(n_stress, seed)]
    return items

"""Stage 1 -- tokenizer alignment. CPU only, no training. This stage is a GATE.

Implements Algorithm 1: greedy window expansion to find token spans in two
different tokenizations that cover identical text.

Pass criteria (from configs/*.yaml, alignment:):
  - overall alignment rate >= 0.99
  - ~zero failures in the `prose` bucket

Failure diagnosis:
  - failures in plain English prose  -> bug in the window-expansion loop.
    Suspect the termination condition and off-by-one on window boundaries.
  - 5%+ overall but prose is clean   -> likely a property of this tokenizer
    pair, not a bug. Re-check against a second pair before editing Algorithm 1.

Do not proceed to Stage 2 until this passes. Misaligned pairs corrupt the
paired activations and surface later as low FVE and a malformed decoder-norm
distribution -- i.e. Checks 2 and 3 failing for the wrong reason.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlignmentResult:
    aligned: bool
    spans: list[tuple[int, int, int, int]]  # (a_start, a_end, b_start, b_end)
    failure_reason: str | None = None


def align_sequence(text: str, tok_a, tok_b, max_window: int) -> AlignmentResult:
    """Align one sequence's two tokenizations via greedy window expansion."""
    raise NotImplementedError("Stage 1")


def run_alignment_eval(config: dict) -> dict:
    """Run Algorithm 1 over the stratified eval set.

    Returns per-bucket rates (prose / chat / stress) plus overall. Writes every
    failing sequence to runs/<id>/alignment_failures.jsonl for inspection --
    the paper's claim is not just the rate but that failures concentrate in
    chat-formatted text with smart quotes, box-drawing characters, and emoji.
    """
    raise NotImplementedError("Stage 1")

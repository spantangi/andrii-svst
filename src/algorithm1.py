"""Stage 1 -- tokenizer alignment. CPU only, no training. This stage is a GATE.

Implements Algorithm 1: greedy window expansion to find spans of tokens in two
different tokenizations that cover byte-identical text.

The invariant: walk both tokenizations left to right holding a window on each
side. Whenever the two windows end at the same character offset, that pair of
windows is an aligned span; emit it and start a fresh window at the next token.
When they end at different offsets, expand whichever side ends earlier. A span
that grows past `max_window` on either side is a failure, not a wider search --
the cap is what keeps this linear and what makes failures diagnostic.

Pass criteria (alignment: in the config):
  - overall alignment rate >= 0.99
  - ~zero failures in the `prose` bucket

Failure diagnosis:
  - failures in plain English prose  -> bug in the window-expansion loop.
    Suspect the termination condition and off-by-one on window boundaries.
  - 5%+ overall but prose is clean   -> likely a property of this tokenizer
    pair, not a bug. Re-check against a second pair before editing Algorithm 1.

Do not proceed to Stage 2 until this passes. Misaligned pairs corrupt the
paired activations and surface later as low FVE and a malformed decoder-norm
distribution -- Checks 2 and 3 failing for the wrong reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Failure taxonomy. Kept explicit because *which* failure fires is the whole
# diagnostic value of this stage -- an aggregate rate tells you nothing.
WINDOW_EXCEEDED = "window_exceeded"   # span grew past max_window on one side
EXHAUSTED = "exhausted"               # ran off the end of one tokenization mid-span
COVERAGE_GAP = "coverage_gap"         # tokenizer dropped/normalised chars; offsets skip text
TEXT_MISMATCH = "text_mismatch"       # spans end at same offset but cover different text
EMPTY = "empty"                       # nothing to align after dropping specials


@dataclass
class AlignmentResult:
    aligned: bool
    spans: list[tuple[int, int, int, int]] = field(default_factory=list)
    failure_reason: str | None = None
    failure_detail: str | None = None
    n_tokens_a: int = 0
    n_tokens_b: int = 0
    max_span_a: int = 0
    max_span_b: int = 0


def _encode_with_offsets(text: str, tok):
    """(offsets, orig_idx) with special/zero-width tokens dropped.

    Zero-width tokens (start == end) carry no text, so they can never close a
    window. Leaving them in makes the expansion loop spin until it hits
    max_window and reports a spurious WINDOW_EXCEEDED.

    orig_idx maps each kept token back to its position in the unfiltered
    encoding. Spans are reported in ORIGINAL indices so Stage 2 can gather
    activations by index without re-deriving the filter.
    """
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offs, idx = [], []
    for k, (s, e) in enumerate(enc["offset_mapping"]):
        if e > s:
            offs.append((s, e))
            idx.append(k)
    return offs, idx


def align_sequence(text: str, tok_a, tok_b, max_window: int = 16,
                   verify_text: bool = True) -> AlignmentResult:
    """Align one sequence's two tokenizations via greedy window expansion."""
    off_a, idx_a = _encode_with_offsets(text, tok_a)
    off_b, idx_b = _encode_with_offsets(text, tok_b)

    if not off_a or not off_b:
        return AlignmentResult(False, failure_reason=EMPTY,
                               n_tokens_a=len(off_a), n_tokens_b=len(off_b))

    spans: list[tuple[int, int, int, int]] = []
    i = j = 0
    max_span_a = max_span_b = 0

    while i < len(off_a) and j < len(off_b):
        i0, j0 = i, j

        # Both windows must *start* at the same character, or the two
        # tokenizers disagree about which text exists (normalisation dropping
        # a character, say). That is a coverage gap, not a windowing problem.
        if off_a[i0][0] != off_b[j0][0]:
            return AlignmentResult(
                False, spans, COVERAGE_GAP,
                f"span start mismatch: a@{off_a[i0][0]} vs b@{off_b[j0][0]}",
                len(off_a), len(off_b), max_span_a, max_span_b)

        a_end, b_end = off_a[i][1], off_b[j][1]

        while a_end != b_end:
            if a_end < b_end:
                i += 1
                if i >= len(off_a):
                    return AlignmentResult(
                        False, spans, EXHAUSTED, "ran out of A tokens mid-span",
                        len(off_a), len(off_b), max_span_a, max_span_b)
                a_end = off_a[i][1]
            else:
                j += 1
                if j >= len(off_b):
                    return AlignmentResult(
                        False, spans, EXHAUSTED, "ran out of B tokens mid-span",
                        len(off_a), len(off_b), max_span_a, max_span_b)
                b_end = off_b[j][1]

            # Cap counts tokens in the window, hence the +1 on each side.
            if (i - i0 + 1) > max_window or (j - j0 + 1) > max_window:
                return AlignmentResult(
                    False, spans, WINDOW_EXCEEDED,
                    f"window {i - i0 + 1}x{j - j0 + 1} > {max_window} "
                    f"at char {off_a[i0][0]}",
                    len(off_a), len(off_b), max_span_a, max_span_b)

        # Several tokens can end at the same character: SentencePiece emits a
        # prefix marker overlapping the next token, and byte fallback splits one
        # character into one token per UTF-8 byte, all sharing an offset span.
        # They carry no additional coverage, so they belong to THIS span --
        # advancing past only the first desyncs the walk and shows up as a
        # spurious start mismatch on the next iteration.
        while i + 1 < len(off_a) and off_a[i + 1][1] <= a_end:
            i += 1
        while j + 1 < len(off_b) and off_b[j + 1][1] <= b_end:
            j += 1

        if (i - i0 + 1) > max_window or (j - j0 + 1) > max_window:
            return AlignmentResult(
                False, spans, WINDOW_EXCEEDED,
                f"window {i - i0 + 1}x{j - j0 + 1} > {max_window} after "
                f"absorbing co-terminating tokens at char {off_a[i0][0]}",
                len(off_a), len(off_b), max_span_a, max_span_b)

        if verify_text:
            ta = text[off_a[i0][0]:off_a[i][1]]
            tb = text[off_b[j0][0]:off_b[j][1]]
            if ta != tb:
                return AlignmentResult(
                    False, spans, TEXT_MISMATCH, f"{ta!r} != {tb!r}",
                    len(off_a), len(off_b), max_span_a, max_span_b)

        spans.append((idx_a[i0], idx_a[i] + 1, idx_b[j0], idx_b[j] + 1))
        max_span_a = max(max_span_a, i - i0 + 1)
        max_span_b = max(max_span_b, j - j0 + 1)
        i += 1
        j += 1

    # A trailing remainder on either side means the tokenizations cover
    # different amounts of text -- treat as a coverage gap, not success.
    if i != len(off_a) or j != len(off_b):
        return AlignmentResult(
            False, spans, COVERAGE_GAP,
            f"trailing tokens: a {len(off_a) - i}, b {len(off_b) - j}",
            len(off_a), len(off_b), max_span_a, max_span_b)

    return AlignmentResult(True, spans, None, None,
                           len(off_a), len(off_b), max_span_a, max_span_b)

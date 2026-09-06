"""Stage 1 runner: the alignment gate.

  python scripts/run_stage1.py --config configs/stage1_ungated.yaml --run-id <id>

Writes runs/<id>/config.yaml, runs/<id>/metrics.json (both committed) and
runs/<id>/alignment_failures.jsonl (gitignored -- can be large).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algorithm1 import align_sequence          # noqa: E402
from src.config import load_config, write_metrics  # noqa: E402
from src.data import build_eval_set                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1_ungated.yaml")
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config, args.run_id)
    from transformers import AutoTokenizer

    tok_a = AutoTokenizer.from_pretrained(cfg["models"]["a"]["name"])
    tok_b = AutoTokenizer.from_pretrained(cfg["models"]["b"]["name"])
    for tag, tok in (("a", tok_a), ("b", tok_b)):
        if not tok.is_fast:
            raise SystemExit(
                f"model {tag} has a slow tokenizer; Algorithm 1 needs offset "
                f"mappings, which only fast tokenizers provide")

    src = {s["bucket"]: s["n"] for s in cfg["data"]["sources"]}
    items = build_eval_set(src.get("prose", 0), src.get("chat", 0),
                           src.get("stress", 0), seed=cfg["run"]["seed"])

    max_window = cfg["alignment"]["max_window"]
    run_dir = Path("runs") / args.run_id
    fail_path = run_dir / "alignment_failures.jsonl"

    totals: Counter = Counter()
    ok: Counter = Counter()
    reasons: dict[str, Counter] = defaultdict(Counter)
    span_max_a = span_max_b = 0

    with fail_path.open("w") as fh:
        for bucket, text in items:
            r = align_sequence(text, tok_a, tok_b, max_window)
            totals[bucket] += 1
            if r.aligned:
                ok[bucket] += 1
                span_max_a = max(span_max_a, r.max_span_a)
                span_max_b = max(span_max_b, r.max_span_b)
            else:
                reasons[bucket][r.failure_reason] += 1
                fh.write(json.dumps({
                    "bucket": bucket,
                    "reason": r.failure_reason,
                    "detail": r.failure_detail,
                    "n_tokens_a": r.n_tokens_a,
                    "n_tokens_b": r.n_tokens_b,
                    "text": text[:600],
                }, ensure_ascii=False) + "\n")

    # The gate rate covers the realistic mix only. The stress bucket is an
    # adversarial probe whose density of pathological Unicode is nothing like
    # FineWeb/LMSYS; folding it in would make the 0.99 threshold arbitrary.
    gate_buckets = cfg["alignment"].get("gate_buckets", sorted(totals))
    n = sum(totals[b] for b in gate_buckets)
    n_ok = sum(ok[b] for b in gate_buckets)
    rate = n_ok / n if n else 0.0
    prose_fail = totals["prose"] - ok["prose"]

    thr = cfg["alignment"]["pass_threshold_overall"]
    max_prose = cfg["alignment"]["pass_max_prose_failures"]
    passed = rate >= thr and prose_fail <= max_prose

    n_probe = sum(totals[b] for b in totals if b not in gate_buckets)
    ok_probe = sum(ok[b] for b in totals if b not in gate_buckets)

    metrics = {
        "gate_buckets": list(gate_buckets),
        "n_samples": n,
        "alignment_rate": round(rate, 4),
        "n_failures": n - n_ok,
        "probe": {"n": n_probe, "aligned": ok_probe,
                  "rate": round(ok_probe / n_probe, 4) if n_probe else None},
        "per_bucket": {
            b: {"n": totals[b], "aligned": ok[b],
                "rate": round(ok[b] / totals[b], 4) if totals[b] else None,
                "reasons": dict(reasons[b])}
            for b in sorted(totals)
        },
        "max_span_tokens": {"a": span_max_a, "b": span_max_b,
                            "max_window": max_window},
        "gate": {"threshold_overall": thr, "max_prose_failures": max_prose,
                 "prose_failures": prose_fail, "passed": passed},
    }
    write_metrics(args.run_id, "stage1_alignment", metrics)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nfailures written to {fail_path}")
    print("GATE: PASS" if passed else "GATE: FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

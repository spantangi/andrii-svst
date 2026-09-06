"""Stage 2 -- paired activation harvesting + calibration. Needs a GPU.

Runs both models over the aligned corpus and caches paired activations at the
configured layer. Before training, one calibration pass computes:

  1. Per-model normalization scale factors (normalization: unit_expected_l2).
     Required for Check 3 -- relative decoder norm is a ratio, so unequal raw
     activation scales shift the distribution off 0.5 independently of the loss.

  2. Batch-median L2 per model, and the first-token norm ratio.
     SANITY CHECK: Qwen first-token activations are expected to run ~10x the
     median. If that spike is absent, the hook is on the wrong tensor. This is
     a free correctness check on the entire harvesting path -- do not skip it.

Outlier masking is PAIRED: if either side exceeds threshold_x_median, the pair
is dropped from the loss. Masking one side leaves the crosscoder fitting an
unpaired target. Log the masked fraction; above warn_masked_frac_above the
threshold or the normalization is off.
"""

from __future__ import annotations


def harvest(config: dict) -> None:
    """Cache paired activations to runs/<id>/activations/ (gitignored)."""
    raise NotImplementedError("Stage 2")


def calibrate(config: dict) -> dict:
    """Compute scale factors, batch medians, and the first-token norm ratio.

    Writes scale_a / scale_b back into runs/<id>/config.yaml.
    """
    raise NotImplementedError("Stage 2")


def outlier_mask(acts_a, acts_b, threshold_x_median: float, drop_bos: bool):
    """Paired mask: True where the pair should be KEPT in the loss."""
    raise NotImplementedError("Stage 2")

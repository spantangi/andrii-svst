"""Stage 3 -- crosscoder training. Needs a GPU.

Standard crosscoder with a joint loss over both models' activations.

Two things that must not be skipped, or Checks 2 and 3 look broken for the
wrong reason:
  - sparsity annealing: k ramps linearly k_start -> k_end over anneal_frac of
    total steps (expressed as a fraction so it completes at reduced scale).
  - outlier masking: applied in harvest.outlier_mask, paired.

Log FVE, dead-feature fraction, masked fraction, and the CURRENT k at every
eval step. Watch FVE through the anneal, not only at the end: it should degrade
smoothly as k drops. A cliff means the anneal is too fast for the step budget.
"""

from __future__ import annotations


def k_at_step(step: int, total_steps: int, k_start: int, k_end: int,
              anneal_frac: float) -> int:
    """Linear sparsity anneal, expressed as a fraction of the run."""
    raise NotImplementedError("Stage 3")


def train(config: dict) -> None:
    raise NotImplementedError("Stage 3")

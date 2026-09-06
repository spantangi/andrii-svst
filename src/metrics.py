"""Stages 4-5 -- the two metric checks.

Check 2 (reconstruction quality):
  - fraction of variance explained; paper target ~0.817 at full scale
  - dead features; paper target ~5-6%, defined as no activation for 10M
    consecutive tokens. At reduced scale we use dead_feature_frac_of_run and
    report the absolute token count alongside it, since a fixed 10M window is
    longer than the whole run and would report 0% dead trivially.
  FAIL: FVE well below 0.70, or dead fraction above 0.15. Suspect the sparsity
  schedule first (did k actually reach k_end?), normalization second.

Check 3 (relative decoder norm) -- the most diagnostic of the three:
  Expected: unimodal, concentrated near 0.5, thin tails. Report the mean of the
  extreme_features_topk features on each side and compare directionally to the
  paper's 0.888 / 0.165.
  FAIL: flat        -> joint loss likely decomposed into two independent
                       per-model losses.
        bimodal or  -> the shared latent is not actually shared; check that
        pinned at     both decoders read the same latent tensor.
        0/1
"""

from __future__ import annotations


def fraction_variance_explained(model, acts_a, acts_b) -> float:
    raise NotImplementedError("Stage 4")


def dead_feature_fraction(activation_counts, window_tokens: int) -> float:
    raise NotImplementedError("Stage 4")


def relative_decoder_norms(model) -> "list[float]":
    """Per-feature ||W_dec_a|| / (||W_dec_a|| + ||W_dec_b||), one per feature."""
    raise NotImplementedError("Stage 5")


def report(run_id: str) -> str:
    """Stage 6 -- metric / paper value / our value / scale factor / verdict."""
    raise NotImplementedError("Stage 6")

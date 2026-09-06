# andrii-svst

Plumbing check for crosscoders trained across models with **different
tokenizers**. The goal is not to replicate the paper's findings — it is to
confirm the numbers land roughly where the paper says they should, so that a
later real run can be trusted.

## The three checks

| # | Check | Paper value (full scale) | Fail condition |
|---|-------|--------------------------|----------------|
| 1 | Tokenizer alignment (Algorithm 1, CPU, no training) | 99.2% of 1,000 sequences align; 8 failures | <99%, or any failures in plain prose |
| 2 | Reconstruction quality | FVE ≈ 0.817; dead features ≈ 5–6% | FVE well below 0.70, or dead >15% |
| 3 | Relative decoder norm distribution | unimodal near 0.5; top-500 extremes avg 0.888 / 0.165 | flat, bimodal, or pinned at extremes |

Check 3 is the most diagnostic — it catches joint-loss wiring bugs that the
loss curve alone will not surface.

## Stage order (the gates matter)

0. **Scaffold** — resolve config, stamp git SHA, create `runs/<id>/`.
1. **Alignment** — `src/algorithm1.py`. **GATE**: do not proceed until it
   passes. Misaligned pairs corrupt the paired activations and surface later as
   low FVE and a malformed decoder-norm distribution.
2. **Harvest + calibrate** — `src/harvest.py`. **GATE**: the calibration pass
   must show the expected ~10× first-token norm spike on the Qwen side. If it
   does not, the hook is on the wrong tensor.
3. **Train** — `src/train.py`.
4. **Check 2** — `src/metrics.py`.
5. **Check 3** — `src/metrics.py`.
6. **Report** — comparison table + the AuxK note + Stage 1 per-bucket rates.

## Things that break silently at reduced scale

- **Sparsity anneal.** Endpoints k=1000→200 are from the spec, but the 5,000-step
  duration is not: expressed here as `anneal_frac` of total steps so the anneal
  actually completes. If k never reaches 200, Check 2 is uninterpretable.
- **Dead-feature definition.** The paper's "no activation for 10M consecutive
  tokens" is longer than a reduced run, which would report 0% dead trivially.
  Rescaled to a fraction of total training tokens; the absolute count is
  reported alongside.
- **Per-model normalization.** Relative decoder norm is a *ratio*. Unequal raw
  activation scales shift the Check 3 distribution off 0.5 for reasons that
  have nothing to do with the joint loss.
- **Outlier masking is paired.** If either side exceeds 2× its batch median,
  drop the pair. Masking one side leaves the crosscoder fitting an unpaired
  target.

## Settled ambiguities

- **AuxK feature count**: the paper gives **1744** (Table 3) and **512**
  (appendix text). We use **512** — it matches the Gao et al. TopK auxk
  convention this builds on, and 1744 appears derived from a dictionary size we
  are not using at reduced scale. Recorded in every run as
  `crosscoder.auxk_source: appendix_text`.
- **Model pair**: the spec names Gemma-2 2B, but every target number quoted is
  for Llama-3.1-8B vs Qwen3-8B. Alignment rate is a property of the tokenizer
  *pair*, so those targets are directional only. Default pair here is
  `google/gemma-2-2b` + `Qwen/Qwen2.5-1.5B` (different tokenizer families, both
  small enough to harvest on one GPU).

## Layout

```
configs/          committed — the config IS the experiment
src/              algorithm1 · harvest · train · metrics
runs/<id>/
  config.yaml     committed — resolved copy, written at launch
  metrics.json    committed — small, diffable, the metric history
  activations/    gitignored — tens of GB
```

Configs are never mutated in place; the resolved copy in `runs/<id>/` is what
each stage reads.

## Environment

Stage 1 is CPU-only and runs anywhere. Stages 2–3 need a GPU — this repo is
also the transport for moving the config and alignment code to that box.

# Dataset B protocol

These files record the contract Dataset B ran under. **They are a
specification, not a runnable configuration.**

## What is missing

`scripts/run.py` performs one train / calibrate / test cycle against a single
holdout. Dataset B needs four things this repository does not have:

| Missing | Where it would go |
|---|---|
| a per-manufacturer × 5-fold driver | above `run_from_config` |
| the fold split rule in `splits:` below | `src/datas/` |
| top-`rho` vehicle aggregation, `rho` searched on calibration | `src/metrics/scoring.py` — `aggregate_vehicles` takes a fixed quantile |
| macro-over-manufacturer reporting | `src/system/evaluator.py` |

Nothing under `src/` mentions a fold or a manufacturer. Pointing `--config` at
a file here will not run.

## Why it is here anyway

The README reports Dataset B results, and a reader is entitled to know exactly
what produced them. Every value in `_base.yaml` is copied from
`config/tsinghua_matched_models.json` of the run that produced those numbers
(schema `hybridtrad-tsinghua-paper-protocol-v3`); none is reconstructed.

## Two differences from Dataset A worth noticing

Both are easy to miss, and both matter if you compare the two columns of the
results tables.

**Optimiser.** Dataset A trains with AdamW at `3e-4` and weight decay `1e-4`,
with early stopping on validation loss. Dataset B trains with Adam at `1e-3`,
no weight decay, for a fixed 100 epochs, keeping the last checkpoint. The two
datasets share no optimiser; every *model* shares one within a dataset, which
is what makes each column an internally valid comparison.

**Vehicle aggregation.** Dataset A takes a fixed 0.99 quantile of a vehicle's
snippet scores. Dataset B takes the mean of the largest `rho` fraction, with
`rho` chosen per fold on calibration vehicle AUROC. `aggregate_vehicles` in
this repository implements the first only.

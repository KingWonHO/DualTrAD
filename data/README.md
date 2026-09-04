# Datasets and preprocessing

Two public datasets are used. Neither is introduced here; both must be cited
from their original publications, and the data records cited alongside them.

| | QAS | Tsinghua |
|---|---|---|
| Paper | Cao et al., *Nat. Commun.* **16**, 1651 (2025) | Zhang et al., *Nat. Commun.* **14**, 5940 (2023) |
| Data record | Zenodo, `10.5281/zenodo.10656500` (CC BY 4.0) | figshare, `10.6084/m9.figshare.23659323` |
| Vehicles used here | 393 | 347 (292 normal / 55 abnormal) |
| Channels | 4 — `V, I, T, SOC` | 7 inputs → 5 scored responses |
| Sampling period | 30 s | 10 s |
| Labels | per vehicle | per vehicle |

Neither dataset carries a name of its own, so the paper refers to them as
Dataset A and Dataset B. Both anonymise their manufacturers.

The Zenodo record covers **515 vehicles from three anonymised manufacturers**,
which its description names `DTI`, `QAS` and `GIS`. This repository uses **QAS
only**: `DTI` was excluded at the data owner's request, and `GIS` has no
verifiable time axis, so windows over it cannot be built safely. A reader who
downloads the record will therefore find more vehicles than the table above.

The Tsinghua release is published as one dataset per manufacturer, which is why
a separate model is fitted per manufacturer rather than one model over the
pooled set.

## How the labels were made

Faults were confirmed **retroactively, by disassembling the pack** and
characterising the failure — electrolyte leakage, thermal runaway, internal
short circuit, excessive ageing. A label therefore says *this vehicle failed*,
never *the fault began at time t*.

Two consequences follow, and they shape everything downstream:

1. **Decisions are made per vehicle**, not per timestep. There is no ground
   truth for when a fault starts, so a timestep-level protocol would have to
   invent one.
2. **Point adjustment does not apply.** The ambiguity that inflates timestep
   protocols cannot arise here, because there is no segment to adjust.

---

## From raw export to split NPZ

```
raw per-vehicle export
        │  drop invalid rows using valid_mask
        ▼
contiguity check ──► reject any window whose source rows are not consecutive
        │
        ▼
vehicle-disjoint split assignment
        │
        ▼
aggregate NPZ per split:  {train, validation, calibration_normal,
                           test_normal, test_abnormal}.npz
```

Each aggregate NPZ carries the columns the loader needs:

| Array | Shape | Meaning |
|---|---|---|
| `X` | `[N, F]` | channel values in `columns` order |
| `valid_mask` | `[N]` | rows that survived cleaning |
| `source_row_index` | `[N]` | original row number, used for the contiguity check |
| `vehicle_offsets` | `[V+1]` | slice boundaries per vehicle |
| `vehicle_ids` | `[V]` | vehicle identifiers |
| `vehicle_binary_labels` | `[V]` | 0 normal, 1 abnormal |
| `columns` | `[F]` | channel names, e.g. `['V','I','T','SOC']` |

### Splits

| Split | Role | Sees labels? |
|---|---|---|
| `train` | fit model weights | normal vehicles only |
| `validation` | early stopping **and** evidence scales | normal vehicles only |
| `calibration_normal` | fit the decision threshold | normal vehicles only |
| `test_normal` / `test_abnormal` | final evaluation | **read once, at the end** |

Vehicles never appear in more than one split. The scaler is fitted on
`train` only, so no calibration or test statistic leaks into standardisation.

---

## Windowing

A window is `context_length` frames of context followed by targets at each
horizon. Two rules are enforced when building them:

- **A window never spans two vehicles.** Windows are built inside one vehicle's
  rows only.
- **A window never bridges a gap.** After invalid rows are dropped, the
  remaining `source_row_index` values must be consecutive across the whole
  window. A window that would silently join two separate driving sessions is
  discarded rather than kept.

Vehicles too short to yield a single window are reported, not padded.

### Offsets

`offset_mode: series_first` subtracts each vehicle's first valid frame from the
context and the targets. This makes the model predict *change from a per-vehicle
baseline* rather than absolute level, which removes the between-vehicle offset
that carries no fault information. `window_first` and `none` are available for
comparison.

---

## Standardisation

Statistics are fitted **per channel** on training-normal vehicles only, then
applied unchanged to every other split.

Per-channel is not optional on this dataset. The four QAS channels differ by four
orders of magnitude:

| Channel | mean | std |
|---|---:|---:|
| `V` (mean cell voltage) | 3.33 | 0.057 |
| `I` (current) | −134.6 | **341.4** |
| `T` (temperature) | 20.9 | 6.50 |
| `SOC` | 69.3 | 19.0 |

A single shared scaler would leave current dominating every distance, and the
same imbalance reappears later in scoring — which is why residuals are also
normalised per channel before they are combined (see the main
[`README.md`](../README.md)).

This scale spread is also why the multi-channel contract trains in **fp32**:
under mixed precision the same configuration produced a non-finite gradient norm
partway through training.

---

## Reproducing the splits

```bash
python -m data.preprocessing.build_splits \
    --raw-dir  <path to the raw per-vehicle export> \
    --out-dir  data/qas_only \
    --manufacturer QAS
```

The script writes the five aggregate NPZs above plus a manifest recording the
vehicle-to-split assignment and a SHA-256 fingerprint of every source file, so a
later run can verify it is reading the same data.

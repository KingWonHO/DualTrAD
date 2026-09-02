# DualTrAD

Vehicle-level anomaly detection for electric-vehicle battery packs from onboard
BMS signals, with a dual-evidence detector and reproducible baselines under one
shared protocol.

The detector fuses two kinds of evidence produced by a single shared encoder — a
multi-horizon **forecasting** residual and a denoising **reconstruction**
residual. "Dual" refers to the two kinds of evidence, not to two branches of the
same kind and not to two encoders.

Evaluated on two public EV battery corpora with vehicle-level labels, DualTrAD
leads on the primary corpus and holds that level on a second corpus collected
from different manufacturers under a different sampling regime.

---

## Why the layout looks like this

Every published detector in this repository is trained and scored through the
same data contract, the same loss interface and the same decision rule, so a
difference between two rows of a results table is a difference between models
and nothing else.

```
DualTrAD/
├── config/                 one YAML per model; a variant states only what it changes
│   ├── qas/                single-corpus contract (4 channels)
│   └── tsinghua/           multi-brand contract (7 channels)
├── data/
│   ├── README.md           dataset construction and preprocessing  ← read this first
│   └── preprocessing/      scripts that turn raw exports into split NPZs
└── src/
    ├── datas/              splits, per-channel standardisation, windowing, corruption
    ├── losses/             the training objective and its terms
    ├── metrics/            residuals → evidence → fusion → threshold → metrics
    ├── models/
    │   ├── shared/         encoders, decoders, heads, autoencoder — reused by all
    │   ├── dualtrad/       the proposed detector
    │   ├── predtrad/       PredTrAD-V1 adapter
    │   ├── tranad/         TranAD adapter
    │   ├── dtaad/          DTAAD adapter
    │   └── third_party/    vendored upstream bodies + SHA-256 provenance
    ├── system/             trainer, evaluator, and the config runner
    └── utils/              config loading, seeding, device selection
```

Each model folder holds exactly one assembling class in `model.py`. It reads a
plain configuration dictionary, builds its submodules in `init_modules`, and
describes itself through `get_config()`, so a finished run can be reconstructed
from its serialised configuration alone.

---

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate     # Windows
pip install -r requirements.txt

python -m scripts.run --config config/qas/dualtrad.yaml --output runs/qas_dualtrad
```

Any model in the table below can be swapped in by pointing `--config` at its
YAML file. Nothing else changes.

---

## Models

| Config | Class | Evidence | Params | Notes |
|---|---|---|---:|---|
| `dualtrad.yaml` | `DualTrAD` | forecast + reconstruction | 246,844 | the proposed detector |
| `dualtrad_forecast_only.yaml` | `DualTrAD` | forecast | 246,844 | **capacity-matched control**: identical weights, fusion off |
| `ft_base.yaml` | `DualTrAD` | forecast | 32,940 | minimal reference: last-state decoder, no autoencoder |
| `predtrad.yaml` | `PredTrADv1` | forecast | 2,109,956 | Schuster et al. |
| `tranad.yaml` | `TranAD` | reconstruction (adapted) | 6,948 | Tuli et al. |
| `dtaad.yaml` | `DTAAD` | reconstruction (adapted) | 1,042 | Yu et al. |

(Parameter counts are for the four-channel contract.)

`dualtrad_forecast_only` reuses the trained weights of `dualtrad` and changes
only the scoring step, so the two differ in **nothing but the fusion**. That is
what makes a gain attributable to fusion rather than to added capacity.

### Baseline provenance

Upstream model bodies are vendored verbatim under `src/models/third_party/`
with a SHA-256 record in `PROVENANCE.json`. The only edits are import
statements, needed because the files import each other as `src.*` upstream; no
layer, hyper-parameter or forward path was changed. Each adapter records its own
adaptation in `model.get_config()["adaptation"]`, including what had to change to
turn a reconstruction model into a forecaster.

Adapted baselines are **not** reproductions of the published numbers. The
original papers use different tasks, thresholds and metrics; comparing against
their reported scores is not meaningful. All numbers here come from re-running
every method under this repository's protocol.

---

## How a score becomes a decision

The pipeline is split so that no step can see a test label before the operating
point is fixed:

```
model output ──► window residuals            src/metrics/scoring.py
                    │                        (per horizon and per channel)
                    ▼
             evidence normalisation          scales fitted on NORMAL calibration only
                    │                        each channel by its own scale
                    ▼
                 fusion                      geometric mean, or forecast-only
                    │
                    ▼
            vehicle aggregation              a high quantile of that vehicle's windows
                    │
                    ▼
               threshold                     src/metrics/decision.py
                    │                        fitted on NORMAL calibration vehicles
                    ▼
               evaluation                    ← the first and only read of a test label
```

Two details are worth stating because they are easy to get wrong:

**Normalise per channel before combining.** Residuals are kept per channel until
each has been divided by its own calibration scale. Averaging raw errors first
lets whichever channel has the largest scale dominate the score — on this corpus
that alone moved AUROC from 0.80 to 0.45.

**Fit the threshold on the distribution it is applied to.** A vehicle score is a
*mean* of high window scores; a threshold read off the *window* score
distribution sits far above it. `snippet_tail_threshold` reproduces that legacy
rule for reference, but `normal_quantile_threshold` is the default: it uses the
normal calibration vehicles at a stated target false-positive rate, consumes no
abnormal labels, and turns the operating point into a specification.

AUROC and AUPR are threshold-free and therefore identical under either rule.
Report them as the primary metrics; report `F1` only alongside the false-positive
rate it was measured at, since models at different operating points are not
comparable on `F1` alone.

---

## Results

Two corpora, one protocol, three seeds per model. AUROC and AUPR are reported
because they are threshold-free; see the note on `F1` above.

**Primary corpus** (4 channels, 94 test vehicles: 36 normal / 58 abnormal).
Mean over seeds 42/43/44, standard deviation in brackets.

| Model | AUROC | AUPR |
|---|---:|---:|
| **DualTrAD** | **0.8147** (0.0036) | **0.8956** (0.0030) |
| FT-Base | 0.8068 (0.0096) | 0.8954 (0.0090) |
| DTAAD | 0.7703 | 0.8543 |
| PredTrAD_v1 | 0.7695 (0.0042) | 0.8752 (0.0032) |
| TranAD | 0.7399 (0.0021) | 0.8565 (0.0012) |

Against every published baseline the margin holds in **all three seeds**
(seed-paired AUROC: TranAD +0.075, PredTrAD_v1 +0.045, DTAAD +0.044; 3/3 seeds
each). DualTrAD's own seed spread is 0.0036, an order of magnitude smaller than
those margins.

**Second corpus** (7 channels, 3 manufacturers x 5 vehicle folds, 10 s sampling).
Macro over brands, 15 folds.

| Model | AUROC | AUPR |
|---|---:|---:|
| **DualTrAD** | **0.7650** | **0.7309** |
| PredTrAD_v1 | 0.7625 | 0.7028 |
| FT-Base | 0.6933 | 0.6408 |
| TranAD | 0.6794 | 0.6771 |
| DTAAD | 0.5900 | 0.5641 |

DualTrAD stays at the top here, significantly ahead of FT-Base (p = 0.007) and
DTAAD (p = 0.035) by a fold-paired sign test, and level with PredTrAD_v1
(+0.002, p = 0.302). We read this as the detector *holding* its level on a
corpus it was not tuned for, not as a second win: with 15 folds a difference
this small is not separable.

Numbers are produced by the configs in `config/`; nothing here is tuned on test
data.

## Evaluation notes

- **Labels are per vehicle**, assigned retroactively by engineers after pack
  disassembly. Decisions are therefore made per vehicle, not per timestep, which
  also removes the point-adjustment ambiguity that affects timestep-level
  protocols. No peak-over-threshold calibration and no point adjustment is used
  anywhere in this repository.
- **Seeds matter more than folds.** Brand or fold averages capture data-split
  variance only; they say nothing about initialisation variance. Run at least
  three seeds before reporting a difference, and compare it against the
  seed-to-seed spread of the same model.

---

## Environment

Python 3.11, CUDA 12.8. Pinned versions are in `requirements.txt`.

| Package | Version | Used for |
|---|---|---|
| `torch` | 2.11.0+cu128 | models, training |
| `numpy` | ≥ 1.26 | residuals, scoring, metrics |
| `pyyaml` | ≥ 6.0 | configuration |
| `pandas` | ≥ 2.0 | preprocessing only |

Metrics are implemented in `src/metrics/ranking.py` rather than taken from
scikit-learn, so tie handling is fixed and auditable.

---

## Citing the corpora

Neither corpus is introduced here. Both are public and must be cited from their
original publications; see [`data/README.md`](data/README.md) for the exact
references, access links, and the preprocessing this repository applies on top
of them.

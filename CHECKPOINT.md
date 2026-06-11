# ECLIPSE Instrumentation — Development Checkpoint

## Status Overview

| Phase | Module | Status | Tests |
|-------|--------|--------|-------|
| 1 | `void_fraction.py` | ✅ Complete | 5/5 passing |
| 2 | `sensor_sim.py` | ✅ Complete | 4/4 passing |
| 3 | `flow_regime.py` | ✅ Complete | 4/4 passing |
| 4 | `demo.py` + visualizations | ✅ Complete | 5 figures generated |
| 5 | `README.md` + `requirements.txt` | ⬜ Not Started | — |

---

## Phase 1: `void_fraction.py` — ✅ Complete

**Purpose:** Implement all sensor math equations from the ECLIPSE paper as typed Python functions with physical units documented in docstrings. No ANSYS data required — pure physics.

**Equations implemented:**

| Function | Equation | Paper Reference |
|---|---|---|
| `linear_void_fraction` | `α = (C_L - C_M) / (C_L - C_G) × 100` | Sec. V.A |
| `corrected_void_fraction` | `α_c = k·α² + (1 − 100k)·α` | Eq. 1, Sec. V.A |
| `thermal_correction_length` | `ΔL = L₀·α_CTE·ΔT` | Eq. 9, App. XII.G |
| `gas_permittivity` | `ε_g = 1 + A_g·(P/T)` | Eq. 10, App. XII.G |
| `liquid_permittivity` | `ε_l = A_l + B_l/T` | Eq. 11, App. XII.G |
| `process_signal` | chains linear → corrected | — |

**Physical constants (module-level, from existing notebook):**
- `C_LIQUID_DEFAULT = 1.113e-10 * 1.51 * (10*10)/10` — ε₀·ε_r_liquid·(A/d) for 10-inch pipe
- `C_GAS_DEFAULT = 1.113e-10 * 1.000494 * (10*10)/10`

**Tests (`instrumentation/tests/test_void_fraction.py`):**
- `test_linear_void_fraction_bounds` — 0% at C_liquid, 100% at C_gas
- `test_linear_void_fraction_midpoint` — 50% at midpoint capacitance
- `test_corrected_void_fraction_identity` — k=0 leaves signal unchanged
- `test_thermal_correction_sign` — negative delta_T → negative delta_L
- `test_permittivity_physical_range` — gas permittivity > 1 always

**Test results:** 5/5 passing (`pytest instrumentation/tests/test_void_fraction.py -v`)

**Infrastructure fix:** macOS case-insensitive APFS caused the scaffold's `Instrumentation/` (uppercase) to shadow all writes to `instrumentation/` (lowercase). Python 3.14 treats these case-sensitively. Fixed with a two-step `git mv Instrumentation Instrumentation_temp && git mv Instrumentation_temp instrumentation`, permanently resolving the package name to lowercase.

---

## Phase 2: `sensor_sim.py` — ✅ Complete

Synthetic capacitance signal generator for slug, intermittent, and annular flow regimes. Makes the full pipeline runnable without ANSYS CFD data.

**Signal design (all converted to capacitance via `C = C_L − VF × (C_L − C_G)`):**

| Regime | Base VF | Dynamics | Noise σ |
|---|---|---|---|
| Slug | 0.3 | Gaussian spikes, height 0.4–0.6, width σ=0.02s, rate 2–5 Hz | 0.02 |
| Intermittent | 0.5 | Sinusoidal ±0.15 at 3–8 Hz + small spikes | 0.015 |
| Annular | 0.85–0.95 | Slow sinusoidal drift ±0.03 at 0.5–1 Hz | 0.005 |

**Ordering guarantee (by construction, not seed-dependent):**
- Mean capacitance: slug > intermittent > annular (slug max mean VF ≈ 0.44 < intermittent min ≈ 0.50)
- Variance: slug (tall narrow spikes) > intermittent (sinusoidal) > annular (slow drift)

**Tests (`instrumentation/tests/test_sensor_sim.py`):**
- `test_signal_length` — `len == int(duration_s * sample_rate_hz)` for all three regimes
- `test_signal_regime_ordering` — mean capacitance: slug > intermittent > annular
- `test_variance_ordering` — variance: slug > intermittent > annular
- `test_reproducibility` — same seed → bitwise-identical output

**Test results:** 4/4 passing. Full suite 9/9 (no regressions in Phase 1).

---

## Phase 3: `flow_regime.py` — ✅ Complete

Feature extraction + Fuzzy c-means classifier. Refactor of `Instrumentation/clustering.ipynb`.

**Functions implemented:**

| Function | Purpose |
|---|---|
| `load_ansys_data(csv_path)` | Load CSV, scale time ×0.01, assign regime labels, cap annular at 3.5s |
| `augment_with_gpr(df_regime, n_points_between)` | GPR interpolation with RBF+WhiteKernel, replicates notebook exactly |
| `extract_features(capacitance, window_size, step_size)` | Windowed [mean, variance, kurtosis] — kurtosis via `moment(x, 4)` |
| `fit_classifier(features, n_clusters, random_state)` | StandardScaler + FCM(3) + cluster relabeling. Returns 4-tuple including label_map |
| `predict_regime(features, fcm, scaler, label_map)` | Inference on new features via `fcm.predict()` → raw indices → relabeled |
| `FlowRegimeClassifier` | sklearn-style wrapper: `fit`, `predict`, `predict_proba`, `regime_name` |

**Key implementation notes:**
- Kurtosis: `scipy.stats.moment(x, moment=4)` (raw 4th central moment, not Fisher-corrected)
- ANSYS time scaling: ×0.01, regime cutoffs at 0.85s / 1.5s / 3.5s cap
- GPR augmentation on slug + intermittent only, `n_points_between=2`
- FCM relabeling: `cluster_means` sorted ascending; [0]→label 2 (Annular), [2]→label 0 (Slug)
- `fit_classifier` returns 4-tuple `(fcm, scaler, labels, label_map)` — extended from spec to enable inference
- `fcm.predict(X)` returns hard labels; `fcm.soft_predict(X)` returns membership matrix
- Package name: `fuzzy-c-means` on PyPI (not `fcmeans`); import as `from fcmeans import FCM`

**Tests (`instrumentation/tests/test_flow_regime.py`):**
- `test_extract_features_shape` — `n_windows = n // window_size` for non-overlapping
- `test_extract_features_mean` — mean feature of constant signal equals that constant
- `test_classifier_three_regimes` — fit on ANSYS+GPR data produces exactly 3 distinct labels
- `test_regime_relabeling` — slug cluster (label 0) has higher mean capacitance than annular (label 2)

**Test results:** 4/4 passing. Full suite 13/13 (no regressions).

---

## Phase 4: `demo.py` — ✅ Complete

End-to-end showcase producing 5 figures in `instrumentation/outputs/`.

**Figures generated:**
1. `signal_examples.png` — 3-panel time series per regime (1s each, title shows mean VF and variance)
2. `void_fraction_pipeline.png` — raw C → corrected void fraction % with equation annotations
3. `flow_regime_map_3d.png` — 3D scatter (ANSYS ● + Synthetic ▲), log-transformed variance/kurtosis
4. `flow_regime_map_2d.png` — 2-panel 2D projections, same coloring
5. `regime_distribution.png` — before/after GPR augmentation pie charts

**Key implementation notes:**
- `sys.path.insert(0, project_root)` at top of demo.py so it runs as `python instrumentation/demo.py` from any working directory
- Scatter plots use log10-transformed variance and kurtosis (both axes span 5+ orders of magnitude) with a combined StandardScaler fit on ANSYS + synthetic features
- ANSYS points use true time-based labels (not FCM labels) for scatter visualization; FCM labels are reported in the pipeline step only
- Cluster separation (slug/annular centroid Euclidean distance in normalized space) = 3.69
- Package `fuzzy-c-means` on PyPI (not `fcmeans`) for `from fcmeans import FCM`; added to requirements.txt as `fuzzy-c-means>=1.5`

---

## Phase 5: `README.md` + `requirements.txt` — ⬜ Not Started

Software-engineer-facing README and dependency file.

---

## Decisions & Non-Obvious Choices

| Decision | Reason |
|---|---|
| `scipy.stats.moment(x, moment=4)` for kurtosis | Notebook uses raw 4th central moment; `scipy.stats.kurtosis()` applies Fisher correction, changing values vs. paper |
| Annular GPR cap at t=3.5s | Notebook explicitly filters annular data to max_annular_time=3.5; annular is well-represented without augmentation |
| FCM relabeling: sorted ascending, highest mean = Slug | Physics: slug = low void fraction = mostly liquid = highest capacitance |
| Notebook relabeling comments are wrong | Notebook maps `sorted_clusters[0] → label 2` with comment `# Slug`, but label 2 = Annular in labels_str. Comments inverted; code logic is physically correct. |

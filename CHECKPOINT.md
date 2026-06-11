# ECLIPSE Instrumentation — Development Checkpoint

## Status Overview

| Phase | Module | Status | Tests |
|-------|--------|--------|-------|
| 1 | `void_fraction.py` | ✅ Complete | 5/5 passing |
| 2 | `sensor_sim.py` | ⬜ Not Started | — |
| 3 | `flow_regime.py` | ⬜ Not Started | — |
| 4 | `demo.py` + visualizations | ⬜ Not Started | — |
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

## Phase 2: `sensor_sim.py` — ⬜ Not Started

Synthetic capacitance signal generator for slug, intermittent, and annular flow regimes. Makes the full pipeline runnable without ANSYS CFD data.

**Key signals to generate:**
- Slug: base VF ~0.3, Gaussian-enveloped spikes at 2–5 Hz driving VF → 0.8–1.0, σ=0.02 noise
- Intermittent: base VF ~0.5, oscillations at 3–8 Hz, σ=0.015 noise
- Annular: VF ~0.85–0.95, slow sinusoidal drift at 0.5–1 Hz, σ=0.005 noise

---

## Phase 3: `flow_regime.py` — ⬜ Not Started

Feature extraction + Fuzzy c-means classifier. Refactor of `Instrumentation/clustering.ipynb`.

**Critical implementation notes:**
- Kurtosis: `scipy.stats.moment(x, moment=4)` (raw 4th central moment, not Fisher-corrected)
- ANSYS time scaling: ×0.01, regime cutoffs at 0.85s / 1.5s / 3.5s cap
- GPR augmentation on slug + intermittent only, `n_points_between=2`
- FCM relabeling: highest mean C → Slug (label 0), lowest mean C → Annular (label 2)

---

## Phase 4: `demo.py` — ⬜ Not Started

End-to-end showcase producing 5 figures in `instrumentation/outputs/`. Depends on Phases 1–3 all passing tests.

**Figures:**
1. `signal_examples.png` — 3-panel time series per regime
2. `void_fraction_pipeline.png` — raw C → corrected void fraction %
3. `flow_regime_map_3d.png` — 3D scatter, ANSYS + synthetic overlaid
4. `flow_regime_map_2d.png` — 2-panel 2D projections
5. `regime_distribution.png` — before/after GPR pie charts

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

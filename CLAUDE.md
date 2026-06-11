# Claude Code Brief: ECLIPSE Instrumentation Rebuild

Project: ECLIPSE — NASA Human Lander Challenge (HuLC 2026), UIUC
Owner: Aneesh Ganti (aneeshg5) — Instrumentation Team Lead
Goal: Refactor and significantly expand the instrumentation codebase from a single research notebook into a clean, demonstrable Python package that showcases ML/signal processing skills to any software or ML engineer, with no aerospace background required to appreciate it.

## PART 1: REPO SETUP (Do This First)

### Step 1 — Fork the repo to Aneesh's GitHub account

The original repo lives at `https://github.com/ISSUIUC/HuLC_2025_UIUC`. You need to fork it to `aneeshg5`'s account so Aneesh owns it on his profile.

```bash
# Using the GitHub MCP or gh CLI:
gh repo fork ISSUIUC/HuLC_2025_UIUC --clone --remote
# This creates github.com/aneeshg5/HuLC_2025_UIUC
# and clones it locally with `upstream` pointed at the original
```

If using the GitHub MCP server instead of gh CLI, call the fork endpoint for `ISSUIUC/HuLC_2025_UIUC` targeting the `aneeshg5` account.

### Step 2 — Clone and set up local environment

```bash
git clone https://github.com/aneeshg5/HuLC_2025_UIUC.git
cd HuLC_2025_UIUC
python3 -m venv .venv
source .venv/bin/activate
pip install -r instrumentation/requirements.txt  # you will create this
```

### Step 3 — Verify the existing files are intact

The repo should contain:

```
Instrumentation/
  clustering.ipynb          ← existing notebook (the core prior work)
  volume-average-rfile.csv  ← ANSYS simulation output (1001 rows: time, Volume fraction)
trigger_point_temperature.py
GFSSP Simulations/          ← large binary simulation results, do not touch
```

## PART 2: WHAT TO BUILD

### Target directory structure

```
instrumentation/
  __init__.py
  sensor_sim.py         ← NEW: synthetic capacitance signal generator
  void_fraction.py      ← NEW: void fraction pipeline + thermal corrections
  flow_regime.py        ← NEW: feature extraction + Fuzzy c-means classifier
  data/
    volume-average-rfile.csv   ← move/copy from Instrumentation/
  tests/
    __init__.py
    test_void_fraction.py
    test_flow_regime.py
    test_sensor_sim.py
  notebooks/
    flow_regime_map.ipynb   ← cleaned-up replacement for clustering.ipynb
  demo.py               ← single entrypoint: run this to see everything
  requirements.txt
README.md               ← project-level, written for a software engineer audience
```

Keep the old `Instrumentation/clustering.ipynb` in place — don't delete prior work.

## PART 3: MODULE SPECIFICATIONS

### `instrumentation/sensor_sim.py` — Synthetic Signal Generator

Purpose: Generate realistic time-domain capacitance signals for each of the three flow regimes (slug, intermittent, annular). This makes the entire pipeline self-contained — no one needs to run ANSYS to use or demo the code.

**Physics background (for correct implementation):**

- The sensor measures capacitance between two asymmetric parallel electrodes around a cryogenic transfer pipe
- Capacitance is related to void fraction (fraction of pipe cross-section that is vapor vs liquid) by:

```
C_linear = C_liquid - void_fraction * (C_liquid - C_gas)
```

where `C_liquid = 1.113e-10 * 1.51 * (100/10)` F and `C_gas = 1.113e-10 * 1.000494 * (100/10)` F (These exact values come from the existing notebook — use them)

- Each flow regime has a characteristic statistical signature in the time-domain signal:
  - Slug flow: high variance, high kurtosis — intermittent large bubbles cause sharp spikes
  - Intermittent flow: moderate variance, moderate kurtosis — transition regime
  - Annular flow: low variance, low kurtosis — stable vapor core, smooth signal near C_gas

**What to implement:**

```python
def generate_signal(
    regime: str,           # 'slug', 'intermittent', or 'annular'
    duration_s: float,     # signal duration in seconds
    sample_rate_hz: float, # must be >= 200 Hz per paper spec
    seed: int = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (time_array, capacitance_array).
    Signal characteristics must match the statistical profile described
    in the paper: slug has highest variance/kurtosis, annular has lowest.
    """
```

**Implementation approach per regime:**

- Slug: base void fraction ~0.3, with random large spikes (void fraction → 0.8–1.0) occurring at ~2–5 Hz, using a smooth spike shape (Gaussian envelope). Add Gaussian noise σ=0.02.
- Intermittent: base void fraction ~0.5, moderate oscillations at ~3–8 Hz, smaller spikes. Add Gaussian noise σ=0.015.
- Annular: void fraction ~0.85–0.95, slow sinusoidal drift at ~0.5–1 Hz, very low noise σ=0.005.

Convert all void fraction signals to capacitance using the formula above.

Also implement:

```python
def generate_dataset(
    n_samples_per_regime: int = 200,
    duration_s: float = 2.0,
    sample_rate_hz: float = 200.0,
    seed: int = 42
) -> dict:
    """
    Returns dict with keys 'slug', 'intermittent', 'annular',
    each containing a list of (time, capacitance) tuples.
    """
```

### `instrumentation/void_fraction.py` — Void Fraction Pipeline

Purpose: Implement all the sensor math from the paper cleanly with docstrings, type hints, and physical units documented. These equations exist only as math in the appendix — making them runnable code is a real contribution.

**Implement these functions:**

```python
def linear_void_fraction(
    C_measured: float | np.ndarray,
    C_liquid: float,
    C_gas: float
) -> float | np.ndarray:
    """
    Eq. from Section V.A of ECLIPSE paper.
    alpha_linear = (C_L - C_M) / (C_L - C_G) * 100%
    
    Returns void fraction in [0, 100] percent.
    C_liquid: capacitance when pipe is full liquid (F)
    C_gas: capacitance when pipe is full vapor (F)
    C_measured: measured capacitance (F)
    """

def corrected_void_fraction(
    alpha_linear: float | np.ndarray,
    k: float
) -> float | np.ndarray:
    """
    Eq. 1 from ECLIPSE paper Section V.A.
    alpha_corrected = k * alpha_linear^2 + (1 - 100k) * alpha_linear  [%]
    
    k is proportional to electrode separation distance.
    Typical range for 10-inch pipe: k ≈ 0.001–0.005 (calibrate via FEM).
    """

def thermal_correction_length(
    L0: float,
    alpha_cte: float,
    delta_T: float
) -> float:
    """
    Eq. 9 from ECLIPSE paper Appendix XII.G.
    Delta_L = L0 * alpha * Delta_T
    
    Predicts dimensional change in sensor components due to thermal contraction.
    L0: initial length (m)
    alpha_cte: thermal expansion coefficient (m/m/K)
    delta_T: temperature change (K), negative for cooling
    Returns: change in length (m)
    """

def gas_permittivity(
    A_g: float,
    P: float,
    T: float
) -> float:
    """
    Eq. 10 from ECLIPSE paper Appendix XII.G.
    epsilon_g = 1 + A_g * (P / T)
    
    Models permittivity variation of cryogenic vapor with temperature/pressure.
    A_g: material-specific constant
    P: pressure (Pa)
    T: temperature (K)
    """

def liquid_permittivity(
    A_l: float,
    B_l: float,
    T: float
) -> float:
    """
    Eq. 11 from ECLIPSE paper Appendix XII.G.
    epsilon_l = A_l + B_l / T
    
    Models permittivity variation of cryogenic liquid with temperature.
    """

def process_signal(
    capacitance: np.ndarray,
    C_liquid: float,
    C_gas: float,
    k: float = 0.002
) -> np.ndarray:
    """
    Full pipeline: raw capacitance → corrected void fraction.
    Applies linear_void_fraction then corrected_void_fraction.
    """
```

**Use these physical constants (from the existing notebook):**

```python
C_LIQUID_DEFAULT = 1.113e-10 * 1.51 * (10 * 10) / 10   # ~1.682e-10 F
C_GAS_DEFAULT    = 1.113e-10 * 1.000494 * (10 * 10) / 10 # ~1.113e-10 F
```

### `instrumentation/flow_regime.py` — Feature Extraction + Classification

Purpose: Extract the three statistical moments from windowed capacitance signals, then classify them into flow regimes using Fuzzy c-means. This is a refactor + significant upgrade of the existing `clustering.ipynb`.

**What the existing notebook does (understand this first):**

1. Loads `volume-average-rfile.csv` (1001 rows: time, Volume fraction from ANSYS)
2. Scales time by 0.01 so final time ≈ 10s
3. Converts void fraction → capacitance using linear formula
4. Time-based categorization: slug ≤ 0.85s, intermittent ≤ 1.5s, annular > 1.5s
5. GPR (Gaussian Process Regression) interpolation on slug + intermittent to augment data
6. Windows signals with Δt = 0.1s (interval_size=20 at 200Hz equivalent), extracts mean/variance/kurtosis per window
7. StandardScaler normalization
8. Fuzzy c-means (FCM) with 3 clusters
9. Relabels clusters by mean capacitance: lowest=slug, middle=intermittent, highest=annular
10. Produces 3D scatter plot + 2D projections

**What to implement in `flow_regime.py`:**

```python
def extract_features(
    capacitance: np.ndarray,
    window_size: int,
    step_size: int = None  # defaults to window_size (non-overlapping)
) -> np.ndarray:
    """
    Slide a window over the capacitance signal.
    For each window, compute: [mean, variance, kurtosis].
    Returns array of shape (n_windows, 3).
    
    Uses scipy.stats.moment(x, moment=4) for kurtosis (4th central moment / sigma^4).
    This matches the existing notebook exactly — do NOT use scipy.stats.kurtosis
    which applies Fisher's correction. Use moment(x, moment=4) directly.
    """

def fit_classifier(
    features: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42
) -> tuple:
    """
    Normalize features with StandardScaler, fit Fuzzy c-means.
    Returns (fcm_model, scaler, labels) where labels are relabeled
    so 0=Annular, 1=Intermittent, 2=Slug (by ascending mean capacitance).
    
    Use fcmeans.FCM from the fcm-means package.
    Relabeling logic: sort cluster centroids by mean capacitance (feature 0).
    Lowest mean = slug (most liquid), highest mean = annular (most vapor).
    This matches the existing notebook's relabel_clusters() function.
    """

def predict_regime(
    features: np.ndarray,
    fcm_model,
    scaler,
    label_map: dict = None
) -> np.ndarray:
    """
    Predict flow regime labels for new feature vectors.
    Returns integer labels and string regime names.
    """

def load_ansys_data(csv_path: str) -> pd.DataFrame:
    """
    Load the ANSYS volume-average output CSV.
    Applies time scaling (×0.01), void fraction→capacitance conversion,
    and time-based flow regime labeling.
    
    Time boundaries (from paper/notebook):
      slug: time <= 0.85s
      intermittent: 0.85s < time <= 1.5s  
      annular: time > 1.5s
    """

def augment_with_gpr(
    df_regime: pd.DataFrame,
    n_points_between: int = 2
) -> pd.DataFrame:
    """
    GPR interpolation to augment data-sparse regimes.
    Replicates interpolate_selected_data() from the notebook.
    Kernel: 1.0 * RBF(length_scale=0.2) + WhiteKernel(noise_level=0.05)
    """
```

Also add a `FlowRegimeClassifier` class that wraps fit + predict with a clean sklearn-style API:

```python
class FlowRegimeClassifier:
    def fit(self, features): ...
    def predict(self, features): ...
    def predict_proba(self, features): ...  # FCM membership values
    def regime_name(self, label: int) -> str: ...
```

### `instrumentation/demo.py` — The Showcase Entrypoint

Purpose: A single script anyone can run (`python demo.py`) that produces all key outputs. No prior knowledge of cryogenics required to understand what it's doing.

**What it should do:**

```
$ python instrumentation/demo.py

ECLIPSE Instrumentation Demo
=============================
Sensor: Asymmetric capacitance sensor for cryogenic two-phase flow monitoring
Pipeline: Capacitance signal → Void fraction → Flow regime classification

[1/4] Generating synthetic capacitance signals...
  ✓ 200 slug flow samples (2.0s each @ 200Hz)
  ✓ 200 intermittent flow samples
  ✓ 200 annular flow samples

[2/4] Processing void fraction...
  ✓ Linear void fraction extraction
  ✓ Thermal correction applied (k=0.002)
  
[3/4] Training flow regime classifier...
  ✓ Feature extraction: mean, variance, kurtosis per 0.1s window
  ✓ GPR augmentation on ANSYS simulation data
  ✓ Fuzzy c-means clustering (3 regimes)
  Cluster separation: slug/annular centroid distance = X.XX

[4/4] Generating visualizations...
  ✓ Saved: outputs/signal_examples.png
  ✓ Saved: outputs/void_fraction_pipeline.png
  ✓ Saved: outputs/flow_regime_map_3d.png
  ✓ Saved: outputs/flow_regime_map_2d.png
  ✓ Saved: outputs/regime_distribution.png

Done. Open outputs/ to view results.
```

**Figures to generate (save to `instrumentation/outputs/`):**

1. `signal_examples.png` — 3-panel subplot, one per regime. Each panel shows a 1-second capacitance signal time series. Title each panel with regime name + key stats (mean void fraction, variance). Clear legend, axes labeled "Time (s)" and "Capacitance (F)". This is the most visually intuitive figure — anyone immediately sees slug is spiky, annular is smooth.
2. `void_fraction_pipeline.png` — 2-panel figure showing one signal going through the pipeline: raw capacitance (top) → corrected void fraction % (bottom). Annotate with the equations being applied. Shows the math is real and running.
3. `flow_regime_map_3d.png` — The 3D scatter plot of (normalized mean capacitance, variance, kurtosis) colored by regime. This should look at least as good as Fig. 11 in the paper, ideally better (use matplotlib `rcParams` for larger fonts, tight layout, good viewing angle). Show both ANSYS-derived points AND synthetic signal points as different marker styles to show the two data sources agree.
4. `flow_regime_map_2d.png` — 2-panel figure with mean vs variance and variance vs kurtosis projections. Same coloring.
5. `regime_distribution.png` — Before/after pie charts showing original ANSYS data distribution vs. after GPR augmentation. Already exists in the notebook — just clean it up.

### `instrumentation/tests/` — Test Suite

**Tests for `void_fraction.py`:**

- `test_linear_void_fraction_bounds`: void fraction = 0% when C_measured = C_liquid, 100% when C_measured = C_gas
- `test_linear_void_fraction_midpoint`: void fraction = 50% when C_measured is exactly midpoint
- `test_corrected_void_fraction_identity`: when k=0, corrected = linear
- `test_thermal_correction_sign`: delta_T negative → delta_L negative
- `test_permittivity_physical_range`: gas permittivity > 1 (physical requirement)

**Tests for `flow_regime.py`:**

- `test_extract_features_shape`: n windows = floor(n_samples / window_size), features shape is (n_windows, 3)
- `test_extract_features_mean`: mean of a constant signal equals that constant
- `test_classifier_three_regimes`: fit on full ANSYS data → exactly 3 clusters
- `test_regime_relabeling`: after relabeling, slug cluster has lowest mean capacitance of the three

**Tests for `sensor_sim.py`:**

- `test_signal_length`: output arrays have length = duration_s * sample_rate_hz
- `test_signal_regime_ordering`: mean(slug capacitance) < mean(intermittent) < mean(annular) — physics check
- `test_variance_ordering`: var(slug) > var(intermittent) > var(annular) — slug is spikiest
- `test_reproducibility`: same seed → same output

Run with: `pytest instrumentation/tests/ -v`

## PART 4: README.md

Write a top-level `README.md` (replacing or supplementing any existing one) targeted at a software or ML engineer who has never heard of HuLC. Structure:

```markdown
# ECLIPSE — Cryogenic Two-Phase Flow Instrumentation

**What this is:** A Python implementation of the sensor signal processing pipeline 
from UIUC's submission to NASA's Human Lander Challenge (HuLC 2026). The pipeline 
classifies flow regimes in cryogenic propellant transfer lines using capacitance 
sensing + unsupervised ML.

**Why it matters:** [1-2 sentences on the mission context — propellant transfer for Artemis]

## The Problem
[Explain inverted annular film boiling in 2 sentences max. Then explain why monitoring 
the flow regime matters operationally.]

## The Pipeline
[Simple diagram or ASCII art showing: Raw capacitance signal → Void fraction → 
Statistical features → Fuzzy c-means → Regime label]

## Flow Regimes
[3-bullet explanation: slug, intermittent, annular — with what each means physically]

## Quickstart
pip install -r instrumentation/requirements.txt
python instrumentation/demo.py

## Output
[Show the 3D flow regime map figure inline]

## Technical Details
- Sensor: asymmetric capacitance electrodes, 200Hz+ sampling, 25V potential difference
- Void fraction: corrected per Sakamoto et al. (2018) calibration model
- Augmentation: Gaussian Process Regression on sparse regimes before clustering
- Classifier: Fuzzy c-means (scikit-fuzzy / fcm-means), 3 clusters
- Test coverage: pytest, covers physics invariants + ML pipeline

## My Role
Instrumentation Team Lead — I designed and implemented the sensor algorithm, 
void fraction pipeline, and flow regime identification approach described in 
Section V of the technical paper.
```

## PART 5: `requirements.txt`

```
numpy>=1.24
pandas>=2.0
scipy>=1.11
scikit-learn>=1.3
fcmeans>=1.5        # for FCM clustering
matplotlib>=3.7
pytest>=7.4
```

## PART 6: KEY DECISIONS + CONSTRAINTS

### Do not break existing work:

- Keep `Instrumentation/clustering.ipynb` and `Instrumentation/volume-average-rfile.csv` exactly as-is
- Keep `trigger_point_temperature.py` as-is
- Keep all GFSSP simulation files untouched
- The new `instrumentation/` package is additive

### Physical correctness non-negotiables:

- Kurtosis must use `scipy.stats.moment(x, moment=4)` — NOT `scipy.stats.kurtosis()`. The notebook uses raw 4th central moment, not Fisher-corrected kurtosis. This matters for consistency with the paper.
- The time scaling in the ANSYS CSV is `time × 0.01`. The raw CSV has integer time steps 0–1000; scaled time is 0–10s. The regime boundaries (0.85s, 1.5s) apply to scaled time.
- The capacitance formula from the notebook: `C = C_L - void_fraction * (C_L - C_G)`. Note the notebook uses `C_liquid = 1.113e-10 * 1.51 * (10*10)/10` — this is `epsilon_0 * epsilon_r * A/d` for the electrode geometry. Keep these exact values.

### Code quality targets:

- Type hints on all public functions
- Docstrings with units documented for every physics parameter
- No magic numbers — define physical constants at module level with comments
- All figures: minimum font size 12pt, tight layout, saved at 300 DPI
- `demo.py` must work from a clean venv with just `requirements.txt` installed

### Commit strategy: Make meaningful atomic commits, not one giant commit:

1. `feat: add void_fraction module with thermal correction equations`
2. `feat: add sensor_sim synthetic signal generator`
3. `feat: refactor flow regime classifier from notebook`
4. `test: add pytest suite for instrumentation modules`
5. `feat: add demo.py end-to-end showcase`
6. `docs: add README with software-engineer-friendly explanation`

## PART 7: CONTEXT FOR CLAUDE CODE

This is Aneesh Ganti's project. He was the Instrumentation Team Lead on ECLIPSE, a NASA Human Lander Challenge submission from UIUC's Illinois Space Society. The instrumentation section (Section V of the paper) is the part he owns — it covers a capacitance-based void fraction sensor for monitoring cryogenic propellant flow (LOX/LCH4) during transfer in microgravity.

The existing `clustering.ipynb` is real research code — it works but it's not clean, not documented, and not runnable without understanding the context. The goal is to transform that work into production-quality Python that demonstrates ML engineering skills (not just aerospace knowledge) to anyone who looks at the repo.

The synthetic signal generator (`sensor_sim.py`) is the key addition that didn't exist before — it makes the pipeline fully self-contained so anyone can run `python demo.py` and see meaningful output without needing ANSYS CFD data.

The paper PDF is the ground truth for all equations and physical constants. When in doubt about a formula, refer to Sections V.A–V.D and Appendices XII.E–XII.G.

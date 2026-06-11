from __future__ import annotations

import numpy as np
import pandas as pd
from fcmeans import FCM
from scipy.stats import moment as _moment
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

from instrumentation.void_fraction import C_GAS_DEFAULT, C_LIQUID_DEFAULT

REGIME_NAMES: tuple[str, ...] = ("Slug", "Intermittent", "Annular")

_SLUG_CUTOFF: float = 0.85
_INTERMITTENT_CUTOFF: float = 1.5
_ANNULAR_CAP: float = 3.5


def load_ansys_data(csv_path: str) -> pd.DataFrame:
    """
    Load ANSYS volume-average CSV and label each row by flow regime.

    Applies time scaling (×0.01), converts void fraction to capacitance (F),
    and assigns flow regime labels based on scaled time boundaries:
      Slug ≤ 0.85s, Intermittent ≤ 1.5s, Annular > 1.5s (capped at 3.5s).

    csv_path: path to volume-average-rfile.csv (1001 rows: time, Volume fraction)
    Returns DataFrame with columns: time (s), Volume fraction, capacitance (F), flow_regime.
    """
    df = pd.read_csv(csv_path)
    df["time"] = df["time"] * 0.01
    df["capacitance"] = C_LIQUID_DEFAULT - df["Volume fraction"] * (C_LIQUID_DEFAULT - C_GAS_DEFAULT)

    def _label(t: float) -> str:
        if t <= _SLUG_CUTOFF:
            return "Slug"
        if t <= _INTERMITTENT_CUTOFF:
            return "Intermittent"
        return "Annular"

    df["flow_regime"] = df["time"].apply(_label)

    mask = (df["flow_regime"] != "Annular") | (df["time"] <= _ANNULAR_CAP)
    return df[mask].reset_index(drop=True)


def augment_with_gpr(df_regime: pd.DataFrame, n_points_between: int = 2) -> pd.DataFrame:
    """
    Augment sparse flow regime data with Gaussian Process Regression interpolation.

    Fits a GPR on the existing (time, void fraction) pairs, then predicts at
    n_points_between evenly-spaced intermediate times between every consecutive
    pair of original time steps.

    Kernel: 1.0 × RBF(length_scale=0.2) + WhiteKernel(noise_level=0.05).
    Replicates interpolate_selected_data() from the original clustering notebook.

    df_regime: DataFrame with 'time' (s) and 'Volume fraction' columns for one regime.
    n_points_between: number of new time points to insert between each original pair.
    Returns DataFrame with columns: time (s), Volume fraction, uncertainty, capacitance (F).
    """
    kernel = 1.0 * RBF(length_scale=0.2) + WhiteKernel(noise_level=0.05)
    gpr = GaussianProcessRegressor(kernel=kernel, optimizer="fmin_l_bfgs_b", random_state=0)

    X = df_regime["time"].values.reshape(-1, 1)
    y = df_regime["Volume fraction"].values
    gpr.fit(X, y)

    new_times: list[float] = []
    for i in range(len(X) - 1):
        between = np.linspace(X[i, 0], X[i + 1, 0], n_points_between + 2)[1:-1]
        new_times.extend(between.tolist())

    all_times = np.sort(np.concatenate([X.flatten(), np.asarray(new_times)]))
    predictions, std = gpr.predict(all_times.reshape(-1, 1), return_std=True)

    result = pd.DataFrame({
        "time": all_times,
        "Volume fraction": predictions,
        "uncertainty": std,
    })
    result["capacitance"] = (
        C_LIQUID_DEFAULT - result["Volume fraction"] * (C_LIQUID_DEFAULT - C_GAS_DEFAULT)
    )
    return result


def extract_features(
    capacitance: np.ndarray,
    window_size: int,
    step_size: int | None = None,
) -> np.ndarray:
    """
    Extract [mean, variance, kurtosis] features from a capacitance signal via a sliding window.

    window_size: number of samples per window
    step_size: samples between successive window starts; defaults to window_size (non-overlapping)

    Kurtosis is the raw 4th central moment: E[(X − E[X])⁴], computed via
    scipy.stats.moment(x, moment=4). This matches the original notebook exactly
    and must NOT be replaced with scipy.stats.kurtosis() (Fisher-corrected).

    Returns array of shape (floor((n − window_size) / step_size) + 1, 3).
    For the default non-overlapping case this equals (n // window_size, 3).
    """
    if step_size is None:
        step_size = window_size

    n = len(capacitance)
    n_windows = (n - window_size) // step_size + 1
    features = np.empty((n_windows, 3), dtype=float)

    for i in range(n_windows):
        start = i * step_size
        w = capacitance[start : start + window_size]
        features[i, 0] = np.mean(w)
        features[i, 1] = np.var(w)
        features[i, 2] = _moment(w, moment=4)

    return features


def fit_classifier(
    features: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
) -> tuple:
    """
    Normalise features with StandardScaler and fit a Fuzzy c-means classifier.

    Relabels raw FCM clusters by ascending mean capacitance (feature column 0):
      label 0 = Slug (highest mean capacitance, mostly liquid)
      label 1 = Intermittent
      label 2 = Annular (lowest mean capacitance, mostly vapor)

    Returns (fcm_model, scaler, labels, label_map).
      labels: integer regime label per input window, shape (n_windows,)
      label_map: dict mapping raw FCM cluster index → regime label (0/1/2)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    fcm = FCM(n_clusters=n_clusters, random_state=random_state)
    fcm.fit(X_scaled)

    raw_labels: np.ndarray = fcm.u.argmax(axis=1)

    cluster_means = sorted(
        [(i, float(np.mean(X_scaled[raw_labels == i, 0]))) for i in range(n_clusters)],
        key=lambda pair: pair[1],
    )

    label_map: dict[int, int] = {
        cluster_means[0][0]: 2,
        cluster_means[1][0]: 1,
        cluster_means[2][0]: 0,
    }

    labels = np.array([label_map[lbl] for lbl in raw_labels])
    return fcm, scaler, labels, label_map


def predict_regime(
    features: np.ndarray,
    fcm_model: FCM,
    scaler: StandardScaler,
    label_map: dict[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict flow regime for new feature vectors.

    label_map: mapping from raw FCM cluster index → regime label (0/1/2).
               Pass the value returned by fit_classifier for consistent labeling.
               If None, raw FCM cluster indices are returned unchanged.

    Returns (integer_labels, string_regime_names), both shape (n_windows,).
    """
    X_scaled = scaler.transform(features)
    raw_labels: np.ndarray = fcm_model.predict(X_scaled)

    if label_map is not None:
        labels = np.array([label_map[int(lbl)] for lbl in raw_labels])
    else:
        labels = raw_labels

    names = np.array([REGIME_NAMES[lbl] for lbl in labels])
    return labels, names


class FlowRegimeClassifier:
    """
    Sklearn-style wrapper for the ECLIPSE Fuzzy c-means flow regime classifier.

    Fit on pre-extracted [mean, variance, kurtosis] feature arrays from capacitance windows.
    Predicts Slug (0), Intermittent (1), or Annular (2) flow regimes.
    """

    def __init__(self) -> None:
        self._fcm: FCM | None = None
        self._scaler: StandardScaler | None = None
        self._label_map: dict[int, int] | None = None

    def fit(self, features: np.ndarray) -> "FlowRegimeClassifier":
        """Fit the classifier on pre-extracted features. Returns self."""
        self._fcm, self._scaler, _, self._label_map = fit_classifier(features)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return integer regime labels (0=Slug, 1=Intermittent, 2=Annular)."""
        if self._fcm is None:
            raise RuntimeError("call fit() before predict()")
        labels, _ = predict_regime(features, self._fcm, self._scaler, self._label_map)
        return labels

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Return FCM soft membership values, shape (n_samples, n_clusters).
        Column ordering matches REGIME_NAMES after applying the internal label_map.
        """
        if self._fcm is None:
            raise RuntimeError("call fit() before predict_proba()")
        X_scaled = self._scaler.transform(features)
        soft = self._fcm.soft_predict(X_scaled)
        inv_map = {v: k for k, v in self._label_map.items()}
        return np.stack([soft[:, inv_map[j]] for j in range(len(REGIME_NAMES))], axis=1)

    def regime_name(self, label: int) -> str:
        """Return the string name for an integer regime label."""
        return REGIME_NAMES[label]

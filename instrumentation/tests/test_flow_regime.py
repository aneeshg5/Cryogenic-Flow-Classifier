import numpy as np
import pandas as pd
import pytest

from instrumentation.flow_regime import (
    augment_with_gpr,
    extract_features,
    fit_classifier,
    load_ansys_data,
)

_CSV = "instrumentation/data/volume-average-rfile.csv"
_WINDOW = 20


@pytest.fixture(scope="module")
def ansys_features() -> np.ndarray:
    df = load_ansys_data(_CSV)
    slug = augment_with_gpr(df[df["flow_regime"] == "Slug"].copy(), 2)
    intermittent = augment_with_gpr(df[df["flow_regime"] == "Intermittent"].copy(), 2)
    annular = df[df["flow_regime"] == "Annular"].copy()
    combined = pd.concat([slug, intermittent, annular]).sort_values("time")
    return extract_features(combined["capacitance"].values, _WINDOW)


def test_extract_features_shape():
    n = 100
    cap = np.random.default_rng(0).random(n)
    features = extract_features(cap, _WINDOW)
    assert features.shape == (n // _WINDOW, 3)


def test_extract_features_mean():
    value = 1.5e-9
    cap = np.full(100, value)
    features = extract_features(cap, _WINDOW)
    np.testing.assert_allclose(features[:, 0], value)


def test_classifier_three_regimes(ansys_features):
    _, _, labels, _ = fit_classifier(ansys_features)
    assert len(np.unique(labels)) == 3


def test_regime_relabeling(ansys_features):
    _, _, labels, _ = fit_classifier(ansys_features)
    slug_mean_C = np.mean(ansys_features[labels == 0, 0])
    annular_mean_C = np.mean(ansys_features[labels == 2, 0])
    assert slug_mean_C > annular_mean_C

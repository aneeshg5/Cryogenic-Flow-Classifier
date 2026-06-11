import numpy as np

from instrumentation.sensor_sim import generate_signal


def test_signal_length():
    expected = int(2.0 * 200.0)
    for regime in ("slug", "intermittent", "annular"):
        t, C = generate_signal(regime, 2.0, 200.0, seed=0)
        assert len(t) == expected
        assert len(C) == expected


def test_signal_regime_ordering():
    _, C_slug = generate_signal("slug", 2.0, 200.0, seed=0)
    _, C_int = generate_signal("intermittent", 2.0, 200.0, seed=0)
    _, C_ann = generate_signal("annular", 2.0, 200.0, seed=0)
    assert np.mean(C_slug) > np.mean(C_int)
    assert np.mean(C_int) > np.mean(C_ann)


def test_variance_ordering():
    _, C_slug = generate_signal("slug", 2.0, 200.0, seed=0)
    _, C_int = generate_signal("intermittent", 2.0, 200.0, seed=0)
    _, C_ann = generate_signal("annular", 2.0, 200.0, seed=0)
    assert np.var(C_slug) > np.var(C_int)
    assert np.var(C_int) > np.var(C_ann)


def test_reproducibility():
    t1, C1 = generate_signal("slug", 2.0, 200.0, seed=42)
    t2, C2 = generate_signal("slug", 2.0, 200.0, seed=42)
    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(C1, C2)

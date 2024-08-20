from typing import Any

import numpy as np

from src.tedi.means import (
    Constant,
    Cosine,
    Cubic,
    Keplerian,
    Linear,
    Parabola,
    Sine,
    UdHO,
)


def test_Constant() -> None:
    model = Constant(c=5.0)
    t = np.array([0, 1, 2])
    result = model(t)
    expected = np.full_like(t, 5.0)
    np.testing.assert_array_equal(result, expected)


def test_Linear() -> None:
    model = Linear(slope=2.0, intercept=3.0)
    t = np.array([0, 1, 2])
    result = model(t)
    expected = 2.0 * (t - t.mean()) + 3.0
    np.testing.assert_array_almost_equal(result, expected)


def test_Parabola() -> None:
    model = Parabola(quad=1.0, slope=0.0, intercept=0.0)
    t = np.array([0, 1, 2])
    result = model(t)
    expected = np.polyval([1.0, 0.0, 0.0], t)
    np.testing.assert_array_almost_equal(result, expected)


def test_Cubic() -> None:
    model = Cubic(cub=1.0, quad=0.0, slope=0.0, intercept=0.0)
    t = np.array([0, 1, 2])
    result = model(t)
    expected = np.polyval([1.0, 0.0, 0.0, 0.0], t)
    np.testing.assert_array_almost_equal(result, expected)


def test_Sine() -> None:
    model = Sine(amp=1.0, p=2.0, phi=np.pi / 4, d=0.0)
    t = np.array([0, 1, 2])
    result = model(t)
    expected = 1.0 * np.sin((2 * np.pi * t / 2.0) + np.pi / 4) + 0.0
    np.testing.assert_array_almost_equal(result, expected)


def test_Cosine() -> None:
    model = Cosine(amp=1.0, p=2.0, phi=np.pi / 4, d=0.0)
    t = np.array([0, 1, 2])
    result = model(t)
    expected = (1.0**2) * np.cos((2 * np.pi * t / 2.0) + np.pi / 4) + 0.0
    np.testing.assert_array_almost_equal(result, expected)


def test_Keplerian() -> None:
    # Dummy check to validate the process
    model = Keplerian(p=1.0, k=1.0, ecc=0.1, w=np.pi / 4, phi=0.0, offset=0.0)
    t = np.array([0, 0.5, 1.0])
    result = model(t)
    assert np.all(np.isfinite(result))


def test_udho() -> None:
    model = UdHO(a=1.0, b=0.5, w=1.0, phi=0.0)
    t = np.array([0, 0.5, 1.0])
    result = model(t)
    expected = 1.0**2 * np.exp(-0.5 * t) * np.cos(1.0 * t + 0.0)
    np.testing.assert_array_almost_equal(result, expected)

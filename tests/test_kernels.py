import numpy as np

from src.tedi.kernels import (
    RQP,
    Constant,
    Cosine,
    Exponential,
    Matern32,
    Matern52,
    Paciorek,
    Periodic,
    PiecewiseRQ,
    PiecewiseSE,
    QuasiPeriodic,
    RationalQuadratic,
    SquaredExponential,
    WhiteNoise,
)


def test_constant_kernel() -> None:
    c = 2.0
    r = np.array([0.5, 1.0, 1.5])
    kernel = Constant(c)
    result = kernel(r)
    expected = c**2 * np.ones_like(r)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_white_noise_kernel() -> None:
    wn = 0.5
    r = np.array([0.5, 1.0, 1.5])
    kernel = WhiteNoise(wn)
    result = kernel(r)
    expected = wn**2 * np.diag(np.diag(np.ones_like(r)))
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_squared_exponential_kernel() -> None:
    amp, ell = 1.0, 1.0
    r = np.array([0.5, 1.0, 1.5])
    kernel = SquaredExponential(amp, ell)
    result = kernel(r)
    expected = amp**2 * np.exp(-0.5 * r**2 / ell**2)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_periodic_kernel() -> None:
    amp, p, ell = 1.0, 2.0, 1.0
    r = np.array([0.5, 1.0, 1.5])
    kernel = Periodic(amp, p, ell)
    result = kernel(r)
    expected = amp**2 * np.exp(-2 * np.sin(np.pi * np.abs(r) / p) ** 2 / ell**2)  # NOQA
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_quasi_periodic_kernel() -> None:
    amp, ell_e, p, ell_p = 1.0, 1.0, 2.0, 2.0
    r = np.array([0.5, 1.0, 1.5])
    kernel = QuasiPeriodic(amp, ell_e, p, ell_p)
    result = kernel(r)
    expected = amp**2 * np.exp(
        -2 * np.sin(np.pi * np.abs(r) / p) ** 2 / ell_p**2
        - r**2 / (2 * ell_e**2)  # NOQA
    )
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_rational_quadratic_kernel() -> None:
    amp, alpha, ell = 1.0, 0.5, 1.0
    r = np.array([0.5, 1.0, 1.5])
    kernel = RationalQuadratic(amp, alpha, ell)
    result = kernel(r)
    expected = amp**2 * (1 + 0.5 * r**2 / (alpha * ell**2)) ** (-alpha)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_cosine_kernel() -> None:
    amp, p = 1.0, 2.0
    r = np.array([0.5, 1.0, 1.5])
    kernel = Cosine(amp, p)
    result = kernel(r)
    expected = amp**2 * np.cos(2 * np.pi * np.abs(r) / p)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_exponential_kernel() -> None:
    amp, ell = 1.0, 1.0
    r = np.array([0.5, 1.0, 1.5])
    kernel = Exponential(amp, ell)
    result = kernel(r)
    expected = amp**2 * np.exp(-np.abs(r) / ell)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_matern32_kernel() -> None:
    amp, ell = 1.0, 1.0
    r = np.array([0.5, 1.0, 1.5])
    kernel = Matern32(amp, ell)
    sqrt_3_r_ell = np.sqrt(3) * np.abs(r) / ell
    expected = amp**2 * (1 + sqrt_3_r_ell) * np.exp(-sqrt_3_r_ell)
    result = kernel(r)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_matern52_kernel() -> None:
    amp, ell = 1.0, 1.0
    r = np.array([0.5, 1.0, 1.5])
    kernel = Matern52(amp, ell)
    sqrt_5 = np.sqrt(5)
    abs_r = np.abs(r)
    expected = (
        amp**2
        * (1 + sqrt_5 * abs_r / ell + 5 * abs_r**2 / (3 * ell**2))
        * np.exp(-sqrt_5 * abs_r / ell)
    )
    result = kernel(r)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_rqp_kernel() -> None:
    amp, alpha, ell_e, p, ell_p = 1.0, 0.5, 1.0, 2.0, 2.0
    r = np.array([0.5, 1.0, 1.5])
    kernel = RQP(amp, alpha, ell_e, p, ell_p)
    per_component = np.exp(-2 * np.sin(np.pi * np.abs(r) / p) ** 2 / ell_p**2)
    rq_component = 1 + r**2 / (2 * alpha * ell_e**2)
    expected = amp**2 * per_component / np.abs(rq_component) ** alpha
    result = kernel(r)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_paciorek_kernel() -> None:
    amp, ell_1, ell_2 = 1.0, 1.0, 2.0
    r = np.array([0.5, 1.0, 1.5])
    kernel = Paciorek(amp, ell_1, ell_2)
    length_scales = np.sqrt(2 * ell_1 * ell_2 / (ell_1**2 + ell_2**2))
    exp_decay = np.exp(-2 * r**2 / (ell_1**2 + ell_2**2))
    expected = amp**2 * length_scales * exp_decay
    result = kernel(r)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_piecewise_se_kernel() -> None:
    eta1, eta2, eta3 = 1.0, 2.0, 3.0
    r = np.array([0.5, 1.0, 1.5])
    kernel = PiecewiseSE(eta1, eta2, eta3)
    SE_term = eta1**2 * np.exp(-0.5 * np.abs(r) ** 2 / eta2**2)
    abs_r_normalized = np.abs(r / (0.5 * eta3))
    piecewise = (3 * abs_r_normalized + 1) * (1 - abs_r_normalized) ** 3
    piecewise = np.where(abs_r_normalized > 1, 0, piecewise)
    expected = SE_term * piecewise
    result = kernel(r)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_piecewise_rq_kernel() -> None:
    eta1, alpha, eta2, eta3 = 1.0, 0.5, 2.0, 3.0
    r = np.array([0.5, 1.0, 1.5])
    kernel = PiecewiseRQ(eta1, alpha, eta2, eta3)
    rq_term = eta1**2 * (1 + 0.5 * np.abs(r) ** 2 / (alpha * eta2**2)) ** (
        -alpha
    )  # NOQA
    abs_r_normalized = np.abs(r / (0.5 * eta3))
    piecewise = (3 * abs_r_normalized + 1) * (1 - abs_r_normalized) ** 3
    piecewise = np.where(abs_r_normalized > 1, 0, piecewise)
    expected = rq_term * piecewise
    result = kernel(r)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"

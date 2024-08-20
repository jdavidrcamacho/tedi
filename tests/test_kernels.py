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
    WhiteNoise, NewPeriodic, QuasiNewPeriodic, NewRQP, 
    HarmonicPeriodic, QuasiHarmonicPeriodic
)


def test_Constant() -> None:
    c = 2.0
    r = np.array([0.5, 1.0, 1.5])
    kernel = Constant(c)
    result = kernel(r)
    expected = c**2 * np.ones_like(r)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_WhiteNoise() -> None:
    wn = 0.5
    r = np.array([0.5, 1.0, 1.5])
    kernel = WhiteNoise(wn)
    result = kernel(r)
    expected = wn**2 * np.diag(np.diag(np.ones_like(r)))
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_SquaredExponential() -> None:
    amp, ell = 1.0, 1.0
    r = np.array([0.5, 1.0, 1.5])
    kernel = SquaredExponential(amp, ell)
    result = kernel(r)
    expected = amp**2 * np.exp(-0.5 * r**2 / ell**2)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_Periodic() -> None:
    amp, p, ell = 1.0, 2.0, 1.0
    r = np.array([0.5, 1.0, 1.5])
    kernel = Periodic(amp, p, ell)
    result = kernel(r)
    expected = amp**2 * np.exp(-2 * np.sin(np.pi * np.abs(r) / p) ** 2 / ell**2)  # NOQA
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_QuasiPeriodic() -> None:
    amp, ell_e, p, ell_p = 1.0, 1.0, 2.0, 2.0
    r = np.array([0.5, 1.0, 1.5])
    kernel = QuasiPeriodic(amp, ell_e, p, ell_p)
    result = kernel(r)
    expected = amp**2 * np.exp(
        -2 * np.sin(np.pi * np.abs(r) / p) ** 2 / ell_p**2
        - r**2 / (2 * ell_e**2)  # NOQA
    )
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_RationalQuadratic() -> None:
    amp, alpha, ell = 1.0, 0.5, 1.0
    r = np.array([0.5, 1.0, 1.5])
    kernel = RationalQuadratic(amp, alpha, ell)
    result = kernel(r)
    expected = amp**2 * (1 + 0.5 * r**2 / (alpha * ell**2)) ** (-alpha)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_Cosine() -> None:
    amp, p = 1.0, 2.0
    r = np.array([0.5, 1.0, 1.5])
    kernel = Cosine(amp, p)
    result = kernel(r)
    expected = amp**2 * np.cos(2 * np.pi * np.abs(r) / p)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_Exponential() -> None:
    amp, ell = 1.0, 1.0
    r = np.array([0.5, 1.0, 1.5])
    kernel = Exponential(amp, ell)
    result = kernel(r)
    expected = amp**2 * np.exp(-np.abs(r) / ell)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_Matern32() -> None:
    amp, ell = 1.0, 1.0
    r = np.array([0.5, 1.0, 1.5])
    kernel = Matern32(amp, ell)
    sqrt_3_r_ell = np.sqrt(3) * np.abs(r) / ell
    expected = amp**2 * (1 + sqrt_3_r_ell) * np.exp(-sqrt_3_r_ell)
    result = kernel(r)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_Matern52() -> None:
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


def test_RQP() -> None:
    amp, alpha, ell_e, p, ell_p = 1.0, 0.5, 1.0, 2.0, 2.0
    r = np.array([0.5, 1.0, 1.5])
    kernel = RQP(amp, alpha, ell_e, p, ell_p)
    per_component = np.exp(-2 * np.sin(np.pi * np.abs(r) / p) ** 2 / ell_p**2)
    rq_component = 1 + r**2 / (2 * alpha * ell_e**2)
    expected = amp**2 * per_component / np.abs(rq_component) ** alpha
    result = kernel(r)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_Paciorek() -> None:
    amp, ell_1, ell_2 = 1.0, 1.0, 2.0
    r = np.array([0.5, 1.0, 1.5])
    kernel = Paciorek(amp, ell_1, ell_2)
    length_scales = np.sqrt(2 * ell_1 * ell_2 / (ell_1**2 + ell_2**2))
    exp_decay = np.exp(-2 * r**2 / (ell_1**2 + ell_2**2))
    expected = amp**2 * length_scales * exp_decay
    result = kernel(r)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_PiecewiseSE() -> None:
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


def test_PiecewiseRQ() -> None:
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


def test_NewPeriodic():
    amp, alpha, p, ell = 1.0, 0.5, 2.0, 1.0
    r = np.array([0.5, 1.0, 1.5])
    kernel = NewPeriodic(amp, alpha, p, ell)
    
    expected = amp**2 * (1 + 2 * np.sin(np.pi * np.abs(r) / p)**2 / (alpha * ell**2))**(-alpha)
    
    result = kernel(r)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_QuasiNewPeriodic():
    amp, alpha, ell_e, p, ell_p = 1.0, 0.5, 1.5, 2.0, 1.0
    r = np.array([0.5, 1.0, 1.5])
    kernel = QuasiNewPeriodic(amp, alpha, ell_e, p, ell_p)
    
    per_component = (1 + 2 * np.sin(np.pi * np.abs(r) / p)**2 / (alpha * ell_p**2))**(-alpha)
    exp_component = np.exp(-0.5 * np.abs(r)**2 / ell_e**2)
    expected = amp**2 * per_component * exp_component
    
    result = kernel(r)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"

def test_NewRQP():
    amp, alpha1, alpha2, ell_e, p, ell_p = 1.0, 0.5, 0.5, 1.5, 2.0, 1.0
    r = np.array([0.5, 1.0, 1.5])
    kernel = NewRQP(amp, alpha1, alpha2, ell_e, p, ell_p)
    
    abs_r = np.abs(r)
    alpha1_component = (1 + 0.5 * abs_r**2 / (alpha1 * ell_e**2))**(-alpha1)
    alpha2_component = (1 + 2 * np.sin(np.pi * abs_r / p)**2 / (alpha2 * ell_p**2))**(-alpha2)
    expected = amp**2 * alpha1_component * alpha2_component
    
    result = kernel(r)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_HarmonicPeriodic():
    n, amp, p, ell = 2, 1.0, 2.0, 1.0
    r = np.array([0.5, 1.0, 1.5])
    s = np.array([0.25, 0.75, 1.25])
    kernel = HarmonicPeriodic(n, amp, p, ell)
    
    first_sin = np.sin((n + 0.5) * 2 * np.pi * r / p) / 2 * np.sin(np.pi * r / p)
    second_sin = np.sin((n + 0.5) * 2 * np.pi * s / p) / 2 * np.sin(np.pi * s / p)
    sine_component = (first_sin - second_sin)**2
    
    first_cot = 0.5 / np.tan(np.pi * r / p)
    first_cos = np.cos((n + 0.5) * 2 * np.pi * r / p) / 2 * np.sin(np.pi * r / p)
    second_cot = 0.5 / np.tan(np.pi * s / p)
    second_cos = np.cos((n + 0.5) * 2 * np.pi * s / p) / 2 * np.sin(np.pi * s / p)
    cot_cos_component = (first_cot - first_cos - second_cot + second_cos)**2
    
    expected = amp**2 * np.exp(-0.5 * (sine_component + cot_cos_component) / ell**2)
    
    result = kernel(r, s)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"

def test_QuasiHarmonicPeriodicv():
    n, amp, ell_e, p, ell_p = 2, 1.0, 1.5, 2.0, 1.0
    r = np.array([0.5, 1.0, 1.5])
    s = np.array([0.25, 0.75, 1.25])
    kernel = QuasiHarmonicPeriodic(n, amp, ell_e, p, ell_p)
    
    first_sin = np.sin((n + 0.5) * 2 * np.pi * r / p) / 2 * np.sin(np.pi * r / p)
    second_sin = np.sin((n + 0.5) * 2 * np.pi * s / p) / 2 * np.sin(np.pi * s / p)
    sine_component = (first_sin - second_sin)**2
    
    first_cot = 0.5 / np.tan(np.pi * r / p)
    first_cos = np.cos((n + 0.5) * 2 * np.pi * r / p) / 2 * np.sin(np.pi * r / p)
    second_cot = 0.5 / np.tan(np.pi * s / p)
    second_cos = np.cos((n + 0.5) * 2 * np.pi * s / p) / 2 * np.sin(np.pi * s / p)
    cot_cos_component = (first_cot - first_cos - second_cot + second_cos)**2
    
    hp_kernel = np.exp(-0.5 * (sine_component + cot_cos_component) / ell_p**2)
    se_kernel = np.exp(-0.5 * np.abs(r - s)**2 / ell_e**2)
    expected = amp**2 * hp_kernel * se_kernel
    
    result = kernel(r, s)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"

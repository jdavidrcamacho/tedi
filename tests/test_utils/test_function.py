import numpy as np
import pytest

from src.tedi.utils.function import (
    keplerian,
    minimum_mass,
    phase_folding,
    rms,
    semi_amplitude,
    wrms,
)


def test_semi_amplitude() -> None:
    period = 1.0  # in years
    m_planet = 1.0  # in Jupiter masses
    m_star = 1.0  # in Solar masses
    ecc = 0.0  # circular orbit
    result = semi_amplitude(period, m_planet, m_star, ecc)
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_minimum_mass() -> None:
    period = 365.25  # in days
    k = 10.0  # in m/s
    ecc = 0.0  # circular orbit
    m_star = 1.0  # in Solar masses
    result = minimum_mass(period, k, ecc, m_star)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    assert np.isfinite(result).all()


def test_keplerian() -> None:
    period = 365.25  # in days
    k = 10.0  # in m/s
    ecc = 0.1  # slight eccentricity
    w = np.pi  # longitude of periastron
    tt = 0  # zero phase
    t = np.linspace(0, 365, 100)  # time array
    t, rv = keplerian(period, k, ecc, w, tt, t=t)
    assert isinstance(t, np.ndarray)
    assert isinstance(rv, np.ndarray)
    assert t.shape == rv.shape
    assert np.isfinite(rv).all()


def test_keplerian_ValueError() -> None:
    with pytest.raises(ValueError, match="Time is None"):
        keplerian()


def test_phase_folding() -> None:
    t = np.linspace(0, 365, 100)  # time array
    y = np.sin(2 * np.pi * t / 365)  # mock measurements
    yerr = np.random.normal(0, 0.1, size=100)  # mock errors
    period = 365.0  # period
    phase, folded_y, folded_yerr = phase_folding(t, y, yerr, period)
    assert isinstance(phase, np.ndarray)
    assert isinstance(folded_y, np.ndarray)
    assert isinstance(folded_yerr, np.ndarray)
    assert len(phase) == len(folded_y) == len(folded_yerr)
    assert np.all((phase >= 0) & (phase < 1))


def test_phase_folding_no_yerr() -> None:
    t = np.linspace(0, 365, 100)  # time array
    y = np.sin(2 * np.pi * t / 365)  # mock measurements
    period = 365.0  # period
    phase, folded_y, folded_yerr = phase_folding(t, y, None, period)
    assert isinstance(phase, np.ndarray)
    assert isinstance(folded_y, np.ndarray)
    assert isinstance(folded_yerr, np.ndarray)
    assert len(phase) == len(folded_y) == len(folded_yerr)
    assert np.all(folded_yerr == 0)


def test_rms() -> None:
    array = np.array([1, 2, 3, 4, 5])
    result = rms(array)
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_wrms() -> None:
    array = np.array([1, 2, 3, 4, 5])
    weights = np.array([1, 1, 1, 1, 1])
    result = wrms(array, weights)
    assert isinstance(result, float)
    assert np.isfinite(result)

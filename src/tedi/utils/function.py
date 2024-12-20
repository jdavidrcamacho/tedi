"""Collection of useful functions."""

from typing import Optional, Tuple

import numpy as np


def semi_amplitude(
    period: float, m_planet: float, m_star: float, ecc: float
) -> float:  # NOQA
    """
    Calculate the semi-amplitude (K) of a planet's radial velocity signal.

    Args:
        period (float): Orbital period in years.
        m_planet (float): Planet's mass in Jupiter masses (M*sin(i)).
        m_star (float): Star mass in Solar masses.
        ecc (float): Orbital eccentricity.

    Returns:
        float: The semi-amplitude k (in m/s) of the planet signal.
    """
    per = float(np.power(1 / period, 1 / 3))
    p_mass = m_planet / 1
    s_mass = float(np.power(1 / m_star, 2 / 3))
    ecc = 1 / np.sqrt(1 - ecc**2)
    k = 28.435 * per * p_mass * s_mass * ecc
    return k


def minimum_mass(p: float, k: float, ecc: float, m_star: float) -> np.ndarray:
    """
    Calculate the minimum mass (m*sin(i)) of a planet.

    Args:
        p (float): Orbital period in days.
        k (float): Semi-amplitude in m/s.
        ecc (float): Orbital eccentricity.
        m_star (float): Star mass in Solar masses.

    Returns:
        np.ndarray: Minimum mass of the planet in Jupiter and Earth masses.
    """
    msini = (
        4.919e-3 * k * np.sqrt(1 - ecc**2) * np.cbrt(p) * np.cbrt(m_star) ** 2
    )  # NOQA
    return np.array([msini, msini * 317.8])


def keplerian(
    p: float = 365,
    k: float = 0.1,
    ecc: float = 0,
    w: float = np.pi,
    tt: float = 0,
    phi: Optional[float] = None,
    gamma: float = 0,
    t: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate the radial velocity signal of a planet in a Keplerian orbit.

    Args:
        p (float, optional): Orbital period in days.
            Defaults to 365.
        k (float, optional): Radial velocity amplitude.
            Defaults to 0.1.
        ecc (float, optional): Orbital eccentricity.
            Defaults to 0.
        w (float, optional): Longitude of the periastron.
            Defaults to pi.
        tt (float, optional): Zero phase.
            Defaults to 0.
        phi (Optional[float], optional): Orbital phase.
            Defaults to None.
        gamma (float, optional): Constant system radial velocity.
            Defaults to 0.
        t (Optional[np.ndarray], optional): Time of measurements.
            Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            - `t` (np.ndarray): Time of measurements.
            - `RV` (np.ndarray): Radial velocity signal generated in m/s.

    Raises:
        ValueError: If `t` is None.
    """
    if t is None:
        raise ValueError("Time is None")

    # Mean anomaly
    if phi is None:
        mean_anom = [2 * np.pi * (x1 - tt) / p for x1 in t]
    else:
        tt = t[0] - (p * phi) / (2.0 * np.pi)
        mean_anom = [2 * np.pi * (x1 - tt) / p for x1 in t]

    # Eccentric anomaly: E0=M + e*sin(M) + 0.5*(e**2)*sin(2*M)
    e0 = [
        x + ecc * np.sin(x) + 0.5 * (ecc**2) * np.sin(2 * x) for x in mean_anom
    ]  # NOQA

    # Mean anomaly: M0=E0 - e*sin(E0)
    m0 = [x - ecc * np.sin(x) for x in e0]
    i = 0
    while i < 1000:
        calc_aux = [x - y for x, y in zip(mean_anom, m0)]
        e1 = [x + y / (1 - ecc * np.cos(x)) for x, y in zip(e0, calc_aux)]
        m1 = [x - ecc * np.sin(x) for x in e0]
        e0, m0 = e1, m1
        i += 1

    nu = [
        2 * np.arctan(np.sqrt((1 + ecc) / (1 - ecc)) * np.tan(x / 2))
        for x in e0  # NOQA
    ]
    rv = np.array([gamma + k * (ecc * np.cos(w) + np.cos(w + x)) for x in nu])
    return t, rv


def phase_folding(
    t: np.ndarray, y: np.ndarray, yerr: Optional[np.ndarray], period: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform phase folding of the given data according to the specified period.

    Args:
        t (np.ndarray): Time array.
        y (np.ndarray): Measurements array.
        yerr (Optional[np.ndarray]): Measurement errors array.
        period (float): Period to fold the data.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing:
            - `phase` (np.ndarray): Folded phase array.
            - `folded_y` (np.ndarray): Measurements sorted.
            - `folded_yerr` (np.ndarray): Errors sorted.
    """
    foldtimes = t / period
    foldtimes = foldtimes % 1
    if yerr is None:
        yerr = np.zeros_like(y)

    phase, folded_y, folded_yerr = zip(*sorted(zip(foldtimes, y, yerr)))
    return np.array(phase), np.array(folded_y), np.array(folded_yerr)


def rms(array: np.ndarray) -> float:
    """
    Compute the root mean square (RMS) of an array of measurements.

    Args:
        array (np.ndarray): Array of measurements.

    Returns:
        float: Root mean square of the measurements.
    """
    mu = np.mean(array)
    return np.sqrt(np.mean((array - mu) ** 2))


def wrms(array: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute the weighted root mean square (WRMS) of an array of measurements.

    Args:
        array (np.ndarray): Array of measurements.
        weights (np.ndarray): Weights corresponding to the measurements
            typically 1/errors**2.

    Returns:
        float: Weighted root mean square of the measurements.
    """
    mu = np.average(array, weights=weights)
    return np.sqrt(np.sum(weights * (array - mu) ** 2) / np.sum(weights))

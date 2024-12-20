"""Mean functions for Gaussian or Student-t processes regression."""

import numpy as np

from .utils.means import MeanModel, array_input

pi, exp, sine, cosine, sqrt = np.pi, np.exp, np.sin, np.cos, np.sqrt
__all__ = ["Constant", "Linear", "Parabola", "Cubic", "Keplerian", "UdHO"]


class Constant(MeanModel):
    """
    Constant offset mean function.

    Attributes:
        _parsize (int): The number of parameters in the mean model.
    """

    _parsize = 1

    def __init__(self, c: float) -> None:
        """
        Initialize the constant mean function with a given offset.

        Args:
            c (float): The constant offset value.
        """
        super(Constant, self).__init__(c)

    @array_input
    def __call__(self, t: np.ndarray) -> np.ndarray:
        """
        Evaluatenstant mean function at the given input.

        Args:
            t (np.ndarray): Input data.

        Returns:
            np.ndarray: An array with the constant value.
        """
        return np.full(t.shape, self.pars[0])


class Linear(MeanModel):
    """
    Linear mean function.

    This class models a mean function of the form:
        m(t) = slope * t + intercept.

    Attributes:
        _parsize (int): The number of parameters in the mean model.
    """

    _parsize = 2

    def __init__(self, slope: float, intercept: float) -> None:
        """
        Initialize the linear mean function with a given slope and intercept.

        Args:
            slope (float): The slope of the linear function.
            intercept (float): The intercept of the linear function.
        """
        super(Linear, self).__init__(slope, intercept)

    @array_input
    def __call__(self, t: np.ndarray) -> np.ndarray:
        """
        Evaluate the linear mean function at the given input.

        Args:
            t (np.ndarray): Input data.

        Returns:
            np.ndarray: An array with the linear mean values.
        """
        t_mean = t.mean()
        return self.pars[0] * (t - t_mean) + np.full(t.shape, self.pars[1])


class Parabola(MeanModel):
    """
    Second degree polynomial mean function.

    This class models a mean function of the form:
        m(t) = quad * t**2 + slope * t + intercept

    Attributes:
        _parsize (int): The number of parameters in the mean model.
    """

    _parsize = 3

    def __init__(self, quad: float, slope: float, intercept: float) -> None:
        """
        Initializee parabolic mean function.

        Args:
            quad (float): The coefficient of the quadratic term.
            slope (float): The coefficient of the linear term.
            intercept (float): The intercept of the polynomial.
        """
        super(Parabola, self).__init__(quad, slope, intercept)

    @array_input
    def __call__(self, t: np.ndarray) -> np.ndarray:
        """
        Evaluate the parabolic mean function at the given input.

        Args:
            t (np.ndarray): Input data.

        Returns:
            np.ndarray: An array with the parabolic mean values.
        """
        return np.polyval(self.pars, t)


class Cubic(MeanModel):
    """
    Third degree polynomial mean function.

    This class models a mean function of the form:
        m(t) = cub * t**3 + quad * t**2 + slope * t + intercept

    Attributes:
        _parsize (int): The number of parameters in the mean model.
    """

    _parsize = 4

    def __init__(
        self, cub: float, quad: float, slope: float, intercept: float
    ) -> None:  # NOQA
        """
        Initialize the cubic mean function.

        Args:
            cub (float): The coefficient of the cubic term.
            quad (float): The coefficient of the quadratic term.
            slope (float): The coefficient of the linear term.
            intercept (float): The intercept of the polynomial.
        """
        super(Cubic, self).__init__(cub, quad, slope, intercept)

    @array_input
    def __call__(self, t: np.ndarray) -> np.ndarray:
        """
        Evaluate the cubic mean function at the given input.

        Args:
            t (np.ndarray): Input data.

        Returns:
            np.ndarray: An array with the cubic mean values.
        """
        return np.polyval(self.pars, t)


class Sine(MeanModel):
    """
    Sinusoidal mean function.

    This class models a mean function of the form:
        m(t) = amplitude * sin((2 * pi * t / p) + phase) + displacement

    Attributes:
        _parsize (int): The number of parameters in the mean model.
    """

    _parsize = 4

    def __init__(self, amp: float, p: float, phi: float, d: float) -> None:
        """
        Initializehe sinusoidal mean function.

        Args:
            amp (float): The amplitude of the sine wave.
            p (float): The period of the sine wave.
            phi (float): The phase shift of the sine wave.
            d (float): The displacement of the sine wave.
        """
        super(Sine, self).__init__(amp, p, phi, d)

    @array_input
    def __call__(self, t: np.ndarray) -> np.ndarray:
        """
        Evaluate the sinusoidal mean function at the given input.

        Args:
            t (np.ndarray): Input data.

        Returns:
            np.ndarray: An array with the sinusoidal mean values.
        """
        return (
            self.pars[0] * np.sin((2 * np.pi * t / self.pars[1]) + self.pars[2])  # NOQA
            + self.pars[3]
        )


class Cosine(MeanModel):
    """
    Cosine mean function.

    This class models a mean function of the form:
        m(t) = amplitude^2 * cos((2 * pi * t / P) + phase) + displacement

    Attributes:
        _parsize (int): The number of parameters in the mean model.
    """

    _parsize = 4

    def __init__(self, amp: float, p: float, phi: float, d: float) -> None:
        """
        Initialize the cosine mean function.

        Args:
            amp (float): The amplitude of the cosine wave.
            p (float): The period of the cosine wave.
            phi (float): The phase shift of the cosine wave.
            d (float): The displacement of the cosine wave.
        """
        super(Cosine, self).__init__(amp, p, phi, d)

    @array_input
    def __call__(self, t: np.ndarray) -> np.ndarray:
        """
        Evaluate the cosine mean function at the given input.

        Args:
            t (np.ndarray): Input data.

        Returns:
            np.ndarray: An array with the cosine mean values.
        """
        return (
            self.pars[0] ** 2
            * np.cos((2 * np.pi * t / self.pars[1]) + self.pars[2])  # NOQA
            + self.pars[3]
        )


class Keplerian(MeanModel):
    """
    Keplerian mean function for modeling radial velocity (RV) signals.

    This class adapts the Keplerian model for RV signals, using parameters
    related to orbital mechanics. It calculates the RV signal based on the
    provided parameters.

    The mathematical model used is:
        tan[phi(t) / 2] = sqrt(1 + ecc) / sqrt(1 - ecc) * tan[E(t) / 2]
        E(t) - ecc * sin[E(t)] = M(t) = eccentric anomaly
        M(t) = (2 * pi * t / P) + M0 = mean anomaly
        RV = K * [cos(w + nu) + ecc * cos(w)] + offset
    where:
        nu = 2 * atan(sqrt((1 + ecc) / (1 - ecc)) * tan(E(t) / 2))

    Attributes:
        _parsize (int): Number of parameters in the Keplerian model (5).

    Parameters:
        p (float): Period in days.
        ecc (float): Eccentricity.
        k (float): RV amplitude in m/s.
        w (float): Longitude of the periastron.
        phi (float): Orbital phase.
        offset (float): Offset.

    Returns:
        np.ndarray: RV signal computed from the model.
    """

    _parsize = 6

    _parsize = 5

    def __init__(self, p, k, ecc, w, phi, offset) -> None:
        """Initialize Keplerian."""
        super(Keplerian, self).__init__(p, k, ecc, w, phi, offset)

    def __call__(self, t):
        """Calculate Keplerian."""
        p, k, ecc, w, phi, offset = self.pars
        t0 = t[0] - (p * phi) / (2.0 * np.pi)
        m0 = 2 * np.pi * (t - t0) / p  # first guess at M and E
        e0 = m0 + ecc * np.sin(m0) + 0.5 * (ecc**2) * np.sin(2 * m0)
        m1 = e0 - ecc * np.sin(e0) - m0  # goes to zero when converges
        criteria = 1e-10
        convd = np.where(np.abs(m1) > criteria)[0]  # indices not converged
        nd = len(convd)  # number of unconverged elements
        count = 0
        while nd > 0:
            count += 1
            e = e0
            m1p = 1 - ecc * np.cos(e)
            m1pp = ecc * np.sin(e)
            m1ppp = 1 - m1p
            d1 = -m1 / m1p
            d2 = -m1 / (m1p + d1 * m1pp / 2.0)
            d3 = -m1 / (m1p + d2 * m1pp / 2.0 + d2 * d2 * m1ppp / 6.0)
            e = e + d3
            e0 = e
            m0 = e0 - ecc * np.sin(e0)
            m1 = e0 - ecc * np.sin(e0) - m0
            convergence_criteria = np.abs(m1) > criteria
            nd = np.sum(convergence_criteria is True)
        nu = 2 * np.arctan(np.sqrt((1 + ecc) / (1 - ecc)) * np.tan(e0 / 2))
        rv = k * (ecc * np.cos(w) + np.cos(w + nu)) + offset
        return rv


class UdHO(MeanModel):
    """
    Underdamped Harmonic Oscillator mean function.

    This class models a mean function based on an underdamped harmonic
    oscillator:
        m(t) = a * exp(-b * t) * cos(w * t + phi)

    Attributes:
        _parsize (int): Number of parameters in the model.

    Parameters:
        a (float): Amplitude-like parameter that scales the oscillation.
        b (float): Damping coefficient, controls the exponential decay.
        w (float): Angular frequency of the oscillation.
        phi (float): Phase shift, determines the initial phase of the
            oscillation.
    """

    _parsize = 4

    def __init__(self, a: float, b: float, w: float, phi: float) -> None:
        """
        Initialize the Underdamped Harmonic Oscillator mean function.

        Args:
            a (float): Amplitude-like parameter that scales the oscillation.
            b (float): Damping coefficient, controls the exponential decay.
            w (float): Angular frequency of the oscillation.
            phi (float): Phase shift, determines the initial phase of the
                oscillation.
        """
        super(UdHO, self).__init__(a, b, w, phi)

    _parsize = 4

    @array_input
    def __call__(self, t: np.ndarray) -> np.ndarray:
        """
        Compute the mean function for the given time array.

        Args:
            t (np.ndarray): Array of time values.

        Returns:
            np.ndarray: Computed mean values based on the model.
        """
        a, b, w, phi = self.pars
        return a**2 * np.exp(-b * t) * np.cos(w * t + phi)

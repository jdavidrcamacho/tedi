"""Covariance functions for Gaussian or Student-t processes regression."""

import numpy as np

from .utils.kernels import Kernel

pi, exp, sine, cosine, sqrt = np.pi, np.exp, np.sin, np.cos, np.sqrt
__all__ = [
    "Constant",
    "WhiteNoise",
    "SquaredExponential",
    "Periodic",
    "QuasiPeriodic",
    "RationalQuadratic",
    "Cosine",
    "Exponential",
    "Matern32",
    "Matern52",
    "RQP",
    "Paciorek",
    "PiecewiseSE",
    "PiecewiseRQ",
    "NewPeriodic",
    "QuasiNewPeriodic",
    "NewRQP",
    "HarmonicPeriodic",
    "QuasiHarmonicPeriodic",
]


class Constant(Kernel):
    """
    Constant kernel representing a constant offset.

    Attributes:
        c (float): Constant value of the kernel.
        params_number (int): Number of hyperparameters.
    """

    def __init__(self, c: float) -> None:
        """Initialize the constant kernel with its value.

        Args:
            c (float): Constant value of the kernel.
        """
        super().__init__(c)
        self.c: float = c
        self.params_number: int = 1

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """Compute the constant kernel value.

        Args:
            r (np.ndarray): Difference between two data points (ignored).

        Returns:
            np.ndarray: Array filled with the constant value squared.
        """
        return self.c**2 * np.ones_like(r)


class WhiteNoise(Kernel):
    """
    White noise kernel.

    Attributes:
        wn (float): White noise amplitude.
        type (str): Type of the kernel.
        params_number (int): Number of hyperparameters.
    """

    def __init__(self, wn: float) -> None:
        """Initialize the white noise kernel with its amplitude.

        Args:
            wn (float): White noise amplitude.
        """
        super().__init__(wn)
        self.wn: float = wn
        self.type: str = "stationary"
        self.params_number: int = 1

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """Compute the white noise kernel value.

        Args:
            r (np.ndarray): Difference between two data points.

        Returns:
            np.ndarray: Diagonal matrix with the white noise amplitude squared.
        """
        return self.wn**2 * np.diag(np.diag(np.ones_like(r)))


class SquaredExponential(Kernel):
    """
    Squared Exponential kernel.

    Also known as radial basis function (RBF) kernel, is commonly used in
    Gaussian processes for regression and classification. It is defined by its
    amplitude and length-scale parameters.

    Attributes:
        amp  (float): Amplitude of the kernel.
        ell (float): Length-scale parameter.
        params_number (int): Number of hyperparameters.
    """

    def __init__(self, amp: float, ell: float) -> None:
        """Initialize the Squared Exponential kernel.

        Args:
            amp  (float): Amplitude of the kernel.
            ell (float): Length-scale.
        """
        super().__init__(amp, ell)
        self.amp = amp
        self.ell = ell
        self.params_number = 2

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """Compute the Squared Exponential kernel value.

        Args:
            r (np.ndarray): Difference between two data points.

        Returns:
            np.ndarray: Kernel value computed with the Squared Exponential.
        """
        return self.amp**2 * exp(-0.5 * r**2 / self.ell**2)


class Periodic(Kernel):
    """
    Periodic kernel.

    This kernel is used to model periodic data, defined by its amplitude,
    length-scale, and period.

    Attributes:
        amp  (float): Amplitude of the kernel.
        ell (float): Length-scale parameter.
        p (float): Period of the kernel.
        params_number (int): Number of hyperparameters.
    """

    def __init__(self, amp: float, p: float, ell: float) -> None:
        """Initialize the Periodic kernel.

        Args:
            amp  (float): Amplitude of the kernel.
            p (float): Period of the kernel.
            ell (float): Length-scale.
        """
        super().__init__(amp, p, ell)
        self.amp: float = amp
        self.ell: float = ell
        self.p: float = p
        self.params_number: int = 3

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """Compute the Periodic kernel value.

        Args:
            r (np.ndarray): Difference between two data points.

        Returns:
            np.ndarray: Kernel value computed using the Periodic function.
        """
        return self.amp**2 * exp(
            -2 * sine(pi * abs(r) / self.p) ** 2 / self.ell**2
        )  # NOQA


class QuasiPeriodic(Kernel):
    """
    Quasi-periodic kernel.

    This kernel is the product of the periodic and squared exponential kernels.

    Attributes:
        amp(float): Amplitude of the kernel.
        ell_e (float): Evolutionary time scale.
        ell_p (float): Length scale of the periodic component.
        p (float): Kernel periodicity.
        params_number (int): Number of hyperparameters.
    """

    def __init__(
        self, amp: float, ell_e: float, p: float, ell_p: float
    ) -> None:  # NOQA
        """Initialize the QuasiPeriodic kernel with its parameters.

        Args:
            amp (float): Amplitude of the kernel.
            ell_e (float): Evolutionary time scale.
            p (float): Kernel periodicity.
            ell_p (float): Length scale of the periodic component.
        """
        super().__init__(amp, ell_e, p, ell_p)
        self.amp: float = amp
        self.ell_e: float = ell_e
        self.p: float = p
        self.ell_p: float = ell_p
        self.params_number: int = 4

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """Compute the QuasiPeriodic kernel value.

        Args:
            r (np.ndarray): Difference between two data points.

        Returns:
            np.ndarray: Kernel value computed using the QuasiPeriodic function.
        """
        return self.amp**2 * exp(
            -2 * sine(pi * abs(r) / self.p) ** 2 / self.ell_p**2
            - r**2 / (2 * self.ell_e**2)
        )


class RationalQuadratic(Kernel):
    """
    Rational Quadratic kernel.

    This kernel can be seen as a scale mixture (infinite sum) of squared
    exponential kernels with different characteristic length scales.
    It is used to model functions with varying length scales.

    Attributes:
        amp (float): Amplitude of the kernel.
        alpha (float): Weight of large-scale and small-scale variations.
        ell (float): Characteristic length scale that defines smoothness.
        params_number (int): Number of hyperparameters.
    """

    def __init__(self, amp: float, alpha: float, ell: float) -> None:
        """Initialize the Rational Quadratic kernel with its parameters.

        Args:
            amp (float): Amplitude of the kernel.
            alpha (float): Weight of large-scale and small-scale variations.
            ell (float): Characteristic length scale that defines smoothness.
        """
        super().__init__(amp, alpha, ell)
        self.amp: float = amp
        self.alpha: float = alpha
        self.ell: float = ell
        self.params_number: int = 3

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """Compute the Rational Quadratic kernel value.

        Args:
            r (np.ndarray): Difference between two data points.

        Returns:
            np.ndarray: Kernel value computed.
        """
        return self.amp**2 * (1 + 0.5 * r**2 / (self.alpha * self.ell**2)) ** (
            -self.alpha
        )


class Cosine(Kernel):
    """
    Cosine kernel.

    This kernel is used to model periodic functions with a specified amplitude
    and period.

    Attributes:
        amp (float): Amplitude of the kernel.
        P (float): Period of the kernel.
        params_number (int): Number of hyperparameters.
    """

    def __init__(self, amp: float, p: float) -> None:
        """Initialize the Cosine kernel with its amp and period.

        Args:
            amp (float): Amplitude of the kernel.
            p (float): Period of the kernel.
        """
        super().__init__(amp, p)
        self.amp: float = amp
        self.p: float = p
        self.params_number: int = 2

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """Compute the Cosine kernel value.

        Args:
            r (np.ndarray): Difference between two data points.

        Returns:
            np.ndarray: Kernel value computed using the Cosine function.
        """
        return self.amp**2 * cosine(2 * pi * abs(r) / self.p)


class Exponential(Kernel):
    """
    Exponential kernel.

    This kernel is a special case of the Matern family of kernels with
    parameter v = 1/2.

    Attributes:
        amp (float): Amplitude of the kernel.
        ell (float): Characteristic length scale.
        params_number (int): Number of hyperparameters.
    """

    def __init__(self, amp: float, ell: float) -> None:
        """Initialize the Exponential kernel with its amp and length scale.

        Args:
            amp (float): Amplitude of the kernel.
            ell (float): Characteristic length scale.
        """
        super().__init__(amp, ell)
        self.amp: float = amp
        self.ell: float = ell
        self.params_number: int = 2

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """Compute the Exponential kernel value.

        Args:
            r (np.ndarray): Difference between two data points.

        Returns:
            np.ndarray: Kernel value computed using the Exponential function.
        """
        return self.amp**2 * exp(-abs(r) / self.ell)


class Matern32(Kernel):
    """
    Matern 3/2 kernel.

    This kernel is a special case of the Matern family with parameter v = 3/2.

    Attributes:
        amp (float): Amplitude of the kernel.
        ell (float): Characteristic length scale.
        params_number (int): Number of hyperparameters.
    """

    def __init__(self, amp: float, ell: float) -> None:
        """Initialize the Matern 3/2 kernel with its amp and length scale.

        Args:
            amp (float): Amplitude of the kernel.
            ell (float): Characteristic length scale.
        """
        super().__init__(amp, ell)
        self.amp: float = amp
        self.ell: float = ell
        self.params_number: int = 2

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """Compute the Matern 3/2 kernel value.

        Args:
            r (np.ndarray): Difference between two data points.

        Returns:
            np.ndarray: Kernel value computed using the Matern 3/2 function.
        """
        sqrt_3_r_ell = sqrt(3) * abs(r) / self.ell
        return self.amp**2 * (1 + sqrt_3_r_ell) * exp(-sqrt_3_r_ell)


class Matern52(Kernel):
    """
    Matern 5/2 kernel.

    This kernel is a special case of the Matern family with parameter v = 5/2.

    Attributes:
        amp (float): Amplitude of the kernel.
        ell (float): Characteristic length scale.
        params_number (int): Number of hyperparameters.
    """

    def __init__(self, amp: float, ell: float) -> None:
        """Initialize the Matern 5/2 kernel with its amp and length scale.

        Args:
            amp (float): Amplitude of the kernel.
            ell (float): Characteristic length scale.
        """
        super().__init__(amp, ell)
        self.amp: float = amp
        self.ell: float = ell
        self.params_number: int = 2

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """Compute the Matern 5/2 kernel value.

        Args:
            r (np.ndarray): Difference between two data points.

        Returns:
            np.ndarray: Kernel value computed using the Matern 5/2 function.
        """
        abs_r, sqrt_5 = abs(r), sqrt(5)
        minus_sqrt_5_r_ell = -sqrt_5 * abs_r / self.ell
        return (
            self.amp**2
            * (
                1
                + (3 * sqrt_5 * self.ell * abs_r + 5 * abs_r**2)
                / (3 * self.ell**2)  # NOQA
            )
            * exp(minus_sqrt_5_r_ell)
        )


class RQP(Kernel):
    """
    Product of the periodic kernel and the rational quadratic kernel.

    This kernel combines the properties of the periodic kernel and the rational
    quadratic kernel. The behavior of the RQP kernel changes with the parameter
    alpha:

    - As alpha approaches infinity, the RQP kernel approaches the
    quasi-periodic kernel.
    - As alpha approaches zero, the RQP kernel approaches the periodic kernel.

    There exists an optimal range for alpha where the RQP kernel often performs
    better than the quasi-periodic kernel.

    Attributes:
        amp (float): Amplitude of the kernel.
        alpha (float): Parameter of the rational quadratic kernel.
        ell_e (float): Aperiodic length scale.
        P (float): Periodicity of the kernel.
        ell_p (float): Periodic length scale.
        params_number (int): Number of hyperparameters.
    """

    def __init__(
        self, amp: float, alpha: float, ell_e: float, p: float, ell_p: float
    ) -> None:
        """Initialize the RQP kernel with its parameters.

        Args:
            amp  (float): Amplitude of the kernel.
            alpha (float): Parameter of the rational quadratic kernel.
            ell_e (float): Aperiodic length scale.
            P (float): Periodicity of the kernel.
            ell_p (float): Periodic length scale.
        """
        super().__init__(amp, alpha, ell_e, p, ell_p)
        self.amp = amp
        self.alpha = alpha
        self.ell_e = ell_e
        self.p = p
        self.ell_p = ell_p
        self.params_number = 5

    def __call__(self, r):
        """Calculate Kernel."""
        per_component = exp(
            -2 * sine(pi * abs(r) / self.p) ** 2 / self.ell_p**2
        )  # NOQA
        rq_component = 1 + r**2 / (2 * self.alpha * self.ell_e**2)
        return (
            self.amp**2
            * per_component
            / (np.sign(rq_component) * abs(rq_component) ** self.alpha)
        )


class Paciorek(Kernel):
    """
    Modified Paciorek's kernel (stationary version).

    Attributes:
        amp  (float): Amplitude of the kernel.
        ell_1 (float): First length scale.
        ell_2 (float): Second length scale.
        params_number (int): Number of hyperparameters.
    """

    def __init__(self, amp: float, ell_1: float, ell_2: float) -> None:
        """Initialize the Paciorek kernel with its amp  and length scales.

        Args:
            amp  (float): Amplitude of the kernel.
            ell_1 (float): First length scale.
            ell_2 (float): Second length scale.
        """
        super().__init__(amp, ell_1, ell_2)
        self.amp: float = amp
        self.ell_1: float = ell_1
        self.ell_2: float = ell_2
        self.params_number: int = 3

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """Compute the Paciorek kernel value.

        Args:
            r (np.ndarray): Difference between two data points.

        Returns:
            np.ndarray: Kernel value computed using the Paciorek function.
        """
        length_scales = sqrt(
            2 * self.ell_1 * self.ell_2 / (self.ell_1**2 + self.ell_2**2)
        )
        exp_decay = exp(-2 * r * r / (self.ell_1**2 + self.ell_2**2))
        return self.amp**2 * length_scales * exp_decay


class PiecewiseSE(Kernel):
    """
    Product of the Squared Exp kernel and a piecewise polynomial kernel.

    This kernel combines a squared exponential (SE) kernel with a piecewise
    polynomial kernel to model data with both smooth and piecewise behaviors.
    The resulting kernel adapts to both smooth variations and abrupt changes.

    Attributes:
        eta1 (float): Amplitude of the squared exponential kernel.
        eta2 (float): Length scale of the squared exponential kernel.
        eta3 (float): Periodic repetition scale for the piecewise kernel.
        params_number (int): Number of hyperparameters.
    """

    def __init__(self, eta1: float, eta2: float, eta3: float) -> None:
        """Initialize the PiecewiseSE kernel with its parameters.

        Args:
            eta1 (float): Amplitude of the squared exponential kernel.
            eta2 (float): Length scale of the squared exponential kernel.
            eta3 (float): Periodic repetition scale for the piecewise kernel.
        """
        super().__init__(eta1, eta2, eta3)
        self.eta1: float = eta1
        self.eta2: float = eta2
        self.eta3: float = eta3
        self.params_number: int = 3

    def __call__(self, r):
        """Compute the PiecewiseSE kernel value.

        Args:
            r (np.ndarray): Difference between two data points.

        Returns:
            np.ndarray: Kernel value computed.
        """
        SE_term = self.eta1**2 * exp(-0.5 * abs(r) ** 2 / self.eta2**2)
        abs_r_normalized = abs(r / (0.5 * self.eta3))
        piecewise = (3 * abs_r_normalized + 1) * (1 - abs_r_normalized) ** 3
        piecewise = np.where(abs_r_normalized > 1, 0, piecewise)
        return SE_term * piecewise


class PiecewiseRQ(Kernel):
    """
    Product of the Rational Quadratic and Piecewise kernels.

    Attributes:
        eta1 (float): Amplitude of the Rational Quadratic kernel.
        alpha (float): Parameter of the Rational Quadratic kernel that
            controls the scale of variations.
        eta2 (float): Length scale of the Rational Quadratic kernel.
        eta3 (float): Periodic repetition scale for the piecewise kernel.
        params_number (int): Number of hyperparameters.
    """

    def __init__(
        self, eta1: float, alpha: float, eta2: float, eta3: float
    ) -> None:  # NOQA
        """Initialize the PiecewiseRQ kernel with its parameters.

        Args:
            eta1 (float): Amplitude of the Rational Quadratic kernel.
            alpha (float): Parameter of the Rational Quadratic kernel.
            eta2 (float): Length scale of the Rational Quadratic kernel.
            eta3 (float): Periodic repetition scale for the piecewise kernel.
        """
        super().__init__(eta1, alpha, eta2, eta3)
        self.eta1: float = eta1
        self.alpha: float = alpha
        self.eta2: float = eta2
        self.eta3: float = eta3
        self.params_number: int = 4

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """Compute the PiecewiseRQ kernel value.

        Args:
            r (np.ndarray): Difference between two data points.

        Returns:
            np.ndarray: Kernel value computed.
        """
        rq_term = self.eta1**2 * (
            1 + 0.5 * abs(r) ** 2 / (self.alpha * self.eta2**2)
        ) ** (-self.alpha)
        abs_r_normalized = abs(r / (0.5 * self.eta3))
        piecewise = (3 * abs_r_normalized + 1) * (1 - abs_r_normalized) ** 3
        piecewise = np.where(abs_r_normalized > 1, 0, piecewise)
        k = rq_term * piecewise
        return k


class NewPeriodic(Kernel):
    """
    Definition of a periodic kernel.

    Derived from mapping the Rational Quadratic kernel to the 2D space
    defined by u(x) = (cos(x), sin(x)).


    Args:
        amp  (float): Amplitude of the kernel.
        alpha2 (float): Alpha parameter of the Rational Quadratic mapping.
        p (float): Period of the kernel.
        ell (float): Length scale of the periodic component.
    """

    def __init__(self, amp: float, alpha: float, p: float, ell: float) -> None:
        """
        Initialize the NewPeriodic kernel with its parameters.

        Args:
            amp  (float): Amplitude of the kernel.
            alpha (float): Alpha parameter of the Rational Quadratic kernel.
            p (float): Period of the kernel.
            ell (float): Length scale of the periodic component.
        """
        super().__init__(amp, alpha, p, ell)
        self.amp: float = amp
        self.alpha: float = alpha
        self.p: float = p
        self.ell: float = ell
        self.params_number: int = 4

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """
        Compute the NewPeriodic kernel value.

        The kernel value is computed as a Rational Quadratic kernel applied to
        the 2D mapped space using the function u(x) = (cos(x), sin(x)).

        Args:
            r (np.ndarray): Array of differences between data points.

        Returns:
            np.ndarray: Kernel value computed.
        """
        return self.amp**2 * (
            1 + 2 * sine(pi * abs(r) / self.p) ** 2 / (self.alpha * self.ell**2)  # NOQA
        ) ** (-self.alpha)


class QuasiNewPeriodic(Kernel):
    """
    Definition of a quasi-periodic kernel.

    Derived from mapping the Rational Quadratic kernel to the 2D space
    u(x) = (cos(x), sin(x)) to multiply it by a Squared Exponential kernel.


    Args:
        amp  (float): Amplitude of the kernel.
        alpha (float): Alpha parameter of the Rational Quadratic kernel.
        ell_e (float): Length scale of the aperiodic component.
        p (float): Period of the periodic component.
        ell_p (float): Length scale of the periodic component.
    """

    def __init__(
        self, amp: float, alpha: float, ell_e: float, p: float, ell_p: float
    ) -> None:
        """
        Initialize the QuasiNewPeriodic kernel with its parameters.

        Args:
            amp (float): Amplitude of the kernel.
            alpha2 (float): Alpha parameter of the Rational Quadratic kernel
                used in the mapping.
            ell_e (float): Length scale of the aperiodic component.
            P (float): Period of the periodic component.
            ell_p (float): Length scale of the periodic component.
        """
        super().__init__(amp, alpha, ell_e, p, ell_p)
        self.amp: float = amp
        self.alpha: float = alpha
        self.ell_e: float = ell_e
        self.p: float = p
        self.ell_p: float = ell_p
        self.params_number: int = 5

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """
        Compute the QuasiNewPeriodic kernel value.

        Args:
            r (np.ndarray): Array of differences between data points.

        Returns:
            np.ndarray: Computed kernel value.
        """
        abs_r = abs(r)
        per_component = (
            1
            + 2 * sine(pi * abs_r / self.p) ** 2 / (self.alpha * self.ell_p**2)  # NOQA
        ) ** (-self.alpha)
        exp_component = exp(-0.5 * abs_r**2 / self.ell_e**2)
        return self.amp**2 * per_component * exp_component


class NewRQP(Kernel):
    """
    Definition of a new quasi-periodic kernel.

    Derived from mapping the rational quadratic kernel to the 2D space
    u(x) = (cos x, sin x) to then multiply it by a rational quadratic kernel.

    Args:
        amp  (float): Amplitude of the kernel.
        alpha1 (float): Alpha parameter of the Rational Quadratic kernel.
        alpha2 (float): Alpha parameter of the Rational Quadratic mapping.
        ell_e (float): Length scale of the aperiodic component.
        p (float): Period of the periodic component.
        ell_p (float): Length scale of the periodic component.
    """

    def __init__(
        self,
        amp: float,
        alpha1: float,
        alpha2: float,
        ell_e: float,
        p: float,
        ell_p: float,
    ) -> None:
        """
        Initialize the NewRQP kernel with its parameters.

        Args:
            amp  (float): Amplitude of the kernel.
            alpha1 (float): Alpha parameter of the Rational Quadratic kernel.
            alpha2 (float): Alpha parameter of the Rational Quadratic mapping.
            ell_e (float): Length scale of the aperiodic component.
            P (float): Period of the periodic component.
            ell_p (float): Length scale of the periodic component.
        """
        super().__init__(amp, alpha1, alpha2, ell_e, p, ell_p)
        self.amp: float = amp
        self.alpha1: float = alpha1
        self.alpha2: float = alpha2
        self.ell_e: float = ell_e
        self.p: float = p
        self.ell_p: float = ell_p
        self.params_number: int = 5

    def __call__(self, r: np.ndarray) -> np.ndarray:
        """
        Compute the NewRQP kernel value.

        Args:
            r (np.ndarray): Array of differences between data points.

        Returns:
            np.ndarray: Computed kernel value.
        """
        abs_r = np.abs(r)
        alpha1_component = (
            1 + 0.5 * abs_r**2 / (self.alpha1 * self.ell_e**2)
        ) ** (  # NOQA
            -self.alpha1
        )
        alpha2_component = (
            1
            + 2 * sine(pi * abs_r / self.p) ** 2 / (self.alpha2 * self.ell_p**2)  # NOQA
        ) ** (-self.alpha2)
        return self.amp**2 * alpha1_component * alpha2_component


class HarmonicPeriodic(Kernel):
    """
    Definition of a periodic kernel.

    Models a periodic signal with a specified number of harmonics.

    Args:
        n (int): Number of harmonics in the periodic signal.
        amp  (float): Amplitude of the kernel.
        p (float): Period of the kernel.
        ell (float): Length scale for the periodic component.
    """

    def __init__(self, n: int, amp: float, p: float, ell: float) -> None:
        """
        Initialize the HarmonicPeriodic kernel with its parameters.

        Args:
            n (int): Number of harmonics in the periodic signal.
            amp  (float): Amplitude of the kernel.
            p (float): Period of the kernel.
            ell (float): Length scale for the periodic component.
        """
        super().__init__(n, amp, p, ell)
        self.n: int = n
        self.amp: float = amp
        self.ell: float = ell
        self.p: float = p
        self.params_number: int = 4

    def __call__(self, r: np.ndarray, s: np.ndarray) -> np.ndarray:  # type: ignore  # NOQA
        """
        Compute the HarmonicPeriodic kernel value for given input arrays.

        Args:
            r (np.ndarray): Array of data points.
            s (np.ndarray): Array of data points.

        Returns:
            np.ndarray: Computed kernel values.
        """
        first_sin = (
            sine((self.n + 0.5) * 2 * pi * r / self.p)
            / 2
            * sine(pi * r / self.p)  # NOQA
        )
        second_sin = (
            sine((self.n + 0.5) * 2 * pi * s / self.p)
            / 2
            * sine(pi * s / self.p)  # NOQA
        )
        sine_component = (first_sin - second_sin) ** 2

        first_cot = 0.5 / np.tan(pi * r / self.p)
        first_cos = (
            cosine((self.n + 0.5) * 2 * pi * r / self.p)
            / 2
            * sine(pi * r / self.p)  # NOQA
        )
        second_cot = 0.5 / np.tan(pi * s / self.p)
        second_cos = (
            cosine((self.n + 0.5) * 2 * pi * s / self.p)
            / 2
            * sine(pi * s / self.p)  # NOQA
        )
        cot_cos_component = (
            first_cot - first_cos - second_cot + second_cos
        ) ** 2  # NOQA

        return self.amp**2 * exp(
            -0.5 * (sine_component + cot_cos_component) / self.ell**2
        )


class QuasiHarmonicPeriodic(Kernel):
    """
    Definition of a quasi-periodic kernel.

    Models periodic signals with a specified number of harmonics.

    Args:
        n (int): Number of harmonics in the periodic signal.
        amp  (float): Amplitude of the kernel.
        ell_e (float): Aperiodic length scale.
        p (float): Period of the kernel.
        ell_p (float): Periodic length scale.
    """

    def __init__(
        self, n: int, amp: float, ell_e: float, p: float, ell_p: float
    ) -> None:
        """
        Initialize the QuasiHarmonicPeriodic kernel with its parameters.

        Args:
            n (int): Number of harmonics in the periodic signal.
            amp  (float): Amplitude of the kernel.
            ell_e (float): Aperiodic length scale.
            p (float): Period of the kernel.
            ell_p (float): Periodic length scale.
        """
        super().__init__(n, amp, ell_e, p, ell_p)
        self.n: int = n
        self.amp: float = amp
        self.ell_e: float = ell_e
        self.p: float = p
        self.ell_p: float = ell_p
        self.params_number: int = 5

    def __call__(self, r: np.ndarray, s: np.ndarray) -> np.ndarray:  # type: ignore  # NOQA
        """
        Compute the QuasiHarmonicPeriodic kernel value for given input arrays.

        Args:
            r (np.ndarray): Array of data points.
            s (np.ndarray): Array of data points.

        Returns:
            np.ndarray: Computed kernel values.
        """
        first_sin = (
            sine((self.n + 0.5) * 2 * pi * r / self.p)
            / 2
            * sine(pi * r / self.p)  # NOQA
        )
        second_sin = (
            sine((self.n + 0.5) * 2 * pi * s / self.p)
            / 2
            * sine(pi * s / self.p)  # NOQA
        )

        sine_component = (first_sin - second_sin) ** 2

        first_cot = 0.5 / np.tan(pi * r / self.p)
        first_cos = (
            cosine((self.n + 0.5) * 2 * pi * r / self.p)
            / 2
            * sine(pi * r / self.p)  # NOQA
        )
        second_cot = 0.5 / np.tan(pi * s / self.p)
        second_cos = (
            cosine((self.n + 0.5) * 2 * pi * s / self.p)
            / 2
            * sine(pi * s / self.p)  # NOQA
        )
        cot_cos_component = (
            first_cot - first_cos - second_cot + second_cos
        ) ** 2  # NOQA

        hp_kernel = exp(
            -0.5 * (sine_component + cot_cos_component) / self.ell_p**2
        )  # NOQA
        se_kernel = exp(-0.5 * abs(r - s) ** 2 / self.ell_e**2)
        return self.amp**2 * hp_kernel * se_kernel

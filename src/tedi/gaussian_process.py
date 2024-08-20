from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.special import loggamma
from scipy.stats import multivariate_normal, norm

from src.tedi import kernels
from src.tedi.utils.kernels import Product, Sum


class CreateProcess:
    """
    A class to represent a Gaussian Process (GP).

    Attributes:
        kernel (Callable): The covariance function of the Gaussian process.
        mean (Callable): The mean function of the Gaussian process.
        time (np.ndarray): The time points at which the Gaussian process is
            observed.
        y (np.ndarray): The observed measurements.
        yerr (Optional[np.ndarray]): The measurement errors. Defaults to a
            small identity matrix if not provided.
        yerr2 (np.ndarray): Squared measurement errors.

    Args:
        kernel (Callable): The covariance function.
        mean (Callable): The mean function.
        time (np.ndarray): The time array.
        y (np.ndarray): The measurements array.
        yerr (Optional[np.ndarray], default=None): The measurement errors.
            Defaults to a small identity matrix if None.
    """

    def __init__(
        self,
        kernel: Callable,
        mean: Callable,
        time: np.ndarray,
        y: np.ndarray,
        yerr: Optional[np.ndarray] = None,
    ):  # NOQA
        self.kernel = kernel
        self.mean = mean
        self.time = time
        self.y = y
        self.yerr = (
            yerr if yerr is not None else 1e-12 * np.identity(self.time.size)
        )  # NOQA
        self.yerr2 = self.yerr**2

    def _kernel_pars(self, kernel: Callable) -> np.ndarray:
        """
        Retrieve the parameters of the specified kernel.

        Args:
            kernel (Callable): The kernel function.

        Returns:
            np.ndarray: The parameters of the kernel.
        """
        return kernel.pars

    def _kernel_matrix(self, kernel: Callable, time: np.ndarray) -> np.ndarray:
        """
        Compute the covariance matrix given a kernel and time array.

        Args:
            kernel (Callable): The kernel function.
            time (np.ndarray): The time points.

        Returns:
            np.ndarray: The covariance matrix.
        """
        if isinstance(kernel, Sum):
            if isinstance(
                kernel.base_kernels[0],
                (kernels.HarmonicPeriodic, kernels.QuasiHarmonicPeriodic),
            ):  # NOQA
                r = time[:, None]
                s = time[None, :]
                k1 = kernel.base_kernels[0](r, s)
                r = time[:, None] - time[None, :]
                return k1 + kernel.base_kernels[1](r)
        if isinstance(
            kernel, (kernels.HarmonicPeriodic, kernels.QuasiHarmonicPeriodic)
        ):  # NOQA
            r = time[:, None]
            s = time[None, :]
            return kernel(r, s)
        r = time[:, None] - time[None, :]
        return kernel(r)

    def _predict_kernel_matrix(
        self, kernel: Callable, time: np.ndarray
    ) -> np.ndarray:  # NOQA
        """
        Compute the prediction kernel matrix for the specified kernel.

        Args:
            kernel (Callable): The kernel function.
            time (np.ndarray): The time points for prediction.

        Returns:
            np.ndarray: The prediction kernel matrix.
        """
        if isinstance(kernel, Sum):
            if isinstance(kernel.base_kernels[1], kernels.WhiteNoise):
                kernel = kernel.base_kernels[0]
        if isinstance(
            kernel,
            (
                kernels.HarmonicPeriodic,
                kernels.QuasiHarmonicPeriodic,
                kernels.unknown,
            ),  # NOQA
        ):  # NOQA
            r = time[:, None]
            s = self.time[None, :]
            return kernel(r, s)
        r = time[:, None] - self.time[None, :]
        return kernel(r)

    def _mean_function(
        self, mean: Callable, time: Optional[np.ndarray] = None
    ) -> np.ndarray:  # NOQA
        """
        Compute the mean function values at the given time points.

        Args:
            mean (Callable): The mean function.
            time (Optional[np.ndarray], default=None): The time points.
                Defaults to the time points provided during initialization.

        Returns:
            np.ndarray: The mean function values.
        """
        if time is None:
            time = self.time
        return mean(time) if mean is not None else np.zeros_like(time)

    def new_kernel(self, kernel: Callable, new_pars: list) -> Callable:
        """
        Update the parameters of the given kernel.

        Args:
            kernel (Callable): The original kernel function.
            new_pars (list): The new hyperparameters for the kernel.

        Returns:
            Callable: The updated kernel function.
        """
        if isinstance(kernel, Sum):
            k1_params = new_pars[: len(kernel.base_kernels[0].pars)]
            k2_params = new_pars[len(kernel.base_kernels[0].pars) :]  # NOQA
            new_k1 = type(kernel.base_kernels[0])(*k1_params)
            new_k2 = type(kernel.base_kernels[1])(*k2_params)
            return new_k1 + new_k2
        elif isinstance(kernel, Product):
            k1_params = new_pars[: len(kernel.base_kernels[0].pars)]
            k2_params = new_pars[len(kernel.base_kernels[0].pars) :]  # NOQA
            new_k1 = type(kernel.base_kernels[0])(*k1_params)
            new_k2 = type(kernel.base_kernels[1])(*k2_params)
            return new_k1 * new_k2
        else:
            return type(kernel)(*new_pars)

    def compute_matrix(
        self,
        kernel: Callable,
        time: np.ndarray,
        nugget: bool = False,
        shift: bool = False,
    ) -> np.ndarray:  # NOQA
        """
        Construct the covariance matrix used in the log marginal likelihood
        calculation.

        Args:
            kernel (Callable): The covariance kernel function.
            time (np.ndarray): The time points.
            nugget (bool, default=False): Whether to add a nugget term to the
                matrix to ensure positive definiteness.
            shift (bool, default=False): Whether to shift the eigenvalues to
                ensure positive definiteness.

        Returns:
            np.ndarray: The covariance matrix.
        """
        K = np.zeros((time.size, time.size))
        k = self._kernel_matrix(kernel, self.time)
        diag = self.yerr**2 * np.identity(self.time.size)
        K = k + diag
        if nugget:
            nugget_value = 0.01
            K = nugget_value * np.identity(self.time.size) + K
        if shift:
            shift_value = 0.01
            K += shift_value * np.identity(self.time.size)
        return K

    def log_likelihood(
        self,
        kernel: Optional[Callable] = None,
        mean: Optional[Callable] = None,
        nugget: bool = False,
        shift: bool = False,
        separate: bool = False,
    ) -> Union[float, Tuple[float, float, float]]:  # NOQA
        """
        Compute the marginal log likelihood of the Gaussian process.

        Args:
            kernel (Optional[Callable], default=None): The covariance function.
                If None, uses the kernel provided during initialization.
            mean (Optional[Callable], default=None): The mean function.
                If None, uses the mean function provided during initialization.
            nugget (bool, default=False): Whether to add a nugget term to the
                covariance matrix.
            shift (bool, default=False): Whether to shift the eigenvalues of
                the covariance matrix.
            separate (bool, default=False): Whether to return the separated
                terms of the log likelihood.

        Returns:
            Union[float, Tuple[float, float, float]]: The marginal log
                likelihood, or a tuple of its separated terms if `separate`
                is True.
        """
        kernel = kernel if kernel else self.kernel
        K = self.compute_matrix(kernel, self.time, nugget=nugget, shift=shift)
        y = self.y - (mean(self.time) if mean else self.mean(self.time))
        try:
            L1 = cho_factor(
                K, overwrite_a=True, lower=False, check_finite=False
            )  # NOQA
            if separate:
                log_like = [
                    -0.5 * np.dot(y.T, cho_solve(L1, y)),
                    -np.sum(np.log(np.diag(L1[0]))),
                    -0.5 * y.size * np.log(2 * np.pi),
                ]
            else:
                log_like = (
                    -0.5 * np.dot(y.T, cho_solve(L1, y))
                    - np.sum(np.log(np.diag(L1[0])))
                    - 0.5 * y.size * np.log(2 * np.pi)
                )
        except LinAlgError:
            return -np.inf
        return log_like

    def marginal_likelihood(
        self,
        kernel1: Optional[Callable] = None,
        mean: Optional[Callable] = None,
        jitter: Optional[float] = None,
        N: int = 1000,
        file: str = "saved_results.txt",
    ) -> float:  # NOQA
        """
        Compute the marginal likelihood by sampling.

        Args:
            kernel1 (Optional[Callable], default=None): The covariance kernel
                function. If None, uses the kernel provided during
                initialization.
            mean (Optional[Callable], default=None): The mean function.
                If None, uses the mean function provided during initialization.
            jitter (Optional[float], default=None): The jitter value to be
                added to the measurement errors.
            N (int, default=1000): Number of samples to compute.
            file (str, default="saved_results.txt"): File to write the
                progress.

        Returns:
            float: The log of the estimated marginal likelihood.
        """
        with open(file, "a") as f:
            m = self._mean_function(mean, self.time)
            err = np.sqrt(jitter**2 + self.yerr2)
            llhood = 0
            for n in range(N):
                sample = self.sample(kernel1, self.time) + m
                normpdf = norm(loc=sample, scale=err).pdf(self.y)
                llhood += normpdf.prod()
                if (n + 1) % 500 == 0:
                    sigmaN = np.std(normpdf)
                    print(
                        n + 1,
                        np.log(llhood / (n + 1)),
                        sigmaN / np.sqrt(n + 1),
                        file=f,  # NOQA
                    )  # NOQA
        return np.log(llhood / N)

    def sample(
        self, kernel: Callable, time: np.ndarray, nugget: bool = False
    ) -> np.ndarray:  # NOQA
        """
        Sample from the Gaussian process given a kernel function.

        Args:
            kernel (Callable): The covariance function.
            time (np.ndarray): The time points for sampling.
            nugget (bool, default=False): Whether to add a nugget term to the
                covariance matrix.

        Returns:
            np.ndarray: A sample from the Gaussian process.
        """
        mean = np.zeros_like(time)
        cov = self._kernel_matrix(kernel, time)
        if nugget:
            nugget_value = 0.01
            cov = (1 - nugget_value) * cov + nugget_value * np.diag(
                np.diag(cov)
            )  # NOQA
        return multivariate_normal(mean, cov, allow_singular=True).rvs()

    def posterior_sample(
        self,
        kernel: Callable,
        mean: Callable,
        a: np.ndarray,
        time: np.ndarray,
        nugget: bool = False,
    ) -> np.ndarray:  # NOQA
        """
        Sample from the posterior distribution of the Gaussian process.

        Args:
            kernel (Callable): The covariance function.
            mean (Callable): The mean function.
            a (np.ndarray): The observed values.
            time (np.ndarray): The time points for sampling.
            nugget (bool, default=False): Whether to add a nugget term to the
                covariance matrix.

        Returns:
            np.ndarray: A sample from the posterior distribution.
        """
        m, _, v, _ = self.prediction(kernel, mean, time)
        return np.random.multivariate_normal(m, v, 1).T

    def prediction(
        self,
        kernel: Optional[Callable] = None,
        mean: Optional[Callable] = None,
        time: Optional[np.ndarray] = None,
        std: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # NOQA
        """
        Predict the conditional distribution of the Gaussian process at new
        time points.

        Args:
            kernel (Optional[Callable], default=None): The covariance function.
                If None, uses the kernel provided during initialization.
            mean (Optional[Callable], default=None): The mean function.
                If None, uses the mean function provided during initialization.
            time (Optional[np.ndarray], default=None): The new time points for
                prediction. If None, uses the time points provided during
                initialization.
            std (bool, default=True): Whether to return the standard deviation.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The predicted mean,
                standard deviation (if `std` is True), and the time points.
        """
        kernel = kernel if kernel else self.kernel
        mean = mean if mean else self.mean
        time = time if time is not None else self.time
        r = self.y - mean(self.time) if mean else self.y
        cov = self._kernel_matrix(kernel, self.time) + np.diag(self.yerr2)
        L1 = cho_factor(cov)
        sol = cho_solve(L1, r)
        Kstar = self._predict_kernel_matrix(kernel, time)
        Kstarstar = self._kernel_matrix(kernel, time)
        y_mean = np.dot(Kstar, sol) + self._mean_function(mean, time)
        kstarT_k_kstar = [
            np.dot(Kstar, cho_solve(L1, Kstar[i, :])) for i in range(time.size)
        ]  # NOQA
        y_cov = Kstarstar - np.array(kstarT_k_kstar)
        y_var = np.diag(y_cov)
        y_std = np.sqrt(y_var) if std else np.zeros_like(y_mean)
        return y_mean, y_std, time

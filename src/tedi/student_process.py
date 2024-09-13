"""Student-t process class."""

from typing import Callable, Optional, Tuple

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve  # type: ignore
from scipy.special import loggamma  # type: ignore

from src.tedi.utils.kernels import Product, Sum


class CreateProcess:
    """
    A class to create a Student-t Process (TP).

    Warning:
        This implementation is not as developed and tested as the Gaussian
        Process class.

    Args:
        kernel (Callable): The covariance function.
        degrees (int): The degrees of freedom for the Student-t process.
        mean (Callable): The mean function.
        time (np.ndarray): The time array.
        y (np.ndarray): The measurements array.
        yerr (Optional[np.ndarray], default=None): The measurement errors.
            Defaults to a small identity matrix if None.
    """

    def __init__(
        self,
        kernel: Callable,
        degrees: int,
        mean: Callable,
        time: np.ndarray,
        y: np.ndarray,
        yerr: Optional[np.ndarray] = None,
    ):  # NOQA
        """Initialize Student-t process."""
        self.kernel = kernel
        self.degrees = degrees
        self.mean = mean
        self.time = time
        self.y = y
        self.yerr = (
            yerr if yerr is not None else 1e-12 * np.identity(self.time.size)
        )  # NOQA

    def _kernel_pars(self, kernel: Callable) -> np.ndarray:
        """
        Retrieve the parameters of the specified kernel.

        Args:
            kernel (Callable): The kernel function.

        Returns:
            np.ndarray: The parameters of the kernel.
        """
        return kernel.pars  # type: ignore

    def _kernel_matrix(
        self, kernel: Callable, time: Optional[np.ndarray] = None
    ) -> np.ndarray:  # NOQA
        """
        Compute the covariance matrix given a kernel and time array.

        Args:
            kernel (Callable): The kernel function.
            time (Optional[np.ndarray], default=None): The time points.
                If None, uses the time points provided during initialization.

        Returns:
            np.ndarray: The covariance matrix.
        """
        if time is None:
            r = self.time[:, None] - self.time[None, :]
        else:
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
            k1_params = new_pars[: len(kernel.base_kernels[0].pars)]  # NOQA
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
            return type(kernel)(*new_pars)  # type: ignore

    def compute_matrix(
        self,
        kernel: Callable,
        time: np.ndarray,
        nugget: bool = False,
        shift: bool = False,
    ) -> np.ndarray:  # NOQA
        """
        Construct the covariance matrix used in the log marginal likelihood.

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
        # Our K starts empty
        K = np.zeros((time.size, time.size))
        # Then we calculate the covariance matrix
        k = self._kernel_matrix(kernel, self.time)
        # addition of the measurement errors
        diag = self.yerr * np.identity(self.time.size)
        K = k + diag
        # more "weight" to the diagonal to avoid a ill-conditioned matrix
        if nugget:
            nugget_value = 0.01  # might be too big
            K = (1 - nugget_value) * K + nugget_value * np.diag(np.diag(K))
        # shifting eigenvalues to avoid a ill-conditioned matrix
        if shift:
            shift_value = 0.01  # might be too big
            K = K + shift_value * np.identity(self.time.size)
        return K

    def log_likelihood(
        self,
        kernel: Callable,
        degrees: int,
        mean: Optional[Callable] = None,
        nugget: bool = False,
        shift: bool = False,
    ) -> float:  # NOQA
        """
        Compute the marginal log likelihood of the Student-t process.

        Args:
            kernel (Callable): The covariance function.
            degrees (int): The degrees of freedom.
            mean (Optional[Callable], default=None): The mean function.
            nugget (bool, default=False): Whether to add a nugget term to the
                covariance matrix.
            shift (bool, default=False): Whether to shift the eigenvalues of
                the covariance matrix.

        Returns:
            float: The marginal log likelihood.
        """
        # covariance matrix calculation
        K = self.compute_matrix(kernel, self.time, nugget, shift)
        # calculation of y having into account the mean funtion
        if mean:
            y = self.y - mean(self.time)
        else:
            y = self.y
        # log marginal likelihood calculation
        try:
            L1 = cho_factor(K, overwrite_a=True, lower=False)
            beta = np.dot(y.T, cho_solve(L1, y))
            log_like = (
                loggamma(0.5 * (degrees + y.size))
                - 0.5 * y.size * np.log((degrees - 2) * np.pi)
                - np.sum(np.log(np.diag(L1[0])))
                - 0.5 * (degrees + y.size) * np.log(1 + beta / (degrees - 2))
                - loggamma(0.5 * degrees)
            )
        except LinAlgError:
            return -np.inf
        return np.real(log_like)

    def sample(
        self, kernel: Callable, degrees: int, time: np.ndarray
    ) -> np.ndarray:  # NOQA
        """
        Sample from the Student-t process.

        Args:
            kernel (Callable): The covariance function.
            degrees (int): The degrees of freedom.
            time (np.ndarray): The time points.

        Returns:
            np.ndarray: A sample from the Student-t process.
        """
        mean = np.zeros_like(self.time)
        x = (
            1
            if degrees == np.inf
            else np.random.chisquare(degrees, time.size) / degrees
        )
        cov = self.compute_matrix(kernel, time)
        z = np.random.multivariate_normal(mean, cov, 1)
        sample = mean + z / np.sqrt(x)  # type: ignore
        return sample.flatten()

    def prediction(
        self,
        kernel: Optional[Callable] = None,
        degrees: Optional[int] = None,
        mean: Optional[Callable] = None,
        time: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the conditional distribution of the Student-t.

        Args:
            kernel (Optional[Callable], default=None): The covariance function.
                If None, uses the kernel provided during initialization.
            degrees (Optional[int], default=None): The degrees of freedom.
                If None, uses the degrees provided during initialization.
            mean (Optional[Callable], default=None): The mean function.
                If None, uses the mean function provided during initialization.
            time (Optional[np.ndarray], default=None): The new time points for
                prediction. If None, uses the time points provided during
                initialization.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The
                predicted mean, standard deviation, and covariance matrix,
                and time used.
        """
        kernel = kernel if kernel else self.kernel
        degrees = degrees if degrees is not None else self.degrees
        mean = mean if mean else self.mean
        time = time if time is not None else self.time
        r = self.y - mean(time) if mean else self.y  # type: ignore
        cov = self._kernel_matrix(kernel, self.time)
        L1 = cho_factor(cov)
        sol = cho_solve(L1, r)
        Kstar = self._predict_kernel_matrix(kernel, time)
        Kstarstar = self._kernel_matrix(kernel, time)
        y_mean = (
            np.dot(Kstar, sol) + self._mean_function(mean, time)
            if mean  # type: ignore
            else np.dot(Kstar, sol)
        )
        kstarT_k_kstar = [
            np.dot(Kstar, cho_solve(L1, Kstar[i, :])) for i in range(time.size)
        ]
        y_cov = Kstarstar - np.array(kstarT_k_kstar)
        var1 = degrees - 2 + np.dot(r.T, sol)
        var2 = degrees - 2 + r.size
        # To check np.abs, without it becomes negative
        y_var = np.abs(var1 * np.diag(y_cov) / var2)
        y_std = np.sqrt(y_var)
        return y_mean, y_std, y_cov

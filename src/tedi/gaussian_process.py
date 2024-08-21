from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.stats import multivariate_normal, norm

from src.tedi import kernels
from src.tedi.utils.kernels import Product, Sum


class CreateProcess:
    """
    A class to represent a Gaussian Process (GP).

    Attributes:
        kernel (Callable[[np.ndarray, np.ndarray], np.ndarray]): The covariance
            function of the Gaussian process.
        mean (Callable[[np.ndarray], np.ndarray]): The mean function of the
            Gaussian process.
        time (np.ndarray): The time points at which the Gaussian process is
            observed.
        y (np.ndarray): The observed measurements.
        yerr (Optional[np.ndarray]): The measurement errors. Defaults to a
            small identity matrix if not provided.
        yerr2 (np.ndarray): Squared measurement errors.
    """

    def __init__(
        self,
        kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
        mean: Callable[[np.ndarray], np.ndarray],
        time: np.ndarray,
        y: np.ndarray,
        yerr: Optional[np.ndarray] = None,
    ):
        self.kernel = kernel
        self.mean = mean
        self.time = time
        self.y = y
        self.yerr = (
            yerr if yerr is not None else 1e-12 * np.identity(self.time.size)
        )  # NOQA
        self.yerr2 = self.yerr**2

    def _kernel_parameters(
        self, kernel: Callable[[np.ndarray, np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Retrieve the parameters of the specified kernel.

        Args:
            kernel (Callable[[np.ndarray, np.ndarray], np.ndarray]): The kernel
                function.

        Returns:
            np.ndarray: The parameters of the kernel.
        """
        return kernel.pars

    def _compute_kernel_matrix(
        self,
        kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
        time: np.ndarray,  # NOQA
    ) -> np.ndarray:
        """
        Compute the covariance matrix given a kernel and time array.

        Args:
            kernel (Callable[[np.ndarray, np.ndarray], np.ndarray]): The kernel
                function.
            time (np.ndarray): The time points.

        Returns:
            np.ndarray: The covariance matrix.
        """
        if isinstance(kernel, Sum):
            if isinstance(
                kernel.base_kernels[0],
                (kernels.HarmonicPeriodic, kernels.QuasiHarmonicPeriodic),
            ):
                r = time[:, None]
                s = time[None, :]
                k1 = kernel.base_kernels[0](r, s)
                r = time[:, None] - time[None, :]
                return k1 + kernel.base_kernels[1](r)

        if isinstance(
            kernel, (kernels.HarmonicPeriodic, kernels.QuasiHarmonicPeriodic)
        ):
            r = time[:, None]
            s = time[None, :]
            return kernel(r, s)

        r = time[:, None] - time[None, :]
        return kernel(r)

    def _compute_predictive_kernel_matrix(
        self,
        kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
        time: np.ndarray,  # NOQA
    ) -> np.ndarray:
        """
        Compute the predictive kernel matrix for the specified kernel.

        Args:
            kernel (Callable[[np.ndarray, np.ndarray], np.ndarray]): The kernel
                function.
            time (np.ndarray): The time points for prediction.

        Returns:
            np.ndarray: The predictive kernel matrix.
        """
        if isinstance(kernel, Sum):
            if isinstance(kernel.base_kernels[1], kernels.WhiteNoise):
                kernel = kernel.base_kernels[0]
        if isinstance(
            kernel,
            (
                kernels.HarmonicPeriodic,
                kernels.QuasiHarmonicPeriodic,
            ),
        ):
            r = time[:, None]
            s = self.time[None, :]
            return kernel(r, s)
        r = time[:, None] - self.time[None, :]
        return kernel(r)

    def _compute_mean_function(
        self,
        mean: Callable[[np.ndarray], np.ndarray],
        time: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute the mean function values at the given time points.

        Args:
            mean (Callable[[np.ndarray], np.ndarray]): The mean function.
            time (Optional[np.ndarray], default=None): The time points.
                Defaults to the time points provided during initialization.

        Returns:
            np.ndarray: The mean function values.
        """
        if time is None:
            time = self.time
        return mean(time) if mean is not None else np.zeros_like(time)

    def update_kernel(
        self,
        kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
        new_params: list,  # NOQA
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """
        Update the parameters of the given kernel.

        Args:
            kernel (Callable[[np.ndarray, np.ndarray], np.ndarray]): The
                original kernel function.
            new_params (list): The new hyperparameters for the kernel.

        Returns:
            Callable[[np.ndarray, np.ndarray], np.ndarray]: The updated kernel
                function.
        """
        if isinstance(kernel, Sum):
            k1_params = new_params[: len(kernel.base_kernels[0].pars)]  # NOQA
            k2_params = new_params[len(kernel.base_kernels[0].pars) :]  # NOQA
            new_k1 = type(kernel.base_kernels[0])(*k1_params)
            new_k2 = type(kernel.base_kernels[1])(*k2_params)
            return new_k1 + new_k2
        elif isinstance(kernel, Product):
            k1_params = new_params[: len(kernel.base_kernels[0].pars)]
            k2_params = new_params[len(kernel.base_kernels[0].pars) :]  # NOQA
            new_k1 = type(kernel.base_kernels[0])(*k1_params)
            new_k2 = type(kernel.base_kernels[1])(*k2_params)
            return new_k1 * new_k2
        else:
            return type(kernel)(*new_params)

    def compute_covariance_matrix(
        self,
        kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
        time: np.ndarray,
        add_nugget: bool = False,
        add_shift: bool = False,
    ) -> np.ndarray:
        """
        Construct the covariance matrix used in the log marginal likelihood
            calculation.

        Args:
            kernel (Callable[[np.ndarray, np.ndarray], np.ndarray]): The
                covariance kernel function.
            time (np.ndarray): The time points.
            add_nugget (bool, default=False): Whether to add a nugget term to
                the matrix to ensure positive definiteness.
            add_shift (bool, default=False): Whether to shift the eigenvalues
                to ensure positive definiteness.

        Returns:
            np.ndarray: The covariance matrix.
        """
        covariance_matrix = np.zeros((time.size, time.size))
        kernel_matrix = self._compute_kernel_matrix(kernel, self.time)
        diag_matrix = self.yerr2 * np.identity(self.time.size)
        covariance_matrix = kernel_matrix + diag_matrix
        if add_nugget:
            nugget_value = 0.01
            covariance_matrix = (
                nugget_value * np.identity(self.time.size) + covariance_matrix
            )
        if add_shift:
            shift_value = 0.01
            covariance_matrix += shift_value * np.identity(self.time.size)
        return covariance_matrix

    def log_marginal_likelihood(
        self,
        kernel: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,  # NOQA
        mean: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        add_nugget: bool = False,
        add_shift: bool = False,
        return_separated: bool = False,
    ) -> Union[float, Tuple[float, float, float]]:
        """
        Compute the marginal log likelihood of the Gaussian process.

        Args:
            kernel (Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]],
                default=None): The covariance function. If None, uses the
                kernel provided during initialization.
            mean (Optional[Callable[[np.ndarray], np.ndarray]], default=None):
                The mean function. If None, uses the mean function provided
                during initialization.
            add_nugget (bool, default=False): Whether to add a nugget term to
                the covariance matrix.
            add_shift (bool, default=False): Whether to shift the eigenvalues
                of the covariance matrix.
            return_separated (bool, default=False): Whether to return the
                separated terms of the log likelihood.

        Returns:
            Union[float, Tuple[float, float, float]]: The marginal log
                likelihood, or a tuple of its separated terms if
                `return_separated` is True.
        """
        kernel = kernel if kernel else self.kernel
        covariance_matrix = self.compute_covariance_matrix(
            kernel, self.time, add_nugget=add_nugget, add_shift=add_shift
        )
        residuals = self.y - (
            self._compute_mean_function(mean) if mean else self.mean(self.time)
        )
        try:
            cholesky_factor = cho_factor(
                covariance_matrix,
                overwrite_a=True,
                lower=False,
                check_finite=False,  # NOQA
            )
            if return_separated:
                log_likelihood = [
                    -0.5
                    * np.dot(
                        residuals.T, cho_solve(cholesky_factor, residuals)
                    ),  # NOQA
                    -np.sum(np.log(np.diag(cholesky_factor[0]))),
                    -0.5 * residuals.size * np.log(2 * np.pi),
                ]
            else:
                log_likelihood = (
                    -0.5
                    * np.dot(residuals.T, cho_solve(cholesky_factor, residuals))  # NOQA
                    - np.sum(np.log(np.diag(cholesky_factor[0])))
                    - 0.5 * residuals.size * np.log(2 * np.pi)
                )
        except LinAlgError:
            return float("-inf")
        return log_likelihood

    def compute_marginal_likelihood_sample(
        self,
        kernel: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,  # NOQA
        mean: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        jitter: Optional[float] = None,
        num_samples: int = 1000,
        output_file: str = "results.txt",
    ) -> float:
        """
        Compute the marginal likelihood by sampling.

        Args:
            kernel (Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]],
                default=None): The covariance kernel function. If None, uses
                the kernel provided during initialization.
            mean (Optional[Callable[[np.ndarray], np.ndarray]], default=None):
                The mean function. If None, uses the mean function provided
                during initialization.
            jitter (Optional[float], default=None): The jitter value to be
                added to the measurement errors.
            num_samples (int, default=1000): Number of samples to compute.
            output_file (str, default="results.txt"): File to write the
                progress.

        Returns:
            float: The log of the estimated marginal likelihood.
        """
        with open(output_file, "a") as f:
            sample_mean = self._compute_mean_function(mean, self.time)
            sample_error = np.sqrt(
                (jitter if jitter else 0.0) ** 2 + self.yerr2
            )  # NOQA
            log_likelihood_sum = 0
            for n in range(num_samples):
                sample = self.sample(kernel, self.time) + sample_mean
                norm_pdf = norm(loc=sample, scale=sample_error).pdf(self.y)
                log_likelihood_sum += norm_pdf.prod()
                if (n + 1) % 500 == 0:
                    sigma_n = np.std(norm_pdf)
                    print(
                        n + 1,
                        np.log(log_likelihood_sum / (n + 1)),
                        sigma_n / np.sqrt(n + 1),
                        file=f,
                    )
        return np.log(log_likelihood_sum / num_samples)

    def sample(
        self,
        kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
        time: np.ndarray,
        add_nugget: bool = False,
    ) -> np.ndarray:
        """
        Sample from the Gaussian process given a kernel function.

        Args:
            kernel (Callable[[np.ndarray, np.ndarray], np.ndarray]): The
                covariance function.
            time (np.ndarray): The time points for sampling.
            add_nugget (bool, default=False): Whether to add a nugget term to
                the covariance matrix.

        Returns:
            np.ndarray: A sample from the Gaussian process.
        """
        sample_mean = np.zeros_like(time)
        covariance_matrix = self._compute_kernel_matrix(kernel, time)
        if add_nugget:
            nugget_value = 0.01
            covariance_matrix = (
                1 - nugget_value
            ) * covariance_matrix + nugget_value * np.diag(
                np.diag(covariance_matrix)
            )  # NOQA
        return multivariate_normal(
            sample_mean, covariance_matrix, allow_singular=True
        ).rvs()

    def posterior_sample(
        self,
        kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
        mean: Callable[[np.ndarray], np.ndarray],
        time: np.ndarray,
    ) -> np.ndarray:
        """
        Sample from the posterior distribution of the Gaussian process.

        Args:
            kernel (Callable[[np.ndarray, np.ndarray], np.ndarray]): The
                covariance function.
            mean (Callable[[np.ndarray], np.ndarray]): The mean function.
            time (np.ndarray): The time points for sampling.

        Returns:
            np.ndarray: A sample from the posterior distribution.
        """
        predicted_mean, _, predicted_covariance, _ = self.prediction(
            kernel, mean, time
        )  # NOQA
        return np.random.multivariate_normal(
            predicted_mean, predicted_covariance, 1
        ).T  # NOQA

    def prediction(
        self,
        kernel: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,  # NOQA
        mean: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        time: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the conditional distribution of the Gaussian process at new
        time points.

        Args:
            kernel (Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]],
                default=None): The covariance function. If None, uses the
                kernel provided during initialization.
            mean (Optional[Callable[[np.ndarray], np.ndarray]], default=None):
                The mean function. If None, uses the mean function provided
                during initialization.
            time (Optional[np.ndarray], default=None): The new time points for
                prediction. If None, uses the time points provided during
                initialization.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The 
                predicted mean, standard deviation, and covariance matrix, 
                and time used.
        """
        kernel = kernel if kernel else self.kernel
        mean = mean if mean else self.mean
        time = time if time is not None else self.time
        residuals = self.y - mean(self.time) if mean else self.y
        covariance_matrix = self._compute_kernel_matrix(
            kernel, self.time
        ) + np.diag(  # NOQA
            self.yerr2
        )
        cholesky_factor = cho_factor(covariance_matrix)
        solution = cho_solve(cholesky_factor, residuals)
        predictive_kernel = self._compute_predictive_kernel_matrix(kernel, time)  # NOQA
        predictive_cov_matrix = self._compute_kernel_matrix(kernel, time)
        mean_prediction = np.dot(
            predictive_kernel, solution
        ) + self._compute_mean_function(mean, time)
        kstarT_k_kstar = [
            np.dot(
                predictive_kernel,
                cho_solve(cholesky_factor, predictive_kernel[i, :]),  # NOQA
            )
            for i in range(time.size)
        ]
        covariance_prediction = predictive_cov_matrix - np.array(kstarT_k_kstar)  # NOQA
        variance_prediction = np.diag(covariance_prediction)
        std_deviation = np.sqrt(variance_prediction)

        return mean_prediction, std_deviation, covariance_prediction, time

import numpy as np

from src.tedi import kernels, means
from src.tedi.gaussian_process import CreateProcess

kernel = kernels.SquaredExponential(1.0, 1.0)
mean = means.Constant(0.0)
time = np.linspace(0, 10, 10)
y = np.sin(time)
yerr = 0.1 * np.ones_like(y)

# Initialize the Gaussian process
gp = CreateProcess(kernel=kernel, mean=mean, time=time, y=y, yerr=yerr)
# NOQA


def test_kernel_parameters() -> None:  # type: ignore
    kernel_params = gp._kernel_parameters()
    assert len(kernel_params) == 2, "Kernel should have 2 parameters"
    assert kernel_params[0] == 1.0, "First parameter should be 1.0"
    assert kernel_params[1] == 1.0, "Second parameter should be 1.0"


def test_compute_kernel_matrix() -> None:  # type: ignore
    kernel_matrix = gp._compute_kernel_matrix(kernel, time)
    assert kernel_matrix.shape == (10, 10), "Kernel matrix shape mismatch"
    assert np.all(
        kernel_matrix >= 0
    ), "Kernel matrix should be positive semi-definite"  # NOQA


def test_update_kernel() -> None:  # type: ignore
    new_params = [2.0, 0.5]
    gp.update_kernel(new_params)
    updated_params = gp._kernel_parameters()
    assert updated_params[0] == 2.0, "First parameter should be updated to 2.0"
    assert updated_params[1] == 0.5, "Second parameter should be updated to 0.5"  # NOQA


def test_compute_covariance_matrix() -> None:  # type: ignore
    covariance_matrix = gp.compute_covariance_matrix(kernel, time)
    assert covariance_matrix.shape == (
        10,
        10,
    ), "Covariance matrix shape mismatch"  # NOQA
    assert np.all(
        covariance_matrix >= 0
    ), "Covariance matrix should be positive semi-definite"


def test_log_marginal_likelihood() -> None:  # type: ignore
    log_marg_likelihood = gp.log_marginal_likelihood()
    assert isinstance(
        log_marg_likelihood, float
    ), "Log marginal likelihood should be a float"


def test_sample() -> None:  # type: ignore
    sample = gp.sample(kernel, time)
    assert sample.shape == time.shape, "Sample shape mismatch"


def test_prediction() -> None:  # type: ignore
    mean_pred, std_pred, _, _ = gp.prediction(kernel, mean, time)
    assert mean_pred.shape == time.shape, "Predicted mean shape mismatch"
    assert std_pred.shape == time.shape, "Predicted std shape mismatch"
    assert np.all(std_pred >= 0), "Predicted std should be non-negative"


def test_posterior_sample() -> None:  # type: ignore
    posterior_sample = gp.posterior_sample(kernel, mean, time)
    assert (
        posterior_sample.shape[0] == time.size
    ), "Posterior sample shape mismatch"  # NOQA


def test_compute_marginal_likelihood_sample() -> None:  # type: ignore
    log_marg_likelihood_sample = gp.compute_marginal_likelihood_sample(
        kernel, num_samples=10
    )
    assert isinstance(
        log_marg_likelihood_sample, float
    ), "Log marginal likelihood sample should be a float"

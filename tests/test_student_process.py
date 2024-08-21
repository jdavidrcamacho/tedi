import numpy as np

from src.tedi import kernels, means
from src.tedi.student_process import CreateProcess

# Setup common variables for all tests
kernel = kernels.SquaredExponential(1.0, 1.0)
degrees = 5
mean = means.Constant(0)
time = np.linspace(0, 10, 10)
y = np.sin(time)
yerr = 0.1 * np.ones_like(y)

# Initialize the Student-t process
tp = CreateProcess(kernel=kernel, degrees=degrees, mean=mean, time=time, y=y, yerr=yerr)


def test_kernel_parameters():
    kernel_params = tp._kernel_pars(kernel)
    assert len(kernel_params) == 2, "Kernel should have 2 parameters"
    assert kernel_params[0] == 1.0, "First parameter should be 1.0"
    assert kernel_params[1] == 1.0, "Second parameter should be 1.0"


def test_kernel_matrix():
    kernel_matrix = tp._kernel_matrix(kernel, time)
    assert kernel_matrix.shape == (10, 10), "Kernel matrix shape mismatch"
    assert np.all(kernel_matrix >= 0), "Kernel matrix should be positive semi-definite"


def test_predict_kernel_matrix():
    pred_kernel_matrix = tp._predict_kernel_matrix(kernel, time)
    assert pred_kernel_matrix.shape == (
        10,
        10,
    ), "Prediction kernel matrix shape mismatch"
    assert np.all(
        pred_kernel_matrix >= 0
    ), "Prediction kernel matrix should be positive semi-definite"


def test_mean_function():
    mean_values = tp._mean_function(mean, time)
    assert mean_values.shape == time.shape, "Mean function values shape mismatch"
    assert np.all(mean_values == 0), "Mean function values should be zeros"


def test_new_kernel():
    new_params = [2.0, 0.5]
    updated_kernel = tp.new_kernel(kernel, new_params)
    updated_params = tp._kernel_pars(updated_kernel)
    assert updated_params[0] == 2.0, "First parameter should be updated to 2.0"
    assert updated_params[1] == 0.5, "Second parameter should be updated to 0.5"


def test_compute_matrix():
    cov_matrix = tp.compute_matrix(kernel, time)
    assert cov_matrix.shape == (10, 10), "Covariance matrix shape mismatch"
    assert np.all(cov_matrix >= 0), "Covariance matrix should be positive semi-definite"


def test_log_likelihood():
    log_like = tp.log_likelihood(kernel=kernel, degrees=degrees)
    assert isinstance(log_like, float), "Log likelihood should be a float"
    assert not np.isnan(log_like), "Log likelihood should not be NaN"


def test_sample():
    sample = tp.sample(kernel=kernel, degrees=degrees, time=time)
    assert sample.shape == time.shape, "Sample shape mismatch"


def test_prediction():
    mean_pred, std_pred, cov_pred = tp.prediction(
        kernel=kernel, degrees=degrees, time=time
    )
    assert mean_pred.shape == time.shape, "Predicted mean shape mismatch"
    assert std_pred.shape == time.shape, "Predicted std shape mismatch"
    assert cov_pred.shape == (10, 10), "Predicted covariance matrix shape mismatch"
    assert np.all(std_pred >= 0), "Predicted standard deviation should be non-negative"

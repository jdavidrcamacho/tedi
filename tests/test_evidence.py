import numpy as np

from src.tedi.evidence import (
    _perrakis_error,
    compute_perrakis_estimate,
    estimate_density,
)


def test_compute_perrakis_estimate() -> None:
    marginal_samples = np.random.normal(size=(100, 2))

    def mock_lnlike(x):
        return -0.5 * np.sum(x**2, axis=1)

    def mock_lnprior(x):
        return -0.5 * np.sum(x**2, axis=1)

    result = compute_perrakis_estimate(
        marginal_samples, mock_lnlike, mock_lnprior
    )  # NOQA
    assert isinstance(result, float), "Expected result to be a float."

    result_with_error = compute_perrakis_estimate(
        marginal_samples, mock_lnlike, mock_lnprior, errorestimation=True
    )
    assert isinstance(
        result_with_error, tuple
    ), "Expected result to be a tuple."  # NOQA
    assert (
        len(result_with_error) == 2
    ), "Expected tuple to contain two elements."  # NOQA
    assert isinstance(
        result_with_error[0], float
    ), "First element should be a float."  # NOQA
    assert isinstance(
        result_with_error[1], float
    ), "Second element should be a float."  # NOQA


def test_perrakis_error() -> None:
    marginal_samples = np.random.normal(size=(100, 2))

    def mock_lnlike(x):
        return -0.5 * np.sum(x**2, axis=1)

    def mock_lnprior(x):
        return -0.5 * np.sum(x**2, axis=1)

    result = _perrakis_error(marginal_samples, mock_lnlike, mock_lnprior)
    assert isinstance(result, float), "Expected result to be a float."


def test_estimate_density_histogram() -> None:
    data = np.random.normal(size=100)

    density = estimate_density(data, method="histogram", nbins=10)
    assert (
        density.shape == data.shape
    ), "Density shape should match input data shape."  # NOQA


def test_estimate_density_kde() -> None:
    data = np.random.normal(size=100)

    density = estimate_density(data, method="kde")
    assert (
        density.shape == data.shape
    ), "Density shape should match input data shape."  # NOQA


def test_estimate_density_normal() -> None:
    data = np.random.normal(size=100)

    density = estimate_density(data, method="normal")
    assert (
        density.shape == data.shape
    ), "Density shape should match input data shape."  # NOQA


def test_estimate_density_invalid_method() -> None:
    data = np.random.normal(size=100)

    try:
        estimate_density(data, method="invalid_method")
    except ValueError as e:
        assert str(e) == "Invalid method specified for density estimation."

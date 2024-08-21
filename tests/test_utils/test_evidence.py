import numpy as np
import pytest

from src.tedi.utils.evidence import (
    MultivariateGaussian,
    compute_harmonicmean,
    log_sum,
    multivariate_normal,
    run_hme_mc,
)


def test_multivariate_normal_cholesky() -> None:
    r = np.array([1.0, 2.0])
    c = np.array([[2.0, 0.5], [0.5, 1.0]])
    result = multivariate_normal(r, c, method="cholesky")
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_multivariate_normal_solve() -> None:
    r = np.array([1.0, 2.0])
    c = np.array([[2.0, 0.5], [0.5, 1.0]])
    result = multivariate_normal(r, c, method="solve")
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_multivariate_normal_invalid_method() -> None:
    r = np.array([1.0, 2.0])
    c = np.array([[2.0, 0.5], [0.5, 1.0]])
    with pytest.raises(ValueError, match="Invalid method"):
        multivariate_normal(r, c, method="invalid")  # type: ignore


def test_MultivariateGaussian_pdf_single_sample() -> None:
    mu = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    x = np.array([1.0, 2.0])
    mvg = MultivariateGaussian(mu, cov)
    result = mvg.pdf(x)
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_MultivariateGaussian_pdf_multiple_samples() -> None:
    mu = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    x = np.array([[1.0, 2.0], [0.5, 1.5], [-1.0, -2.0]])
    mvg = MultivariateGaussian(mu, cov)
    result = mvg.pdf(x)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert np.all(np.isfinite(result))


def test_MultivariateGaussian_pdf_invalid_dimensions() -> None:
    mu = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    x = np.array([[1.0], [2.0]])  # Mismatched dimensions
    mvg = MultivariateGaussian(mu, cov)
    with pytest.raises(
        ValueError, match="Input array not aligned with covariance"
    ):  # NOQA
        mvg.pdf(x)


def test_MultivariateGaussian_rvs() -> None:
    mu = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    mvg = MultivariateGaussian(mu, cov)
    samples = mvg.rvs(5)
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (5, 2)
    assert np.all(np.isfinite(samples))


def test_log_sum() -> None:
    log_values = np.array([0.0, -1.0, -2.0])
    result = log_sum(log_values)
    assert isinstance(result, float)
    assert np.isfinite(result)
    assert result > max(log_values)


def test_compute_harmonicmean_with_lnlike_post() -> None:
    lnlike_post = np.array([-10.0, -5.0, -2.0, -0.5])
    result = compute_harmonicmean(lnlike_post)
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_compute_harmonicmean_with_posterior_sample() -> None:
    posterior_sample = np.random.normal(0, 1, size=(100, 2))

    def lnlikefunc(samples):
        return -0.5 * np.sum(samples**2, axis=1)

    result = compute_harmonicmean(np.array([]), posterior_sample, lnlikefunc)
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_run_hme_mc() -> None:
    log_likelihood = np.array([-10.0, -5.0, -2.0, -0.5])
    nmc = 10
    samplesize = 3
    result = run_hme_mc(log_likelihood, nmc, samplesize)
    assert isinstance(result, np.ndarray)
    assert result.shape == (nmc,)
    assert np.all(np.isfinite(result))

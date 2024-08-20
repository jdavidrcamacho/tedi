import numpy as np
import pytest

from src.tedi.utils.evidence import MultivariateGaussian, multivariate_normal


def test_multivariate_normal_cholesky():
    r = np.array([1.0, 2.0])
    c = np.array([[2.0, 0.5], [0.5, 1.0]])
    result = multivariate_normal(r, c, method="cholesky")
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_multivariate_normal_solve():
    r = np.array([1.0, 2.0])
    c = np.array([[2.0, 0.5], [0.5, 1.0]])
    result = multivariate_normal(r, c, method="solve")
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_multivariate_normal_invalid_method():
    r = np.array([1.0, 2.0])
    c = np.array([[2.0, 0.5], [0.5, 1.0]])
    with pytest.raises(ValueError, match="Invalid method"):
        multivariate_normal(r, c, method="invalid")


def test_MultivariateGaussian_pdf_single_sample():
    mu = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    x = np.array([1.0, 2.0])
    mvg = MultivariateGaussian(mu, cov)
    result = mvg.pdf(x)
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_MultivariateGaussian_pdf_multiple_samples():
    mu = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    x = np.array([[1.0, 2.0], [0.5, 1.5], [-1.0, -2.0]])
    mvg = MultivariateGaussian(mu, cov)
    result = mvg.pdf(x)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert np.all(np.isfinite(result))


def test_MultivariateGaussian_pdf_invalid_dimensions():
    mu = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    x = np.array([[1.0], [2.0]])  # Mismatched dimensions
    mvg = MultivariateGaussian(mu, cov)
    with pytest.raises(ValueError, match="Input array not aligned with covariance"):
        mvg.pdf(x)


def test_MultivariateGaussian_rvs():
    mu = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    mvg = MultivariateGaussian(mu, cov)
    samples = mvg.rvs(5)
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (5, 2)
    assert np.all(np.isfinite(samples))

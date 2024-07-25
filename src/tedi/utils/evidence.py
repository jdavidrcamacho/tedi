"""Functions for the evidence calculation."""

from typing import Literal

import numpy as np
import scipy as sp


def multivariate_normal(
    r: np.ndarray,
    c: np.ndarray,
    method: Literal["cholesky", "solve"] = "cholesky",  # NOQA
) -> float:
    """
    Computes the multivariate normal density for a given residual vector `r`
    and covariance matrix `c`.

    Args:
        r (np.ndarray): A 1-D array of shape (k,) representing the residual
            vector.
        c (np.ndarray): A 2-D array or matrix of shape (k, k) representing the
            covariance matrix.
        method (Literal["cholesky", "solve"], optional): The method used to
            compute the multivariate density.
                - "cholesky": Uses Cholesky decomposition via
                scipy.linalg.cho_factor` and `scipy.linalg.cho_solve`.
                - "solve": Uses `np.linalg.solve` and `np.linalg.slogdet`.
            Default is "cholesky".

    Returns:
        float: The multivariate density value at the residual vector `r`.

    Raises:
        ValueError: If the specified method is invalid.
    """
    # Compute normalization factor used for all methods.
    normalization_factor = len(r) * np.log(2 * np.pi)

    # Use Cholesky decomposition of covariance.
    if method == "cholesky":
        cho, lower = sp.linalg.cho_factor(c)
        alpha = sp.linalg.cho_solve((cho, lower), r)
        return -0.5 * (
            normalization_factor
            + np.dot(r, alpha)
            + 2 * np.sum(np.log(np.diag(cho)))  # NOQA
        )
    # Use slogdet and solve
    if method == "solve":
        (_, d) = np.linalg.slogdet(c)
        alpha = np.linalg.solve(c, r)
        return -0.5 * (normalization_factor + np.dot(r, alpha) + d)
    raise ValueError("Invalid method. Choose either 'cholesky' or 'solve'.")


class MultivariateGaussian(sp.stats.rv_continuous):
    """
    Multivariate Gaussian distribution.

    Args:
        mu (np.ndarray): Mean vector of shape (k,) for the multivariate
            Gaussian distribution.
        cov (np.ndarray): Covariance matrix of shape (k, k) for the
            multivariate Gaussian distribution.
    """

    def __init__(self, mu: np.ndarray, cov: np.ndarray) -> None:
        super().__init__()
        self.mu = mu
        self.cov = cov + 1e-10 * np.eye(len(mu))
        self.dimensions = len(mu)

    def pdf(
        self, x: np.ndarray, method: Literal["cholesky", "solve"] = "cholesky"
    ) -> np.ndarray:
        """
        Computes the probability density function (PDF) of the multivariate
        Gaussian distribution.

        Args:
            x (np.ndarray): Input data with shape (n, k) for n samples of k
                dimensions, or (k,) for a single sample.
            method (Literal["cholesky", "solve"], optional): Method to use for
                computation. Defaults to "cholesky".

        Returns:
            np.ndarray: Probability density function values for the input data.
                Shape will be (n,) if `x` is 2D, otherwise (1,).

        Raises:
            ValueError: If the input array `x` is not 1-D or 2-D, or if the
                dimensions do not match the covariance matrix.
        """
        if 1 < len(x.shape) < 3:
            if x.T.shape[0] != len(self.cov):
                raise ValueError(
                    "Input array not aligned with covariance. "
                    "It must have dimensions (n x k), where k is "
                    "the dimension of the multivariate Gaussian."
                )
            mvg = np.zeros(len(x))
            for s, rr in enumerate(x):
                mvg[s] = multivariate_normal(rr - self.mu, self.cov, method)
            return mvg
        if len(x.shape) == 1:
            return multivariate_normal(x - self.mu, self.cov, method)
        raise ValueError("Input array must be 1- or 2-D.")

    def rvs(self, nsamples: int) -> np.ndarray:
        """
        Generates random samples from the multivariate Gaussian distribution.

        Args:
            nsamples (int): Number of samples to generate.

        Returns:
            np.ndarray: Random samples with shape (nsamples, k).
        """
        return np.random.multivariate_normal(self.mu, self.cov, nsamples)

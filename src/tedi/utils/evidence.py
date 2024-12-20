"""Functions for the evidence calculation."""

from math import log
from typing import Callable, Literal, Optional, Tuple

import numpy as np
import scipy as sp  # type: ignore


def multivariate_normal(
    r: np.ndarray,
    c: np.ndarray,
    method: Literal["cholesky", "solve"] = "cholesky",  # NOQA
) -> float:
    """
    Compute the multivariate normal density.

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
        """Initialize multivariate gaussian distribution."""
        super().__init__()
        self.mu = mu
        self.cov = cov + 1e-10 * np.eye(len(mu))
        self.dimensions = len(mu)

    def pdf(
        self, x: np.ndarray, method: Literal["cholesky", "solve"] = "cholesky"
    ) -> np.ndarray:
        """
        Compute the probability density function (PDF).

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
            return np.array(multivariate_normal(x - self.mu, self.cov, method))
        raise ValueError("Input array must be 1- or 2-D.")

    def rvs(self, nsamples: int) -> np.ndarray:
        """
        Generate random samples from the multivariate Gaussian distribution.

        Args:
            nsamples (int): Number of samples to generate.

        Returns:
            np.ndarray: Random samples with shape (nsamples, k).
        """
        return np.random.multivariate_normal(self.mu, self.cov, nsamples)


def log_sum(log_summands: np.ndarray) -> float:
    """
    Compute the logarithm of the sum of exponentials of input elements.

    Args:
        log_summands (np.ndarray): Array of log values to sum.

    Returns:
        float: Logarithm of the sum of exponentials of the input elements.
    """
    a = np.inf
    x = log_summands.copy()
    while a == np.inf or a == -np.inf or np.isnan(a):
        a = x[0] + np.log(1 + np.sum(np.exp(x[1:] - x[0])))
        x = np.random.permutation(x)  # Instead of random.shuffle(x)
    return a


def compute_harmonicmean(
    lnlike_post: np.ndarray,
    posterior_sample: Optional[np.ndarray] = None,
    lnlikefunc: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    lnlikeargs: Tuple = (),
    **kwargs
) -> float:
    """
    Compute the harmonic mean estimate of the marginal likelihood.

    The estimation is based on n posterior samples (indexed by s, with s = 0,
    ..., n-1), but can be done directly if the log(likelihood) in this sample
    is passed.

    Args:
        lnlike_post (np.ndarray): Log-likelihood computed over a posterior
            sample. 1-D array of length n. If an empty array is given, then
            compute from posterior sample.
        posterior_sample (Optional[np.ndarray], optional): A sample from the
            parameter posterior distribution. Dimensions are (n x k), where k
            is the number of parameters. If None, the computation is done using
            the log(likelihood) obtained from the posterior sample.
        lnlikefunc (Optional[Callable[[np.ndarray], np.ndarray]], optional):
            Function to compute ln(likelihood) on the marginal samples.
        lnlikeargs (Tuple, optional): Extra arguments passed to the likelihood
            function.
        **kwargs: Additional parameters. The `size` parameter is expected to
            specify the size of the sample used for computation. If none is
            given, use the size of the given array or posterior sample.

    Returns:
        float: The harmonic mean estimate of the marginal likelihood.

    References:
        Kass & Raftery (1995), JASA vol. 90, N. 430, pp. 773-795
    """
    if len(lnlike_post) == 0 and posterior_sample is not None:
        if lnlikefunc is None:
            raise ValueError(
                "Likelihood function must be provided if lnlike_post is empty."
            )

        samplesize = kwargs.pop("size", len(posterior_sample))
        if samplesize < len(posterior_sample):
            posterior_subsample = np.random.choice(
                posterior_sample, size=samplesize, replace=False
            )
        else:
            posterior_subsample = posterior_sample.copy()
        # Compute log likelihood in posterior sample.
        log_likelihood = lnlikefunc(posterior_subsample, *lnlikeargs)
    elif len(lnlike_post) > 0:
        samplesize = kwargs.pop("size", len(lnlike_post))
        log_likelihood = np.random.choice(
            lnlike_post, size=samplesize, replace=False
        )  # NOQA
    else:
        raise ValueError(
            "At least one of lnlike_post or posterior_sample must be provided."
        )

    hme = -log_sum(-log_likelihood) + log(len(log_likelihood))
    return hme


def run_hme_mc(
    log_likelihood: np.ndarray, nmc: int, samplesize: int
) -> np.ndarray:  # NOQA
    """
    Run Monte Carlo simulations to compute the harmonic mean estimate.

    Args:
        log_likelihood (np.ndarray): Array of log-likelihood values.
        nmc (int): Number of Monte Carlo simulations.
        samplesize (int): Size of the sample used in each simulation.

    Returns:
        np.ndarray: Array of harmonic mean estimates from each simulation.
    """
    hme = np.zeros(nmc)
    for i in range(nmc):
        hme[i] = compute_harmonicmean(log_likelihood, size=samplesize)
    return hme

"""Computation of the evidence using the method of Perrakis et al. (2014)."""

from math import log, sqrt
from typing import Callable, Optional, Tuple, Union

import numpy as np
import scipy.stats  # type: ignore

from .utils.evidence import log_sum


def compute_perrakis_estimate(
    marginal_sample: np.ndarray,
    lnlikefunc: Callable[[np.ndarray], np.ndarray],
    lnpriorfunc: Callable[[np.ndarray], np.ndarray],
    nsamples: int = 1000,
    lnlikeargs: Tuple = (),
    lnpriorargs: Tuple = (),
    densityestimation: str = "kde",
    errorestimation: bool = False,
    **kwargs
) -> Union[float, Tuple[float, float]]:
    """
    Compute the Perrakis estimate of the Bayesian evidence.

    The estimation is based on `m` marginal posterior samples.

    Args:
        marginal_sample (np.ndarray): A sample from the parameter marginal
            posterior distribution. Dimensions are (n x k), where k is the
            number of parameters.
        lnlikefunc (Callable[[np.ndarray], np.ndarray]): Function to
            compute ln(likelihood) on the marginal samples.
        lnpriorfunc (Callable[[np.ndarray], np.ndarray]): Function to
            compute ln(prior density) on the marginal samples.
        nsamples (int, optional): Number of samples to produce.
            Defaults to 1000.
        lnlikeargs (Tuple, optional): Extra arguments passed to the likelihood
            function. Defaults to empty tuple.
        lnpriorargs (Tuple, optional): Extra arguments passed to the lnprior
            function. Defaults to empty tuple.
        densityestimation (str, optional): Method to estimate the marginal
            posterior density ("normal", "kde", or "histogram").
            Defaults to "kde".
        errorestimation (bool, optional): Whether to estimate the error of the
            Perrakis method. Defaults to False.
        **kwargs: Additional arguments passed to estimate_density function.

    Returns:
        Union[float, Tuple[float, float]]: The Perrakis estimate of the
            Bayesian evidence. If `errorestimation` is True, also returns the
            standard error.

    References:
        Perrakis et al. (2014; arXiv:1311.0674)
    """
    if errorestimation:
        initial_sample = marginal_sample
    marginal_sample = make_marginal_samples(marginal_sample, nsamples)
    if not isinstance(marginal_sample, np.ndarray):
        marginal_sample = np.array(marginal_sample)
    number_parameters = marginal_sample.shape[1]
    print("Estimating marginal posterior density for each parameter...")
    marginal_posterior_density = np.zeros(marginal_sample.shape)
    for parameter_index in range(number_parameters):
        x = marginal_sample[:, parameter_index]
        # Estimate density with method "densityestimation".
        marginal_posterior_density[:, parameter_index] = estimate_density(
            x, method=densityestimation, **kwargs
        )
    print("Computing produt of marginal posterior densities for parameters")
    prod_marginal_densities = marginal_posterior_density.prod(axis=1)
    print("Computing lnprior and likelihood in marginal sample")
    log_prior = lnpriorfunc(marginal_sample, *lnpriorargs)
    log_likelihood = lnlikefunc(marginal_sample, *lnlikeargs)
    print("Masking values with zero likelihood")
    cond = log_likelihood != 0
    log_summands = (
        log_likelihood[cond]
        + log_prior[cond]
        - np.log(prod_marginal_densities[cond])  # NOQA
    )
    perr = log_sum(log_summands) - log(len(log_summands))
    # error estimation
    K = 10
    if errorestimation:
        batchSize = initial_sample.shape[0] // K
        meanErr = [
            _perrakis_error(
                initial_sample[0:batchSize, :],
                lnlikefunc,
                lnpriorfunc,
                nsamples=nsamples,
                densityestimation=densityestimation,
            )
        ]
        for i in range(K):
            meanErr.append(
                _perrakis_error(
                    initial_sample[i * batchSize : (i + 1) * batchSize, :],  # NOQA
                    lnlikefunc,
                    lnpriorfunc,
                    nsamples=nsamples,
                    densityestimation=densityestimation,
                )
            )
        stdErr = np.std(meanErr)
        return perr, float(stdErr)
    return perr


def _perrakis_error(
    marginal_samples: np.ndarray,
    lnlikefunc: Callable[[np.ndarray], np.ndarray],
    lnpriorfunc: Callable[[np.ndarray], np.ndarray],
    nsamples: int = 1000,
    densityestimation: str = "histogram",
    errorestimation: bool = False,
) -> Union[float, Tuple[float, float]]:
    """
    Estimate the error of the Perrakis method.

    Args:
        marginal_samples (np.ndarray): A sample from the parameter marginal
            posterior distribution. Dimensions are (n x k), where k is the
            number of parameters.
        lnlikefunc (Callable[[np.ndarray], np.ndarray]): Function to
            compute ln(likelihood) on the marginal samples.
        lnpriorfunc (Callable[[np.ndarray], np.ndarray]): Function to
            compute ln(prior density) on the marginal samples.
        nsamples (int, optional): Number of samples to produce.
            Defaults to 1000.
        densityestimation (str, optional): Method to estimate the marginal
            posterior density ("normal", "kde", or "histogram").
            Defaults to "histogram".
        errorestimation (bool, optional): Whether to estimate the error.
            Defaults to False.

    Returns:
        float: The Perrakis estimate of the Bayesian evidence.
    """
    return compute_perrakis_estimate(
        marginal_samples,
        lnlikefunc,
        lnpriorfunc,
        nsamples=nsamples,
        densityestimation=densityestimation,
        errorestimation=errorestimation,
    )


def estimate_density(
    x: np.ndarray, method: str = "histogram", **kwargs
) -> np.ndarray:  # NOQA
    """
    Estimate probability density based on a sample.

    Args:
        x (np.ndarray): Sample data.
        method (str, optional): Method for density estimation
            ("histogram", "kde", or "normal"). Defaults to "histogram".
        **kwargs: Additional parameters for the density estimation method.

    Returns:
        np.ndarray: Density estimation at the sample points.

    Raises:
        ValueError: If an invalid method is specified.
    """
    nbins = kwargs.pop("nbins", 100)
    if method == "normal":
        return scipy.stats.norm.pdf(x, loc=x.mean(), scale=sqrt(x.var()))
    if method == "kde":
        return scipy.stats.gaussian_kde(x)(x)
    if method == "histogram":
        density, bin_edges = np.histogram(x, nbins, density=True)
        density_indexes = np.searchsorted(bin_edges, x, side="left")
        density_indexes = np.where(
            density_indexes > 0, density_indexes, density_indexes + 1
        )
        return density[density_indexes - 1]
    raise ValueError("Invalid method specified for density estimation.")


def make_marginal_samples(
    joint_samples: np.ndarray, nsamples: Optional[int] = None
) -> np.ndarray:  # NOQA
    """
    Marginal Samples.

    Reshuffles samples from joint distribution to obtain samples from the
    marginal distribution of each parameter.

    Args:
        joint_samples (np.ndarray): Samples from the joint distribution of
            parameters. Dimensions are (n x k).
        nsamples (Optional[int], optional): Number of samples to produce.

    Returns:
        np.ndarray: Samples from the marginal distribution of each parameter.
    """
    if nsamples is None:
        nsamples = len(joint_samples)
    if nsamples > len(joint_samples):
        nsamples = len(joint_samples)
    marginal_samples = joint_samples[-nsamples:, :].copy()
    number_parameters = marginal_samples.shape[-1]
    # Reshuffle joint posterior samples to obtain _marginal_ posterior samples
    for parameter_index in range(number_parameters):

        marginal_samples[:, parameter_index] = np.random.permutation(
            marginal_samples[:, parameter_index]
        )
    return marginal_samples

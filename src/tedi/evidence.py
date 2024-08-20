"""Computation of the evidence using the method of Perrakis et al. (2014)"""

import random
from math import log, sqrt
from typing import Callable, Optional, Tuple, Union

import numpy as np
import scipy.stats

from .utils.evidence import MultivariateGaussian


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
    Computes the Perrakis estimate of the Bayesian evidence.

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
        meanErr = np.mean(meanErr)
        return perr, stdErr
    return perr


def _perrakis_error(
    marginal_samples: np.ndarray,
    lnlikefunc: Callable[[np.ndarray], np.ndarray],
    lnpriorfunc: Callable[[np.ndarray], np.ndarray],
    nsamples: int = 1000,
    densityestimation: str = "histogram",
    errorestimation: bool = False,
) -> float:
    """
    Helper function to estimate the error of the Perrakis method.

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


def make_marginal_samples(
    joint_samples: np.ndarray, nsamples: Optional[int] = None
) -> np.ndarray:  # NOQA
    """
    Reshuffles samples from joint distribution to obtain samples from the
    marginal distribution of each parameter.

    Args:
        joint_samples (np.ndarray): Samples from the joint distribution of
            parameters. Dimensions are (n x k).
        nsamples (Optional[int], optional): Number of samples to produce.

    Returns:
        np.ndarray: Samples from the marginal distribution of each parameter.
    """
    if nsamples > len(joint_samples) or nsamples is None:
        nsamples = len(joint_samples)
    marginal_samples = joint_samples[-nsamples:, :].copy()
    number_parameters = marginal_samples.shape[-1]
    # Reshuffle joint posterior samples to obtain _marginal_ posterior samples
    for parameter_index in range(number_parameters):
        random.shuffle(marginal_samples[:, parameter_index])
    return marginal_samples


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
        random.shuffle(x)
    return a


def compute_harmonicmean(
    lnlike_post: np.ndarray,
    posterior_sample: Optional[np.ndarray] = None,
    lnlikefunc: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    lnlikeargs: Tuple = (),
    **kwargs
) -> float:
    """
    Computes the harmonic mean estimate of the marginal likelihood.

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
    hme = -log_sum(-log_likelihood) + log(len(log_likelihood))
    return hme


def run_hme_mc(
    log_likelihood: np.ndarray, nmc: int, samplesize: int
) -> np.ndarray:  # NOQA
    """
    Runs Monte Carlo simulations to compute the harmonic mean estimate.

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


def compute_cj_estimate(
    posterior_sample: np.ndarray,
    lnlikefunc: Callable[[np.ndarray], np.ndarray],
    lnpriorfunc: Callable[[np.ndarray], np.ndarray],
    param_post: np.ndarray,
    nsamples: int,
    qprob: Optional[
        Union[scipy.stats.rv_continuous, MultivariateGaussian]
    ] = None,  # NOQA
    lnlikeargs: Tuple = (),
    lnpriorargs: Tuple = (),
    lnlike_post: Optional[np.ndarray] = None,
    lnprior_post: Optional[np.ndarray] = None,
) -> float:
    """
    Computes the Chib & Jeliazkov estimate of the Bayesian evidence.

    The estimation is based on a posterior sample with n elements and a sample
    from the proposal distribution used in MCMC (`qprob`) of size `nsamples`.
    If `qprob` is None, it is estimated as a multivariate Gaussian.

    Args:
        posterior_sample (np.ndarray): A sample from the parameter posterior
            distribution. Dimensions are (n x k), where k is the number of
            parameters.
        lnlikefunc (Callable[[np.ndarray], np.ndarray]): Function to compute
            ln(likelihood) on the marginal samples.
        lnpriorfunc (Callable[[np.ndarray, np.ndarray]): Function to compute
            ln(prior density) on the marginal samples.
        param_post (np.ndarray): Posterior parameter sample used to obtain the
            fixed point needed by the algorithm.
        nsamples (int): Size of sample drawn from the proposal distribution.
        qprob (Optional[Union[scipy.stats.rv_continuous, MultivariateGaussian]], optional):  # NOQA
            Proposal distribution function. If None, it will be estimated as a
            multivariate Gaussian. If not None, it must possess the methods
            `pdf` and `rvs`. See `scipy.stats.rv_continuous`.
        lnlikeargs (Tuple, optional): Arguments passed to the lnlikefunc.
        lnpriorargs (Tuple, optional): Arguments passed to the lnpriorfunc.
        lnlike_post (Optional[np.ndarray], optional): Log-likelihood computed
            over a posterior sample. 1-D array of length n.
        lnprior_post (Optional[np.ndarray], optional): Log-prior computed over
            a posterior sample. 1-D array of length n.

    Returns:
        float: Natural logarithm of the estimated Bayesian evidence.

    Raises:
        AttributeError: If `qprob` does not have method 'pdf' or 'rvs'.
        TypeError: If methods 'pdf' or 'rvs' from `qprob` are not callable.

    References:
        Chib & Jeliazkov (2001): Journal of the Am. Stat. Assoc.; Mar 2001; 96, 453
    """
    # Find fixed point on which to estimate posterior ordinate.
    if lnlike_post is not None:
        # Pass values of log(likelihood) in posterior sample.
        arg_fp = [
            lnlike_post,
        ]
    else:
        # Pass function that computes log(likelihood).
        arg_fp = [
            lnlikefunc,
        ]
    if lnlike_post is not None:
        # Pass values of log(prior) in posterior sample.
        arg_fp.append(lnprior_post)
    else:
        # Pass function that computes log(prior).
        arg_fp.append(lnpriorfunc)
    fp, lnpost0 = get_fixed_point(
        posterior_sample,
        param_post,
        lnlikefunc,
        lnpriorfunc,
        lnlikeargs=lnlikeargs,
        lnpriorargs=lnpriorargs,
    )
    # If proposal distribution is not given, define as multivariate Gaussian.
    if qprob is None:
        # Get covariance from posterior sample
        k = np.cov(posterior_sample.T)
        qprob = MultivariateGaussian(fp, k)
    else:
        # Check that qprob has the necessary attributes
        for method in ("pdf", "rvs"):
            try:
                att = getattr(qprob, method)
            except AttributeError:
                raise AttributeError(
                    "qprob does not have method " "'{}'".format(method)
                )
            if not callable(att):
                raise TypeError(
                    "{} method of qprob is not " "callable".format(method)
                )  # NOQA
    q_post = qprob.pdf(posterior_sample)
    if lnlike_post is None:
        lnlike_post = lnlikefunc(posterior_sample, *lnlikeargs)
    if lnprior_post is None:
        lnprior_post = lnpriorfunc(posterior_sample, *lnpriorargs)

    lnalpha_post = metropolis_ratio(lnprior_post + lnlike_post, lnpost0)
    proposal_sample = qprob.rvs(nsamples)
    lnprior_prop = lnpriorfunc(proposal_sample, *lnpriorargs)
    if np.all(lnprior_prop == -np.inf):
        raise ValueError(
            "All samples from proposal density have zero prior"
            "probability. Increase nsample."
        )

    lnlike_prop = np.full_like(lnprior_prop, -np.inf)
    ind = lnprior_prop != -np.inf
    lnlike_prop[ind] = lnlikefunc(proposal_sample[ind, :], *lnlikeargs)
    lnalpha_prop = metropolis_ratio(lnpost0, lnprior_prop + lnlike_prop)
    num = log_sum(lnalpha_post + q_post) - log(len(posterior_sample))
    den = log_sum(lnalpha_prop) - log(len(proposal_sample))
    lnpostord = num - den

    return lnpost0 - lnpostord


def metropolis_ratio(lnpost0: np.ndarray, lnpost1: np.ndarray) -> np.ndarray:
    """
    Computes the Metropolis ratio for two states.

    Args:
        lnpost0 (np.ndarray): Value of ln(likelihood*prior) for initial state.
        lnpost1 (np.ndarray): Value of ln(likelihood*prior) for proposal state.

    Returns:
        np.ndarray: Log of the Metropolis ratio.

    Raises:
        ValueError: If `lnpost0` and `lnpost1` have different lengths.
    """
    if (
        hasattr(lnpost0, "__iter__")
        and hasattr(lnpost1, "__iter__")
        and len(lnpost0) != len(lnpost1)
    ):
        raise ValueError("lnpost0 and lnpost1 have different lenghts.")
    return np.minimum(lnpost1 - lnpost0, 0.0)


def get_fixed_point(
    posterior_samples: np.ndarray,
    param_post: Optional[np.ndarray],
    lnlike: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]],
    lnprior: Union[np.ndarray, Callable[[np.ndarray], np.ndarray]],
    lnlikeargs: Tuple = (),
    lnpriorargs: Tuple = (),
) -> Tuple[np.ndarray, float]:
    """
    Finds the posterior point closest to the model of the lnlike distribution.

    Args:
        posterior_samples (np.ndarray): A sample from the parameters posterior
            distribution. Dimensions must be (n x k), where n is the number of
            elements in the sample and k is the number of parameters.
        param_post (Optional[np.ndarray]): A sample from the marginal posterior
            distribution of the parameter chosen to identify the high-density
            point to use as a fixed point. This is typically one of the columns
            of `posterior_samples`, but could be any 1-D array of size n.
            If None, then a multivariate Gaussian kernel estimate of the joint
            posterior distribution is used.
        lnlike (Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]):
            Function to compute log(likelihood). If an array is given, this is
            simply the log(likelihood) values at the posterior samples, and the
            best value will be chosen from this array.
        lnprior (Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]):
            Function to compute log(prior). If an array is given, this is
            simply the log(prior) values at the posterior samples, and the best
            value will be chosen from this array.
        lnlikeargs (Tuple, optional): Extra arguments passed to lnlike
            functions.
        lnpriorargs (Tuple, optional): Extra arguments passed to lnprior
            functions.

    Returns:
        Tuple[np.ndarray, float]: The fixed point in parameter space and the
            value of log(prior * likelihood) evaluated at this point.

    Raises:
        IndexError: If either `lnlike` or `lnprior` are arrays with length not
            matching the number of posterior samples.
    """
    if param_post is not None:
        # Use median of param_post as fixed point.
        param0 = np.median(param_post)
        # Find argument closest to median.
        ind0 = np.argmin(np.abs(param_post - param0))
        fixed_point = posterior_samples[ind0, :]
        # Compute log(likelihood) at fixed_point
        if hasattr(lnlike, "__iter__"):
            if len(lnlike) != len(posterior_samples):
                raise IndexError(
                    "Number of elements in lnlike array and in "
                    "posterior sample must match."
                )
            lnlike0 = lnlike[ind0]
        else:
            # Evaluate lnlike function at fixed point.
            lnlike0 = lnlike(fixed_point, *lnlikeargs)
        # Compute log(prior) at fixed_point
        if hasattr(lnprior, "__iter__"):
            if len(lnprior) != len(posterior_samples):
                raise IndexError(
                    "Number of elements in lnprior array and in "
                    "posterior sample must match."
                )
            lnprior0 = lnprior[ind0]
        else:
            # Evaluate lnlike function at fixed point.
            lnprior0 = lnprior(fixed_point, *lnpriorargs)
        return fixed_point, lnlike0 + lnprior0
    raise NotImplementedError

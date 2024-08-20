"""Run a MCMC example"""

from multiprocessing import Pool

import corner  # type: ignore
import emcee  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats  # type: ignore

from src.tedi import gaussian_process
from src.tedi.kernels import Exponential, WhiteNoise
from src.tedi.means import Constant

np.random.seed(23011990)

max_n = 1000  # Iterations

# Data
time = np.linspace(1, 10, 50)
y = 10 * np.sin(time)
yerr = np.random.uniform(0, 0.5, time.size)

# Covariance and mean functions
kernel = Exponential(10, 1) + WhiteNoise(0.1)
mean = Constant(0)

gp = gaussian_process.CreateProcess(kernel, mean, time, y, yerr)


# Priors
def priors():
    neta1 = stats.uniform(0, 50)
    neta2 = stats.uniform(0, 10)
    offset = stats.uniform(y.min(), y.max() - y.min())
    jitter = stats.uniform(0, 1)
    return np.array(
        [
            neta1,
            neta2,
            offset,
            jitter,
        ]
    )


def prior_transform():
    return np.array(
        [
            priors()[0].rvs(),
            priors()[1].rvs(),
            priors()[2].rvs(),
            priors()[3].rvs(),
        ]
    )


# log_transform calculates our posterior
def log_transform(theta):
    neta1, neta2, offset, jitter = theta

    logprior = priors()[0].logpdf(neta1)
    logprior += priors()[1].logpdf(neta2)
    logprior += priors()[2].logpdf(offset)
    logprior += priors()[3].logpdf(jitter)
    if np.isinf(logprior):
        return -np.inf

    kernel = Exponential(neta1, neta2) + WhiteNoise(jitter)
    mean = Constant(offset)

    gp = gaussian_process.CreateProcess(kernel, mean, time, y, yerr)
    logpost = logprior + gp.log_marginal_likelihood()
    return logpost


# Sampler definition
ndim = prior_transform().size
nwalkers = 2 * ndim
ncpu = 4

pool = Pool(ncpu)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_transform, pool=pool)

# Initialize the walkers
p0 = [prior_transform() for i in range(nwalkers)]

print("\nRunning MCMC...")
for sample in sampler.sample(p0, iterations=max_n, progress=True):
    samples = sampler.get_chain(
        discard=100,
        flat=True,
        thin=10,
    )
print("Chain shape: {0}".format(samples.shape))


labels = np.array(["eta1", "eta2", "offset", "jitter"])
corner.corner(
    sampler.get_chain(flat=True),
    labels=labels,
    color="k",
    bins=50,
    quantiles=[0.16, 0.5, 0.84],
    smooth=True,
    smooth1d=True,
    show_titles=True,
    plot_density=True,
    plot_contours=True,
    fill_contours=True,
    plot_datapoints=True,
)
plt.show()

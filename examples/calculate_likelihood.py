"""Calculate log-likelihood example."""

import numpy as np

from src.tedi import gaussian_process, kernels, means, student_process

np.random.seed(23011990)

# Data
time = np.linspace(1, 10, 50)
y = 10 * np.sin(time)
yerr = np.random.uniform(0, 0.5, time.size)


# Covariance and mean functions
k = kernels.Exponential(10, 1) + kernels.WhiteNoise(0.1)
m = means.Cosine(1, 10, 0, 0)


# Gaussian processes
gp = gaussian_process.CreateProcess(k, m, time, y, yerr)  # type: ignore

gp_loglike = gp.log_marginal_likelihood()
print(f"GP log-likelihood = {gp_loglike}")
assert gp_loglike == -134.6115049347123, f"{gp_loglike} != -134.6115049347123"

# Trying with a new kernels
gp_loglike = gp.log_marginal_likelihood(
    kernels.Matern32(10, 1) + kernels.WhiteNoise(0.1)  # type: ignore
)  # NOQA
print(f"GP log-likelihood = {gp_loglike}")
assert gp_loglike == -88.75761546043923, f"{gp_loglike} != -88.75761546043923"

gp_loglike = gp.log_marginal_likelihood(
    kernels.PiecewiseSE(1, 2, 10) + kernels.WhiteNoise(0.1)  # type: ignore
)  # NOQA
print(f"GP log-likelihood = {gp_loglike}")
assert gp_loglike == -190.0126293943043, f"{gp_loglike} != -190.0126293943043"


# Student-t processes
tp = student_process.CreateProcess(k, 5, m, time, y, yerr)

tp_loglike = tp.log_likelihood(k, 5)
print(f"TP log-likelihood = {tp_loglike}")
assert tp_loglike == -106.93685196354443, f"{tp_loglike} != -106.93685196354443"  # NOQA

# Trying with a new kernels
tp_loglike = tp.log_likelihood(
    kernels.Matern32(10, 1) + kernels.WhiteNoise(0.1), 5
)  # NOQA
print(f"TP log-likelihood = {tp_loglike}")
assert tp_loglike == -61.26479588055934, f"{tp_loglike} != -61.26479588055934"

tp_loglike = tp.log_likelihood(
    kernels.PiecewiseSE(1, 2, 10) + kernels.WhiteNoise(0.1), 5
)
print(f"TP log-likelihood = {tp_loglike}")
assert tp_loglike == -101.83522277011839, f"{tp_loglike} != -101.83522277011839"  # NOQA

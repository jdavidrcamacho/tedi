"""Calculate log-likelihood example"""

import numpy as np

from src.tedi import gaussian_process, student_process, kernels, means

np.random.seed(23011990)

# Data
time = np.linspace(1, 10, 50)
y = 10 * np.sin(time)
yerr = np.random.uniform(0, 0.5, time.size)


# Covariance and mean functions
kernel = kernels.Exponential(10, 1) + kernels.WhiteNoise(0.1)
mean = means.Cosine(1, 10, 0, 0)


# Gaussian processes
gpOBJ = gaussian_process.CreateProcess(kernel, mean, time, y, yerr)

gp_loglike = gpOBJ.log_likelihood()
print(f"GP log-likelihood = {gp_loglike}")
assert gp_loglike == -134.6115049347123, f"{gp_loglike} != -134.6115049347123"

# Trying with a new kernels
gp_loglike = gpOBJ.log_likelihood(
    kernels.Matern32(10, 1) + kernels.WhiteNoise(0.1)
)  # NOQA
print(f"GP log-likelihood = {gp_loglike}")
assert gp_loglike == -88.75761546043923, f"{gp_loglike} != -88.75761546043923"

gp_loglike = gpOBJ.log_likelihood(
    kernels.PiecewiseSE(1, 2, 10) + kernels.WhiteNoise(0.1)
)
print(f"GP log-likelihood = {gp_loglike}")
assert gp_loglike == -190.0126293943043, f"{gp_loglike} != -190.0126293943043"


# Student-t processes
tpOBJ = student_process.CreateProcess(kernel, 5, mean, time, y, yerr)

tp_loglike = tpOBJ.log_likelihood(kernel, 5)
print(f"TP log-likelihood = {tp_loglike}")
assert tp_loglike == -106.93685196354443, f"{tp_loglike} != -106.93685196354443"  # NOQA

# Trying with a new kernels
tp_loglike = tpOBJ.log_likelihood(
    kernels.Matern32(10, 1) + kernels.WhiteNoise(0.1), 5
)  # NOQA
print(f"TP log-likelihood = {tp_loglike}")
assert tp_loglike == -61.26479588055934, f"{tp_loglike} != -61.26479588055934"

tp_loglike = tpOBJ.log_likelihood(
    kernels.PiecewiseSE(1, 2, 10) + kernels.WhiteNoise(0.1), 5
)
print(f"TP log-likelihood = {tp_loglike}")
assert tp_loglike == -101.83522277011839, f"{tp_loglike} != -101.83522277011839"  # NOQA

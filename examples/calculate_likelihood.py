"""Calculate log-likelihood example"""

import numpy as np

from src.tedi import kernels, means, process

np.random.seed(23011990)

# Data
time = np.linspace(1, 10, 50)
y = 10 * np.sin(time)
yerr = np.random.uniform(0, 0.5, time.size)


kernel = kernels.Exponential(10, 1) + kernels.WhiteNoise(0.1)
mean = means.Cosine(1, 10, 0, 0)
print(kernel)

# Gaussian processes
gpOBJ = process.GP(kernel, mean, time, y, yerr)
gp_loglike = gpOBJ.log_likelihood(kernel)
print(f"GP log-likelihood = {gp_loglike}")
assert gp_loglike == -134.6115049347123

# Student-t processes
tpOBJ = process.TP(kernel, 5, mean, time, y, yerr)
tp_loglike = tpOBJ.log_likelihood(kernel, 5)
print(f"TP log-likelihood = {tp_loglike}")
assert tp_loglike == -106.93685196354443


kernel = kernels.Matern32(10, 1) + kernels.WhiteNoise(0.1)
gpOBJ = process.GP(kernel, mean, time, y, yerr)
gp_loglike = gpOBJ.log_likelihood(kernel)
print(f"GP log-likelihood = {gp_loglike}")
assert gp_loglike == -88.75761546043923

kernel = kernels.PiecewiseSE(1, 2, 10) + kernels.WhiteNoise(0.1)
gpOBJ = process.GP(kernel, mean, time, y, yerr)
gp_loglike = gpOBJ.log_likelihood(kernel)
print(f"GP log-likelihood = {gp_loglike}")
assert gp_loglike == -190.0126293943043

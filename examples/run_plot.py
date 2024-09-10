"""Calculate log-likelihood and plot prediction example"""

import matplotlib.pyplot as plt
import numpy as np

from src.tedi import gaussian_process, kernels, means

np.random.seed(23011990)

# Data
time = np.linspace(1, 10, 50)
y = 10 * np.sin(time)
yerr = np.random.uniform(0, 0.5, time.size)

# Covariance and mean functions
k1 = kernels.Exponential(10, 1) + kernels.WhiteNoise(0.1)
m1 = means.Cosine(1, 10, 0, 0)

# Gaussian processes
gp1 = gaussian_process.CreateProcess(k1, m1, time, y, yerr)

gp_loglike = gp1.log_marginal_likelihood()
print(f"GP log-likelihood = {gp_loglike}")
assert gp_loglike == -134.6115049347123, f"{gp_loglike} != -134.6115049347123"

# Prediction
tstar = np.linspace(0, 11, 1000)
mean, std, _, t = gp1.prediction(time=tstar)
std_min, std_max = mean - std, mean + std

plt.figure()
plt.errorbar(time, y, yerr, fmt=".b")
plt.plot(t, mean, "-r", alpha=0.75, label="Exponential GP")
plt.fill_between(tstar, std_max.T, std_min.T, color="red", alpha=0.25)

# New covariance and mean functions
k2 = kernels.HarmonicPeriodic(1, 1, 10, 1) + kernels.WhiteNoise(0.1)
m2 = means.Constant(0)

gp2 = gaussian_process.CreateProcess(k2, m2, time, y, yerr)
gp_loglike = gp2.log_marginal_likelihood()
print(f"GP log-likelihood = {gp_loglike}")
assert gp_loglike == -1057.0059050526309, f"{gp_loglike} != -1057.0059050526309"  # NOQA

mean, std, _, t = gp2.prediction(time=tstar)
std_min, std_max = mean - std, mean + std
plt.plot(t, mean, "-g", alpha=0.75, label="Harmonic-periodic GP")
plt.fill_between(tstar, std_max.T, std_min.T, color="green", alpha=0.25)

# plt.show()

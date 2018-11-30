#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 08:44:28 2018

@author: joaocamacho
"""
import numpy as np
import emcee
import matplotlib.pylab as plt
plt.close('all')

from scipy import stats
from tedi import process, kernels, means



### Data
time, rv, rverr = np.loadtxt("barnardsStar_carmenes.dat", skiprows=1,
                             usecols=(0, 1, 2), unpack=True)


#because we need to define a "initial" kernel and mean
kernel = kernels.QuasiPeriodic(1, 1, 1, 1, 1)
mean = means.Keplerian(P = 1, K = 1, e = 0.5, w = np.pi, T0 = 1000) 

GPobj = process.GP(kernel,mean, time, rv, rverr)


### Preparing our MCMC
burns, runs= 10000, 10000

#defining our priors
def logprob(p):
    global kernel
    if any([p[0] < np.log(0.1), p[0] > np.log(50), 
            p[1] < np.log(1), p[1] > np.log(100),
            p[2] < np.log(100), p[2] > np.log(200),
            p[3] < np.log(0.1), p[3] > np.log(10),

            p[5] < np.log(100), p[5] > np.log(300),
            p[6] < np.log(0.1), p[6] > np.log(10), 
            p[7] < -10, p[7] > np.log(0.99),
            p[8] < -10, p[8] > np.log(2*np.pi),
            p[9] < -5, p[9] > 5
            ]):
        return -np.inf
    logprior = 0.0

    p = np.exp(p)
    # Update the kernel and compute the log marginal likelihood.
    new_kernel = kernels.QuasiPeriodic(p[0], p[1], p[2], p[3], p[4])

    new_mean = means.Keplerian(p[5], p[6], p[7], p[8], p[9]) 
    new_likelihood = GPobj.log_likelihood(new_kernel, new_mean)
    return logprior + new_likelihood

amp_prior = stats.uniform(0.1, 50 - 0.1)            #amplitude
eta2_prior= stats.uniform(1, 100 - 1)              # le
eta3_prior= stats.uniform(100, 200 - 100)     # period
eta4_prior= stats.uniform(0.1, 10 - 0.1)                              # lp
wn_prior= stats.halfcauchy(0, 1)             # White noise

p_prior=stats.uniform(100, 300 - 100)            # period
k_prior=stats.uniform(0.1, 10 - 0.1)                           # semi-amplitude
e_prior=stats.uniform(np.exp(-10), 0.99 - np.exp(-10))        # eccentricity
w_prior=stats.uniform(np.exp(-10), 2*np.pi - np.exp(-10))     # omega
t0_prior=stats.uniform(np.exp(-5), np.exp(5) - np.exp(-5))       # T0

def from_prior():
    return np.array([amp_prior.rvs(), eta2_prior.rvs(), eta3_prior.rvs(), 
                     eta4_prior.rvs(), wn_prior.rvs(),
                     p_prior.rvs(), k_prior.rvs(), e_prior.rvs(), w_prior.rvs(),
                     t0_prior.rvs()])

#Setingt up the sampler
nwalkers, ndim = 2*10, 10
sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, threads= 4)

#Initialize the walkers
p0=[np.log(from_prior()) for i in range(nwalkers)]

print("Running burn-in")
p0, _, _ = sampler.run_mcmc(p0, burns)
print("Running production chain")
sampler.run_mcmc(p0, runs);

from matplotlib.ticker import MaxNLocator
fig, axes = plt.subplots(5, 1, sharex=True, figsize=(8, 9))
axes[0].plot(np.exp(sampler.chain[:, :, 0]).T, color="k", alpha=0.4)
axes[0].yaxis.set_major_locator(MaxNLocator(5))
axes[0].set_ylabel("$eta1$")
axes[1].plot(np.exp(sampler.chain[:, :, 1]).T, color="k", alpha=0.4)
axes[1].yaxis.set_major_locator(MaxNLocator(5))
axes[1].set_ylabel("$eta4$")
axes[2].plot(np.exp(sampler.chain[:, :, 2]).T, color="k", alpha=0.4)
axes[2].yaxis.set_major_locator(MaxNLocator(5))
axes[2].set_ylabel("$eta2$")
axes[3].plot(np.exp(sampler.chain[:, :, 3]).T, color="k", alpha=0.4)
axes[3].yaxis.set_major_locator(MaxNLocator(5))
axes[3].set_ylabel("$eta3$")
axes[4].plot(np.exp(sampler.chain[:, :, 4]).T, color="k", alpha=0.4)
axes[4].yaxis.set_major_locator(MaxNLocator(5))
axes[4].set_ylabel("$wn$")
axes[4].set_xlabel("step number")
fig.tight_layout(h_pad=0.0)

fig, axes = plt.subplots(5, 1, sharex=True, figsize=(8, 9))
axes[0].plot(np.exp(sampler.chain[:, :, 5]).T, color="k", alpha=0.4)
axes[0].yaxis.set_major_locator(MaxNLocator(5))
axes[0].set_ylabel("$P$")
axes[1].plot(np.exp(sampler.chain[:, :, 6]).T, color="k", alpha=0.4)
axes[1].yaxis.set_major_locator(MaxNLocator(5))
axes[1].set_ylabel("$k$")
axes[2].plot(np.exp(sampler.chain[:, :, 7]).T, color="k", alpha=0.4)
axes[2].yaxis.set_major_locator(MaxNLocator(5))
axes[2].set_ylabel("$ecc3$")
axes[3].plot(np.exp(sampler.chain[:, :, 8]).T, color="k", alpha=0.4)
axes[3].yaxis.set_major_locator(MaxNLocator(5))
axes[3].set_ylabel("$omega$")
axes[4].plot(np.exp(sampler.chain[:, :, 9]).T, color="k", alpha=0.4)
axes[4].yaxis.set_major_locator(MaxNLocator(5))
axes[4].set_ylabel("$T0$")
axes[4].set_xlabel("step number")
fig.tight_layout(h_pad=0.0)

burnin = burns
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))


##### MCMC analysis #####
burnin = burns
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

samples[:, :] = np.exp(samples[:, :])
e1_mcmc,e4_mcmc,e2_mcmc,e3_mcmc,wn_mcmc, p_mcmc,k_mcmc,e_mcmc,w_mcmc,t0_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))

print('eta1 = {0[0]} +{0[1]} -{0[2]}'.format(e1_mcmc))
print('eta4 = {0[0]} +{0[1]} -{0[2]}'.format(e4_mcmc))
print('eta2 = {0[0]} +{0[1]} -{0[2]}'.format(e2_mcmc))
print('eta3 = {0[0]} +{0[1]} -{0[2]}'.format(e3_mcmc))
print('white noise = {0[0]} +{0[1]} -{0[2]}'.format(wn_mcmc))
print()
print('P = {0[0]} +{0[1]} -{0[2]}'.format(p_mcmc))
print('K = {0[0]} +{0[1]} -{0[2]}'.format(k_mcmc))
print('ecc = {0[0]} +{0[1]} -{0[2]}'.format(e_mcmc))
print('omega = {0[0]} +{0[1]} -{0[2]}'.format(w_mcmc))
print('T0 = {0[0]} +{0[1]} -{0[2]}'.format(t0_mcmc))

plt.figure()
for i in range(sampler.lnprobability.shape[0]):
    plt.plot(sampler.lnprobability[i, :])

import corner
##### likelihood calculations #####
likes=[]
labels = ['eta1', 'eta4', 'eta2', 'eta3', 'wn']
corner.corner(samples[:,0:5], labels=labels, show_titles=True,
                         plot_contours=False, plot_datapoints=True, plot_density=False,
                         hexbin_kwargs={'cmap':plt.get_cmap('afmhot_r'), 'bins':'log'},
                         hist_kwargs={'normed':True}, data_kwargs={'alpha':1})

labels1= ['period', 'K', 'ecc', 'w', 'T0']
corner.corner(samples[:,5:], labels=labels1, show_titles=True,
                         plot_contours=False, plot_datapoints=True, plot_density=False,
                         hexbin_kwargs={'cmap':plt.get_cmap('afmhot_r'), 'bins':'log'},
                         hist_kwargs={'normed':True}, data_kwargs={'alpha':1},)


likes=[]
for i in range(samples[:,0].size):
    new_kernel = kernels.QuasiPeriodic(samples[i,0], samples[i,1], samples[i,2], samples[i,3], samples[i,4])
    new_mean = means.Keplerian(samples[i,5], samples[i,6], samples[i,7], samples[i,8], samples[i,9]) 
    likes.append(GPobj.log_likelihood(new_kernel, new_mean))
plt.figure()
plt.hist(likes, bins = 15, label='likelihood')

datafinal = np.vstack([samples.T,np.array(likes).T]).T
np.save('samples_carmenes.npy', datafinal)

##### checking the likelihood that matters to us #####
samples = datafinal
values = np.where(samples[:,-1] > -500)
#values = np.where(samples[:,-1] < -300)
likelihoods = samples[values,-1].T
plt.figure()
plt.hist(likelihoods)
plt.title("Likelihoood")
plt.xlabel("Value")
plt.ylabel("Samples")

samples = samples[values,:]
samples = samples.reshape(-1, 11)

e1_mcmc,e4_mcmc,e2_mcmc,e3_mcmc,wn_mcmc, p_mcmc,k_mcmc,e_mcmc,w_mcmc,t0_mcmc, likes = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))

print('FINAL SOLUTION')
print('eta1 = {0[0]} +{0[1]} -{0[2]}'.format(e1_mcmc))
print('eta4 = {0[0]} +{0[1]} -{0[2]}'.format(e4_mcmc))
print('eta2 = {0[0]} +{0[1]} -{0[2]}'.format(e2_mcmc))
print('eta3 = {0[0]} +{0[1]} -{0[2]}'.format(e3_mcmc))
print('white noise = {0[0]} +{0[1]} -{0[2]}'.format(wn_mcmc))
print()
print('P = {0[0]} +{0[1]} -{0[2]}'.format(p_mcmc))
print('K = {0[0]} +{0[1]} -{0[2]}'.format(k_mcmc))
print('ecc = {0[0]} +{0[1]} -{0[2]}'.format(e_mcmc))
print('omega = {0[0]} +{0[1]} -{0[2]}'.format(w_mcmc))
print('T0 = {0[0]} +{0[1]} -{0[2]}'.format(t0_mcmc))


##### Keplerian #####
from tedi import astro

#phase folding the data
#phase, folded_rv, folder_err = astro.phase_folding(time, rv, rverr, p_mcmc[0])

#making the keplerian
times, points = astro.keplerian(P=p_mcmc[0], K=k_mcmc[0], e=e_mcmc[0], 
                            w=w_mcmc[0], T=t0_mcmc[0], t=time)
#phase_2, folded_kep, folder_kep2 = astro.phase_folding(times, points, np.zeros_like(points),
#                                                       period= p_mcmc[0])

plt.figure()
plt.errorbar(time, rv, rverr, fmt='.')
plt.plot(times, points, 'k-')
plt.xlabel('time')
plt.ylabel('RV (m/s)')
plt.show()
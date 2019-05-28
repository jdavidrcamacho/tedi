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
time, rv, rverr = np.loadtxt("corot7.txt", skiprows=112-3,
                             usecols=(0, 1, 2), unpack=True)

#removinhg 'planets'
from tedi import astro
_, p1 = astro.keplerian(P = 0.85359165, K = 3.42, e = 0.12, w = 105*np.pi/180, 
                        T = 4398.21, t = time)
_, p2 = astro.keplerian(P = 3.70, K = 6.01, e = 0.12, w = 140*np.pi/180, 
                        T = 5953.3, t=time)
rv = rv - p1 -p2



#because we need to define a "initial" kernel and mean
kernel = kernels.QuasiPeriodic(1, 1, 1, 1, 1)
mean = None

GPobj = process.GP(kernel, mean, time, rv, rverr)


### Preparing our MCMC
burns, runs= 50, 50

#defining our priors
def logprob(p):
    global kernel
    if any([p[0] < np.log(0.1), p[0] > np.log(50), 
            p[1] < np.log(1), p[1] > np.log(100),
            p[2] < np.log(10), p[2] > np.log(40),
            p[3] < np.log(0.1), p[3] > np.log(10),
            

            ]):
        return -np.inf
    logprior = 0.0

    p = np.exp(p)
    # Update the kernel and compute the log marginal likelihood.
    new_kernel = kernels.QuasiPeriodic(p[0], p[1], p[2], p[3], p[4])
    new_mean = mean
    new_likelihood = GPobj.log_likelihood(new_kernel, new_mean)

    return logprior + new_likelihood

amp_prior = stats.uniform(0.1, 50 - 0.1)                                        #amplitude
eta2_prior= stats.uniform(1, 100 - 1)                                           #le
eta3_prior= stats.uniform(10, 40 - 10)                                          #period
eta4_prior= stats.uniform(0.1, 10 - 0.1)                                        #lp
wn_prior= stats.halfcauchy(0,1)                                                 #white noise

p_prior=stats.uniform(np.exp(-10), 5 - np.exp(-10))                            #period
k_prior=stats.uniform(0.1, 100 - 0.1)                                           #semi-amplitude
e_prior=stats.uniform(np.exp(-10), 0.99 - np.exp(-10))                          #eccentricity
w_prior=stats.uniform(np.exp(-10), 2*np.pi - np.exp(-10))                       #omega
t0_prior=stats.uniform(np.exp(-5), np.exp(5) - np.exp(-5))                      #T0

def from_prior():
    return np.array([amp_prior.rvs(), eta2_prior.rvs(), eta3_prior.rvs(), eta4_prior.rvs(), wn_prior.rvs()])

#Setingt up the sampler
nwalkers, ndim = 2*5, 5
sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, threads= 4)

#Initialize the walkers
p0=[np.log(from_prior()) for i in range(nwalkers)]

print("Running burn-in")
p0, _, _ = sampler.run_mcmc(p0, burns)
print("Running production chain")
sampler.run_mcmc(p0, runs);


##### MCMC analysis #####
burnin = burns
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
samples = np.exp(samples)

#median and quantiles
amp1,l1,p1,l2,wn1 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],axis=0)))

#printing results
print()
print('Amplitude = {0[0]} +{0[1]} -{0[2]}'.format(amp1))
print('Aperiodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l1))
print('Kernel period = {0[0]} +{0[1]} -{0[2]}'.format(p1))
print('Periodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l2))
print('Kernel wn = {0[0]} +{0[1]} -{0[2]}'.format(wn1))
print()


plt.figure()
for i in range(sampler.lnprobability.shape[0]):
    plt.plot(sampler.lnprobability[i, :])


##### likelihood calculations #####
likes=[]
for i in range(samples[:,0].size):
    new_kernel = kernels.QuasiPeriodic(samples[i,0], samples[i,1], samples[i,2], 
                                       samples[i,3], samples[i,4])
    new_mean = mean
    likes.append(GPobj.log_likelihood(new_kernel, new_mean))

#plt.figure()
#plt.hist(likes, bins = 15, label='likelihood')

datafinal = np.vstack([samples.T,np.array(likes).T]).T
np.save('samples_corot7_tediGP_2.npy', datafinal)


##### checking the likelihood that matters to us #####
samples = datafinal
values = np.where(samples[:,-1] > -220)
#values = np.where(samples[:,-1] < -300)
likelihoods = samples[values,-1].T
plt.figure()
plt.hist(likelihoods)
plt.title("Likelihoood")
plt.xlabel("Value")
plt.ylabel("Samples")

samples = samples[values,:]
samples = samples.reshape(-1, 6)

amp1,l1,p1,l2,wn1, likes = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],axis=0)))

#printing results
print('FINAL SOLUTION')
print()
print('Amplitude = {0[0]} +{0[1]} -{0[2]}'.format(amp1))
print('Aperiodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l1))
print('Kernel period = {0[0]} +{0[1]} -{0[2]}'.format(p1))
print('Periodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l2))
print('Kernel wn = {0[0]} +{0[1]} -{0[2]}'.format(wn1))
print()

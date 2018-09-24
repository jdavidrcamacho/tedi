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

from matplotlib.ticker import MaxNLocator
from scipy import stats
from tedi import process, kernels, means



### Data
time, rv, rverr = np.loadtxt("corot7.txt", skiprows=2,
                             usecols=(0, 1, 2), unpack=True)


#because we need to define a "initial" kernel and mean
kernel = kernels.QuasiPeriodic(1, 1, 1, 1, 1)
mean = means.Keplerian(1, 1, 0.1, 0 ,0) + means.Keplerian(1, 1, 0.1, 0 ,0)

GPobj = process.GP(kernel, mean, time, rv, rverr)


### Preparing our MCMC
burns, runs= 25000, 25000


#/* GP parameters */
#Uniform *log_eta1_prior = new Uniform(-5, 5);
#Uniform *log_eta2_prior = new Uniform(0, 5);
#Uniform *eta3_prior = new Uniform(10., 40.);
#Uniform *log_eta4_prior = new Uniform(-5, 0);

#LogUniform *Pprior = new LogUniform(1.0, 1E5); // days
#ModifiedLogUniform *Kprior = new ModifiedLogUniform(1.0, 2E3); // m/s
#
#Uniform *eprior = new Uniform(0., 1.);
#Uniform *phiprior = new Uniform(0.0, 2*M_PI);
#Uniform *wprior = new Uniform(0.0, 2*M_PI);


#defining our priors
def logprob(p):
    global kernel
    if any([p[0] < np.log(0.1), p[0] > np.log(50), 
            p[1] < np.log(1), p[1] > np.log(100),
            p[2] < 10, p[2] > 40,
            p[3] < np.log(0.1), p[3] > np.log(10),
            
            p[5] < -10, p[5] > np.log(50),
            p[6] < np.log(0.1), p[6] > np.log(100),
            p[7] < -10, p[7] > np.log(0.99), 
            p[8] < -10, p[8] > np.log(2*np.pi),
            p[9] < -10, p[9] > 10,
            
            p[10] < -10, p[10] > np.log(50),
            p[11] < np.log(0.1), p[11] > np.log(100),
            p[12] < -10, p[12] > np.log(0.99), 
            p[13] < -10, p[13] > np.log(2*np.pi),
            p[14] < -5, p[14] > 5
            ]):
        return -np.inf
    logprior = 0.0

    p = np.exp(p)
    # Update the kernel and compute the log marginal likelihood.
    new_kernel = kernels.QuasiPeriodic(p[0], p[1], p[2], p[3], p[4])
    new_mean = means.Keplerian(p[5], p[6], p[7], p[8] ,p[9]) \
                + means.Keplerian(p[10], p[11], p[12], p[13] ,p[14])
    new_likelihood = GPobj.log_likelihood(new_kernel, new_mean)
    print(new_likelihood)
    return logprior + new_likelihood

amp_prior = stats.uniform(0.1, 50 - 0.1)            #amplitude
eta2_prior= stats.uniform(1, 100 - 1)              # le
eta3_prior= stats.uniform(np.exp(10), np.exp(40) - np.exp(10))     # period
eta4_prior= stats.uniform(0.1, 10 - 0.1)                              # lp
wn_prior= stats.halfcauchy(0,1)             # White noise

p_prior=stats.uniform(np.exp(-10), 50 - np.exp(-10))            # period
k_prior=stats.uniform(0.1, 100 - 0.1)                           # semi-amplitude
e_prior=stats.uniform(np.exp(-10), 0.99 - np.exp(-10))        # eccentricity
w_prior=stats.uniform(np.exp(-10), 2*np.pi - np.exp(-10))     # omega
t0_prior=stats.uniform(np.exp(-5), np.exp(5) - np.exp(-5))       # T0

def from_prior():
    wn = wn_prior.rvs()
    #to truncate the wn 
    while wn > 1:
        wn = wn_prior.rvs()
    return np.array([amp_prior.rvs(), eta2_prior.rvs(), eta3_prior.rvs(), eta4_prior.rvs(), wn,
                     p_prior.rvs(), k_prior.rvs(), e_prior.rvs(), w_prior.rvs(), t0_prior.rvs(),
                     p_prior.rvs(), k_prior.rvs(), e_prior.rvs(), w_prior.rvs(), t0_prior.rvs()])

#Setingt up the sampler
nwalkers, ndim = 2*15, 15
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
amp1,l1,p1,l2,wn1, pl1,k1,e1,w1,t01, pl2,k2,e2,w2,t02 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],axis=0)))

#printing results
print()
print('Amplitude = {0[0]} +{0[1]} -{0[2]}'.format(amp1))
print('Aperiodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l1))
print('Kernel period = {0[0]} +{0[1]} -{0[2]}'.format(p1))
print('Periodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l2))
print('Kernel wn = {0[0]} +{0[1]} -{0[2]}'.format(wn1))
print()
print('P = {0[0]} +{0[1]} -{0[2]}'.format(pl1))
print('K = {0[0]} +{0[1]} -{0[2]}'.format(k1))
print('ecc = {0[0]} +{0[1]} -{0[2]}'.format(e1))
print('omega = {0[0]} +{0[1]} -{0[2]}'.format(w1))
print('T0 = {0[0]} +{0[1]} -{0[2]}'.format(t01))
print()
print('P = {0[0]} +{0[1]} -{0[2]}'.format(pl2))
print('K = {0[0]} +{0[1]} -{0[2]}'.format(k2))
print('ecc = {0[0]} +{0[1]} -{0[2]}'.format(e2))
print('omega = {0[0]} +{0[1]} -{0[2]}'.format(w2))
print('T0 = {0[0]} +{0[1]} -{0[2]}'.format(t02))
print()

plt.figure()
for i in range(sampler.lnprobability.shape[0]):
    plt.plot(sampler.lnprobability[i, :])


##### likelihood calculations #####
likes=[]
for i in range(samples[:,0].size):
    new_kernel = kernels.QuasiPeriodic(samples[i,0], samples[i,1], samples[i,2], 
                                       samples[i,3], samples[i,4])
    new_mean = means.Keplerian(samples[i,5], samples[i,6], samples[i,7], 
                               samples[i,8] ,samples[i,9]) \
                + means.Keplerian(samples[i,10], samples[i,11], samples[i,12], 
                                  samples[i,13] ,samples[i,14])
    likes.append(GPobj.log_likelihood(new_kernel, new_mean))

#plt.figure()
#plt.hist(likes, bins = 15, label='likelihood')

datafinal = np.vstack([samples.T,np.array(likes).T]).T
np.save('samples_corot7_gp.npy', datafinal)


##### checking the likelihood that matters to us #####
samples = datafinal
values = np.where(samples[:,-1] > -5000)
#values = np.where(samples[:,-1] < -300)
likelihoods = samples[values,-1].T
plt.figure()
plt.hist(likelihoods)
plt.title("Likelihoood")
plt.xlabel("Value")
plt.ylabel("Samples")

samples = samples[values,:]
samples = samples.reshape(-1, 16)

amp1,l1,p1,l2,wn1, pl1,k1,e1,w1,t01, pl2,k2,e2,w2,t02, likes = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],axis=0)))

#printing results
print()
print('Amplitude = {0[0]} +{0[1]} -{0[2]}'.format(amp1))
print('Aperiodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l1))
print('Kernel period = {0[0]} +{0[1]} -{0[2]}'.format(p1))
print('Periodic length scale = {0[0]} +{0[1]} -{0[2]}'.format(l2))
print('Kernel wn = {0[0]} +{0[1]} -{0[2]}'.format(wn1))
print()
print('P = {0[0]} +{0[1]} -{0[2]}'.format(pl1))
print('K = {0[0]} +{0[1]} -{0[2]}'.format(k1))
print('ecc = {0[0]} +{0[1]} -{0[2]}'.format(e1))
print('omega = {0[0]} +{0[1]} -{0[2]}'.format(w1))
print('T0 = {0[0]} +{0[1]} -{0[2]}'.format(t01))
print()
print('P = {0[0]} +{0[1]} -{0[2]}'.format(pl2))
print('K = {0[0]} +{0[1]} -{0[2]}'.format(k2))
print('ecc = {0[0]} +{0[1]} -{0[2]}'.format(e2))
print('omega = {0[0]} +{0[1]} -{0[2]}'.format(w2))
print('T0 = {0[0]} +{0[1]} -{0[2]}'.format(t02))
print()
print('likelihood = {0[0]} +{0[1]} -{0[2]}'.format(likes))
print()

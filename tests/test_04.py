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
time, rv, rverr = np.loadtxt("corot7.txt", skiprows=2,
                             usecols=(0, 1, 2), unpack=True)


#because we need to define a "initial" kernel and mean
kernel = kernels.QuasiPeriodic(1, 1, 1, 1, 1)
degrees = 5
mean = means.Keplerian(P = 0.85359165, K = 3.42, e = 0.12, w = 105*np.pi/180, T0 = 4398.21) \
                    + means.Keplerian(P = 3.70, K = 6.01, e = 0.12, w = 140*np.pi/180, T0 = 5953.3)

TPobj = process.TP(kernel, degrees, mean, time, rv, rverr)


### Preparing our MCMC
burns, runs= 50000, 50000


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
            p[2] < np.log(10), p[2] > np.log(40),
            p[3] < np.log(0.1), p[3] > np.log(10),

            p[5] < np.log(3), p[5] >np.log(1000)
            ]):
        return -np.inf
    logprior = 0.0

    p = np.exp(p)
    # Update the kernel and compute the log marginal likelihood.
    new_kernel = kernels.QuasiPeriodic(p[0], p[1], p[2], p[3], p[4])
    new_degree = p[5]
    new_mean = mean
    new_likelihood = TPobj.log_likelihood(new_kernel, new_degree, new_mean)
    return logprior + new_likelihood

amp_prior = stats.uniform(0.1, 50 - 0.1)            #amplitude
eta2_prior= stats.uniform(1, 100 - 1)              # le
eta3_prior= stats.uniform(10, 40 - 10)     # period
eta4_prior= stats.uniform(0.1, 10 - 0.1)                              # lp
wn_prior= stats.halfcauchy(0, 1)             # White noise

degrees_prior = stats.randint(3, 1000)      #degree > 2 and integer

p_prior=stats.uniform(np.exp(-10), 50 - np.exp(-10))            # period
k_prior=stats.uniform(0.1, 100 - 0.1)                           # semi-amplitude
e_prior=stats.uniform(np.exp(-10), 0.99 - np.exp(-10))        # eccentricity
w_prior=stats.uniform(np.exp(-10), 2*np.pi - np.exp(-10))     # omega
t0_prior=stats.uniform(np.exp(-5), np.exp(5) - np.exp(-5))       # T0

def from_prior():
    return np.array([amp_prior.rvs(), eta2_prior.rvs(), eta3_prior.rvs(), 
                     eta4_prior.rvs(), wn_prior.rvs(), degrees_prior.rvs()])

#Setingt up the sampler
nwalkers, ndim = 2*6, 6
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
amp1,l1,p1,l2,wn1, deg1,t02 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
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
    new_degree = samples[i,5]
    new_mean = mean
    likes.append(TPobj.log_likelihood(new_kernel, new_degree, new_mean))

#plt.figure()
#plt.hist(likes, bins = 15, label='likelihood')

datafinal = np.vstack([samples.T,np.array(likes).T]).T
np.save('samples_corot7_tediTP.npy', datafinal)


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
samples = samples.reshape(-1, 17)

amp1,l1,p1,l2,wn1, deg1, likes = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
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
print('degrees = {0[0]} +{0[1]} -{0[2]}'.format(deg1))
print()
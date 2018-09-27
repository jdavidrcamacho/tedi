#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 08:44:28 2018

@author: joaocamacho
"""
import numpy as np
import matplotlib.pylab as plt
plt.close('all')
from tedi import process, kernels

#Data
time = np.linspace(1,10,50)
y = 10*np.sin(time)
yerr = np.random.uniform(0,0.5,time.size)


kernel = kernels.Exponential(10,1,0.1)
mean = None
print(kernel)

#Gaussian processes
gpOBJ = process.GP(kernel,mean,time,y,yerr)
print(gpOBJ.log_likelihood(kernel))
print(gpOBJ.log_likelihood_gradient(kernel))

new_hyperparms = np.array([20.0, 3.0 ,1.0])
kernel1 = gpOBJ.new_kernel(kernel, new_hyperparms)
print(kernel1)

mean_gp, std_gp, cov_gp = gpOBJ.prediction(time = np.linspace(1,10,1000))

#Student-t processes
tpOBJ = process.TP(kernel, 5, mean,time,y,yerr)
print(tpOBJ.log_likelihood(kernel, 5))
print(tpOBJ.log_likelihood_gradient(kernel, 5))
mean_tp, std_tp, cov_tp = tpOBJ.prediction(time = np.linspace(1,10,1000))


#plots
f, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.set_title(' ')
ax1.fill_between(np.linspace(1,10,1000), mean_gp+std_gp, mean_gp-std_gp, 
                 color="grey", alpha=0.5)
ax1.plot(np.linspace(1,10,1000), mean_gp, "k--", alpha=1, lw=1.5)
ax1.plot(time, y,"b.")
ax1.set_ylabel("GPs")

ax2.fill_between(np.linspace(1,10,1000), mean_tp+std_tp, mean_tp-std_tp, 
                 color="grey", alpha=0.5)
ax2.plot(np.linspace(1,10,1000), mean_tp, "k--", alpha=1, lw=1.5)
ax2.plot(time, y,"b.")
ax2.set_ylabel("TPs")
plt.show()

#samples
plt.figure()
for i in range(30):
    plt.plot(time, y, 'k-')
    plt.plot(time,  gpOBJ.sample(kernel, time), 'b:')
    plt.plot(time,  tpOBJ.sample(kernel, 5, time), 'r-')
    i +=1
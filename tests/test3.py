#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 08:44:28 2018

@author: joaocamacho
"""
import numpy as np
import matplotlib.pylab as plt
plt.close('all')
from tedi import process, kernels, means

#Data
time = np.linspace(1,10,50)
y = 10*np.sin(time)
yerr = np.random.uniform(0,0.5,time.size)


kernel = kernels.Exponential(10,1,0.1)
mean = means.Keplerian(10,10,0.5,0,0, 5) + means.Keplerian(15,15,0.6,0,0, 10)
print(kernel)

#Gaussian processes
gpOBJ = process.GP(kernel,mean,time,y,yerr)
print(gpOBJ.log_likelihood(kernel))
print(gpOBJ.log_likelihood_gradient(kernel))

#ST processes
tpOBJ = process.TP(kernel,mean,time,y,yerr)
print(tpOBJ.log_likelihood(kernel, degrees=3))
print(tpOBJ.log_likelihood_gradient(kernel, degrees=3))

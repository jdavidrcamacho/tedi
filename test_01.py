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

gpOBJ = process.GP(kernel,mean,time,y,yerr)
print(gpOBJ.log_likelihood(kernel))
print(gpOBJ.log_likelihood_gradient(kernel))

new_hyperparms = np.array([20.0, 3.0 ,1.0])
kernel1 = gpOBJ.new_kernel(kernel, new_hyperparms)
print(kernel1)

mean, std, cov = gpOBJ.prediction(time = np.linspace(1,10,10))

tpOBJ = process.TP(kernel,mean,time,y,yerr)
print(tpOBJ.log_likelihood(kernel, 5))
print(tpOBJ.log_likelihood_gradient(kernel, 5))

plt.figure()
for i in range(3):
    plt.plot(time, y, 'k-')
    plt.plot(time,  gpOBJ.sample(kernel, time), 'b:')
    plt.plot(time,  tpOBJ.sample(kernel, 5, time), 'r-')
    i +=1
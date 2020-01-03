#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 15:06:55 2018

@author: joaocamacho
"""

import numpy as np
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


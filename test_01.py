#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 08:44:28 2018

@author: joaocamacho
"""
import numpy as np
from tedi import process, kernels

#Data
time = np.linspace(1,10,50)
y = np.sin(time)
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

mean, std, cov = gpOBJ.predict_gp(time = np.linspace(1,10,10))

tpOBJ = process.TP(kernel,mean,time,y,yerr)
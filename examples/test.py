#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 15:06:55 2018

@author: joaocamacho
"""

import numpy as np
from tedi import process, kernels, means

#Data
time = np.linspace(1,10,50)
y = 10*np.sin(time)
yerr = np.random.uniform(0,0.5,time.size)


kernel = kernels.Exponential(10,1) + kernels.WhiteNoise(0.1)
mean = means.Constant(0)
print(kernel)

#Gaussian processes
gpOBJ = process.GP(kernel,mean,time,y,yerr)
print(gpOBJ.log_likelihood(kernel))

#Student-t processes
tpOBJ = process.TP(kernel, 5, mean,time,y,yerr)
print(tpOBJ.log_likelihood(kernel, 5))


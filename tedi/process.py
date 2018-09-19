#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from tedi import kernels
from scipy.linalg import cho_factor, cho_solve, LinAlgError

class GP(object):
    """ 
        Class to create our Gaussian process.
        Parameters:
            kernel = covariance funtion
            means = mean function 
            time = time array
            y = measurements array
            yerr = measurements errors array
    """
    def __init__(self, kernel, mean, time, y, yerr = None):
        self.kernel = kernel        #covariance function
        self.mean = mean            #mean function
        self.time = time            #time
        self.y = y                  #measurements
        if yerr is None:
            self.yerr = 1e-12 * np.identity(self.t.size)
        else:
            self.yerr = yerr        #measurements errors

    def _kernel_pars(self, kernel):
        """
            Returns a kernel parameters
        """
        return kernel.pars

    def _kernel_matrix(self, kernel, time = None):
        """
            Returns the covariance matrix created by evaluating a given kernel 
        at inputs time.
        """
        #if time is None we use the time of our GP class
        if time is None:
            r = self.time[:, None] - self.time[None, :]
        #if we define a new time we will use it
        else:
            r = time[:, None] - time[None, :]
        K = kernel(r)
        return K

    def _predict_kernel_matrix(self, kernel, time, tstar):
        """
            To be used in prediction()
        """
        r = time[:, None] - self.time[None, :]
        K = kernel(r)
        return K

    def _mean_function(self, mean):
        """
            Returns the value of the mean function
        """
        #if we have a zero mean GP
        if mean is None:
            m = np.zeros_like(self.time)
        #if we defined a mean function to be used
        else:
            m = mean(self.time)
        return m

    def new_kernel(self, kernel, new_pars):
        """
            Updates the parameters of a kernel.
            Parameters:
                kernel = original kernel
                new_pars = new hyperparameters 
        """
        #if we are working with a sum of kernels
        if isinstance(kernel, kernels.Sum):
            k1_params = []
            for i, j in enumerate(kernel.k1.pars):
                k1_params.append(new_pars[i])
            k2_params = []
            for i, j in enumerate(kernel.k2.pars):
                k2_params.append(new_pars[len(kernel.k1.pars)+i])
            new_k1 = type(kernel.k1)(*k1_params)
            new_k2 = type(kernel.k2)(*k2_params)
            return new_k1+new_k2
        #if we are working with the product of kernels
        elif isinstance(kernel, kernels.Product):
            k1_params = []
            for i, e in enumerate(kernel.k1.pars):
                k1_params.append(new_pars[i])
            k2_params = []
            for j, e in enumerate(kernel.k2.pars):
                k2_params.append(new_pars[len(kernel.k1.pars)+j])
            new_k1 = type(kernel.k1)(*k1_params)
            new_k2 = type(kernel.k2)(*k2_params)
            return new_k1*new_k2
        #if we are working with a "single" kernel
        else:
            return type(kernel)(*new_pars)


##### marginal likelihood functions
    def compute_matrix(self, kernel, time, nugget = False, shift = False):
        """
            Creates the big covariance matrix K that will be used in the 
        log marginal likelihood calculation
            Parameters:
                kernel = covariance kernel
                time = time  
                nugget = True if K is not positive definite, False otherwise
                shift = True if K is not positive definite, False otherwise
            Returns:
                K = final covariance matrix 
                
            Note:
                To understand the nugget and shift parameters see 
            http://mathworld.wolfram.com/Ill-ConditionedMatrix.html for more 
            information about ill-conditioned matrices and 
            http://mathworld.wolfram.com/PositiveDefiniteMatrix.html for more
            information about positive defined matrices.
        """
        #Our K starts empty
        K = np.zeros((time.size, time.size))
        #Then we calculate the covariance matrix
        k = self._kernel_matrix(kernel, self.time)
        
        #addition of the measurement errors
        diag = self.yerr * np.identity(self.time.size)
        K = k + diag

        #more "weight" to the diagonal to avoid a ill-conditioned matrix
        if nugget:
            nugget_value = 0.01 #might be too big
            K = (1 - nugget_value)*K + nugget_value*np.diag(np.diag(K))
        #shifting all the eigenvalues up by the positive scalar to avoid a ill-conditioned matrix
        if shift:
            shift = 0.01 #might be too big
            K = K + shift * np.identity(self.time.size)
            
        return K

    def log_likelihood(self, kernel, mean = False, nugget = False, shift = False):
        """ 
            Calculates the marginal log likelihood.
        See Rasmussen & Williams (2006), page 113.
            Parameters:
                kernel = covariance funtion
                mean = mean function 
                nugget = True if K is not positive definite, False otherwise
                shift = True if K is not positive definite, False otherwise
            Returns:
                log_like  = marginal log likelihood
        """
        #covariance matrix calculation
        K = self.compute_matrix(kernel, self.time, 
                                nugget = False, shift = False)

        #calculation of y having into account the mean funtion
        if mean:
            y = self.y - mean(self.time)
        else:
            y = self.y

        #log marginal likelihood calculation
        try:
            L1 = cho_factor(K, overwrite_a=True, lower=False)
            log_like = - 0.5*np.dot(y.T, cho_solve(L1, y)) \
                       - np.sum(np.log(np.diag(L1[0]))) \
                       - 0.5*y.size*np.log(2*np.pi)
        except LinAlgError:
            return -np.inf
        return log_like








class TP(object):
    """ 
        Class to create our t-student process.
        Parameters:
            kernel = covariance funtion
            means = mean function 
            time = time array
            y = measurements array
            yerr = measurements errors array
    """
    def __init__(self, kernel, mean, time, y, yerr = None):
        self.kernel = kernel        #covariance function
        self.mean = mean            #mean function
        self.time = time            #time
        self.y = y                  #measurements
        if yerr is None:
            self.yerr = 1e-12 * np.identity(self.t.size)
        else:
            self.yerr = yerr        #measurements errors


    def _kernel_pars(self, kernel):
        """
            Returns a kernel parameters
        """
        return kernel.pars


    def _kernel_matrix(self, kernel, time = None):
        """
            Returns the covariance matrix created by evaluating a given kernel 
        at inputs time.
        """
        #if time is None we use the time of our GP class
        if time is None:
            r = self.time[:, None] - self.time[None, :]
        #if we define a new time we will use it
        else:
            r = time[:, None] - time[None, :]
        K = kernel(r)
        return K


    def _predict_kernel_matrix(self, kernel, time, tstar):
        """
            To be used in prediction()
        """
        r = time[:, None] - self.time[None, :]
        K = kernel(r)
        return K


    def _mean_function(self, mean):
        """
            Returns the value of the mean function
        """
        #if we have a zero mean GP
        if mean is None:
            m = np.zeros_like(self.time)
        #if we defined a mean function to be used
        else:
            m = mean(self.time)
        return m

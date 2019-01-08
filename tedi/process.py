#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from tedi import kernels
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.special import loggamma, digamma
from scipy.stats import multivariate_normal


##### Gaussian processes #######################################################
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
            self.yerr = 1e-12 * np.identity(self.time.size)
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

    def _mean_function(self, mean, time = None):
        """
            Returns the value of the mean function
        """
        if time is None:
            #if we have a zero mean GP
            if mean is None:
                m = np.zeros_like(self.time)
            #if we defined a mean function to be used
            else:
                m = mean(self.time)
        else:
            #if we have a zero mean GP
            if mean is None:
                m = np.zeros_like(time)
            #if we defined a mean function to be used
            else:
                m = mean(time)
        return m

    def new_kernel(self, kernel, new_pars):
        """
            Updates the parameters of a kernel.
            Parameters:
                kernel = original kernel
                new_pars = new hyperparameters 
        """
        #if we are working with the sum of kernels
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


##### GP sample funtion
    def sample(self, kernel, time):
        """ 
            Returns samples from the kernel
            Parameters:
                kernel = covariance funtion
                time = time array
            Returns:
                Sample of K 
        """
        mean = np.zeros_like(time)
        cov = self.compute_matrix(kernel, time)
        norm = multivariate_normal(mean, cov, allow_singular=True)
        return norm.rvs()


##### marginal likelihood gradient functions
    def _compute_matrix_derivative(self, kernel_derivative, nugget = False):
        """ 
            Creates the covariance matrices of dK/dOmega, the derivatives of the
        kernels.
            Parameters:
                kernel_derivative = derivatives we want to use this round
                nugget = True if K is not positive definite, False otherwise
            Return:
                k = final covariance matrix of dK/dOmega
        """
        #our matrix starts empty
        A = np.zeros((self.time.size, self.time.size))

        #measurement errors, should I add the errors in the derivatives???
        diag = self.yerr * np.identity(self.time.size)

        #derivative
        k = self._kernel_matrix(kernel_derivative, self.time)

        #final matrix
        A = A + k + diag
        #to avoid a ill-conditioned matrix
        if nugget:
            nugget_value = 0.01
            A = (1 - nugget_value) * A + nugget_value * np.diag(np.diag(A))
        return A

    def _log_like_grad(self, kernel_derivative, kernel, mean = False,
                       nugget = False):
        """ 
            Calculates the gradient of the marginal log likelihood for a given
        kernel derivative. 
        See Rasmussen & Williams (2006), page 114.
            Parameters:
                kernel_derivative = derivative we want to use this round
                kernel = covariance function
                nugget = True if K is not positive definite, False otherwise
            Returns:
                log_like  = Marginal log likelihood
        """
        #calculates the  covariance matrix of K and its inverse Kinv
        K = self.compute_matrix(kernel, self.time)
        Kinv = np.linalg.inv(K)
        #calculates the  covariance matrix of dK/dOmega
        dK = self._compute_matrix_derivative(kernel_derivative, nugget)

        #mean funtion
        if mean:
            y = self.y - mean(self.time)
        else:
            y = self.y

        #d(log marginal likelihood)/dOmega calculation
        try:
            alpha = np.dot(Kinv, y) #gives an array
            A = np.einsum('i,j',alpha, alpha) - Kinv #= alpha @ alpha.T - Kinv
            log_like_grad = 0.5 * np.einsum('ij,ij', A, dK) #= trace(a @ dK)

        except LinAlgError:
            return -np.inf
        return log_like_grad

    def log_likelihood_gradient(self, kernel, mean = False, nugget = False):
        """ 
            Returns the marginal log likelihood gradients of a kernel
            Parameters:
                kernel = covariance funtion
                mean = mean function
                nugget = True if K is not positive definite, False otherwise
            Returns:
                grads  = array of gradients
        """
        #First we derive the kernels
        parameters = kernel.pars #kernel parameters to use
        k = type(kernel).__subclasses__() #derivatives list
        derivatives_array = [] #its a list and not an array but thats ok
        for _, j in enumerate(k):
            derivative = j(*parameters)
            loglike = self._log_like_grad(derivative, kernel, nugget)
            derivatives_array.append(loglike)

        #To finalize we merge it into an array
        grads = np.array(derivatives_array)
        return grads


##### GP prediction funtion
    def prediction(self, kernel = False, mean = False, time = None):
        """ 
            Conditional predictive distribution of the Gaussian process
            Parameters:
                kernel = covariance function
                mean = mean function being used
                time = time  
        Returns:
            mean vector, covariance matrix, standard deviation vector
        """
        if kernel:
            #To use a new kernel
            kernel = kernel
        else:
            #To use the one we defined earlier 
            kernel = self.kernel

        #calculate mean and residuals
        if mean:
            r = self.y - mean(time)
        else:
            r = self.y

        #K
        cov = self._kernel_matrix(kernel, self.time)
        L1 = cho_factor(cov)
        sol = cho_solve(L1, r)

        #Kstar calculation
        Kstar = self._predict_kernel_matrix(kernel, time, self.time)
        #Kstarstar
        Kstarstar =  self._kernel_matrix(kernel, time)

        y_mean = np.dot(Kstar, sol) + self._mean(mean, time) #mean
        kstarT_k_kstar = []
        for i, e in enumerate(time):
            kstarT_k_kstar.append(np.dot(Kstar, cho_solve(L1, Kstar[i,:])))
        y_cov = Kstarstar - kstarT_k_kstar
        y_var = np.diag(y_cov) #variance
        y_std = np.sqrt(y_var) #standard deviation
        return y_mean, y_std, y_cov


##### Student-t processes ######################################################
class TP(object):
    """ 
        Class to create our student-t process.
        Parameters:
            kernel = covariance funtion
            degrees = degrees of freedom
            means = mean function 
            time = time array
            y = measurements array
            yerr = measurements errors array
    """
    def __init__(self, kernel, degrees, mean, time, y, yerr = None):
        self.kernel = kernel        #covariance function
        self.degrees = degrees      #degrees of freedom
        self.mean = mean            #mean function
        self.time = time            #time
        self.y = y                  #measurements
        if yerr is None:
            self.yerr = 1e-12 * np.identity(self.time.size)
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
        #if time is None we use the time of our TP class
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

    def _mean_function(self, mean, time = None):
        """
            Returns the value of the mean function
        """
        if time is None:
            #if we have a zero mean GP
            if mean is None:
                m = np.zeros_like(self.time)
            #if we defined a mean function to be used
            else:
                m = mean(self.time)
        else:
            #if we have a zero mean GP
            if mean is None:
                m = np.zeros_like(time)
            #if we defined a mean function to be used
            else:
                m = mean(time)
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
            information about ill-conditioned matrices, and 
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

    def log_likelihood(self, kernel, degrees, mean = False, 
                       nugget = False, shift = False):
        """ 
            Calculates the marginal log likelihood.
        See Solin and Särkkä (2015).
            Parameters:
                kernel = covariance funtion
                degrees = degrees of freedom
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
            beta = np.dot(y.T, cho_solve(L1, y))
            log_like = loggamma(0.5 * (degrees + y.size)) \
                        - 0.5 * y.size * np.log((degrees - 2) * np.pi) \
                        - np.sum(np.log(np.diag(L1[0]))) \
                        - 0.5 * (degrees + y.size)*np.log(1 + beta/(degrees-2)) \
                        - loggamma(0.5 * degrees)
        except LinAlgError:
            return -np.inf
        return np.real(log_like)


##### TP sample funtion
    def sample(self, kernel, degrees, time):
        """ 
            Sample from the kernel
            Parameters:
                kernel = covariance funtion
                degrees_freedom = degrees of freedom
                time = time array
            Returns:
                Sample of K 
                
            Note:
                Adapted from https://github.com/statsmodels/statsmodels
        """
        mean = np.zeros_like(self.time)
        if degrees == np.inf:
            x = 1
        else:
            x = np.random.chisquare(degrees, time.size)/degrees
        cov = self.compute_matrix(kernel, time)
        z = np.random.multivariate_normal(mean, cov, 1)
        sample = mean + z/np.sqrt(x)
        sample = (sample.T).reshape(-1)
        return sample


##### marginal likelihood gradient functions
    def _compute_matrix_derivative(self, kernel_derivative, nugget = False):
        """ 
            Creates the covariance matrices of dK/dOmega, the derivatives of the
        kernels.
            Parameters:
                kernel_derivative = derivatives we want to use this round
                nugget = True if K is not positive definite, False otherwise
            Return:
                k = final covariance matrix of dK/dOmega
        """
        #our matrix starts empty
        A = np.zeros((self.time.size, self.time.size))

        #measurement errors, should I add the errors in the derivatives???
        diag = self.yerr * np.identity(self.time.size)

        #derivative
        k = self._kernel_matrix(kernel_derivative, self.time)

        #final matrix
        A = A + k + diag
        #to avoid a ill-conditioned matrix
        if nugget:
            nugget_value = 0.01
            A = (1 - nugget_value) * A + nugget_value * np.diag(np.diag(A))
        return A

    def _log_like_grad(self, kernel_derivative, kernel, degrees, mean = False,
                       nugget = False):
        """ 
            Calculates the gradient of the marginal log likelihood for a given
        kernel derivative. 
        See Solin and Särkkä (2015) supplementary material.
            Parameters:
                kernel_derivative = derivative we want to use this round
                kernel = covariance function
                degrees = degrees of freedom
                nugget = True if K is not positive definite, False otherwise
            Returns:
                log_like  = Marginal log likelihood
        """
        #calculates the  covariance matrix of K and its inverse Kinv
        K = self.compute_matrix(kernel, self.time)
        Kinv = np.linalg.inv(K)
        #calculates the  covariance matrix of dK/dOmega
        dK = self._compute_matrix_derivative(kernel_derivative, nugget)

        #mean funtion
        if mean:
            y = self.y - mean(self.time)
        else:
            y = self.y

        #d(log marginal likelihood)/dOmega calculation
        try:
            L1 = cho_factor(K, overwrite_a=True, lower=False)
            alpha = np.dot(Kinv, y) #gives an array
            beta = np.dot(y.T, cho_solve(L1, y))
            alphaTalpha = np.einsum('i,j',alpha, alpha)
            log_like_grad = 0.5 * np.einsum('ij,ij', Kinv, dK) \
                            + 0.5*(degrees + y.size)/(degrees - 2 + beta) \
                            * np.einsum('ij,ij', alphaTalpha, dK)
        except LinAlgError:
            return -np.inf
        return log_like_grad

    def log_likelihood_gradient(self, kernel, degrees, mean = False, nugget = False):
        """ 
            Returns the marginal log likelihood gradients of a kernel.
            Parameters:
                kernel = covariance funtion
                degrees = degrees of freedom
                mean = mean function
                nugget = True if K is not positive definite, False otherwise
            Returns:
                grads  = array of gradients
        """
        #First we derive the kernels
        parameters = kernel.pars #kernel parameters to use
        k = type(kernel).__subclasses__() #derivatives list
        derivatives_array = [] #its a list and not an array but thats ok
        for _, j in enumerate(k):
            derivative = j(*parameters)
            loglike = self._log_like_grad(derivative, kernel, degrees, nugget)
            derivatives_array.append(loglike)

        #Then the derivative of the degrees of freedom
        K = self.compute_matrix(kernel, self.time)
        L1 = cho_factor(K, overwrite_a=True, lower=False)
        beta = np.dot(self.y.T, cho_solve(L1, self.y))
        degree_derivative = [0.5 * self.y.size / (degrees-2) \
                            - 0.5 * digamma(0.5*(degrees + self.y.size)) \
                            + 0.5 * digamma(0.5*degrees) \
                            + 0.5 * np.log(1 + beta/(degrees-2)) \
                            - 0.5 * ((degrees+self.y.size) + beta) \
                                        / ((degrees-2)*(degrees-2 + beta))]

        #To finalize we merge it into an array
        grads = np.array(derivatives_array + degree_derivative)
        return grads
    

##### TP predition funtion
    def prediction(self, kernel = False, degrees = False, 
                   mean = False, time = None):
        """ 
            Conditional predictive distribution of the Gaussian process
            Parameters:
                kernel = covariance function
                degrees = degrees of freedom
                mean = mean function being used
                time = time  
        Returns:
            mean vector, covariance matrix, standard deviation vector
        """
        if kernel:
            #To use a new kernel
            kernel = kernel
        else:
            #To use the one defined earlier 
            kernel = self.kernel
        if degrees:
            #To use a new degree of freedom
            degrees = degrees
        else:
            #To use the one defined earlier 
            degrees = self.degrees
        if mean:
            #If we are using a mean
            r = self.y - mean(time)
        else:
            r = self.y

        #K
        cov = self._kernel_matrix(kernel, self.time)
        L1 = cho_factor(cov)
        sol = cho_solve(L1, r)

        #Kstar calculation
        Kstar = self._predict_kernel_matrix(kernel, time, self.time)
        #Kstarstar
        Kstarstar =  self._kernel_matrix(kernel, time)

        y_mean = np.dot(Kstar, sol) + self._mean(mean, time) #mean
        kstarT_k_kstar = []
        for i, e in enumerate(time):
            kstarT_k_kstar.append(np.dot(Kstar, cho_solve(L1, Kstar[i,:])))
        y_cov = Kstarstar - kstarT_k_kstar

        var1 = degrees -2 + np.dot(r.T, sol)
        var2 = (degrees -2 + r.size)
        y_var =  var1 * np.diag(y_cov) / var2 #variance
        y_std = np.sqrt(y_var) #standard deviation
        return y_mean, y_std, y_cov

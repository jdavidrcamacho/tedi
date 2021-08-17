"""
Gaussian and Student-t processes functions
"""
import numpy as np
from tedi import kernels
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.special import loggamma
from scipy.stats import multivariate_normal, norm


##### Gaussian processes #######################################################
class GP(object):
    """ 
    Class to create our Gaussian process.
    
    Parameters
    ----------
    kernel: func
        Covariance funtion
    means: func
        Mean function 
    time: array
        Time array
    y: array
        Measurements array
    yerr: array
        Measurements errors array
    """
    def __init__(self, kernel, mean, time, y, yerr=None):
        self.kernel = kernel        #covariance function
        self.mean = mean            #mean function
        self.time = time            #time
        self.y = y                  #measurements
        if yerr is None:
            self.yerr = 1e-12 * np.identity(self.time.size)
        else:
            self.yerr = yerr        #measurements errors
        self.yerr2 = yerr**2
        
    def _kernel_pars(self, kernel):
        """ Returns a kernel parameters """
        return kernel.pars

    def _kernel_matrix(self, kernel, time):
        """ Returns a cov matrix when evaluating a given kernel at inputs time """
        r = time[:, None] - time[None, :]
        K = kernel(r)
        return K

    def _predict_kernel_matrix(self, kernel, time):
        """ To be used in prediction() """
        if isinstance(kernel, kernels.Sum):
            if isinstance(kernel.k2, kernels.WhiteNoise):
                kernel = kernel.k1
        r = time[:, None] - self.time[None, :]
        K = kernel(r)
        return K

    def _mean_function(self, mean, time=None):
        """ Returns the value of the mean function """
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
        Updates the parameters of a kernel
        
        Parameters
        ----------
        kernel: func
            Original kernel
        new_pars: list
            New hyperparameters
        
        Returns
        -------
        new_k: func
            Updated kernel
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
        elif isinstance(kernel, kernels.Multiplication):
            k1_params = []
            for i, _ in enumerate(kernel.k1.pars):
                k1_params.append(new_pars[i])
            k2_params = []
            for j, _ in enumerate(kernel.k2.pars):
                k2_params.append(new_pars[len(kernel.k1.pars)+j])
            new_k = type(kernel.k1)(*k1_params) * type(kernel.k1)(*k2_params)
            return new_k
        #if we are working with a "single" kernel
        else:
            new_k = type(kernel)(*new_pars)
            return new_k


##### marginal likelihood functions
    def compute_matrix(self, kernel, time, nugget=False, shift=False):
        """
        Creates the big covariance matrix K that will be used in the log 
        marginal likelihood calculation
        
        Parameters
        ----------
        kernel: func
            Covariance kernel
        time: array
            Time array
        nugget: bool
            True if K is not positive definite, False otherwise
        shift: bool
            True if K is not positive definite, False otherwise
        
        To understand the nugget and shift parameters see 
        http://mathworld.wolfram.com/Ill-ConditionedMatrix.html
        for more information about ill-conditioned matrices and 
        http://mathworld.wolfram.com/PositiveDefiniteMatrix.html
        for more information about positive defined matrices.
        
        Returns
        -------
        K: array
            Final covariance matrix 
        """
        #Our K starts empty
        K = np.zeros((time.size, time.size))
        #Then we calculate the covariance matrix
        k = self._kernel_matrix(kernel, self.time)
        #addition of the measurement errors
        diag = self.yerr**2 * np.identity(self.time.size)
        K = k + diag
        #more "weight" to the diagonal to avoid a ill-conditioned matrix
        if nugget:
            nugget_value = 0.01 #might be too big
            K = nugget_value*np.identity(self.time.size) + K
            #K = (1 - nugget_value)*K + nugget_value*np.diag(np.diag(K))
        #shifting all the eigenvalues up by the positive scalar to avoid a ill-conditioned matrix
        if shift:
            shift = 0.01 #might be too big
            K = K + shift * np.identity(self.time.size)
        return K

    def log_likelihood(self, kernel=None, mean=None, nugget=False, shift=False,
                       separate=False):
        """ 
        Calculates the marginal log likelihood. See Rasmussen & Williams (2006),
        page 113.
        
        Parameters
        ----------
        kernel:func
            Covariance funtion
        Mean: func
            Mean function 
        nugget: bool
            True if K is not positive definite, False otherwise
        shift: bool
            True if K is not positive definite, False otherwise
        shift: bool
            True to return the separated terms of the marginal log likelihood,
            False otherwise
        Returns
        -------
        log_like: float
            Marginal log likelihood
        """
        if kernel:
            #To use a new kernel
            kernel = kernel
        else:
            #To use the one defined earlier 
            kernel = self.kernel
        #covariance matrix calculation
        K = self.compute_matrix(kernel, self.time, 
                                nugget = nugget, shift = shift)
        #calculation of y having into account the mean funtion
        if mean:
            y = self.y - mean(self.time)
        else:
            y = self.y - self.mean(self.time)
        #log marginal likelihood calculation
        try:
            L1 = cho_factor(K, overwrite_a=True, lower=False, check_finite=False)
            if separate:
                log_like = [- 0.5*np.dot(y.T, cho_solve(L1, y)),
                            - np.sum(np.log(np.diag(L1[0]))),
                            - 0.5*y.size*np.log(2*np.pi)]
            else:
                log_like = - 0.5*np.dot(y.T, cho_solve(L1, y)) \
                           - np.sum(np.log(np.diag(L1[0]))) \
                           - 0.5*y.size*np.log(2*np.pi)
        except LinAlgError:
            return -np.inf
        return log_like
    
    def marginalLikelihood(self, kernel1=None, mean=None, jitter=None, 
                           N=1000 , file = "saved_results.txt"):
        f = open(file, "a")
        m = mean(self.time) 
        err = np.sqrt(jitter**2 + self.yerr2) 
        n = 0
        llhood = 0
        while n<N:
            sample = self.sample(kernel1, self.time) + m
            normpdf = norm(loc=sample, scale=err).pdf(self.y)
            llhood += normpdf.prod()
            sigmaN = np.std(normpdf)
            n += 1
            if n%500 ==0:
                 print(n, np.log(llhood/n), sigmaN/np.sqrt(n), file=f)
        f.close()
        return np.log(llhood/N)


##### GP sample funtion
    def sample(self, kernel, time, nugget=False):
        """
        Returns samples from the kernel
        
        Parameters
        ----------
        kernel: func
            Covariance function
        time: array
            Time array
        nugget: bool
            True if K is not positive definite, False otherwise
        seed: bool
            True to include a seed
        seedValue: int
            Integer of the seed
        Returns
        -------
        norm: array
            Sample of K 
        """
        mean = np.zeros_like(time)
        cov = self._kernel_matrix(kernel, time)
        if nugget:
            nugget_value = 0.01 #might be too big
            cov = (1 - nugget_value)*cov + nugget_value*np.diag(np.diag(cov))
        norm = multivariate_normal(mean, cov, allow_singular=True).rvs()
        return norm

    def posteriorSample(self, kernel, mean, a, time, nugget=False):
        """
        Returns samples from the kernel
        
        Parameters
        ----------
        kernel: func
            Covariance function
        time: array
            Time array
        nugget: bool
            True if K is not positive definite, False otherwise
        seed: bool
            True to include a seed
        seedValue: int
            Integer of the seed
        Returns
        -------
        norm: array
            Sample of K 
        """
        m, _, v = self.prediction(kernel, mean, time)
        return np.random.multivariate_normal(m, v, 1).T

##### GP prediction funtion
    def prediction(self, kernel=None, mean=None, time=False, std=True):
        """ 
        Conditional predictive distribution of the Gaussian process
        
        Parameters
        ----------
        kernel: func
            Covariance function
        mean: func
            Mean function being used
        time: array
            Time array
        std: bool
            True to calculate standard deviation, False otherwise
            
        Returns
        -------
        y_mean: array
            Mean vector
        y_std: array
            Standard deviation vector
        time: array
            Time array
        """
        if not kernel:
            #To use the one we defined earlier 
            kernel = self.kernel
        #calculate mean and residuals
        if mean:
            r = self.y - mean(self.time)
        else:
            r = self.y
        cov = self._kernel_matrix(kernel, self.time) + np.diag(self.yerr2) #K
        L1 = cho_factor(cov)
        sol = cho_solve(L1, r)
        #Kstar
        Kstar = self._predict_kernel_matrix(kernel, time)
        #Kstarstar
        Kstarstar =  self._kernel_matrix(kernel, time)
        if mean:
            y_mean = np.dot(Kstar, sol) + self._mean_function(mean, time) #mean
        else:
            y_mean = np.dot(Kstar, sol) #mean
        kstarT_k_kstar = []
        for i, _ in enumerate(time):
            kstarT_k_kstar.append(np.dot(Kstar, cho_solve(L1, Kstar[i,:])))
        y_cov = Kstarstar - kstarT_k_kstar
        y_var = np.diag(y_cov) #variance
        if std:
            y_std = np.sqrt(y_var) #standard deviation
            return y_mean, y_std, time
        return y_mean, np.zeros_like(y_mean), time


##### Student-t processes ######################################################
class TP(object):
    """ 
    Class to create our Student-t process
    WARNING: Not as developed and tested as the Gaussian process class
    
        Parameters
    ----------
    kernel: func
        Covariance funtion
    degrees: int
        Degrees of freedom
    means: func
        Mean function 
    time: array
        Time array
    y: array
        Measurements array
        else:
    yerr: array
        Measurements errors array
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
        """ Returns a kernel parameters """
        return kernel.pars

    def _kernel_matrix(self, kernel, time = None):
        """ Returns a cov matrix evaluating a given kernel at inputs time """
        #if time is None we use the time of our TP class
        if time is None:
            r = self.time[:, None] - self.time[None, :]
        #if we define a new time we will use it
        else:
            r = time[:, None] - time[None, :]
        K = kernel(r)
        return K

    def _predict_kernel_matrix(self, kernel, time):
        """ To be used in prediction() """
        r = time[:, None] - self.time[None, :]
        K = kernel(r)
        return K

    def _mean_function(self, mean, time = None):
        """ Returns the value of the mean function """
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
        Updates the parameters of a kernel
        
        Parameters
        ----------
        kernel: func
            Original kernel
        new_pars: list
            New hyperparameters
        
        Returns
        -------
        new_k: func
            Updated kernel
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
        elif isinstance(kernel, kernels.Multiplication):
            k1_params = []
            for i, _ in enumerate(kernel.k1.pars):
                k1_params.append(new_pars[i])
            k2_params = []
            for j, _ in enumerate(kernel.k2.pars):
                k2_params.append(new_pars[len(kernel.k1.pars)+j])
            new_k = type(kernel.k1)(*k1_params) * type(kernel.k2)(*k2_params)
            return new_k
        #if we are working with a "single" kernel
        else:
            new_k = type(kernel)(*new_pars)
            return new_k


##### marginal likelihood functions
    def compute_matrix(self, kernel, time, nugget = False, shift = False):
        """
        Creates the big covariance matrix K that will be used in the log 
        marginal likelihood calculation
        
        Parameters
        ----------
        kernel: func
            Covariance kernel
        time: array
            Time array
        nugget: bool
            True if K is not positive definite, False otherwise
        shift: bool
            True if K is not positive definite, False otherwise
        
        To understand the nugget and shift parameters see 
        http://mathworld.wolfram.com/Ill-ConditionedMatrix.html
        for more information about ill-conditioned matrices and 
        http://mathworld.wolfram.com/PositiveDefiniteMatrix.html
        for more information about positive defined matrices.
        
        Returns
        -------
        K: array
            Final covariance matrix 
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

    def log_likelihood(self, kernel, degrees, mean = False, nugget = False, 
                       shift = False):
        """ 
        Calculates the marginal log likelihood. See Solin and Särkkä (2015).
        
        Parameters
        ----------
        kernel: func
            Covariance funtion
        degrees: int
            Degrees of freedom
        mean: func
            Mean function 
        nugget: bool
            True if K is not positive definite, False otherwise
        shift: bool
            True if K is not positive definite, False otherwise
        
        Returns
        -------
        log_like: int
            Marginal log likelihood
        """
        #covariance matrix calculation
        K = self.compute_matrix(kernel, self.time, nugget = False, shift = False)
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
        Note: adapted from https://github.com/statsmodels/statsmodels
        
        Parameters
        ----------
        kernel: func
            Covariance funtion
        degrees_freedom: int
            Degrees of freedom
        time: array
            Time array
        
        Returns
        -------
        sample: array
            Sample of K 
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


##### TP predition funtion
    def prediction(self, kernel = None, degrees = None, mean = None, time = None):
        """ 
        Conditional predictive distribution of our process
        
        Parameters
        ----------
        kernel: func
            Covariance function
        degrees: int
            Degrees of freedom
        mean: func
            Mean function being used
        time: array
            Time array
        
        Returns
        -------
        y_mean: array
            Mean vector
        y_std: array
            Standard deviation vector
        y_cov: array
            Covariance matrix
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
        cov = self._kernel_matrix(kernel, self.time) #K
        L1 = cho_factor(cov)
        sol = cho_solve(L1, r)
        #Kstar calculation
        Kstar = self._predict_kernel_matrix(kernel, time)
        #Kstarstar
        Kstarstar =  self._kernel_matrix(kernel, time)
        if mean:
            y_mean = np.dot(Kstar, sol) + self._mean_function(mean, time) #mean
        else:
            y_mean = np.dot(Kstar, sol)
        kstarT_k_kstar = []
        for i, _ in enumerate(time):
            kstarT_k_kstar.append(np.dot(Kstar, cho_solve(L1, Kstar[i,:])))
        y_cov = Kstarstar - kstarT_k_kstar
        var1 = degrees -2 + np.dot(r.T, sol)
        var2 = (degrees -2 + r.size)
        y_var =  var1 * np.diag(y_cov) / var2 #variance
        y_std = np.sqrt(y_var) #standard deviation
        return y_mean, y_std, y_cov


### END

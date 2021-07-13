"""
Covariance functions
"""
import numpy as np
#because it makes life easier down the line
pi, exp, sine, cosine, sqrt = np.pi, np.exp, np.sin, np.cos, np.sqrt
__all__ = ['Constant', 'WhiteNoise', 'SquaredExponential' , 'Periodic', 
            'QuasiPeriodic', 'RationalQuadratic', 'Cosine', 'Exponential',
            'Matern32', 'Matern52', 'RQP']


class kernel(object):
    """
        Definition the kernels that will be used. To simplify my life all the
    kernels defined are the sum of kernel + white noise
    """
    def __init__(self, *args):
        """ Puts all kernel arguments in an array pars. """
        self.pars = np.array(args, dtype=float)
    def __call__(self, r):
        """ r = t - t' """
        raise NotImplementedError
    def __repr__(self):
        """ Representation of each kernel instance """
        return "{0}({1})".format(self.__class__.__name__,
                                 ", ".join(map(str, self.pars)))
    def __add__(self, b):
        return Sum(self, b)
    def __radd__(self, b):
        return self.__add__(b)

    def __mul__(self, b):
        return Multiplication(self, b)
    def __rmul__(self, b):
        return self.__mul__(b)


class _operator(kernel):
    """ To allow operations between two kernels """
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2
        self.kerneltype = 'complex'
    @property
    def pars(self):
        return np.append(self.k1.pars, self.k2.pars)


class Sum(_operator):
    """ To allow the sum of kernels """
    def __repr__(self):
        return "{0} + {1}".format(self.k1, self.k2)
    def __call__(self, r):
        return self.k1(r) + self.k2(r)


class Multiplication(_operator):
    """ To allow the multiplication of kernels """
    def __repr__(self):
        return "{0} * {1}".format(self.k1, self.k2)
    def __call__(self, r):
        return self.k1(r) * self.k2(r)


##### Constant kernel #########################################################
class Constant(kernel):
    """
    This kernel returns its constant argument c 
    
    Parameters
    ----------
    c: float
        Constant
    """
    def __init__(self, c):
        super(Constant, self).__init__(c)
        self.c = c
        self.params_number = 1  #number of hyperparameters
    def __call__(self, r):
        return self.c * np.ones_like(r)


##### White noise kernel ######################################################
class WhiteNoise(kernel):
    """
    Definition of the white noise kernel.
    
    Parameters
    ----------
    wn: float
        White noise amplitude
    """
    def __init__(self, wn):
        super(WhiteNoise, self).__init__(wn)
        self.wn = wn
        self.type = 'stationary'
        self.derivatives = 1    #number of derivatives in this kernel
        self.params_number = 1  #number of hyperparameters
    def __call__(self, r):
#        return self.wn**2 * np.identity(len(r))
        return self.wn**2 * np.diag(np.diag(np.ones_like(r)))


##### Squared exponential kernel ##############################################
class SquaredExponential(kernel):
    """
    Squared Exponential kernel, also known as radial basis function or RBF 
    kernel in other works.
    
    Parameters
    ----------
    amplitude: float
        Amplitude of the kernel
    ell: float
        Length-scale
    """
    def __init__(self, amplitude, ell):
        super(SquaredExponential, self).__init__(amplitude, ell)
        self.amplitude = amplitude
        self.ell = ell
        self.params_number = 2
    def __call__(self, r):
        return self.amplitude**2 * exp(-0.5 * r**2 / self.ell**2)


##### Periodic kernel #########################################################
class Periodic(kernel):
    """
    Definition of the periodic kernel.
    
    Parameters
    ----------
    amplitude: float
        Amplitude of the kernel
    ell: float
        Lenght scale
    P: float
        Period
    """
    def __init__(self, amplitude, P, ell):
        super(Periodic, self).__init__(amplitude, P, ell)
        self.amplitude = amplitude
        self.ell = ell
        self.P = P
        self.params_number = 3  #number of hyperparameters
    def __call__(self, r):
        return self.amplitude**2*exp(-2*sine(pi*np.abs(r)/self.P)**2/self.ell**2)


##### Quasi periodic kernel ###################################################
class QuasiPeriodic(kernel):
    """
    This kernel is the product between the exponential sine squared kernel 
    and the squared exponential kernel, commonly known as the quasi-periodic 
    kernel.
    
    Parameters
    ----------
    amplitude: float
        Amplitude of the kernel
    ell_e: float
        Evolutionary time scale
    ell_p: float
        Length scale of the periodic component
    P: float
        Kernel periodicity
    """
    def __init__(self, amplitude, ell_e, P, ell_p):
        super(QuasiPeriodic, self).__init__(amplitude, ell_e, P, ell_p)
        self.amplitude = amplitude
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.params_number = 4
    def __call__(self, r):
        return self.amplitude**2 *exp(- 2*sine(pi*np.abs(r)/self.P)**2 \
                                      /self.ell_p**2 - r**2/(2*self.ell_e**2))


##### Rational quadratic kernel ###############################################
class RationalQuadratic(kernel):
    """
    Definition of the rational quadratic kernel.
    
    Parameters
    ----------
    amplitude: float
        Amplitude of the kernel
    alpha: float
        Amplitude of large and small scale variations
    ell: float
        Characteristic lenght scale to define the kernel "smoothness"
    """
    def __init__(self, amplitude, alpha, ell):
        super(RationalQuadratic, self).__init__(amplitude, alpha, ell)
        self.amplitude = amplitude
        self.alpha = alpha
        self.ell = ell
        self.params_number = 3
    def __call__(self, r):
        return self.amplitude**2*(1+0.5*r**2/(self.alpha*self.ell**2))**(-self.alpha)


##### Cosine kernel ###########################################################
class Cosine(kernel):
    """
    Definition of the cosine kernel.
    
    Parameters
    ----------
    amplitude: float
        Amplitude of the kernel
    P: float
        Period
    """
    def __init__(self, amplitude, P):
        super(Cosine, self).__init__(amplitude, P)
        self.amplitude = amplitude
        self.P = P
        self.params_number = 2
    def __call__(self, r):
        return self.amplitude**2 * cosine(2*pi*np.abs(r) / self.P)


##### Exponential kernel ######################################################
class Exponential(kernel):
    """
    Definition of the exponential kernel. This kernel arises when setting v=1/2
    in the matern family of kernels
    
    Parameters
    ----------
    amplitude: float
        Amplitude of the kernel
    ell: float
        Characteristic lenght scale
    """
    def __init__(self, amplitude, ell):
        super(Exponential, self).__init__(amplitude, ell)
        self.amplitude = amplitude
        self.ell = ell
        self.params_number = 2
    def __call__(self, r):
        return self.amplitude**2 * exp(- np.abs(r)/self.ell)


##### Matern 3/2 kernel #######################################################
class Matern32(kernel):
    """
    Definition of the Matern 3/2 kernel. This kernel arise when setting v=3/2 
    in the matern family of kernels
    
    Parameters
    ----------
    amplitude: float
        Amplitude of the kernel
    ell: float
        Characteristic lenght scale
    """
    def __init__(self, amplitude, ell):
        super(Matern32, self).__init__(amplitude, ell)
        self.amplitude = amplitude
        self.ell = ell
        self.params_number = 2
    def __call__(self, r):
        return self.amplitude**2 *(1 + np.sqrt(3)*np.abs(r)/self.ell) \
                        *np.exp(-np.sqrt(3)*np.abs(r) / self.ell)


#### Matern 5/2 kernel ########################################################
class Matern52(kernel):
    """
    Definition of the Matern 5/2 kernel. This kernel arise when setting v=5/2 
    in the matern family of kernels

    Parameters
    ----------
    amplitude: float
        Amplitude of the kernel
    ell: float
        Characteristic lenght scale
    """
    def __init__(self, amplitude, ell):
        super(Matern52, self).__init__(amplitude, ell)
        self.amplitude = amplitude
        self.ell = ell
        self.params_number = 2
    def __call__(self, r):
        return self.amplitude**2 * (1 + (3*np.sqrt(5)*self.ell*np.abs(r) \
                                         +5*np.abs(r)**2)/(3*self.ell**2) ) \
                                         *exp(-np.sqrt(5.0)*np.abs(r)/self.ell)


##### Paciorek's kernel #######################################################
class Paciorek(kernel):
    """
    Definition of the modified Paciorek's kernel (stationary version). 
    
    Parameters
    ----------
    amplitude: float
        Amplitude/amplitude of the kernel
    ell_1: float
        First lenght scale
    ell_2: float
        Second lenght scale
    """
    def __init__(self, amplitude, ell_1, ell_2):
        super(Paciorek, self).__init__(amplitude, ell_1, ell_2)
        self.amplitude = amplitude
        self.ell_1 = ell_1
        self.ell_2 = ell_2
        self.params_number = 3
    def __call__(self, r):
        a = sqrt(2*self.ell_1*self.ell_2 / (self.ell_1**2+self.ell_2**2))
        b = exp(-2*r*r / (self.ell_1**2+self.ell_2**2))
        return self.amplitude**2 * a *b


##### RQP kernel ##############################################################
class RQP(kernel):
    """
    Definition of the product between the periodic kernel and the rational 
    quadratic kernel that we called RQP kernel.
    
    Info: Test show that if alpha goes to infinity the RQP tends to the quasi
    periodic kernel, if alpha goes to zero it tends to the periodic kernel.
        There is a goldilocks region of alpha where this kernel is much better 
    than the quasi periodic kernel.
    
    Parameters
    ----------
    amplitude: float
        Amplitude of the kernel
    ell_e and ell_p: float
        Aperiodic and periodic lenght scales
    alpha: float
        alpha of the rational quadratic kernel
    P: float
        Periodic repetitions of the kernel
    """
    def __init__(self, amplitude, alpha, ell_e, P, ell_p):
        super(RQP, self).__init__(amplitude, alpha, ell_e, P, ell_p)
        self.amplitude = amplitude
        self.alpha = alpha
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.params_number = 5
    def __call__(self, r):
        a = exp(- 2*sine(pi*np.abs(r)/self.P)**2 / self.ell_p**2)
        b = (1+ r**2/ (2*self.alpha*self.ell_e**2))#**self.alpha
        return self.amplitude**2 * a / (np.sign(b) * (np.abs(b)) ** self.alpha)


###############################################################################
class PiecewiseSE(kernel):
    """
    Product of the Squared Exponential and Piecewice kernels
    
    Parameters
    ----------
    eta1: float
        Amplitude of the kernel
    eta2: float
        Aperiodic lenght scale
    eta3: float
        Periodic repetitions of the kernel
    """
    def __init__(self, eta1, eta2, eta3):
        super(PiecewiseSE, self).__init__(eta1, eta2, eta3)
        self.eta1 = eta1
        self.eta2 = eta2
        self.eta3 = eta3
        self.params_number = 3
    def __call__(self, r):
        SE_term = self.eta1**2 * exp(-0.5 * r**2 / self.eta2**2)
        r = r/(0.5*self.eta3)
        piecewise = (3*np.abs(r) +1) * (1 - np.abs(r))**3
        piecewise = np.where(np.abs(r)>1, 0, piecewise)
        k = SE_term*piecewise
        return k


###############################################################################
class PiecewiseRQ(kernel):
    """
    Product of the Rational Quadratic and Piecewice kernels
    
    Parameters
    ----------
    eta1: float
        Amplitude of the kernel
    alpha: float
        alpha of the rational quadratic kernel
    eta2: float
        Aperiodic lenght scale
    eta3: float
        Periodic repetitions of the kernel
    """
    def __init__(self, eta1, alpha, eta2, eta3):
        super(PiecewiseRQ, self).__init__(eta1, alpha, eta2, eta3)
        self.eta1 = eta1
        self.alpha = alpha
        self.eta2 = eta2
        self.eta3 = eta3
        self.params_number = 3
    def __call__(self, r):
        RQ_term = self.eta1**2 * (1+0.5*r**2/(self.alpha*self.eta2**2))**(-self.alpha)
        r = r/(0.5*self.eta3)
        piecewise = (3*np.abs(r) +1) * (1 - np.abs(r))**3
        piecewise = np.where(np.abs(r)>1, 0, piecewise)
        k = RQ_term*piecewise
        return k


##### New periodic kernel ######################################################
class NewPeriodic(kernel):
    """
    Definition of a new periodic kernel derived from mapping the rational 
    quadratic kernel to the 2D space u(x) = (cos x, sin x)
    
    Parameters
    ----------
    amplitude: float
        Amplitude of the kernel
    alpha2: float
        Alpha parameter of the rational quadratic mapping
    P: float
        Period
    l: float
        Periodic lenght scale
    """
    def __init__(self, amplitude, alpha2, P, l):
        super(NewPeriodic, self).__init__(amplitude, alpha2, P, l)
        self.amplitude = amplitude
        self.alpha2 = alpha2
        self.P = P
        self.l = l
        self.params_number = 4
    def __call__(self, r):
        a = (1 + 2*sine(pi*np.abs(r)/self.P)**2/(self.alpha2*self.l**2))**(-self.alpha2)
        return self.amplitude**2 * a


##### New periodic kernel ######################################################
class NewQuasiPeriodic(kernel):
    """
    Definition of a new quasi-periodic kernel. Derived from mapping the rational
    quadratic kernel to the 2D space u(x) = (cos x, sin x) and multiplying it by
    a squared exponential kernel
    
    Parameters
    ----------
    amplitude: float
        Amplitude of the kernel
    alpha2: float
        Alpha parameter of the rational quadratic mapping
    ell_e: float
        Aperiodic lenght scale
    P: float
        Period
    ell_p: float
        Periodic lenght scale
    """
    def __init__(self, amplitude, alpha2, ell_e, P, ell_p):
        super(NewQuasiPeriodic, self).__init__(amplitude, alpha2, ell_e, P, ell_p)
        self.amplitude = amplitude
        self.alpha2 = alpha2
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.params_number = 5  #number of hyperparameters
    def __call__(self, r):
        a = (1 + 2*sine(pi*np.abs(r)/self.P)**2/(self.alpha2*self.ell_p**2))**(-self.alpha2)
        b =  exp(-0.5 * r**2 / self.ell_e**2)
        return self.amplitude**2 * a * b


class NewRQP(kernel):
    """
    Definition of a new quasi-periodic kernel. Derived from mapping the rational
    quadratic kernel to the 2D space u(x) = (cos x, sin x) and multiplying it by
    a rational quadratic kernel
    
    Parameters
    ----------
    amplitude: float
        Amplitude of the kernel
    alpha1: float
        Alpha parameter of the rational quadratic kernel
    ell_e: float
        Aperiodic lenght scale
    P: float
        Period
    ell_p: float
        Periodic lenght scale
    alpha2: float
        Another alpha parameter from the mapping 
    """
    def __init__(self, amplitude, alpha1, alpha2, ell_e, P, ell_p):
        super(NewRQP, self).__init__(amplitude, alpha1, alpha2,
                                     ell_e, P, ell_p)
        self.amplitude = amplitude
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.params_number = 5  #number of hyperparameters
    def __call__(self, r):
        a = (1 + 2*sine(pi*np.abs(r)/self.P)**2/(self.alpha2*self.ell_p**2))**(-self.alpha2)
        b = (1+ 0.5*r**2/ (self.alpha1*self.ell_e**2))**(-self.alpha1)
        return self.amplitude**2 * a * b


### END

"""
Covariance functions
"""
import numpy as np
#because it makes my life easier down the line
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


##### Constant kernel ##########################################################
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
        self.type = 'non-stationary and anisotropic'
        self.derivatives = 1    #number of derivatives in this kernel
        self.params_number = 1  #number of hyperparameters
    def __call__(self, r):
        return self.c * np.ones_like(r)

class dConstant_dc(Constant):
    """ Log-derivative in order to c """
    def __init__(self, c):
        super(dConstant_dc, self).__init__(c)
        self.c = c
    def __call__(self, r):
        return self.c * np.ones_like(r)


##### White noise kernel #######################################################
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
        return self.wn**2 * np.diag(np.diag(np.ones_like(r)))

class dWhiteNoise_dwn(WhiteNoise):
    """ Log-derivative in order to the white noise amplitude """
    def __init__(self, wn):
        super(dWhiteNoise_dwn, self).__init__(wn)
        self.wn = wn
    def __call__(self, r):
        return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))


##### Squared exponential kernel ###############################################
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
    wn: float
        White noise amplitude
    """
    def __init__(self, amplitude, ell, wn):
        super(SquaredExponential, self).__init__(amplitude, ell, wn)
        self.amplitude = amplitude
        self.ell = ell
        self.wn = wn
        self.type = 'stationary and anisotropic'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_number = 3  #number of hyperparameters
    def __call__(self, r):
        try:
            return self.amplitude**2 * exp(-0.5 * r**2 / self.ell**2) \
                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.amplitude**2 * exp(-0.5 * r**2 / self.ell**2)

class dSquaredExponential_damplitude(SquaredExponential):
    """ Log-derivative in order to the amplitude """
    def __init__(self, amplitude, ell, wn):
        super(dSquaredExponential_damplitude, self).__init__(amplitude, ell, wn)
        self.amplitude = amplitude
        self.ell = ell
        self.wn = wn
    def __call__(self, r):
        return 2 * self.amplitude**2 * exp(-0.5 * r**2 / self.ell**2)

class dSquaredExponential_dell(SquaredExponential):
    """ Log-derivative in order to the ell """
    def __init__(self, amplitude, ell, wn):
        super(dSquaredExponential_dell, self).__init__(amplitude, ell, wn)
        self.amplitude = amplitude
        self.ell = ell
        self.wn = wn
    def __call__(self, r):
        return (r**2 * self.amplitude**2 / self.ell**2) \
                * exp(-0.5 * r**2 / self.ell**2)

class dSquaredExponential_dwn(SquaredExponential):
    """ Log-derivative in order to the white noise amplitude """
    def __init__(self, amplitude, ell, wn):
        super(dSquaredExponential_dwn, self).__init__(amplitude, ell, wn)
        self.amplitude = amplitude
        self.ell = ell
        self.wn = wn
    def __call__(self, r):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)


##### Periodic kernel ##########################################################
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
    wn: float
        White noise amplitude
    """
    def __init__(self, amplitude, ell, P, wn):
        super(Periodic, self).__init__(amplitude, ell, P, wn)
        self.amplitude = amplitude
        self.ell = ell
        self.P = P
        self.wn = wn
        self.type = 'non-stationary and isotropic'
        self.derivatives = 4    #number of derivatives in this kernel
        self.params_number = 4  #number of hyperparameters
    def __call__(self, r):
        try:
            return self.amplitude**2 * \
                    exp( -2 * sine(pi*np.abs(r)/self.P)**2 /self.ell**2) \
                        + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.amplitude**2 * \
                    exp( -2 * sine(pi*np.abs(r)/self.P)**2 /self.ell**2) 

class dPeriodic_damplitude(Periodic):
    """ Log-derivative in order to the amplitude """
    def __init__(self, amplitude, ell, P, wn):
        super(dPeriodic_damplitude, self).__init__(amplitude, ell, P, wn)
        self.amplitude = amplitude
        self.ell = ell
        self.P = P
        self.wn = wn
    def __call__(self, r):
        return 2 * self.amplitude**2 * exp(-2 * sine(pi * np.abs(r) / self.P)**2 \
                                        / self.ell**2)

class dPeriodic_dell(Periodic):
    """ Log-derivative in order to ell """
    def __init__(self, amplitude, ell, P, wn):
        super(dPeriodic_dell, self).__init__(amplitude, ell, P, wn)
        self.amplitude = amplitude
        self.ell = ell
        self.P = P
        self.wn = wn
    def __call__(self, r):
        return (4* self.amplitude**2 * sine(pi * np.abs(r) / self.P)**2 \
                *exp(-2 * sine(pi * np.abs(r) / self.P)**2 \
                     / self.ell**2)) / self.ell**2

class dPeriodic_dP(Periodic):
    """ Log-derivative in order to P """
    def __init__(self, amplitude, ell, P, wn):
        super(dPeriodic_dP, self).__init__(amplitude, ell, P, wn)
        self.amplitude = amplitude
        self.ell = ell
        self.P = P
        self.wn = wn
    def __call__(self, r):
        return (4 * pi * r * self.amplitude**2 \
                * cosine(pi*np.abs(r) / self.P) *sine(pi*np.abs(r) / self.P) \
                * exp(-2 * sine(pi*np.abs(r) / self.P)**2 / self.ell**2)) \
                / (self.ell**2 * self.P)

class dPeriodic_dwn(Periodic):
    """ Log-derivative in order to the white noise amplitude """
    def __init__(self, amplitude, ell, P, wn):
        super(dPeriodic_dwn, self).__init__(amplitude, ell, P, wn)
        self.amplitude = amplitude
        self.ell = ell
        self.P = P
        self.wn = wn
    def __call__(self, r):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)


##### Quasi periodic kernel ####################################################
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
    wn: float
        White noise amplitude
    """
    def __init__(self, amplitude, ell_e, P, ell_p, wn):
        super(QuasiPeriodic, self).__init__(amplitude, ell_e, P, ell_p, wn)
        self.amplitude = amplitude
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn
        self.type = 'non-stationary and anisotropic'
        self.derivatives = 5    #number of derivatives in this kernel
        self.params_number = 5  #number of hyperparameters
    def __call__(self, r):
        try:
            return self.amplitude**2 *exp(- 2*sine(pi*np.abs(r)/self.P)**2 \
                                          /self.ell_p**2 - r**2/(2*self.ell_e**2)) \
                                          + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.amplitude**2 *exp(- 2*sine(pi*np.abs(r)/self.P)**2 \
                                          /self.ell_p**2 - r**2/(2*self.ell_e**2))

class dQuasiPeriodic_damplitude(Periodic):
    """ Log-derivative in order to the amplitude """
    def __init__(self, amplitude, ell_e, P, ell_p, wn):
        super(dQuasiPeriodic_damplitude, self).__init__(amplitude, ell_e, P, ell_p, wn)
        self.amplitude = amplitude
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn
    def __call(self, r):
        return 2 * self.amplitude**2 *exp(-2 * sine(pi*np.abs(r)/self.P)**2 \
                                   /self.ell_p**2 - r**2/(2*self.ell_e**2))

class dQuasiPeriodic_delle(QuasiPeriodic):
    """ Log-derivative in order to ell_e """
    def __init__(self, amplitude, ell_e, P, ell_p, wn):
        super(dQuasiPeriodic_delle, self).__init__(amplitude, ell_e, P, ell_p, wn)
        self.amplitude = amplitude
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn
    def __call(self, r):
        return (r**2 * self.amplitude**2 / self.ell_e**2) \
                *exp(-2 * sine(pi*np.abs(r)/self.P)**2 \
                     /self.ell_p**2 - r**2/(2*self.ell_e**2))

class dQuasiPeriodic_dP(QuasiPeriodic):
    """ Log-derivative in order to P """
    def __init__(self, amplitude, ell_e, P, ell_p, wn):
        super(dQuasiPeriodic_dP, self).__init__(amplitude, ell_e, P, ell_p, wn)
        self.amplitude = amplitude
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn
    def __call(self, r):
        return 4 * pi * r * self.wn**2 \
                * cosine(pi*np.abs(r)/self.P) * sine(pi*np.abs(r)/self.P) \
                * exp(-2 * sine(pi * np.abs(r)/self.P)**2 \
                      /self.ell_p**2 - r**2/(2*self.ell_e**2)) \
                      / (self.ell_p**2 * self.P)

class dQuasiPeriodic_dellp(QuasiPeriodic):
    """ Log-derivative in order to ell_p """
    def __init__(self, amplitude, ell_e, P, ell_p, wn):
        super(dQuasiPeriodic_dellp, self).__init__(amplitude, ell_e, P, ell_p, wn)
        self.amplitude = amplitude
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn =wn
    def __call(self, r):
        return  4 * self.wn**2 * sine(pi*r/self.P)**2 \
                * exp(-2 * sine(pi*np.abs(r)/self.P)**2 \
                      /self.ell_p**2 - r**2/(2*self.ell_e**2)) / self.ell_p**2

class dQuasiPeriodic_dwn(QuasiPeriodic):
    """ Log-derivative in order to the white noise amplitude
    """
    def __init__(self, amplitude, ell, P, wn):
        super(dQuasiPeriodic_dwn, self).__init__(amplitude, ell, P, wn)
        self.amplitude = amplitude
        self.ell = ell
        self.P = P
        self.wn = wn
    def __call__(self, r):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)


##### Rational quadratic kernel ################################################
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
    wn: float
        White noise amplitude
    """
    def __init__(self, amplitude, alpha, ell, wn):
        super(RationalQuadratic, self).__init__(amplitude, alpha, ell, wn)
        self.amplitude = amplitude
        self.alpha = alpha
        self.ell = ell
        self.wn = wn
        self.type = 'stationary and anisotropic'
        self.derivatives = 4    #number of derivatives in this kernel
        self.params_number = 4  #number of hyperparameters
    def __call__(self, r):
        try: 
            return self.amplitude**2 * (1+ 0.5*r**2/ (self.alpha*self.ell**2))**(-self.alpha) \
                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.amplitude**2 * (1+ 0.5*r**2/ (self.alpha*self.ell**2))**(-self.alpha)

class dRationalQuadratic_damplitude(RationalQuadratic):
    """ Log-derivative in order to the amplitude """
    def __init__(self, amplitude, alpha, ell, wn):
        super(dRationalQuadratic_damplitude, self).__init__(amplitude, alpha, ell, wn)
        self.amplitude = amplitude
        self.alpha = alpha
        self.ell = ell
        self.wn = wn
    def __call__(self, r):
        return 2 * self.amplitude**2 \
                / (1+ r**2/ (2*self.alpha*self.ell**2))**self.alpha

class dRationalQuadratic_dalpha(RationalQuadratic):
    """ Log-derivative in order to alpha """
    def __init__(self, amplitude, alpha, ell, wn):
        super(dRationalQuadratic_dalpha, self).__init__(amplitude, alpha, ell, wn)
        self.amplitude = amplitude
        self.alpha = alpha
        self.ell = ell
        self.wn = wn
    def __call(self, r):
        return ((r**2/(2*self.alpha*self.ell**2*(r**2/(2*self.alpha*self.ell**2)+1))\
                 - np.log(r**2/(2*self.alpha*self.ell**2)+1)) \
                    * self.amplitude**2 * self.alpha) \
                    / (1+r**2/(2*self.alpha*self.ell**2))**self.alpha

class dRationalQuadratic_dell(RationalQuadratic):
    """ Log-derivative in order to ell """
    def __init__(self, amplitude, alpha, ell, wn):
        super(dRationalQuadratic_dell, self).__init__(amplitude, alpha, ell, wn)
        self.amplitude = amplitude
        self.alpha = alpha
        self.ell = ell
        self.wn = wn
    def __call(self, r):
        return r**2 * (1+r**2/(2*self.alpha*self.ell**2))**(-1-self.alpha) \
                * self.wn**2 / self.ell**2

class dRationalQuadratic_dwn(RationalQuadratic):
    """ Log-derivative in order to the white noise amplitude """
    def __init__(self, amplitude, alpha, ell, wn):
        super(dRationalQuadratic_dwn, self).__init__(amplitude, alpha, ell, wn)
        self.amplitude = amplitude
        self.alpha = alpha
        self.ell = ell
        self.wn = wn
    def __call__(self, r):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)


##### Cosine kernel ############################################################
class Cosine(kernel):
    """
    Definition of the cosine kernel.
    
    Parameters
    ----------
    amplitude: float
        Amplitude of the kernel
    P: float
        Period
    wn: float
        White noise amplitude
    """
    def __init__(self, amplitude, P, wn):
        super(Cosine, self).__init__(amplitude, P, wn)
        self.amplitude = amplitude
        self.P = P
        self.wn = wn
        self.type = 'non-stationary and isotropic'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_number = 3  #number of hyperparameters
    def __call__(self, r):
        try:
            return self.amplitude**2 * cosine(2*pi*np.abs(r) / self.P) \
                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.amplitude**2 * cosine(2*pi*np.abs(r) / self.P)

class dCosine_damplitude(Cosine):
    """ Log-derivative in order to the amplitude """
    def __init__(self, amplitude, P, wn):
        super(dCosine_damplitude, self).__init__(amplitude, P, wn)
        self.amplitude = amplitude
        self.P = P
        self.wn = wn
    def __call__(self, r):
        return 2*self.amplitude**2 * cosine(2*pi*np.abs(r) / self.P)

class dCosine_dP(Cosine):
    """ Log-derivative in order to P """
    def __init__(self, amplitude, P, wn):
        super(dCosine_dP, self).__init__(amplitude, P, wn)
        self.amplitude = amplitude
        self.P = P
        self.wn = wn
    def __call__(self, r):
        return self.amplitude**2 * r*pi*sine(2*pi*np.abs(r) / self.P) / self.P

class dCosine_dwn(Cosine):
    """ Log-derivative in order to the white noise amplitude """
    def __init__(self, amplitude, P, wn):
        super(dCosine_dwn, self).__init__(amplitude, P, wn)
        self.amplitude = amplitude
        self.P = P
        self.wn = wn
    def __call__(self, r):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)


##### Exponential kernel #######################################################
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
    wn: float
        White noise amplitude
    """
    def __init__(self, amplitude, ell, wn):
        super(Exponential, self).__init__(amplitude, ell, wn)
        self.amplitude = amplitude
        self.ell = ell
        self.wn = wn
        self.type = 'stationary and isotropic'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_number = 3  #number of hyperparameters
    def __call__(self, r):
        try:
            return self.amplitude**2 * exp(- np.abs(r)/self.ell) \
                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.amplitude**2 * exp(- np.abs(r)/self.ell)


class dExponential_damplitude(Exponential):
    """ Log-derivative in order to the amplitude """
    def __init__(self, amplitude, ell, wn):
        super(dExponential_damplitude, self).__init__(amplitude, ell, wn)
        self.amplitude = amplitude
        self.ell = ell
        self.wn = wn
    def __call__(self, r):
        return 2*self.amplitude**2 * exp(- np.abs(r)/self.ell)

class dExpoential_dell(Exponential):
    """ Log-derivative in order to ell """
    def __init__(self, amplitude, ell, wn):
        super(dExpoential_dell, self).__init__(amplitude, ell, wn)
        self.amplitude = amplitude
        self.ell = ell
        self.wn = wn
    def __call__(self, r):
        return -0.5*self.amplitude**2 * r *exp(- np.abs(r)/self.ell) /self.ell

class dExpoential_dwm(Exponential):
    """ Log-derivative in order to the white noise amplitude """
    def __init__(self, amplitude, ell, wn):
        super(dExpoential_dwm, self).__init__(amplitude, ell, wn)
        self.amplitude = amplitude
        self.ell = ell
        self.wn = wn
    def __call__(self, r):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)


##### Matern 3/2 kernel ########################################################
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
    wn: float
        White noise amplitude
    """
    def __init__(self, amplitude, ell, wn):
        super(Matern32, self).__init__(amplitude, ell, wn)
        self.amplitude = amplitude
        self.ell = ell
        self.wn = wn
        self.type = 'stationary and isotropic'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_number = 3  #number of hyperparameters
    def __call__(self, r):
        try:
            return self.amplitude**2 *(1 + np.sqrt(3)*np.abs(r)/self.ell) \
                        * np.exp(-np.sqrt(3)*np.abs(r) / self.ell) \
                        + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.amplitude**2 *(1 + np.sqrt(3)*np.abs(r)/self.ell) \
                        * np.exp(-np.sqrt(3)*np.abs(r) / self.ell)

class dMatern32_damplitude(Matern32):
    """ Log-derivative in order to the amplitude """
    def __init__(self, amplitude, ell, wn):
        super(dMatern32_damplitude, self).__init__(amplitude, ell, wn)
        self.amplitude = amplitude
        self.ell = ell
        self.wn = wn
    def __call__(self, r):
        return 2*self.amplitude**2 *(1 + np.sqrt(3)*np.abs(r)/self.ell) \
                    *np.exp(-np.sqrt(3)*np.abs(r) / self.ell)

class dMatern32_dell(Matern32):
    """ Log-derivative in order to ell """
    def __init__(self, amplitude, ell, wn):
        super(dMatern32_dell, self).__init__(amplitude, ell, wn)
        self.amplitude = amplitude
        self.ell = ell
        self.wn = wn
    def __call__(self, r):
        return (sqrt(3) * r * (1+ (sqrt(3) * r) / self.ell) \
                *exp(-(sqrt(3)*r) / self.ell) * self.amplitude**2) / self.ell \
                -(sqrt(3) * r * exp(-(sqrt(3)*r) / self.ell)*self.amplitude**2)/self.ell

class dMatern32_dwn(Matern32):
    """ Log-derivative in order to the white noise amplitude """
    def __init__(self, amplitude, ell, wn):
        super(dMatern32_dwn, self).__init__(amplitude, ell, wn)
        self.amplitude = amplitude
        self.ell = ell
        self.wn = wn
    def __call__(self, r):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)


#### Matern 5/2 kernel #########################################################
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
    wn: float
        White noise amplitude
    """
    def __init__(self, amplitude, ell, wn):
        super(Matern52, self).__init__(amplitude, ell, wn)
        self.amplitude = amplitude
        self.ell = ell
        self.wn = wn
        self.type = 'stationary and isotropic'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_number = 3    #number of hyperparameters
    def __call__(self, r):
        try:
            return self.amplitude**2 * (1 + (3*np.sqrt(5)*self.ell*np.abs(r) \
                                               +5*np.abs(r)**2)/(3*self.ell**2) ) \
                                              *exp(-np.sqrt(5.0)*np.abs(r)/self.ell) \
                                    + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return self.amplitude**2 * (1 + (3*np.sqrt(5)*self.ell*np.abs(r) \
                                               +5*np.abs(r)**2)/(3*self.ell**2) ) \
                                              *exp(-np.sqrt(5.0)*np.abs(r)/self.ell) 

class dMatern52_damplitude(Matern52):
    """ Log-derivative in order to the amplitude """
    def __init__(self, amplitude, ell, wn):
        super(dMatern52_damplitude, self).__init__(amplitude, ell, wn)
        self.amplitude = amplitude
        self.ell = ell
        self.wn = wn
    def __call__(self, r):
        return 2*self.amplitude**2 * (1 + ( 3*np.sqrt(5)*self.ell*np.abs(r) \
                                           +5*np.abs(r)**2)/(3*self.ell**2) ) \
                                          *exp(-np.sqrt(5.0)*np.abs(r)/self.ell)

class dMatern52_dell(Matern52):
    """ Log-derivative in order to ell """
    def __init__(self, amplitude, ell, wn):
        super(dMatern52_dell, self).__init__(amplitude, ell, wn)
        self.amplitude = amplitude
        self.ell = ell
        self.wn = wn
    def __call__(self, r):
        return self.ell * ((sqrt(5)*r*(1+(sqrt(5)*r) \
                                 /self.ell+(5*r**2)/(3*self.ell**2)) \
                             *exp(-(sqrt(5)*r)/self.ell)*self.amplitude**2) \
            /self.ell**2 +(-(sqrt(5)*r)/self.ell**2-(10*r**2) \
                           /(3*self.ell**3)) \
                           *exp(-(sqrt(5)*r)/self.ell)*self.amplitude**2)

class dMatern52_dwn(Matern52):
    """ Log-derivative in order to the white noise amplitude """
    def __init__(self, amplitude, ell, wn):
        super(dMatern52_dwn, self).__init__(amplitude, ell, wn)
        self.amplitude = amplitude
        self.ell = ell
        self.wn = wn
    def __call__(self, r):
        try:
            return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            return np.zeros_like(r)


##### RQP kernel ###############################################################
class RQP(kernel):
    """
    WARNING: EXPERIMENTAL KERNEL
    Definition of the product between the exponential sine squared kernel 
    and the rational quadratic kernel that we called RQP kernel.
    If I am thinking this correctly then this kernel should tend to the
    QuasiPeriodic kernel as alpha increases, although I am not sure if we can
    say that it tends to the QuasiPeriodic kernel as alpha tends to infinity.
    
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
    wn: float
        White noise amplitude
    """
    def __init__(self, amplitude, alpha, ell_e, P, ell_p, wn):
        super(RQP, self).__init__(amplitude, alpha, ell_e, P, ell_p, wn)
        self.amplitude = amplitude
        self.alpha = alpha
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn
        self.type = 'non-stationary and anisotropic'
        self.derivatives = 6    #number of derivatives in this kernel
        self.params_number = 6  #number of hyperparameters
    def __call__(self, r):
        try:
            #because of numpy issues
            a = exp(- 2*sine(pi*np.abs(r)/self.P)**2 / self.ell_p**2)
            b = (1+ r**2/ (2*self.alpha*self.ell_e**2))#**self.alpha
            c = self.wn**2 * np.diag(np.diag(np.ones_like(r)))
            return self.amplitude**2 * a / (np.sign(b) * (np.abs(b)) ** self.alpha) + c
        except ValueError:
            a = exp(- 2*sine(pi*np.abs(r)/self.P)**2 / self.ell_p**2)
            b = (1+ r**2/ (2*self.alpha*self.ell_e**2))#**self.alpha
            return self.amplitude**2 * a / (np.sign(b) * (np.abs(b)) ** self.alpha)

class dRQP_damplitude(RQP):
    """ Log-derivative in order to the amplitude """
    def __init__(self, amplitude, alpha, ell_e, P, ell_p, wn):
        super(dRQP_damplitude, self).__init__(amplitude, alpha, ell_e, P, ell_p, wn)
        self.amplitude = amplitude
        self.alpha = alpha
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn
    def __call(self, r):
        raise NotImplementedError

class dRQP_dalpha(RQP):
    """ Log-derivative in order to alpha """
    def __init__(self, amplitude, alpha, ell_e, P, ell_p, wn):
        super(dRQP_dalpha, self).__init__(amplitude, alpha, ell_e, P, ell_p, wn)
        self.amplitude = amplitude
        self.alpha = alpha
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn
    def __call__(self, r):
        raise NotImplementedError

class dRQP_delle(RQP):
    """ Log-derivative in order to ell_e """
    def __init__(self, amplitude, alpha, ell_e, P, ell_p, wn):
        super(dRQP_delle, self).__init__(amplitude, alpha, ell_e, P, ell_p, wn)
        self.amplitude = amplitude
        self.alpha = alpha
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn
    def __call__(self, r):
        raise NotImplementedError

class dRQP_dP(RQP):
    """ Log-derivative in order to P """
    def __init__(self, amplitude, alpha, ell_e, P, ell_p, wn):
        super(dRQP_dP, self).__init__(amplitude, alpha, ell_e, P, ell_p, wn)
        self.amplitude = amplitude
        self.alpha = alpha
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn
    def __call__(self, r):
        raise NotImplementedError

class dRQP_dellp(RQP):
    """ Log-derivative in order to ell_p """
    def __init__(self, amplitude, alpha, ell_e, P, ell_p, wn):
        super(dRQP_dellp, self).__init__(amplitude, alpha, ell_e, P, ell_p, wn)
        self.amplitude = amplitude
        self.alpha = alpha
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn
    def __call(self, r):
        raise NotImplementedError

class dRQP_dwn(RQP):
    """ Log-derivative in order to the white noise amplitude """
    def __init__(self, amplitude, alpha, ell_e, P, ell_p, wn):
        super(dRQP_dwn, self).__init__(amplitude, alpha, ell_e, P, ell_p, wn)
        self.amplitude = amplitude
        self.alpha = alpha
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.wn = wn
    def __call(self, r):
        raise NotImplementedError


#### Wave kernel #########################################################
class Wave(kernel):
    """
    WARNING: EXPERIMENTAL KERNEL
    Definition of the wave kernel. Still don't understand how this is a valid 
    kernel since np.abs(r) needs to be different than 0
    
    Parameters
    ----------
    amplitude: float
        Amplitude/amplitude of the kernel
    theta: float
        Parameter that still don't know what it does
    wn: float
        White noise amplitude
    """
    def __init__(self, amplitude, theta, wn):
        super(Wave, self).__init__(amplitude, theta, wn)
        self.amplitude = amplitude
        self.theta = theta
        self.wn = wn
        self.type = 'unknown'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_number = 3    #number of hyperparameters
    def __call__(self, r):
        try:
            if r == 0:
                return 0
            else:
                return self.amplitude**2 * self.theta/np.abs(r) \
                                * np.sin(-np.abs(r)/self.theta) \
                                + self.wn**2 * np.diag(np.diag(np.ones_like(r)))
        except ValueError:
            if r == 0:
                return 0
            else:
                return self.amplitude**2 * self.theta/np.abs(r) \
                                * np.sin(-np.abs(r)/self.theta)

class dWave_damplitude(Wave):
    """ Log-derivative in order to the amplitude """
    def __init__(self, amplitude, theta, wn):
        super(dWave_damplitude, self).__init__(amplitude, theta, wn)
        self.amplitude = amplitude
        self.theta = theta
        self.wn = wn
    def __call__(self, r):
        raise NotImplementedError

class dWave_dtheta(Wave):
    """ Log-derivative in order to theta """
    def __init__(self, amplitude, theta, wn):
        super(dWave_dtheta, self).__init__(amplitude, theta, wn)
        self.amplitude = amplitude
        self.theta = theta
        self.wn = wn
    def __call__(self, r):
        raise NotImplementedError

class dWave_dwn(Wave):
    """ Log-derivative in order to the white noise amplitude """
    def __init__(self, amplitude, theta, wn):
        super(dWave_dwn, self).__init__(amplitude, theta, wn)
        self.amplitude = amplitude
        self.theta = theta
        self.wn = wn
    def __call__(self, r):
        raise NotImplementedError


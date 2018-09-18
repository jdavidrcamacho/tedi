#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
#because it makes my life easier down the line
pi, exp, sine, cosine, sqrt = np.pi, np.exp, np.sin, np.cos, np.sqrt

class kernel(object):
    """
        Definition the kernels that will be used..

    """
    def __init__(self, *args):
        """
            Puts all kernel arguments in an array pars.
        """
        self.pars = np.array(args)

    def __call__(self, r):
        """
            r = t - t' 
        """
        raise NotImplementedError

    def __repr__(self):
        """
            Representation of each kernel instance
        """
        return "{0}({1})".format(self.__class__.__name__,
                                 ", ".join(map(str, self.pars)))

    def __add__(self, b):
        return Sum(self, b)
    def __radd__(self, b):
        return self.__add__(b)

    def __mul__(self, b):
        return Product(self, b)
    def __rmul__(self, b):
        return self.__mul__(b)


class _operator(kernel):
    """ 
        To allow operations between two kernels 
    """
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2
        self.kerneltype = 'complex'

    @property
    def pars(self):
        return np.append(self.k1.pars, self.k2.pars)


class Sum(_operator):
    """ 
        To allow the sum of kernels
    """
    def __repr__(self):
        return "{0} + {1}".format(self.k1, self.k2)

    def __call__(self, r):
        return self.k1(r) + self.k2(r)


class Product(_operator):
    """ 
        To allow the multiplication of kernels 
    """
    def __repr__(self):
        return "{0} * {1}".format(self.k1, self.k2)

    def __call__(self, r):
        return self.k1(r) * self.k2(r)



##### Constant #################################################################
class Constant(kernel):
    """
        This kernel returns its constant argument c 
        Parameters:
            c = constant
    """
    def __init__(self, c):
        super(Constant, self).__init__(c)
        self.c = c
        self.type = 'non-stationary and anisotropic'
        self.derivatives = 10   #number of derivatives in this kernel
        self.params_number = 1  #number of hyperparameters

    def __call__(self, r):
        return self.c * np.ones_like(r)

class dConstant_dc(Constant):
    """
        Log-derivative in order to c
    """
    def __init__(self, c):
        super(dConstant_dc, self).__init__(c)
        self.c = c

    def __call__(self, r):
        return self.c * np.ones_like(r)

##### White Noise ##############################################################
class WhiteNoise(kernel):
    """
        Definition of the white noise kernel.
        Parameters
            wn = white noise amplitude
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
    """
        Log-derivative in order to the amplitude
    """
    def __init__(self, wn):
        super(dWhiteNoise_dwn, self).__init__(wn)
        self.wn = wn

    def __call__(self, r):
        return 2 * self.wn**2 * np.diag(np.diag(np.ones_like(r)))


##### Squared exponential ######################################################
class SquaredExponential(kernel):
    """
        Squared Exponential kernel, also known as radial basis function or RBF 
    kernel in other works.
        Parameters:
            amplitude = amplitude of the kernel
            ell = length-scale
    """
    def __init__(self, amplitude, ell):
        super(SquaredExponential, self).__init__(amplitude, ell)
        self.amplitude = amplitude
        self.ell = ell
        self.type = 'stationary and anisotropic'
        self.derivatives = 2    #number of derivatives in this kernel
        self.params_number = 2  #number of hyperparameters

    def __call__(self, r):
        return self.amplitude**2 * exp(-0.5 * r**2 / self.ell**2)

class dSquaredExponential_damplitude(SquaredExponential):
    """
        Log-derivative in order to the amplitude
    """
    def __init__(self, amplitude, ell):
        super(dSquaredExponential_damplitude, self).__init__(amplitude, ell)
        self.amplitude = amplitude
        self.ell = ell

    def __call__(self, r):
        return 2 * self.amplitude**2 * exp(-0.5 * r**2 / self.ell**2)

class dSquaredExponential_dell(SquaredExponential):
    """
        Log-derivative in order to the ell
    """
    def __init__(self, amplitude, ell):
        super(dSquaredExponential_dell, self).__init__(amplitude, ell)
        self.amplitude = amplitude
        self.ell = ell

    def __call__(self, r):
        return (r**2 * self.amplitude**2 / self.ell**2) \
                * exp(-0.5 * r**2 / self.ell**2)


##### Periodic #################################################################
class Periodic(kernel):
    """
        Definition of the periodic kernel.
        Parameters:
            amplitude = amplitude of the kernel
            ell = lenght scale
            P = period
    """
    def __init__(self, amplitude, ell, P):
        super(Periodic, self).__init__(amplitude, ell, P)
        self.amplitude = amplitude
        self.ell = ell
        self.P = P
        self.type = 'non-stationary and isotropic'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_number = 3  #number of hyperparameters

    def __call__(self, r):
        return self.amplitude**2 * exp( -2 * sine(pi*np.abs(r)/self.P)**2 /self.ell**2)

class dPeriodic_damplitude(Periodic):
    """
        Log-derivative in order to the amplitude
    """
    def __init__(self, amplitude, ell, P):
        super(dPeriodic_damplitude, self).__init__(amplitude, ell, P)
        self.amplitude = amplitude
        self.ell = ell
        self.P = P

    def __call__(self, r):
        return 2 * self.amplitude**2 * exp(-2 * sine(pi * np.abs(r) / self.P)**2 \
                                        / self.ell**2)

class dPeriodic_dell(Periodic):
    """
        Log-derivative in order to ell
    """
    def __init__(self, amplitude, ell, P):
        super(dPeriodic_dell, self).__init__(amplitude, ell, P)
        self.amplitude = amplitude
        self.ell = ell
        self.P = P

    def __call__(self, r):
        return (4* self.amplitude**2 * sine(pi * np.abs(r) / self.P)**2 \
                *exp(-2 * sine(pi * np.abs(r) / self.P)**2 \
                     / self.ell**2)) / self.ell**2

class dPeriodic_dP(Periodic):
    """
        Log-derivative in order to P
    """
    def __init__(self, amplitude, ell, P):
        super(dPeriodic_dP, self).__init__(amplitude, ell, P)
        self.amplitude = amplitude
        self.ell = ell
        self.P = P

    def __call__(self, r):
        return (4 * pi * r * self.amplitude**2 \
                * cosine(pi*np.abs(r) / self.P) *sine(pi*np.abs(r) / self.P) \
                * exp(-2 * sine(pi*np.abs(r) / self.P)**2 / self.ell**2)) \
                / (self.ell**2 * self.P)


##### Quasi Periodic ###########################################################
class QuasiPeriodic(kernel):
    """
        This kernel is the product between the exponential sine squared kernel 
    and the squared exponential kernel, commonly known as the quasi-periodic 
    kernel.
        Parameters:
            amplitude = amplitude of the kernel
            ell_e = evolutionary time scale
            ell_p = length scale of the periodic component
            P = kernel Periodicity
    """
    def __init__(self, amplitude, ell_e, P, ell_p):
        super(QuasiPeriodic, self).__init__(amplitude, ell_e, P, ell_p)
        self.amplitude = amplitude
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.type = 'non-stationary and anisotropic'
        self.derivatives = 4    #number of derivatives in this kernel
        self.params_number = 4  #number of hyperparameters

    def __call__(self, r):
        return self.amplitude**2 *exp(- 2*sine(pi*np.abs(r)/self.P)**2 \
                                   /self.ell_p**2 - r**2/(2*self.ell_e**2))

class dQuasiPeriodic_damplitude(Periodic):
    """
            Log-derivative in order to the amplitude
    """
    def __init__(self, amplitude, ell_e, P, ell_p):
        super(dQuasiPeriodic_damplitude, self).__init__(amplitude, ell_e, P, ell_p)
        self.amplitude = amplitude
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p

    def __call(self, r):
        return 2 * self.amplitude**2 *exp(-2 * sine(pi*np.abs(r)/self.P)**2 \
                                   /self.ell_p**2 - r**2/(2*self.ell_e**2))

class dQuasiPeriodic_delle(QuasiPeriodic):
    """
        Log-derivative in order to ell_e
    """
    def __init__(self, amplitude, ell_e, P, ell_p):
        super(dQuasiPeriodic_delle, self).__init__(amplitude, ell_e, P, ell_p)
        self.amplitude = amplitude
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p

    def __call(self, r):
        return (r**2 * self.amplitude**2 / self.ell_e**2) \
                *exp(-2 * sine(pi*np.abs(r)/self.P)**2 \
                     /self.ell_p**2 - r**2/(2*self.ell_e**2))

class dQuasiPeriodic_dP(QuasiPeriodic):
    """
        Log-derivative in order to P
    """
    def __init__(self, amplitude, ell_e, P, ell_p):
        super(dQuasiPeriodic_dP, self).__init__(amplitude, ell_e, P, ell_p)
        self.amplitude = amplitude
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p

    def __call(self, r):
        return 4 * pi * r * self.w**2 \
                * cosine(pi*np.abs(r)/self.P) * sine(pi*np.abs(r)/self.P) \
                * exp(-2 * sine(pi * np.abs(r)/self.P)**2 \
                      /self.ell_p**2 - r**2/(2*self.ell_e**2)) \
                      / (self.ell_p**2 * self.P)

class dQuasiPeriodic_dellp(QuasiPeriodic):
    """
        Log-derivative in order to ell_p
    """
    def __init__(self, amplitude, ell_e, P, ell_p):
        super(dQuasiPeriodic_dellp, self).__init__(amplitude, ell_e, P, ell_p)
        self.amplitude = amplitude
        self.ell_e = ell_e
        self.P = P
        self.ell_p = ell_p

    def __call(self, r):
        return  4 * self.w**2 * sine(pi*r/self.P)**2 \
                * exp(-2 * sine(pi*np.abs(r)/self.P)**2 \
                      /self.ell_p**2 - r**2/(2*self.ell_e**2)) / self.ell_p**2


##### Rational Quadratic #######################################################
class RationalQuadratic(kernel):
    """
        Definition of the rational quadratic kernel.
        Parameters:
            amplitude = amplitude of the kernel
            alpha = amplitude of large and small scale variations
            ell = characteristic lenght scale to define the kernel "smoothness"
    """
    def __init__(self, amplitude, alpha, ell):
        super(RationalQuadratic, self).__init__(amplitude, alpha, ell)
        self.amplitude = amplitude
        self.alpha = alpha
        self.ell = ell
        self.type = 'stationary and anisotropic'
        self.derivatives = 3    #number of derivatives in this kernel
        self.params_number = 3  #number of hyperparameters

    def __call__(self, r):
        return self.amplitude**2 / (1+ r**2/ (2*self.alpha*self.ell**2))**self.alpha

class dRationalQuadratic_damplitude(RationalQuadratic):
    """
        Log-derivative in order to the amplitude
    """
    def __init__(self, amplitude, alpha, ell):
        super(dRationalQuadratic_damplitude, self).__init__(amplitude, alpha, ell)
        self.amplitude = amplitude
        self.alpha = alpha
        self.ell = ell

    def __call__(self, r):
        return 2 * self.amplitude**2 \
                / (1+ r**2/ (2*self.alpha*self.ell**2))**self.alpha

class dRationalQuadratic_dalpha(RationalQuadratic):
    """
        Log-derivative in order to alpha
    """
    def __init__(self, amplitude, alpha, ell):
        super(dRationalQuadratic_dalpha, self).__init__(amplitude, alpha, ell)
        self.amplitude = amplitude
        self.alpha = alpha
        self.ell = ell

    def __call(self, r):
        return ((r**2/(2*self.alpha*self.ell**2*(r**2/(2*self.alpha*self.ell**2)+1))\
                 - np.log(r**2/(2*self.alpha*self.ell**2)+1)) \
                    * self.amplitude**2 * self.alpha) \
                    / (1+r**2/(2*self.alpha*self.ell**2))**self.alpha

class dRationalQuadratic_dell(RationalQuadratic):
    """
        Log-derivative in order to ell
    """
    def __init__(self, amplitude, alpha, ell):
        super(dRationalQuadratic_dell, self).__init__(amplitude, alpha, ell)
        self.amplitude = amplitude
        self.alpha = alpha
        self.ell = ell

    def __call(self, r):
        return r**2 * (1+r**2/(2*self.alpha*self.ell**2))**(-1-self.alpha) \
                * self.w**2 / self.ell**2


##### RQP kernel ###############################################################
class RQP(kernel):
    """
        Definition of the product between the exponential sine squared kernel 
    and the rational quadratic kernel that we called RQP kernel.
        If I am thinking this correctly then this kernel should tend to the
    QuasiPeriodic kernel as alpha increases, although I am not sure if we can
    say that it tends to the QuasiPeriodic kernel as alpha tends to infinity.
        Parameters:
            amplitude = amplitude of the kernel
            ell_e and ell_p = aperiodic and periodic lenght scales
            alpha = alpha of the rational quadratic kernel
            P = periodic repetitions of the kernel
    """
    def __init__(self, amplitude, alpha, ell_e, P, ell_p):
        super(RQP, self).__init__(amplitude, alpha, ell_e, P, ell_p)
        self.amplitude = amplitude
        self.alpha = alpha
        self.RQP_ell_e = ell_e
        self.P = P
        self.ell_p = ell_p
        self.type = 'non-stationary and anisotropic'
        self.derivatives = 5    #number of derivatives in this kernel
        self.params_number = 5  #number of hyperparameters

    def __call__(self, r):
        return self.amplitude**2 * exp(- 2*sine(pi*np.abs(r)/self.P)**2 \
                                    / self.ell_p**2) \
                    /(1+ r**2/ (2*self.alpha*self.ell_e**2))**self.alpha

class dRQP_damplitude(RQP):
    """
        Log-derivative in order to the amplitude
    """
    def __init__(self, amplitude, alpha, ell_e, P, ell_p):
        super(dRQP_damplitude, self).__init__(amplitude, alpha, ell_e, P, ell_p)
        self.amplitude = amplitude
        self.alpha = alpha
        self.RQP_ell_e = ell_e
        self.P = P
        self.ell_p = ell_p

    def __call(self, r):
        return 2 * self.amplitude**2 * exp(- 2*sine(pi*np.abs(r)/self.P)**2 \
                                    / self.ell_p**2) \
                    /(1+ r**2/ (2*self.alpha*self.ell_e**2))**self.alpha

class dRQP_dalpha(RQP):
    """
        Log-derivative in order to alpha
    """
    def __init__(self, amplitude, alpha, ell_e, P, ell_p):
        super(dRQP_damplitude, self).__init__(amplitude, alpha, ell_e, P, ell_p)
        self.amplitude = amplitude
        self.alpha = alpha
        self.RQP_ell_e = ell_e
        self.P = P
        self.ell_p = ell_p

    def __call__(self, r):
        return self.alpha * ((r**2 / (2*self.alpha \
                         *self.ell_e**2*(r**2/(2*self.alpha*self.ell_e**2)+1)) \
            -np.log(r**2/(2*self.alpha*self.ell_e**2)+1)) \
            *self.amplitude**2*exp(-2*sine(pi*np.abs(r)/self.P)**2/self.ell_p**2)) \
            /(1+r**2/(2*self.alpha*self.ell_e**2))**self.alpha

class dRQP_delle(RQP):
    """
        Log-derivative in order to ell_e
    """
    def __init__(self, amplitude, alpha, ell_e, P, ell_p):
        super(dRQP_damplitude, self).__init__(amplitude, alpha, ell_e, P, ell_p)
        self.amplitude = amplitude
        self.alpha = alpha
        self.RQP_ell_e = ell_e
        self.P = P
        self.ell_p = ell_p

    def __call__(self, r):
        return (r**2*(1+r**2/(2*self.alpha*self.ell_e**2))**(-1-self.alpha) \
                *self.amplitude**2 \
                *exp(-2*sine(pi*np.abs(r)/self.P)**2/self.ell_p**2))/self.ell_e**2

class dRQP_dP(RQP):
    """
        Log-derivative in order to P
    """
    def __init__(self, amplitude, alpha, ell_e, P, ell_p):
        super(dRQP_damplitude, self).__init__(amplitude, alpha, ell_e, P, ell_p)
        self.amplitude = amplitude
        self.alpha = alpha
        self.RQP_ell_e = ell_e
        self.P = P
        self.ell_p = ell_p

    def __call__(self, r):
        return (4*pi*r*self.amplitude**2*cosine(pi*np.abs(r)/self.P) \
                *sine(pi*np.abs(r)/self.P) \
                *exp(-2*sine(pi*np.abs(r)/self.P)**2/self.ell_p**2)) \
                /(self.ell_p**2*(1+r**2/(2*self.alpha*self.ell_e**2))**self.alpha*self.P)

class dRQP_dellp(RQP):
    """
        Log-derivative in order to ell_p
    """
    def __init__(self, amplitude, alpha, ell_e, P, ell_p):
        super(dRQP_damplitude, self).__init__(amplitude, alpha, ell_e, P, ell_p)
        self.amplitude = amplitude
        self.alpha = alpha
        self.RQP_ell_e = ell_e
        self.P = P
        self.ell_p = ell_p

    def __call(self, r):
        return (4*self.amplitude**2*sine(pi*np.abs(r)/self.P)**2 \
                *exp(-2*sine(pi*np.abs(r)/self.P)**2/self.ell_p**2)) \
                /(self.ell_p**2*(1+r**2/(2*self.alpha*self.ell_e**2))**self.alpha)


##### Cosine ###################################################################
class Cosine(kernel):
    """
        Definition of the cosine kernel.
        Parameters:
            amplitude = amplitude/amplitude of the kernel
            P = period
    """
    def __init__(self, amplitude, P):
        super(Cosine, self).__init__(amplitude, P)
        self.amplitude = amplitude
        self.P = P
        self.type = 'non-stationary and isotropic'
        self.derivatives = 2    #number of derivatives in this kernel
        self.params_number = 2  #number of hyperparameters

    def __call__(self, r):
        return self.amplitude**2 * cosine(2*pi*np.abs(r) / self.P)

class dCosine_damplitude(Cosine):
    """
        Log-derivative in order to the amplitude
    """
    def __init__(self, amplitude, P):
        super(dCosine_damplitude, self).__init__(amplitude, P)
        self.amplitude = amplitude
        self.P = P

    def __call__(self, r):
        return 2*self.amplitude**2 * cosine(2*pi*np.abs(r) / self.P)

class dCosine_dP(Cosine):
    """
        Log-derivative in order to P
    """
    def __init__(self, amplitude, P):
        super(dCosine_dP, self).__init__(amplitude, P)
        self.amplitude = amplitude
        self.P = P

    def __call__(self, r):
        return self.amplitude**2 * r*pi*sine(2*pi*np.abs(r) / self.P) / self.P


##### Exponential ##############################################################
class Exponential(kernel):
    """
        Definition of the exponential kernel. This kernel arises when 
    setting v=1/2 in the matern family of kernels
        Parameters:
            amplitude = amplitude/amplitude of the kernel
            ell = characteristic lenght scale
    """
    def __init__(self, amplitude, ell):
        super(Exponential, self).__init__(amplitude, ell)
        self.amplitude = amplitude
        self.ell = ell
        self.type = 'stationary and isotropic'
        self.derivatives = 2    #number of derivatives in this kernel
        self.params_number = 2  #number of hyperparameters

    def __call__(self, r): 
        return self.amplitude**2 * exp(- np.abs(r)/self.ell)

class dExponential_damplitude(Exponential):
    """
        Log-derivative in order to the amplitude
    """
    def __init__(self, amplitude, ell):
        super(dExponential_damplitude, self).__init__(amplitude, ell)
        self.amplitude = amplitude
        self.ell = ell

    def __call__(self, r):
        return 2*self.amplitude**2 * exp(- np.abs(r)/self.ell)

class dExpoential_dell(Exponential):
    """
        Log-derivative in order to ell
    """
    def __init__(self, amplitude, ell):
        super(dExpoential_dell, self).__init__(amplitude, ell)
        self.amplitude = amplitude
        self.ell = ell

    def __call__(self, r):
        return -0.5*self.amplitude**2 * r *exp(- np.abs(r)/self.ell) /self.ell


##### Matern 3/2 ###############################################################
class Matern32(kernel):
    """
        Definition of the Matern 3/2 kernel. This kernel arise when setting 
    v=3/2 in the matern family of kernels
        Parameters:
            amplitude = amplitude/amplitude of the kernel
            theta = amplitude of the kernel
            ell = characteristic lenght scale
    """
    def __init__(self, amplitude, ell):
        super(Matern32, self).__init__(amplitude, ell)
        self.amplitude = amplitude
        self.ell = ell
        self.type = 'stationary and isotropic'
        self.derivatives = 2    #number of derivatives in this kernel
        self.params_number = 2  #number of hyperparameters

    def __call__(self, r):
        return self.amplitude**2 *(1 + np.sqrt(3)*np.abs(r)/self.ell) \
                    *np.exp(-np.sqrt(3)*np.abs(r) / self.ell)

class dMatern32_damplitude(Matern32):
    """
        Log-derivative in order to the amplitude
    """
    def __init__(self, amplitude, ell):
        super(dMatern32_damplitude, self).__init__(amplitude, ell)
        self.amplitude = amplitude
        self.ell = ell

    def __call__(self, r):
        return 2*self.amplitude**2 *(1 + np.sqrt(3)*np.abs(r)/self.ell) \
                    *np.exp(-np.sqrt(3)*np.abs(r) / self.ell)

class dMatern32_dell(Matern32):
    """
        Log-derivative in order to ell
    """
    def __init__(self, amplitude, ell):
        super(dMatern32_dell, self).__init__(amplitude, ell)
        self.amplitude = amplitude
        self.ell = ell

    def __call__(self, r):
        return (sqrt(3) * r * (1+ (sqrt(3) * r) / self.ell) \
                *exp(-(sqrt(3)*r) / self.ell) * self.amplitude**2) / self.ell \
                -(sqrt(3) * r * exp(-(sqrt(3)*r) / self.ell)*self.amplitude**2)/self.ell


#### Matern 5/2 ################################################################
class Matern52(kernel):
    """
        Definition of the Matern 5/2 kernel. This kernel arise when setting 
    v=5/2 in the matern family of kernels
        Parameters:
            amplitude = amplitude/amplitude of the kernel
            theta = amplitude of the kernel
            ell = characteristic lenght scale  
    """
    def __init__(self, amplitude, ell):
        super(Matern52, self).__init__(amplitude, ell)
        self.amplitude = amplitude
        self.ell = ell
        self.type = 'stationary and isotropic'
        self.derivatives = 2    #number of derivatives in this kernel
        self.params_number = 2    #number of hyperparameters

    def __call__(self, r):
        return self.amplitude**2 * (1 + (3*np.sqrt(5)*self.ell*np.abs(r) \
                                           +5*np.abs(r)**2)/(3*self.ell**2) ) \
                                          *exp(-np.sqrt(5.0)*np.abs(r)/self.ell)

class dMatern52_damplitude(Matern52):
    """
        Log-derivative in order to the amplitude
    """
    def __init__(self, amplitude, ell):
        super(dMatern52_damplitude, self).__init__(amplitude, ell)
        self.amplitude = amplitude
        self.ell = ell

    def __call__(self, r):
        return 2*self.amplitude**2 * (1 + ( 3*np.sqrt(5)*self.ell*np.abs(r) \
                                           +5*np.abs(r)**2)/(3*self.ell**2) ) \
                                          *exp(-np.sqrt(5.0)*np.abs(r)/self.ell)

class dMatern52_dell(Matern52):
    """
        Log-derivative in order to ell
    """
    def __init__(self, amplitude, ell):
        super(dMatern52_dell, self).__init__(amplitude, ell)
        self.amplitude = amplitude
        self.ell = ell

    def __call__(self, r):
        return self.ell * ((sqrt(5)*r*(1+(sqrt(5)*r) \
                                 /self.ell+(5*r**2)/(3*self.ell**2)) \
                             *exp(-(sqrt(5)*r)/self.ell)*self.amplitude**2) \
            /self.ell**2 +(-(sqrt(5)*r)/self.ell**2-(10*r**2) \
                           /(3*self.ell**3)) \
                           *exp(-(sqrt(5)*r)/self.ell)*self.amplitude**2)


### END

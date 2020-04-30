#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from functools import wraps

__all__ = ['Constant', 'Linear', 'Parabola', 'Cubic', 'Keplerian', 'UdHO']

def array_input(f):
    """
        decorator to provide the __call__ methods with an array
    """
    @wraps(f)
    def wrapped(self, t):
        t = np.atleast_1d(t)
        r = f(self, t)
        return r
    return wrapped


class MeanModel(object):
    """
        Definition of the mean funtion that will be used.
    """
    
    _parsize = 0
    def __init__(self, *pars):
        self.pars = list(pars)
        #self.pars = np.array(pars, dtype=float)

    def __repr__(self):
        """ Representation of each instance """
        return "{0}({1})".format(self.__class__.__name__,
                                 ", ".join(map(str, self.pars)))

    @classmethod
    def initialize(cls):
        """ 
            Initialize instance, setting all parameters to 0.
        """
        return cls( *([0.]*cls._parsize) )

    def __add__(self, b):
        return Sum(self, b)
    def __radd__(self, b):
        return self.__add__(b)

    def __mul__(self, b):
        return Multiplication(self, b)
    def __rmul__(self, b):
        return self.__mul__(b)


class Sum(MeanModel):
    """
        Sum of two mean functions.
    """
    def __init__(self, m1, m2):
        self.m1, self.m2 = m1, m2

    @property
    def _parsize(self):
        return self.m1._parsize + self.m2._parsize

    @property
    def pars(self):
        return self.m1.pars + self.m2.pars

    def initialize(self):
        return

    def __repr__(self):
        return "{0} + {1}".format(self.m1, self.m2)

    @array_input
    def __call__(self, t):
        return self.m1(t) + self.m2(t)


class Multiplication(MeanModel):
    """
        Product of two mean functions. Not sure if we will need it...
    """
    def __init__(self, m1, m2):
        self.m1, self.m2 = m1, m2

    @property
    def _parsize(self):
        return self.m1._parsize * self.m2._parsize

    @property
    def pars(self):
        return self.m1.pars * self.m2.pars

    def initialize(self):
        return

    def __repr__(self):
        return "{0} * {1}".format(self.m1, self.m2)

    @array_input
    def __call__(self, t):
        return self.m1(t) * self.m2(t)


##### Constant mean ############################################################
class Constant(MeanModel):
    """ 
        A constant offset mean function
    """
    _parsize = 1
    def __init__(self, c):
        super(Constant, self).__init__(c)

    @array_input
    def __call__(self, t):
        return np.full(t.shape, self.pars[0])


##### Linear mean ##############################################################
class Linear(MeanModel):
    """ 
        A linear mean function
        m(t) = slope * t + intercept 
    """
    _parsize = 2
    def __init__(self, slope, intercept):
        super(Linear, self).__init__(slope, intercept)

    @array_input
    def __call__(self, t):
        tmean = t.mean()
        return self.pars[0] * (t-tmean) + self.pars[1]


##### Parabolic mean ###########################################################
class Parabola(MeanModel):
    """ 
        A 2nd degree polynomial mean function
        m(t) = quad * t**2 + slope * t + intercept 
    """
    _parsize = 3
    def __init__(self, quad, slope, intercept):
        super(Parabola, self).__init__(quad, slope, intercept)

    @array_input
    def __call__(self, t):
        return np.polyval(self.pars, t)


##### Cubic mean ###############################################################
class Cubic(MeanModel):
    """ 
        A 3rd degree polynomial mean function
        m(t) = cub * t**3 + quad * t**2 + slope * t + intercept 
    """
    _parsize = 4
    def __init__(self, cub, quad, slope, intercept):
        super(Cubic, self).__init__(cub, quad, slope, intercept)

    @array_input
    def __call__(self, t):
        return np.polyval(self.pars, t)


##### Sinusoidal means #########################################################
class Sine(MeanModel):
    """ 
        A sinusoidal mean function
        m(t) = amplitude**2 * sine( (2*pi*t/P) + phase) + displacement
    """
    _parsize = 3
    def __init__(self, amp, P, phi, D):
        super(Sine, self).__init__(amp, P, phi, D)

    @array_input
    def __call__(self, t):
        return self.pars[0] * np.sin((2*np.pi*t/self.pars[1]) + self.pars[2]) \
                + self.pars[3]

class Cosine(MeanModel):
    """ 
        Another sinusoidal mean function
        m(t) = amplitude**2 * cosine( (2*pi*t/P) + phase) + displacement
    """
    _parsize = 3
    def __init__(self, amp, P, phi, D):
        super(Cosine, self).__init__(amp, P, phi, D)

    @array_input
    def __call__(self, t):
        return self.pars[0]**2 * np.cos((2*np.pi*t/self.pars[1]) + self.pars[2]) \
                + self.pars[3]


##### Keplerian mean ###########################################################
class oldKeplerian(MeanModel):
    """
        Keplerian function with T0
        tan[phi(t) / 2 ] = sqrt(1+e / 1-e) * tan[E(t) / 2] = true anomaly
        E(t) - e*sin[E(t)] = M(t) = eccentric anomaly
        M(t) = (2*pi*t/tau) + M0 = Mean anomaly
        p  = period in days
        k = RV amplitude in m/s 
        e = eccentricity
        w = longitude of the periastron
        T0 = time of periastron passage

        RV = K[cos(w+v) + e*cos(w)]
    """
    _parsize = 5
    def __init__(self, p, k, e, w, T0):
        super(oldKeplerian, self).__init__(p, k, e, w, T0)

    @array_input
    def __call__(self, t):
        p, k, e, w, T0 = self.pars
        #mean anomaly
        Mean_anom = 2*np.pi*(t-T0)/p
        #eccentric anomaly -> E0=M + e*sin(M) + 0.5*(e**2)*sin(2*M)
        E0 = Mean_anom + e*np.sin(Mean_anom) + 0.5*(e**2)*np.sin(2*Mean_anom)
        #mean anomaly -> M0=E0 - e*sin(E0)
        M0 = E0 - e*np.sin(E0)
        niter=0
        while niter < 500:
            aux = Mean_anom - M0
            E1 = E0 + aux/(1 - e*np.cos(E0))
            M1 = E0 - e*np.sin(E0)
            niter += 1
            E0 = E1
            M0 = M1
        nu = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E0/2))
        RV = k*(e*np.cos(w)+np.cos(w+nu))
        return RV


##### Keplerian mean ###########################################################
class Keplerian(MeanModel):
    """
        Keplerian function with phi
        tan[phi(t) / 2 ] = sqrt(1+e / 1-e) * tan[E(t) / 2] = true anomaly
        E(t) - e*sin[E(t)] = M(t) = eccentric anomaly
        M(t) = (2*pi*t/tau) + M0 = Mean anomaly
        p  = period in days
        k = RV amplitude in m/s 
        e = eccentricity
        w = longitude of the periastron
        phi = orbital phase

        RV = K[cos(w+v) + e*cos(w)]
    """
    _parsize = 5
    def __init__(self, p, k, e, w, phi):
        super(Keplerian, self).__init__(p, k, e, w, phi)

    @array_input
    def __call__(self, t):
        p, k, e, w, phi = self.pars
        #mean anomaly
        T0 = t[0] - (p*phi)/(2.*np.pi)
        Mean_anom = 2*np.pi*(t-T0)/p
        #eccentric anomaly -> E0=M + e*sin(M) + 0.5*(e**2)*sin(2*M)
        E0 = Mean_anom + e*np.sin(Mean_anom) + 0.5*(e**2)*np.sin(2*Mean_anom)
        #mean anomaly -> M0=E0 - e*sin(E0)
        M0 = E0 - e*np.sin(E0)
        niter=0
        while niter < 500:
            aux = Mean_anom - M0
            E1 = E0 + aux/(1 - e*np.cos(E0))
            M1 = E0 - e*np.sin(E0)
            niter += 1
            E0 = E1
            M0 = M1
        nu = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E0/2))
        RV = k*(e*np.cos(w)+np.cos(w+nu))
        return RV


##### Underdamped harmonic oscillator mean #####################################
class UdHO(MeanModel):
    """
        Underdamped harmonic oscillator mean function
        m(t) = A * exp(-b*t) * cos(w*t + phi)
        A = kinda of an amplitude
        b = damping coefficient
        w = kinda of an angular frequency, w**2 = sqrt(w0**2 - b**2), where w0
            is the angular frequency
        phi = phase, determines the starting point of the "wave"
    """
    _parsize = 4
    def __init__(self, A, b, w, phi):
        super(UdHO, self).__init__(A, b, w, phi)

    @array_input
    def __call__(self, t):
        return self.pars[0]**2 * np.exp(-self.pars[1]*t) \
                    * np.cos(self.pars[2]*t + self.pars[3])


##### END

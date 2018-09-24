#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from functools import wraps

__all__ = ['Constant', 'Linear', 'Parabola', 'Cubic', 'Keplerian']

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


class Sum(MeanModel):
    """
        Sum of two mean functions. Not sure if we will need it...
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
        return self.pars[0] * t + self.pars[1]


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


##### Sinusoidal mean ##########################################################
class Sine(MeanModel):
    """ 
        A sinusoidal mean function
        m(t) = amplitude * sine(ang_freq * t + phase)
    """
    _parsize = 3
    def __init__(self, amp, w, phi):
        super(Sine, self).__init__(amp, w, phi)

    @array_input
    def __call__(self, t):
        return self.pars[0] * np.sin(self.pars[1]*t + self.pars[2])


##### Keplerian mean ###########################################################
class Keplerian(MeanModel):
    """
        Keplerian function
        tan[phi(t) / 2 ] = sqrt(1+e / 1-e) * tan[E(t) / 2] = true anomaly
        E(t) - e*sin[E(t)] = M(t) = eccentric anomaly
        M(t) = (2*pi*t/tau) + M0 = Mean anomaly
        P  = period in days
        e = eccentricity
        K = RV amplitude in m/s 
        w = longitude of the periastron
        T0 = time of periastron passage

        RV = K[cos(w+v) + e*cos(w)] + sis_vel
    """
    _parsize = 5
    def __init__(self, P, K, e, w, T0):
        super(Keplerian, self).__init__(P, K, e, w, T0)

    @array_input
    def __call__(self, t):
        P, K, e, w, T0 = self.pars
        #mean anomaly
        Mean_anom = 2*np.pi*(t-T0)/P
        #eccentric anomaly -> E0=M + e*sin(M) + 0.5*(e**2)*sin(2*M)
        E0 = Mean_anom + e*np.sin(Mean_anom) + 0.5*(e**2)*np.sin(2*Mean_anom)
        #mean anomaly -> M0=E0 - e*sin(E0)
        M0 = E0 - e*np.sin(E0)

        niter=0
        while niter < 1000:
            aux = Mean_anom - M0
            E1 = E0 + aux/(1 - e*np.cos(E0))
            M1 = E0 - e*np.sin(E0)

            niter += 1
            E0 = E1
            M0 = M1

        nu = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E0/2))
        RV = K*(e*np.cos(w)+np.cos(w+nu))
        return RV


### END

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from functools import wraps

__all__ = ['Constant', 'Linear', 'Parabola', 'Cubic', 'Keplerian', 'UdHO']

def array_input(f):
    """ decorator to provide the __call__ methods with an array """
    @wraps(f)
    def wrapped(self, t):
        t = np.atleast_1d(t)
        r = f(self, t)
        return r
    return wrapped


class MeanModel(object):
    """ Definition of the mean funtion that will be used """
    
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
        """ Initialize instance, setting all parameters to 0 """
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
    """ Sum of two mean functions """
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
    """ Product of two mean functions. Not sure if we will need it... """
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
    """ A constant offset mean function """
    _parsize = 1
    def __init__(self, c):
        super(Constant, self).__init__(c)

    @array_input
    def __call__(self, t):
        return np.full(t.shape, self.pars[0])


##### Linear mean ##############################################################
class Linear(MeanModel):
    """ Linear mean; m(t) = slope * t + intercept """
    _parsize = 2
    def __init__(self, slope, intercept):
        super(Linear, self).__init__(slope, intercept)

    @array_input
    def __call__(self, t):
        tmean = t.mean()
        return self.pars[0] * (t-tmean) + self.pars[1]


##### Parabolic mean ###########################################################
class Parabola(MeanModel):
    """ 2nd degree polynomial mean; m(t) = quad * t**2 + slope * t + intercept """
    _parsize = 3
    def __init__(self, quad, slope, intercept):
        super(Parabola, self).__init__(quad, slope, intercept)

    @array_input
    def __call__(self, t):
        return np.polyval(self.pars, t)


##### Cubic mean ###############################################################
class Cubic(MeanModel):
    """ 3rd degree polynomial mean; m(t) = cub*t**3 + quad*t**2 + slope*t + intercept """
    _parsize = 4
    def __init__(self, cub, quad, slope, intercept):
        super(Cubic, self).__init__(cub, quad, slope, intercept)

    @array_input
    def __call__(self, t):
        return np.polyval(self.pars, t)


##### Sinusoidal means #########################################################
class Sine(MeanModel):
    """ Sinusoidal mean; m(t) = amplitude**2*sine((2*pi*t/P)+phase) + displacement """
    _parsize = 3
    def __init__(self, amp, P, phi, D):
        super(Sine, self).__init__(amp, P, phi, D)

    @array_input
    def __call__(self, t):
        return self.pars[0] * np.sin((2*np.pi*t/self.pars[1]) + self.pars[2]) \
                + self.pars[3]

class Cosine(MeanModel):
    """ Cosine mean; m(t) = amplitude**2*cosine((2*pi*t/P)+phase) + displacement """
    _parsize = 3
    def __init__(self, amp, P, phi, D):
        super(Cosine, self).__init__(amp, P, phi, D)

    @array_input
    def __call__(self, t):
        return self.pars[0]**2 * np.cos((2*np.pi*t/self.pars[1]) + self.pars[2]) \
                + self.pars[3]


##### Keplerian mean ###########################################################
class Keplerian(MeanModel):
    """
    Keplerian mean function
    tan[phi(t) / 2 ] = sqrt(1+e / 1-e) * tan[E(t) / 2] = true anomaly
    E(t) - e*sin[E(t)] = M(t) = eccentric anomaly
    M(t) = (2*pi*t/tau) + M0 = Mean anomaly
    p  = period in days
    k = RV amplitude in m/s 
    e = eccentricity
    w = longitude of the periastron
    phi = orbital phase
    RV = K[cos(w+v) + e*cos(w)] + C
    """
    _parsize = 5
    def __init__(self, p, k, e, w, phi, offset):
        super(Keplerian, self).__init__(p, k, e, w, phi, offset)

    @array_input
    def __call__(self, t):
        p, k, e, w, phi, offset = self.pars
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
        RV = k*(e*np.cos(w)+np.cos(w+nu)) + offset
        return RV


##### Underdamped harmonic oscillator mean #####################################
class UdHO(MeanModel):
    """
    Underdamped harmonic oscillator mean; m(t) = A*exp(-b*t)*cos(w*t+phi)
        
    Parameters
    ----------
    A: float
        Kinda of an amplitude
    b: flaot
        Damping coefficient
    w: float
        Kinda of an angular frequency, w**2 = sqrt(w0**2 - b**2), where w0 is 
        the angular frequency
    phi: float
        Phase, determines the starting point of the "wave"
    """
    _parsize = 4
    def __init__(self, A, b, w, phi):
        super(UdHO, self).__init__(A, b, w, phi)

    @array_input
    def __call__(self, t):
        return self.pars[0]**2 * np.exp(-self.pars[1]*t) \
                    * np.cos(self.pars[2]*t + self.pars[3])


class TOI175keplerians(MeanModel):
    """ Keplerian function for TOI-175 """
    _parsize = 16
    def __init__(self, Pb,Kb,eb,wb,phib, Pc,Kc,ec,wc,phic, Pd,Kd,ed,wd,phid, C):
        super(TOI175keplerians, self).__init__(Pb,Kb,eb,wb,phib, 
                                               Pc,Kc,ec,wc,phic, 
                                               Pd,Kd,ed,wd,phid, C)

    @array_input
    def __call__(self, t):
        Pb,Kb,eb,wb,phib, Pc,Kc,ec,wc,phic, Pd,Kd,ed,wd,phid, C = self.pars
        #mean anomaly
        T0 = t[0] - (Pb*phib)/(2.*np.pi)
        Mean_anom = 2*np.pi*(t-T0)/Pb
        #eccentric anomaly -> E0=M + e*sin(M) + 0.5*(e**2)*sin(2*M)
        E0 = Mean_anom + eb*np.sin(Mean_anom) + 0.5*(eb**2)*np.sin(2*Mean_anom)
        #mean anomaly -> M0=E0 - e*sin(E0)
        M0 = E0 - eb*np.sin(E0)
        niter=0
        while niter < 500:
            aux = Mean_anom - M0
            E1 = E0 + aux/(1 - eb*np.cos(E0))
            M1 = E0 - eb*np.sin(E0)
            niter += 1
            E0 = E1
            M0 = M1
        nu = 2*np.arctan(np.sqrt((1+eb)/(1-eb))*np.tan(E0/2))
        RVb = Kb*(eb*np.cos(wb)+np.cos(wb+nu))

        #mean anomaly
        T0c = t[0] - (Pc*phic)/(2.*np.pi)
        Mean_anomc = 2*np.pi*(t-T0c)/Pc
        #eccentric anomaly -> E0=M + e*sin(M) + 0.5*(e**2)*sin(2*M)
        E0c = Mean_anomc + ec*np.sin(Mean_anomc) + 0.5*(ec**2)*np.sin(2*Mean_anomc)
        #mean anomaly -> M0=E0 - e*sin(E0)
        M0c = E0c - ec*np.sin(E0c)
        niter=0
        while niter < 100:
            aux = Mean_anomc - M0c
            E1c = E0c + aux/(1 - ec*np.cos(E0c))
            M1c = E0c - ec*np.sin(E0c)
            niter += 1
            E0c = E1c
            M0c = M1c
        nu = 2*np.arctan(np.sqrt((1+ec)/(1-ec))*np.tan(E0c/2))
        RVc = Kc*(ec*np.cos(wc)+np.cos(wc+nu))

        #mean anomaly
        T0d = t[0] - (Pd*phid)/(2.*np.pi)
        Mean_anomd = 2*np.pi*(t-T0d)/Pd
        #eccentric anomaly -> E0=M + e*sin(M) + 0.5*(e**2)*sin(2*M)
        E0d = Mean_anomd + ed*np.sin(Mean_anomd) + 0.5*(ed**2)*np.sin(2*Mean_anomd)
        #mean anomaly -> M0=E0 - e*sin(E0)
        M0d = E0d - ed*np.sin(E0d)
        niter=0
        while niter < 100:
            aux = Mean_anomd - M0d
            E1d = E0d + aux/(1 - ed*np.cos(E0d))
            M1d = E0d - ed*np.sin(E0d)
            niter += 1
            E0d = E1d
            M0d = M1d
        nu = 2*np.arctan(np.sqrt((1+ed)/(1-ed))*np.tan(E0d/2))
        RVd = Kd*(ed*np.cos(wd)+np.cos(wd+nu))
        offset = np.full(t.shape, self.pars[-1])
        finalRV = RVb + RVc + RVd + offset
        return finalRV


# -*- coding: utf-8 -*-
import  numpy as _np

##### Semi amplitude calculation ###############################################
def semi_amplitude(period, Mplanet, Mstar, ecc):
    """
        Calculates the semi-amplitude (K) caused by a planet with a given
    perion and mass Mplanet with a eccentricity ecc.
    
        Parameters:
    period in years
    Mplanet in Jupiter masses, tecnically is the M.sin i
    Mstar in Solar masses
    ecc is the eccentricity

        Returns:
    Semi-amplitude K
    """
    
    per = _np.power(1/period, 1/3)
    Pmass = Mplanet / 1
    Smass = _np.power(1/Mstar, 2/3)
    Ecc = 1 / _np.sqrt(1 - ecc**2)

    return 28.435 * per * Pmass* Smass * Ecc


##### Keplerian function #######################################################
def kepler(P=365, K=.1, e=0,  w=_np.pi, T=0, phi=None, gamma=0, t=None):
    """
        kepler() simulates the radial velocity signal of a planet in a 
    keplerian orbit around a star.

        Parameters:
    P = period in days
    K = RV amplitude
    e = eccentricity
    w = longitude of the periastron
    T = zero phase
    phi = orbital phase
    gamma = constant system RV
    t = time of measurements

        Returns:
    t = time of measurements
    RV = rv signal generated
    """
    if t is  None:
    	print('Time needed')

    #mean anomaly
    if phi is None:
        mean_anom = [2*_np.pi*(x1-T)/P  for x1 in t]
    else:
        T = t[0] - (P*phi)/(2.*_np.pi)
        mean_anom = [2*_np.pi*(x1-T)/P  for x1 in t]

    #eccentric anomaly -> E0=M + e*sin(M) + 0.5*(e**2)*sin(2*M)
    E0 = [x + e*_np.sin(x)  + 0.5*(e**2)*_np.sin(2*x) for x in mean_anom]
    #mean anomaly -> M0=E0 - e*sin(E0)
    M0 = [x - e*_np.sin(x) for x in E0]

    i = 0
    while i < 1000:
        #[x + y for x, y in zip(first, second)]
        calc_aux = [x2-y for x2,y in zip(mean_anom, M0)]    
        E1 = [x3 + y/(1-e*_np.cos(x3)) for x3,y in zip(E0, calc_aux)]
        M1 = [x4 - e*_np.sin(x4) for x4 in E0]   
        i += 1
        E0 = E1
        M0 = M1

    nu = [2*_np.arctan(_np.sqrt((1+e)/(1-e))*_np.tan(x5/2)) for x5 in E0]
    RV = [ gamma + K*(e*_np.cos(w)+_np.cos(w+x6)) for x6 in nu]
    RV = [x for x in RV] #m/s 
    return t, RV


##### Phase-folding function ###################################################
def phase_folding(t, y, yerr, period):
    """
        phase_folding allows the phase folding (duh...) of a given data
    accordingly to a given period
        Parameters:
            t = time
            y = measurements
            yerr = measurement errors
            period = period to fold the data
        Returns:
            phase = phase
            folded_y = sorted measurments according to the phase
            folded_yerr = sorted errors according to the phase
    """
    #divide the time by the period to convert to phase
    foldtimes = t / period
    #remove the whole number part of the phase
    foldtimes = foldtimes % 1
    #sort everything
    phase, folded_y, folded_yerr = zip(*sorted(zip(foldtimes, y, yerr)))
    return phase, folded_y, folded_yerr


### END

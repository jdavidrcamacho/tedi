"""
A collection of useful functions
"""
from scipy.stats import invgamma
from scipy.optimize import minimize

import  numpy as np

##### Semi amplitude calculation ###############################################
def semi_amplitude(period, Mplanet, Mstar, ecc):
    """
    Calculates the semi-amplitude (K) caused by a planet with a given period 
    and mass Mplanet, around a star of mass Mstar, with a eccentricity ecc.
    
    Parameters
    ----------
    period: float
        Period in years
    Mplanet: float
        Planet's mass in Jupiter masses, tecnically is the M.sin i
    Mstar: float
        Star mass in Solar masses
    ecc: float
        Eccentricity
    
    Returns
    -------
    : float
        Semi-amplitude K
    """
    per = np.float(np.power(1/period, 1/3))
    Pmass = Mplanet / 1
    Smass = np.float(np.power(1/Mstar, 2/3))
    Ecc = 1 / np.sqrt(1 - ecc**2)
    return 28.435 * per * Pmass* Smass * Ecc


##### Keplerian function #######################################################
def keplerian(P=365, K=.1, e=0,  w=np.pi, T=0, phi=None, gamma=0, t=None):
    """
    keplerian() simulates the radial velocity signal of a planet in a 
    keplerian orbit around a star.
    
    Parameters
    ----------
    P: float
        Period in days
    K: float
        RV amplitude
    e: float
        Eccentricity
    w: float
        Longitude of the periastron
    T: float
        Zero phase
    phi: float
        Orbital phase
    gamma: float
        Constant system RV
    t: array
        Time of measurements
    
    Returns
    -------
    t: array
        Time of measurements
    RV: array
        RV signal generated
    """
    if t is None:
        print('\n TEMPORAL ERROR, time is nowhere to be found \n')
        return 0, 0
    #mean anomaly
    if phi is None:
        mean_anom = [2*np.pi*(x1-T)/P  for x1 in t]
    else:
        T = t[0] - (P*phi)/(2.*np.pi)
        mean_anom = [2*np.pi*(x1-T)/P  for x1 in t]
    #eccentric anomaly -> E0=M + e*sin(M) + 0.5*(e**2)*sin(2*M)
    E0 = [x + e*np.sin(x)  + 0.5*(e**2)*np.sin(2*x) for x in mean_anom]
    #mean anomaly -> M0=E0 - e*sin(E0)
    M0 = [x - e*np.sin(x) for x in E0]
    i = 0
    while i < 1000:
        #[x + y for x, y in zip(first, second)]
        calc_aux = [x2-y for x2,y in zip(mean_anom, M0)]    
        E1 = [x3 + y/(1-e*np.cos(x3)) for x3,y in zip(E0, calc_aux)]
        M1 = [x4 - e*np.sin(x4) for x4 in E0]   
        i += 1
        E0 = E1
        M0 = M1
    nu = [2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(x5/2)) for x5 in E0]
    RV = [ gamma + K*(e*np.cos(w)+np.cos(w+x6)) for x6 in nu]
    RV = [x for x in RV] #m/s 
    return t, RV


##### Phase-folding function ###################################################
def phase_folding(t, y, yerr, period):
    """
    phase_folding() allows the phase folding (duh...) of a given data
    accordingly to a given period
    
    Parameters
    ----------
    t: array
        Time array
    y: array
        Measurements array
    yerr: array
        Measurement errors arrays
    period: float
        Period to fold the data
    
    Returns
    -------
    phase: array
        Phase
    folded_y: array
        Sorted measurments according to the phase
    folded_yerr: array
        Sorted errors according to the phase
    """
    #divide the time by the period to convert to phase
    foldtimes = t / period
    #remove the whole number part of the phase
    foldtimes = foldtimes % 1
    if yerr is None:
        yerr = 0 * y
    #sort everything
    phase, folded_y, folded_yerr = zip(*sorted(zip(foldtimes, y, yerr)))
    return phase, folded_y, folded_yerr


##### MCMC with dynesty or emcee ###############################################
import dynesty, emcee
from multiprocessing import Pool

def run_mcmc(prior_func, loglike_func, iterations = 1000, sampler = 'emcee'):
    """
    run_mcmc() allow the user to run emcee or dynesty automatically
    
    Parameters
    ----------
    prior_func: func
        Function that return an array with the priors
    loglike_func: func
        Function that calculates the log-likelihood 
    iterations: int
        Number of iterations; in emcee half of it will be used as burn-in
    sampler: string
        'emcee' or 'dynesty'
        
    Returns
    -------
    result: ?
        Sampler's results accordingly to the sampler
    """
    if sampler == 'emcee':
        ndim = prior_func().size
        burns, runs = int(iterations/2), int(iterations/2)
        #defining emcee properties
        nwalkers = 2*ndim
        sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                                        loglike_func, threads= 4)
        #Initialize the walkers
        p0=[prior_func() for i in range(nwalkers)]
        #running burns and runs
        print("Running burn-in")
        p0, _, _ = sampler.run_mcmc(p0, burns)
        print("Running production chain")
        sampler.run_mcmc(p0, runs)
        #preparing samples to return
        samples = sampler.chain[:, burns:, :].reshape((-1, ndim))
        lnprob = sampler.lnprobability[:, burns:].reshape(nwalkers*burns, 1)
        results = np.vstack([samples.T,np.array(lnprob).T]).T
    if sampler == 'dynesty':
        ndim = prior_func(0).size
        dsampler = dynesty.DynamicNestedSampler(loglike_func, prior_func, 
                                        ndim=ndim, nlive = 1000, sample='rwalk',
                                        queue_size=4, pool=Pool(4))
        dsampler.run_nested(nlive_init = 1000, maxiter = iterations)
        results = dsampler.results
    return results


##### inverse gamma distribution ###############################################
f = lambda x, lims: \
    (np.array([invgamma(a=x[0], scale=x[1]).cdf(lims[0]) - 0.01,
               invgamma(a=x[0], scale=x[1]).sf(lims[1]) - 0.01])**2).sum()

def invGamma(lower, upper, x0=[1, 5], showit=False):
    """
    Arguments
    ---------
    lower, upper : float
        The upper and lower limits between which we want 98% of the probability
    x0 : list, length 2
        Initial guesses for the parameters of the inverse gamma (a and scale)
    showit : bool
        Make a plot
    """
    limits = [lower, upper]
    result = minimize(f, x0=x0, args=limits, method='L-BFGS-B',
                      bounds=[(0, None), (0, None)], tol=1e-10)
    a, b = result.x
    if showit:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        d = invgamma(a=a, scale=b)
        x = np.linspace(0.2*limits[0], 2*limits[1], 1000)
        ax.plot(x, d.pdf(x))
        ax.vlines(limits, 0, d.pdf(x).max())
        plt.show()
    return invgamma(a=a, scale=b)

### END

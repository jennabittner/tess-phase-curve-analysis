# TESSPhaseCurve_lib.py
#
# Library of routines for analyzing TESS phase curves.

# Imports
import numpy as np
import scipy.stats
import astropy.units as u
import lightkurve as lk

### Lightcurve component models ###
# Model functions adopted from  Wong et al. 2020

## Stellar pulsation model ##
zeta = lambda t, t_0_pulse, PI: ((t - t_0_pulse) % PI)/PI
Theta = lambda t, t_0_pulse, PI, alpha, beta: 1 + alpha*np.sin(2*np.pi*zeta(t, t_0_pulse, PI)) + \
beta*np.cos(2*np.pi*zeta(t, t_0_pulse, PI))
def Theta_func(params, t):
    return Theta(t, *params)

## Planet phase curve ##
phi = lambda t, t_0, P: ((t - t_0) % P)/P
psi_p = lambda t, t_0, P, fp, B1, delta: fp + B1 * np.cos(2*np.pi*phi(t, t_0, P) + delta)
def psi_p_func(params, t):
    return psi_p(t, *params)

## Stellar harmonics ##
# k = 1 harmonic
psi_star_1 = lambda t, t_0, P, A1: A1*np.sin(2*np.pi*phi(t, t_0, P)) 
# k = 2 harmonic
psi_star_2 = lambda t, t_0, P, A2, B2: A2*np.sin(2*np.pi*2*phi(t, t_0, P)) + B2*np.cos(2*np.pi*2*phi(t, t_0, P)) 
# k = 3 harmonic
psi_star_3 = lambda t, t_0, P, A3, B3: A3*np.sin(2*np.pi*3*phi(t, t_0, P)) + B3*np.cos(2*np.pi*3*phi(t, t_0, P))
# Combined stellar harmonics
psi_star_sum = lambda t, t_0, P, A1, A2, B2, A3, B3: 1 + psi_star_1(t, t_0, P, A1) + psi_star_2(t, t_0, P, A2, B2) + \
psi_star_3(t, t_0, P, A3, B3)
def psi_star_func(params, t):
    return psi_star(t, *params)

## Total lightcurve ##
# With pulsations
psi_tot = lambda t, t_0, P, PI, alpha, beta, fp, delta, A1, B1, A2, B2, A3, B3: \
(psi_p(t, t_0, P, fp, B1, delta) + Theta(t, t_0, PI, alpha, beta) * psi_star_sum(t, t_0, P, A1, A2, B2, A3, B3))/(1. + fp)
def psi_tot_func(params, t):
    return psi_tot(t, *params)

# With no pulsations
psi_tot_no_pulse = lambda t, t_0, P, fp, delta, A1, B1, A2, B2, A3, B3: \
(psi_p(t, t_0, P, fp, B1, delta) + psi_star_sum(t, t_0, P, A1, A2, B2, A3, B3))/(1. + fp)
def psi_tot_func_no_pulse(params, t):
    return psi_tot_no_pulse(t, *params)

### Data transformations to isolate specific components ###
def lc_transform_planet(lc, t_0, P, PI, alpha, beta, fp, A1, A2, B2, A3, B3):
    ''' Uses model parameters to transform TESS lightcurve to isolate planet phase curve.
        
        Parameters
        -----------
        lc : Lightkurve object
            Cleaned and unfolded TESS lightcurve with planet phase curve, stellar harmonics, and stellar pulsations.
        t_0 : float Quantity in units of days
            Mid-transit time
        P : float Quantity in units of days
            Orbital period
        PI : float Quantity in units of days
            Stellar pulsation period
        alpha : float
            Stellar pulsation sine amplitude
        beta : float
            Stellar pulsation cosine amplitude
        fp : float
            Mean planet atmospheric brightness (normalized to stellar flux)
        A1 : float
            Doppler beaming amplitude
        A2 : float
            Stellar k=2 harmonic sine amplitude
        B2 : float
            Ellipsoidal variation amplitude
        A3 : float
            Stellar k=3 harmonic sine amplitude
        B3 : float
            Stellar k=3 harmonic cosine amplitude

        Returns
        -------
        lc_planet : Lightkurve object
            Planet phase curve component extracted from TESS lightcurve
    '''
    lc_planet = lc.copy()
    pulse = Theta(lc_planet.time.value, t_0, PI, alpha, beta)
    star = psi_star_sum(lc_planet.time.value, t_0, P, A1, A2, B2, A3, B3)
    lc_planet.flux = (1+fp) * lc_planet.flux - star * pulse
    lc_planet.flux_err = (1+fp) * lc_planet.flux_err
    return lc_planet

def lc_transform_star(lc, t_0, P, PI, alpha, beta, fp, B1, delta):
    ''' Uses model parameters to transform TESS lightcurve to isolate stellar harmonics.
        
        Parameters
        -----------
        lc : Lightkurve object
            Cleaned and unfolded TESS lightcurve with planet phase curve, stellar harmonics, and stellar pulsations.
        t_0 : float Quantity in units of days
            Mid-transit time
        P : float Quantity in units of days
            Orbital period
        PI : float Quantity in units of days
            Stellar pulsation period
        alpha : float
            Stellar pulsation sine amplitude
        beta : float
            Stellar pulsation cosine amplitude
        fp : float
            Mean planet atmospheric brightness (normalized to stellar flux)
        B1 : float
            Planet phase curve amplitude
        delta : float
            Planet phase curve offset

        Returns
        -------
        lc_star : Lightkurve object
            Stellar harmonics component extracted from TESS lightcurve
    '''
    lc_star = lc.copy()
    pulse = Theta(lc_star.time.value, t_0, PI, alpha, beta)
    planet = psi_p(lc_star.time.value, t_0, P, fp, B1, delta)
    lc_star.flux = ((1+fp) * lc_star.flux - planet) / pulse
    lc_star.flux_err = (1+fp) * lc_star.flux_err / pulse
    return lc_star


def lc_transform_pulse(lc, t_0, P, fp, delta, A1, A2, B1, B2, A3, B3):
    ''' Uses model parameters to transform TESS lightcurve to isolate planet phase curve.
        
        Parameters
        -----------
        lc : Lightkurve object
            Cleaned and unfolded TESS lightcurve with planet phase curve, stellar harmonics, and stellar pulsations.
        t_0 : float Quantity in units of days
            Mid-transit time
        P : float Quantity in units of days
            Orbital period
        fp : float
            Mean planet atmospheric brightness (normalized to stellar flux)
        delta : float
            Planet phase curve offset
        A1 : float
            Doppler beaming amplitude
        A2 : float
            Stellar k=2 harmonic sine amplitude
        B1 : float
            Planet phase curve amplitude
        B2 : float
            Ellipsoidal variation amplitude
        A3 : float
            Stellar k=3 harmonic sine amplitude
        B3 : float
            Stellar k=3 harmonic cosine amplitude

        Returns
        -------
        lc_pulse : Lightkurve object
            Stellar pulsation component extracted from TESS lightcurve
    '''
    lc_pulse = lc.copy()
    planet = psi_p(lc_pulse.time.value, t_0, P, fp, B1, delta)
    star = psi_star_sum(lc_pulse.time.value, t_0, P, A1, A2, B2, A3, B3)
    lc_pulse.flux = ((1+fp) * lc_pulse.flux - planet) / star
    lc_pulse.flux_err = (1+fp) * lc_pulse.flux_err / star
    return lc_pulse

### Lightkurve helper routines ###
def fold_lk(lc, P, epoch_time):
    ''' Phase-folds a Lightkurve object and wraps the phase between 0 (transit) to 1 (0.5 is secondary eclipse).

        Parameters
        ----------
        lc : Lightkurve object
            A Lightkurve object with time in units of days
        P : float with time units
            Period to fold over
        epoch_time : float with time units
            Mid-transit reference time

        Returns
        -------
        lc_fold : Lightkurve object
            Lightkurve object with orbital phase (unitless) given by time attribute
    '''
    lc_fold = lc.fold(P, epoch_time=epoch_time, wrap_phase=1*u.dimensionless_unscaled, 
                                        normalize_phase=True)
    ind_order = np.argsort(lc_fold.time)

    lc_dict = {'time': lc_fold.time[ind_order] * u.day,
               'flux': lc_fold.flux[ind_order],
               'flux_err': lc_fold.flux_err[ind_order]
              }
    lc_fold = lk.LightCurve(lc_dict)
    return lc_fold

def create_transit_mask(lc, period, transit_time, duration):
    '''Creates a mask to identify data points that fall within the transit or eclipse
    
    Parameters
    ----------
    lc : Lightkurve object
        A Lightkurve object with time in units of days
    period : float with time units
        Orbital period of planet
    transit_time : float with time units
        Mid-transit time
    duration : float with time units
        Duration of planetary transit
        
    Returns
    -------
    mask : numpy array
        Mask of points within the transit or eclipse
    '''
    
    # Converting observation times to phases relative to orbital period
    phase = ((lc.time.value*u.day - transit_time) % period) / period
    
    # Removing units
    phase = phase.value
    
    # Converting transit duration from units of days to units of phase
    duration_phase = (duration.to(u.day) / period.to(u.day)).value
    
    # Build of mask of True/False where True = during transit/eclipse
    mask = phase < duration_phase / 2  # Masking second half of transit
    mask += abs(phase - 0.5) < duration_phase / 2  # Masking secondary eclipse
    mask += phase > 1 - duration_phase / 2  # Masking first half of transit
    return mask

def segment_analysis(time_segment, flux_segment, P):
    '''
    Analyzes time segments between momentum dumps and assigns a 0th order polynomials for abnormally short lengths of data within 
    segments, otherwise assigns a 1st order polynomial if data is a good length
    
    Parameters
    ----------
    time_segment : array-like
        Array containing time values for the segment
    flux_segment : array-like
        Array containing flux values corresponding to time values in the segment
    P : float
        Orbital period of planet
        
    Returns
    -------
    normalization_deg : array-like
        An array containing the degree of polynomial
    '''
    # Create a mask for nan values in segment
    nanmask = np.isnan(flux_segment)
    timevalue = time_segment[~nanmask]
    
    # Check if length of time segment is less than 2*P
    if ( (max(timevalue) - min(timevalue)) ) < 2 * P.value:
        # Fit 0th order polynomial
        p = np.polyfit(time_segment, flux_segment, 0)
        normalization_deg = np.polyval(p, time_segment)
    else:
        # Fit 1st order polynomial
        p = np.polyfit(time_segment, flux_segment, 1)
        normalization_deg = np.polyval(p, time_segment)
    return normalization_deg

### Bayesian functions for parameter estimation with dynesty ###
def prior_transform(unif, priors, priors_bool):
    ''' Transforms the uniform random variable unif~[0, 1] to either a uniform prior over a different
        range of interest or a Gaussian prior

        Parameters
        ----------
        unif : array-like
            Array of uniform random variables for each parameter. Length of unif is equal to the number of model
            parameters.
        priors : 2D numpy array
            Pairs of values describing the priors for each model parameter. For uniform priors, the first value
            of the pair is the lower bound and the second value is the upper bound. For Gaussian priors, the
            first value of the pair is the central value of the Gaussian and the second value is the 1-sigma width.
        priors_bool : 1D numpy array
            True or False values ascribed to each model parameter. True if uniform prior, False if Gaussian prior.

        Returns
        -------
        x : array-like
            Prior transforms for each model parameter.
    '''

    x = np.array(unif)

    # Uniform priors
    prior_range = np.diff(priors[priors_bool], axis=1).flatten()/2.
    prior_center = np.mean(priors[priors_bool], axis=1)
    x[priors_bool] = 2. * unif[priors_bool] - 1.  # scale and shift to [-1., 1.)
    x[priors_bool] *= prior_range
    x[priors_bool] += prior_center

    # Bivariate Normal
    t = scipy.stats.norm.ppf(unif[~priors_bool])  # convert to standard normal
    Csqrt = priors[~priors_bool][:,1] * np.identity(np.sum(~priors_bool))
    x[~priors_bool] = np.dot(Csqrt, t)  # correlate with appropriate covariance
    mu = priors[~priors_bool][:,0]  # mean
    x[~priors_bool] += mu  # add mean
    return x

def loglike(theta, x, data, data_err, model_func, y=None, args=None):
    ''' Calculates chi-squared likelihood for parameter estimation with MCMC or nested sampling.

        Parameters
        ----------
        theta : 1-D numpy array
            Model parameters
        x : 1-D numpy array
            Data x coordinates
        data : 1-D/2-D numpy array
            Data values
        data_err : 1-D/2-D numpy array
            Errors associated with data
        model_func : function
            Model function
        y : 1-D numpy array
            Data y coordinates; None if data is 1-D
        args : 1-D numpy array
            Additional arguments taken in by model; None if model does not take in additional arguments

        Returns
        -------
        likelihood : float
            Likelihood probability

    '''
    if y is None:
        if args is None:
            model = model_func(theta, x)
        else:
            model = model_func(theta, x, *args)
    else:
        if args is None:
            model = model_func(theta, x, y)
        else:
            model = model_func(theta, x, y, *args)
    inv_sigma2 = 1.0 / (data_err**2)
    return -0.5 * (np.nansum((data-model)**2 * inv_sigma2 - np.log(inv_sigma2)))


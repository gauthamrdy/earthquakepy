import numpy as np

from scipy.integrate import cumtrapz
from scipy.integrate import trapz

def arias_intensity(ts, g=False):
    '''
    Computes arias intensity:

    Parameters
    ----------
    ts: Acceleration timeseries object

    g: Bool, optional
        g=True multiplies acceleration values with g=9.81 m/sec^2.
        Used when acceleration values in 'g' units are to be converted into 'm/sec^2'
    Returns
    -------
    array like:
        Arias intensity time series

    '''
    acc = ts.y
    dt = ts.dt
    if g:
        acc = acc*9.81

    iaSeries = np.pi / (2*9.81) * cumtrapz(acc**2, dx=dt, initial=0)
    return iaSeries


def total_arias(ts, g= False):
    '''

    Parameters
    ----------
    ts: Acceleration timeseries object

    g: Bool, optional
        g=True multiplies acceleration values with g=9.81 m/sec^2.
        Used when acceleration values in 'g' units are to be converted into 'm/sec^2'

    Returns
    -------
    Scalar:
        Total Arias Intensity

    '''
    acc = ts.y
    dt = ts.dt
    if g:
        acc = acc*9.81

    ia = np.pi / (2*9.81) * trapz(acc**2, dx=dt)
    return ia


def sig_duration(ts, g=False, start=0.05, stop=0.95):
    '''

    Computes significant duration as portion of ground motion encompassing 5% to 95% of total arias intensity

    Parameters
    ----------
    ts: Acceleration timeseries object

    g: Bool, optional
        g=True multiplies acceleration values with g=9.81 m/sec^2.
        Used when acceleration values in 'g' units are to be converted into 'm/sec^2'

    Returns
    -------
    Scalar:
        Significant Duration (5-95)%

    '''
    dt = ts.dt
    cumIa = arias_intensity(ts, g=g)
    index = np.where((cumIa >start*cumIa[-1]) & (cumIa <stop*cumIa[-1]))
    return index[0][-1]*dt - index[0][0]*dt


def destructive_potential(ts, g=False):
    '''
    Computes destructiveness potential according to Araya and Sargoni (1984)

    Parameters
    ----------
    ts: Acceleration timeseries object

    g: Bool, optional
        g=True multiplies acceleration values with g=9.81 m/sec^2.
        Used when acceleration values in 'g' units are to be converted into 'm/sec^2'

    Returns
    -------
    Scalar:
        Destructiveness potential

    '''
    acc = ts.y
    ia = total_arias(ts, g=g)
    u0 = len(np.where(np.diff(np.sign(acc)))[0])
    return ia/u0**2


def cum_abs_vel(ts, g=False):
    '''
    Computes cummulative absolute velocity

    Parameters
    ----------
    ts: Acceleration timeseries object

    g: Bool, optional
        g=True multiplies acceleration values with g=9.81 m/sec^2.
        Used when acceleration values in 'g' units are to be converted into 'm/sec^2'

    Returns
    -------
    Scalar:
        Cummulative Absolute Velocity

    '''
    acc = ts.y
    dt = ts.dt
    if g:
        acc = acc*9.81
    acc = np.absolute(acc)
    return trapz(acc, dx=dt)


def cum_abs_disp(ts):
    '''
    Computes Cummulative Absolute Displacement

    Note
        Please make sure to use velocity time series as input to compute cummulative absolute displacement.

    Parameters
    ----------
    ts: Velocity timeseries object

    Returns
    -------
    Scalar:
        Cummulative Absolute Velocity

    '''
    vel = ts.y
    dt = ts.dt
    vel = np.absolute(vel)
    return trapz(vel, dx=dt)

def specific_energy(ts):
    '''
    
    Computes specific energy density
    
    Note
        Please use velocity time series as input to compute specific energy density.
    
    Parameters
    ----------
    ts: Velocity timeseries object

    Returns
    -------
    Scalar:
        Specify Energy Density
        
    '''
    vel = ts.y
    dt = ts.dt
    return trapz(vel**2, dx=dt)

def rms(ts,g=False):
    '''
    
    Root-mean-square value of acceleration/velocity/displacement time series

    Parameters
    ----------
    ts: Acceleration/Velocity/Displacement time series object 

    g: Bool, optional
        g=True multiplies acceleration values with g=9.81 m/sec^2.
        Used when acceleration values in 'g' units are to be converted into 'm/sec^2'

    Returns
    -------
    Scalar:
        Root-mean-square value of time series values
    

    '''
    val = ts.y
    total_time = ts.time
    dt = ts.dt
    return np.sqrt(trapz(val**2, dx=dt)*total_time**-1)



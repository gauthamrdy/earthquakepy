import numpy as np

from earthquakepy import tsReaders

def arias_intensity(acc, dt, g= False):
    '''
    Computes arias intensity:

    Parameters
    ----------
    acc: 1-d array like 
        Acceleration time series
    
    dt: Time step
        Sampling interval
    
    g: Bool, optional
        g=True multiplies acceleration values with g=9.81 m/sec^2. 
        Used when acceleration values in 'g' units are to be converted into 'm/sec^2'
    Returns
    -------
    array like: 
        Arias intensity time series 

    '''
    from scipy.integrate import cumtrapz
    if g:
        acc = acc*9.81
    
    iaSeries = np.pi / (2*9.81) * cumtrapz(acc**2, dx=dt, initial=0)
    return iaSeries


def total_arias(acc, dt, g= False):
    '''
    
    Parameters
    ----------
    acc: 1-d array like 
        Acceleration time series
    
    dt: Time step
        Sampling interval
    
    g: Bool, optional
        g=True multiplies acceleration values with g=9.81 m/sec^2. 
        Used when acceleration values in 'g' units are to be converted into 'm/sec^2'

    Returns
    -------
    Scalar:
        Total Arias Intensity
    
    '''
    from scipy.integrate import trapz
    if g:
        acc = acc*9.81
    
    ia = np.pi / (2*9.81) * trapz(acc**2, dx=dt)
    return ia


def sig_duration(acc, dt, g = False, start = 0.05, stop = 0.95):
    '''

    Computes significant duration as portion of ground motion encompassing 5% to 95% of total arias intensity

    Parameters
    ----------
    acc: 1-d array like 
        Acceleration time series
    
    dt: Time step
        Sampling interval
    
    g: Bool, optional
        g=True multiplies acceleration values with g=9.81 m/sec^2. 
        Used when acceleration values in 'g' units are to be converted into 'm/sec^2'

    Returns
    -------
    Scalar:
        Significant Duration (5-95)%

    '''
    cumIa = arias_intensity(acc, dt, g = g)
    index = np.where((cumIa >start*cumIa[-1]) & (cumIa <stop*cumIa[-1]))
    return index[0][-1]*dt - index[0][0]*dt
    
def destructive_potential(acc, dt, g = False):
    '''
    Computes destructiveness potential according to Araya and Sargoni (1984)

    Parameters
    ----------
    acc: 1-d array like 
        Acceleration time series
    
    dt: Time step
        Sampling interval
    
    g: Bool, optional
        g=True multiplies acceleration values with g=9.81 m/sec^2. 
        Used when acceleration values in 'g' units are to be converted into 'm/sec^2'

    Returns
    -------
    Scalar:
        Destructiveness potential 

    '''
    ia = total_arias(acc,dt,g=g)
    u0 = len(np.where(np.diff(np.sign(acc)))[0])
    return ia/u0**2

def cum_abs_vel(acc, dt, g=False):
    '''
    Computes cummulative absolute velocity
    
    Parameters
    ----------
    acc: 1-d array like 
        Acceleration time series
    
    dt: Time step
        Sampling interval
    
    g: Bool, optional
        g=True multiplies acceleration values with g=9.81 m/sec^2. 
        Used when acceleration values in 'g' units are to be converted into 'm/sec^2'
    
    Returns
    -------
    Scalar:
        Cummulative Absolute Velocity
        
    '''
    from scipy.integrate import trapz
    if g:
        acc = acc*9.81
    acc = np.absolute(acc)
    return trapz(acc,dx=dt)

def cum_abs_disp(vel, dt):
    '''
    Computes Cummulative Absolute Displacement
    
        Note
            Please make sure to use velocity time series as input to compute cummulative absolute displacement.
        
    Parameters
    ----------
    vel: 1-d array like 
        Velocity time series
    
    dt: Time step
        Sampling interval
    
    Returns
    -------
    Scalar:
        Cummulative Absolute Velocity
        
    '''
    from scipy.integrate import trapz
    vel = np.absolute(vel)
    return trapz(vel,dx=dt)


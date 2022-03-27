import re
import numpy as np
from earthquakepy import timeseries

def read_peer_nga_file(filepath):
    '''
    Reads PEER NGA record file and 
    generates a timeseries object.
    Input :
    filepath (string): PEER NGA file path
    '''
    with open(filepath, 'r') as f:
        lines = f.readlines()
        nlines = len(lines)

    for n in range(nlines):
        line = lines[n]
        if n == 0:
            pass
        elif n == 1:
            eq, eqDate, station, component = line.strip("\n").split(",")
        elif n == 2:
            yunit = line
        elif n == 3:
            npts = int(re.match(r".*= *([0-9]*),.*", line)[1])
            dt = float(re.match(r".*= *(0?\.[0-9]*) SEC", line)[1])
            time = dt*npts
            y = np.zeros(int(npts))
        else:
            elms = line.strip("\n").split()
            nelms = len(elms)
            i = (n-4)*nelms
            j = i + nelms
            y[i:j] = [float(e) for e in elms]

    ts = timeseries.TimeSeries(dt, y)
    ts.set_tunit('s')
    ts.set_yunit(yunit)
    ts.set_eqname(eq)
    ts.set_eqdate(eqDate)
    ts.set_station(station)
    ts.set_component(component)
    ts.set_npts(npts)
    ts.set_dt(dt)
    ts.set_time(time)
    ts.set_filepath(filepath)
    return ts
        

def read_raw_file(filename, **kwargs):
    """
    Reads a raw file readable by numpy.genfromtxt(). The first column is assumed as time and second column as ordinates.
    Input:
        filename (str): filename of the file containing raw data

    Output:
        Timeseries object
    """
    data = np.genfromtxt(filename, **kwargs)
    ts = timeseries.TimeSeries(data[:, 0], data[:, 1])
    return ts

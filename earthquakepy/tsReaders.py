import re
import numpy as np
from earthquakepy import timeseries

def readPeerNgaFile(filepath):
    '''
    Reads PEER NGA record file and 
    generates a timeseries object.
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
            y = np.zeros(int(npts))
        else:
            elms = line.strip("\n").split()
            nelms = len(elms)
            i = (n-4)*nelms
            j = i + nelms
            y[i:j] = [float(e) for e in elms]

    ts = timeseries.TimeSeries(dt, y)
    ts.setTunit('s')
    ts.setYunit(yunit)
    ts.setEqName(eq)
    ts.setEqDate(eqDate)
    ts.setStation(station)
    ts.setComponent(component)
    ts.setNpts(npts)
    ts.setDt(dt)
    ts.setFilepath(filepath)
    return ts
        

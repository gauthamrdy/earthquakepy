import re
import numpy as np
from earthquakepy import timeseries


def read_peer_nga_file(filepath, scale_factor=1):
    """
    Read PEER NGA record file and generate a timeseries object.

    Parameters
    ----------
    filepath (string): PEER NGA file path
    scale_factor: Scaling factor, default = 1

    Returns
    -------
    TimeSeries object
    """
    with open(filepath, "r") as f:
        lines = f.readlines()
        nlines = len(lines)

    for n in range(nlines):
        line = lines[n]
        if n == 0:
            pass
        elif n == 1:
            eq, eqDate, station, component = [l.strip() for l in line.strip("\n").split(",")]
        elif n == 2:
            yunit = line.strip()
        elif n == 3:
            npts = int(re.match(r".*= *([0-9]*),.*", line)[1])
            dt = float(re.match(r".*= *(0?\.[0-9]*) SEC", line)[1])
            duration = dt * npts
            y = np.zeros(int(npts))
        else:
            elms = line.strip("\n").split()
            nelms = len(elms)
            i = (n - 4) * nelms
            j = i + nelms
            y[i:j] = [float(e) for e in elms]

    ts = timeseries.TimeSeries(dt, y*scale_factor)
    ts.set_tunit("s")
    ts.set_yunit(yunit)
    ts.set_eqname(eq)
    ts.set_eqdate(eqDate)
    ts.set_station(station)
    ts.set_component(component)
    ts.set_npts(npts)
    ts.set_dt(dt)
    ts.set_duration(duration)
    ts.set_filepath(filepath)
    return ts


def read_data_between_lines(text, start=0, end=None):
    """
    Read data between line numbers given by start and end.

    Parameters
    ----------
    text (list of strings): data read from file using readlines()
    start (int): line number (0 indexed) to start reading the data
    end (int): line number (0 indexed) to end reading the data (inclusive)

    returns
    -------
    1D numpy array of values
    """
    if end is None:
        end = len(text)

    v = []
    for i in range(start, end):
        line = text[i].strip().split(" ")
        for j in range(len(line)):
            v.append(float(line[j]))
    return np.array(v)


def read_cosmos_vdc_file(filename, **kwargs):
    """
    Read a cosmos db virtual data file and return three timeseries objects corresponding to acceleration, velocity and displacement record, respectively.

    Parameters
    ----------
    filename (string): name of the file to read

    Returns
    -------
    three Timeseries objects
    """
    with open(filename, "r") as f:
        text = f.readlines()

    l1 = text[0].strip()
    eqName = l1.split(",")[0]
    eqDate = ",".join(l1.split(",")[1:]).strip()

    l2 = text[1].strip()
    station = re.search("\s*([A-Za-z0-9]*)\s*", l2)[1]
    latloncomp = re.search("Lat.*Lon([0-9 \-]*[NS])\s*([0-9 \-]*[EW])\s*Comp:\s*(.*)\s*", l2)
    lat = latloncomp[1]
    lon = latloncomp[2]
    comp = latloncomp[3]

    l3 = text[2].strip()
    l4 = text[3].strip()
    meta = l3 + "\n" + "            " + l4

    l6 = text[5].strip()
    nptsdt = re.search("\s*(\d*)\s*.*at\s*([0-9.]*)\s*sec", l6)
    npts = int(nptsdt[1])
    dt = float(nptsdt[2])

    ptsperline = len(text[6].strip().split(" "))
    nlines = int(np.ceil(npts / (ptsperline+0.001)))

    acc_start = 6
    acc_end = acc_start + nlines

    vel_start = acc_end + 2
    vel_end = vel_start + nlines

    disp_start = vel_end + 2
    disp_end = disp_start + nlines

    aunit = re.search(".*\(in\s*(.*)\)\s*.*", text[acc_start-1])[1]
    vunit = re.search(".*\(in\s*(.*)\)\s*.*", text[vel_start-1])[1]
    dunit = re.search(".*\(in\s*(.*)\)\s*.*", text[disp_start-1])[1]

    acc = read_data_between_lines(text, start=acc_start, end=acc_end)
    vel = read_data_between_lines(text, start=vel_start, end=vel_end)
    disp = read_data_between_lines(text, start=disp_start, end=disp_end)

    ats = timeseries.TimeSeries(dt, acc)
    ats.component = comp
    ats.dt = dt
    ats.duration = ats.t[-1]
    ats.eqDate = eqDate
    ats.eqName = eqName
    ats.lat = lat
    ats.lon = lon
    ats.station = station
    ats.yunit = aunit
    ats.filepath = filename
    ats.meta = meta

    vts = timeseries.TimeSeries(dt, vel)
    vts.component = comp
    vts.dt = dt
    vts.duration = vts.t[-1]
    vts.eqDate = eqDate
    vts.eqName = eqName
    vts.lat = lat
    vts.lon = lon
    vts.station = station
    vts.yunit = vunit
    vts.filepath = filename
    vts.meta = meta

    dts = timeseries.TimeSeries(dt, disp)
    dts.component = comp
    dts.dt = dt
    dts.duration = dts.t[-1]
    dts.eqDate = eqDate
    dts.eqName = eqName
    dts.lat = lat
    dts.lon = lon
    dts.station = station
    dts.yunit = dunit
    dts.filepath = filename
    dts.meta = meta
    return ats, vts, dts


def read_raw_timeseries_file(filename, **kwargs):
    """
    Read a raw file readable by numpy.genfromtxt(). The first column is assumed as time and second column as ordinates. Accepts all arguments supported by genfromtxt().

    Parameters
    ----------
    filename: (str) filename of the file containing raw data

    Returns
    -------
    Timeseries object
    """
    data = np.genfromtxt(filename, **kwargs)
    ts = timeseries.TimeSeries(data[:, 0], data[:, 1])
    return ts

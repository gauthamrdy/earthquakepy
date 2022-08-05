import numpy as np
from scipy.integrate import trapz, cumtrapz
import matplotlib.pyplot as plt
import matplotlib as mpl
from .singledof import Sdof, SdofNL
from scipy.fftpack import fft, fftfreq


class TimeSeries:
    """
    TimeSeries object class. Defines time series
    """

    def __init__(self, t, y):
        """
        Parameters
        ----------
        t: (scalar or 1D array) time step (if scalar) or time axis (if 1D array)
        y: (1D array) ordinates
        """
        if (type(t) == float) or (type(t) == int):
            t = np.arange(t, len(y) * t + 0.1 * t, t)
        self.t = t
        self.y = y
        self.npts = len(t)
        self.dt = t[1] - t[0]

    def __repr__(self):
        a = ""
        for key, val in vars(self).items():
            a += "{:>10s}: {}\n".format(key, val)
        return a

    def set_tunit(self, unit):
        """
        Set unit for T(time) coordinates. Unit should be a string.
        """
        self.tunit = unit

    def set_yunit(self, unit):
        """
        Set unit for y coordinates. Unit should be a string.
        """
        self.yunit = unit

    def set_t(self, coords):
        """
        Set T(time) coordinates. Should be a list or numpy array (1D)
        """
        self.t = coords

    def set_eqname(self, name):
        """Set earthquake name"""
        self.eqName = name

    def set_eqdate(self, date):
        """Set earthquake date"""
        self.eqDate = date

    def set_station(self, station):
        """Recording station"""
        self.station = station

    def set_component(self, comp):
        """Directional component of record"""
        self.component = comp

    def set_dt(self, dt):
        """Time step between data points"""
        self.dt = dt

    def set_npts(self, npts):
        """Total number of points in the record"""
        self.npts = npts

    def set_duration(self, duration):
        """Total duration of the record in seconds"""
        self.duration = duration

    def set_filepath(self, filepath):
        """Record filepath"""
        self.filepath = filepath

    def plot(self, log=False, **kwargs):
        """
        Method plots time history of timeseries object. It accepts all the arguments that matplotlib.pyplot.subplots recognizes.

        Returns
        -------
        Matplotlib Figure Object

        """
        mpl.pyplot.style.use('/home/kaushal/PhDwork/earthquakepy/mpl_stylesheet/style1.mplstyle')
        fig, ax = plt.subplots(**kwargs)
        ax.plot(self.t, self.y)
        if hasattr(self, "tunit"):
            ax.set_xlabel("t ({})".format(self.tunit))
        if hasattr(self, "yunit"):
            ax.set_ylabel(str(self.yunit))
        if log:
            ax.set_xscale("log")
        ax.set_xlim(left=0)
        # plt.show()
        return fig

    def get_y(self, t):
        return np.interp(t, self.t, self.y)

    def get_response_spectra(self, T=np.arange(0.1, 20.01, 1.0), xi=0.05):
        """
        Calculates linear elastic response spectra associated with the timeseries.

        Parameters
        ----------
        T: (1D array of floats) Periods corresponding to spectrum width
        xi: (float) damping ratio

        Returns
        -------
        ResponseSpectrum object with T, Sd, Sv, Sa as attributes.
        """
        specLength = len(T)
        Sd = np.empty(specLength)
        Sv = np.empty(specLength)
        Sa = np.empty(specLength)
        for i in range(specLength):
            s = Sdof(T=T[i], xi=xi)
            r = s.get_response(self, tsType="baseExcitation")
            D = np.max(np.abs(r.y[0]))
            V = np.max(np.abs(r.y[1]))
            A = np.max(np.abs(r.acc))
            Sd[i] = D
            Sv[i] = V
            Sa[i] = A
        return ResponseSpectrum(T, Sd, Sv, Sa)

    def get_fourier_spectrum(self):
        """
        Computes fourier spectrum associated with the time series
        """
        N = self.npts
        T = self.dt  # sampling interval
        yf = fft(self.y)
        # FAmp = np.abs(yf[0:N//2])
        freq = fftfreq(N, T)  # [:N//2]
        return FourierSpectrum(freq, yf, N)

    def get_power_spectrum(self):
        """
        Computes power spectrum associated with the time series
        """
        N = self.npts
        fourier_spectrum = self.get_fourier_spectrum()
        freq = fourier_spectrum.frequencies
        df = freq[1] - freq[0]
        powerAmp = 2.0 / df * fourier_spectrum.amplitude**2
        return PowerSpectrum(freq, powerAmp, N)

    def get_mean_period(self):
        """
        Computes the simplified frequency content characterisation parameter according to  Rathje et al. [1998]

        Returns
        -------
        Scalar:
            Mean period

        """
        fourier_spectrum = self.get_fourier_spectrum()
        freq = fourier_spectrum.frequencies
        FAmp = fourier_spectrum.amplitude
        boolArr = (freq > 0.25) & (freq < 20)
        n = FAmp[boolArr] ** 2 / freq[boolArr]
        n = n.sum()
        d = FAmp[boolArr] ** 2
        d = d.sum()
        return n / d

    def get_mean_frequency(self):
        """
        Computes the simplified frequency content characterisation parameter according to  Schnabel [1973]

        Returns
        -------
        Scalar:
            Mean square frequency

        """
        fourier_spectrum = self.get_fourier_spectrum()
        freq = fourier_spectrum.frequencies
        FAmp = fourier_spectrum.amplitude
        boolArr = (freq > 0.25) & (freq < 20)
        n = FAmp[boolArr] ** 2 * freq[boolArr]
        n = n.sum()
        d = FAmp[boolArr] ** 2
        d = d.sum()
        return n / d

    def get_epsilon(self):
        """
        Computes the dimensionless frequency indicator, epsilon as given by Clough and Penzien

        Returns
        -------
        Scalar

        """
        from scipy.integrate import trapz

        power_spectrum = self.get_power_spectrum()
        freq = power_spectrum.frequencies
        powerAmp = power_spectrum.amplitude
        m0 = trapz(powerAmp, freq)
        m2 = trapz(powerAmp * freq**2, freq)
        m4 = trapz(powerAmp * freq**4, freq)
        eps = np.sqrt(1 - m2**2 / (m0 * m4))
        return eps

    def get_arias_intensity(self, g=False):
        """
        Computes arias intensity:

        Parameters
        ----------
        g: Bool, optional
            g=True multiplies acceleration values with g=9.81 m/sec^2.
            Used when acceleration values in 'g' units are to be converted into 'm/sec^2'
        
        Returns
        -------
        array like:
            Arias intensity time series

        """
        acc = self.y
        dt = self.dt
        if g:
            acc = acc * 9.81

        iaSeries = np.pi / (2 * 9.81) * cumtrapz(acc**2, dx=dt, initial=0)
        return iaSeries

    def get_total_arias(self, g=False):
        """

        Parameters
        ----------
        g: Bool, optional
            g=True multiplies acceleration values with g=9.81 m/sec^2.
            Used when acceleration values in 'g' units are to be converted into 'm/sec^2'

        Returns
        -------
        Scalar:
            Total Arias Intensity

        """
        acc = self.y
        dt = self.dt
        if g:
            acc = acc * 9.81

        ia = np.pi / (2 * 9.81) * trapz(acc**2, dx=dt)
        return ia

    def get_sig_duration(self, g=False, start=0.05, stop=0.95):
        """

        Computes significant duration as portion of ground motion encompassing 5% to 95% of total arias intensity

        Parameters
        ----------
        g: Bool, optional
            g=True multiplies acceleration values with g=9.81 m/sec^2.
            Used when acceleration values in 'g' units are to be converted into 'm/sec^2'

        start: float, optional
            defines the start point in fraction of total arias intensity from where significant duration starts.

        stop: float, optional
            defines the end point in fraction of total arias intensity where significant duration ends.

        Returns
        -------
        Scalar:
            Significant Duration (5-95)%

        """
        dt = self.dt
        cumIa = self.get_arias_intensity(g=g)
        index = np.where((cumIa > start * cumIa[-1]) & (cumIa < stop * cumIa[-1]))
        return index[0][-1] * dt - index[0][0] * dt

    def get_destructive_potential(self, g=False):
        """
        Computes destructiveness potential according to Araya and Sargoni (1984)

        Parameters
        ----------
        g: Bool, optional
            g=True multiplies acceleration values with g=9.81 m/sec^2.
            Used when acceleration values in 'g' units are to be converted into 'm/sec^2'
        
        Returns
        -------
        Scalar:
            Destructiveness potential

        """
        acc = self.y
        ia = self.get_total_arias(g=True)
        u0 = len(np.where(np.diff(np.sign(acc)))[0])/self.duration
        return ia / u0**2

    def get_cum_abs_vel(self, g=False):
        """
        Computes cummulative absolute velocity

        Parameters
        ----------
        g: Bool, optional
            g=True multiplies acceleration values with g=9.81 m/sec^2.
            Used when acceleration values in 'g' units are to be converted into 'm/sec^2'

        Returns
        -------
        Scalar:
            Cummulative Absolute Velocity

        """
        acc = self.y
        dt = self.dt
        if g:
            acc = acc * 9.81
        acc = np.absolute(acc)
        return trapz(acc, dx=dt)

    def get_cum_abs_disp(self):
        """
        Computes Cummulative Absolute Displacement

        Note
            Please make sure to use velocity time series as input to compute cummulative absolute displacement.

        Parameters
        ----------
        None
        
        Returns
        -------
        Scalar:
            Cummulative Absolute Velocity

        """
        vel = self.y
        dt = self.dt
        vel = np.absolute(vel)
        return trapz(vel, dx=dt)

    def get_specific_energy(self):
        """

        Computes specific energy density

        Note
            Please use velocity time series as input to compute specific energy density.

        Parameters
        ----------
        None
        
        Returns
        -------
        Scalar:
            Specify Energy Density

        """
        vel = self.y
        dt = self.dt
        return trapz(vel**2, dx=dt)

    def get_rms(self, g=False):
        """

        Root-mean-square value of acceleration/velocity/displacement time series

        Parameters
        ----------
        g: Bool, optional
            g=True multiplies acceleration values with g=9.81 m/sec^2.
            Used when acceleration values in 'g' units are to be converted into 'm/sec^2'

        
        Returns
        -------
        Scalar:
            Root-mean-square value of time series values


        """
        val = self.y
        total_time = self.duration
        dt = self.dt
        return np.sqrt(trapz(val**2, dx=dt) * total_time**-1)

    def get_diff(self, edge_order=1):
        """
        Differentiate the time series with respect to 't' (default).
        Uses numpy.gradient() internally.

        Parameters
        ----------
        edge_order: Same as edge_order in numpy.gradient(). Default: 1

        Returns
        -------
        1D array of same length as that of 't'.
        """
        return np.gradient(self.y, self.t, edge_order=edge_order)

    def get_int(self, **kwargs):
        """
        Integrate the time series with respect to 't' (default).
        Uses numpy.trapz() internally.

        Parameters
        ----------
        All **kwargs supported by numpy.trapz()

        Returns
        -------
        Scalar
        """
        defaultArgs = {"x": self.t}
        kwargs = {**defaultArgs, **kwargs}
        return np.trapz(self.y, **kwargs)


class ResponseSpectrum:
    def __init__(self, T, Sd, Sv, Sa):
        """
        Class for storing response spectra

        Parameters
        ----------
        T: (array of float) Natural period
        Sd: (array of float) Spectral displacement
        Sv: (array of float) Spectral velocity
        Sa: (array of float) Spectral acceleration
        """
        self.T = T
        self.Sd = Sd
        self.Sv = Sv
        self.Sa = Sa

    def __repr__(self):
        a = ""
        for key, val in vars(self).items():
            a += "{:10s}:{}\n".format(key, val)
        return a

    def plot(self, log=False, **kwargs):
        """
        Method plots Response Spectrum of ResponseSpectrum object
        Parameters
        ----------
        Returns
        -------
        Matplotlib Figure Object

        """
        mpl.pyplot.style.use('/home/kaushal/PhDwork/earthquakepy/mpl_stylesheet/style1.mplstyle')
        fig, ax = plt.subplots(nrows=1, ncols=3, constrained_layout=True, **kwargs)
        ax[0].plot(self.T, self.Sd)
        ax[1].plot(self.T, self.Sv)
        ax[2].plot(self.T, self.Sa)
        ax[0].set_xlabel("Period (s)")
        ax[1].set_xlabel("Period (s)")
        ax[2].set_xlabel("Period (s)")
        ax[0].set_ylabel("Spectral displacement")
        ax[1].set_ylabel("Spectral velocity")
        ax[2].set_ylabel("Spectral acceleration")
        if log:
            ax[0].set_xscale("log")
            ax[0].set_yscale("log")
            ax[1].set_xscale("log")
            ax[1].set_yscale("log")
            ax[2].set_xscale("log")
            ax[2].set_yscale("log")
        # plt.show()
        return fig


class FourierSpectrum:
    def __init__(self, frequencies, ordinate, N):
        """
        Class to store fourier spectra

        Parameters
        ----------
        Freqencies: Frequency (Hz)
        Amplitude: Fourier Amplitude
        """
        self.frequencies = frequencies
        self.ordinate = ordinate
        # Famp = np.abs(ordinate[0:N//2])
        # phase = np.angle(ordinate[0:N//2])
        self.amplitude = np.abs(ordinate)
        self.phase = np.angle(ordinate)
        self.N = N
        fSortIndices = np.argsort(self.frequencies)
        self.unwrappedPhase = np.unwrap(self.phase[fSortIndices], np.pi)

    def __repr__(self):
        a = ""
        for key, val in vars(self).items():
            a += "{:10s}:{}\n".format(key, val)
        return a

    def plot(self, log=False, **kwargs):
        """
        Method plots Fourier Spectrum of FourierSpectrum object

        Parameters
        ----------
        Returns
        -------
        Matplotlib Figure Object

        """
        mpl.pyplot.style.use('/home/kaushal/PhDwork/earthquakepy/mpl_stylesheet/style1.mplstyle')
        defaultArgs = {"figsize" : (10,5)}
        kwargs = {**defaultArgs, **kwargs}
        fig = plt.figure(constrained_layout=True, **kwargs)

        gs = plt.GridSpec(2, 2, figure=fig)
        ax0 = fig.add_subplot(gs[0, :])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])

        ax0.set_xlabel("Frequency (Hz)")
        ax0.set_ylabel("Fourier Amplitude")
        ax0.plot(
            self.frequencies[:self.N//2],
            2.0 / self.N * self.amplitude[:self.N//2],
        )
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Phase angle")
        ax1.plot(self.frequencies[:self.N//2], self.phase[:self.N//2])
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Unwrapped phase angle")
        ax2.plot(
            np.sort(self.frequencies), self.unwrappedPhase
        )
        if log:
            ax0.set_xscale("log")

        return fig


class PowerSpectrum:
    def __init__(self, frequencies, amplitude, N):
        """
        Class to store power spectra

        Parameters
        ----------
        Freqencies: Frequency (Hz)
        Amplitude: Power Amplitude
        """
        self.frequencies = frequencies
        self.amplitude = amplitude
        self.N = N

    def __repr__(self):
        a = ""
        for key, val in vars(self).items():
            a += "{:10s}:{}\n".format(key, val)
        return a

    def get_compatible_timehistories(self, m=5):
        """
        Generates m compatible timehistories from power spectrum

        Parameters
        ----------
        m: (int) number of compatible timehistories to generate

        Returns
        -------
        list of timeseries objects
        """
        # get FFT amplitude from power amplitude
        dF = self.frequencies[1] - self.frequencies[0]
        Xamp = np.sqrt(dF / 2.0 * self.amplitude)
        n = len(Xamp)

        dt = 1 / (2 * np.max(self.frequencies))
        endT = 1 / dF
        t = np.linspace(0, endT - dt, len(self.frequencies))

        X = np.complex_(np.zeros((n, m)))
        x = np.complex_(np.zeros((n, m)))
        Xphase = np.complex_(np.zeros(n))

        ts = [None] * m

        for k in range(m):
            if np.mod(n, 2) == 0:  # even number of samples
                Xphase[1 : n // 2] = 2 * np.pi * np.random.rand(n // 2 - 1)
                Xphase[n // 2 + 1 :] = -np.flip(Xphase[1 : n // 2])
            else:
                Xphase[1 : (n + 1) // 2] = n * np.pi * np.random.rand((n + 1) // 2 - 1)
                Xphase[(n + 1) // 2 :] = -np.flip(Xphase[1 : (n + 1) // 2])

            X[:, k] = Xamp * (np.cos(Xphase) + 1j * np.sin(Xphase))
            x[:, k] = np.fft.ifft(X[:, k])

            ts[k] = TimeSeries(t, np.real(x[:, k]))
        return ts

    def plot(self, log=False):
        """
        Method plots Power Spectrum of PowerSpectrum object
        Parameters
        ----------
        Returns
        -------
        Matplotlib Figure Object

        """
        mpl.pyplot.style.use('/home/kaushal/PhDwork/earthquakepy/mpl_stylesheet/style1.mplstyle')
        fig, ax = plt.subplots()
        ax.plot(
            self.frequencies,
            2.0 / self.N * self.amplitude,
            )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power Amplitude")
        if log:
            ax.set_xscale("log")
        # plt.show()
        return fig

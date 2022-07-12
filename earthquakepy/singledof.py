import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
import matplotlib.pyplot as plt


class Sdof:
    """
    Class for single-degree-of-freedom system
    Parameters :
    You should provide at least wn or T or (m and k).

    Parameters
    ----------
    m: scalar
        mass of the system
    c: scalar
        damping constant
    k: scalar
        stiffness
    xi: scalar
            damping ratio
    wn: scalar
            Natural Frequency (rad/s)
    T: scalar
        Natural Period
    """

    def __init__(self, m=None, c=0.0, k=None, xi=0.0, wn=None, T=None):
        if T:
            wn = 2 * np.pi / T
            m = 1.0
            k = m * wn**2
            c = 2 * xi * m * wn
        elif wn:
            T = 2 * np.pi / wn
            m = 1.0
            k = m * wn**2
            c = 2 * xi * m * wn
        elif k and m:
            wn = np.sqrt(k / m)
            T = 2 * np.pi / wn
            if xi:
                c = 2 * xi * m * wn
            elif c:
                xi = c / (2 * m * wn)
        else:
            raise Exception("Incorrect parameters")

        self.m, self.c, self.k = m, c, k
        self.xi, self.wn, self.T = xi, wn, T

    def __repr__(self):
        a = ""
        for key, val in vars(self).items():
            a += "{:10s}:{}\n".format(key, val)
        return a

    def sdof_grad(self, t, y, ts, a):
        m, c, k = self.m, self.c, self.k

        A = np.array([[0, 1], [-k / m, -c / m]])
        x = np.array(y)
        f = np.array([[0.0], [ts.get_y(t) / m]])
        dy = np.matmul(A, x) + f

        return dy

    def get_response(self, ts, tsType="baseExcitation", **kwargs):
        """
        Wrapper around solve_ivp module from scipy.integrate. It supports all the arguments supported by solve_ivp.

        Parameters
        ----------
        ts: (timeseries object) timeseries defining loading/base excitation

        tsType: (string) "baseExcitation" or "force"

        **kwargs: arguments acceptable to scipy solve_ivp module

        Returns
        -------
        SdofResponseTimeSeries object

        By default the solution will be obtained for duration = 2 * (ts.t duration). This can be changed using t_span argument. Default method : BDF.
        """
        if tsType == "baseExcitation":
            f = -self.m * ts.y
        elif tsType == "force":
            f = ts.y
        else:
            raise Exception("Incorrect timeseries type given")

        defaultArgs = {
            "t_span": (ts.t[0], ts.t[-1] * 2),
            "y0": [0.0, 0.0],
            "method": "BDF",
            "t_eval": None,
            "dense_output": False,
            "events": None,
            "vectorized": True,
            "args": (ts, 1), # additional element 1 required in tuple due to bug in solve_ivp with only one argument
            "jac": np.array([[0, 1], [-self.k / self.m, -self.c / self.m]]),
        }
        kwargs = {**defaultArgs, **kwargs}

        # r = solve_ivp(self.sdof_grad, **kwargs)
        r = solve_ivp(self.sdof_grad, **kwargs)
        m, c, k = self.m, self.c, self.k
        fv = np.interp(r.t, ts.t, f)
        acc = 1 / m * (fv - c * r.y[1] - k * r.y[0])
        return SdofResponseTimeSeries(r, acc)


class SdofResponseTimeSeries(OdeResult):
    """
    Class for response object of SDOF system
    """

    def __init__(self, ode_result, acc):
        self.acc = acc
        for key, val in ode_result.items():
            self.__setattr__(key, val)

    def plot(self, **kwargs):
        """
        A quick and dirty way to plot and inspect the output. It accepts all the arguments supported by matplotlib.pyplot.subplots()
        """
        y = np.transpose(self.y)
        fig, ax = plt.subplots(nrows=1, ncols=len(self.y), **kwargs)
        for i in range(len(self.y)):
            ax[i].plot(self.t, self.y[i], linewidth=0.5, color="black")
            ax[i].set_xlabel("Time (s)")
        return fig


class SdofNL:
    """
    Class for nonlinear SDOF system.
    """
    def __init__(self, m=1, dampForce=None, springForce=None):
        self.m = m
        self.dampForce = dampForce
        self.springForce = springForce

    def __repr__(self):
        a = ""
        for key, val in vars(self).items():
            a += "{:10s}:{}\n".format(key, val)
        return a

    def damping_force(self):
        return self.dampForce

    def spring_force(self):
        return self.springForce


    def sdof_grad(self, t, y, m, dampForce, springForce, extForce):
        """
        Defines state space form of nonlinear sdof system

        Parameters
        ----------
        m: (scalar) mass
        dampForce: (function or timeseries object) Damping force
        springForce: (function or timeseries object) Spring force
        extForce: (function or timeseries object) external force
        """
        dy = np.zeros(2)
        if callable(dampForce):
            dampF = dampForce(t, y[0], y[1])
        else:
            dampF = dampForce.get_y(t)

        if callable(springForce):
            sprF = springForce(t, y[0], y[1])
        else:
            sprF = springForce.get_y(t)

        if callable(extForce):
            extF = extForce(t)
        else:
            extF = extForce.get_y(t)
        dy[0] = y[1]
        dy[1] = 1/m*(extF - dampF - sprF)
        return dy

    def get_response(self, ts, tsType="baseExcitation", **kwargs):
        """
        Wrapper around solve_ivp module from scipy.integrate. It supports all the arguments supported by solve_ivp.

        Parameters
        ----------
        ts: (timeseries object) timeseries defining loading/base excitation

        tsType: (string) "baseExcitation" or "force"

        **kwargs: arguments acceptable to scipy solve_ivp module

        Returns
        -------
        SdofResponseTimeSeries object

        By default the solution will be obtained for duration = 2 * (ts.t duration). This can be changed using t_span argument. Default method : BDF.
        """
        if tsType == "baseExcitation":
            f = -self.m * ts.y
        elif tsType == "force":
            f = ts.y
        else:
            raise Exception("Incorrect timeseries type given")

        defaultArgs = {
            "t_span": (ts.t[0], ts.t[-1] * 2),
            "y0": [0.0, 0.0],
            "method": "BDF",
            "t_eval": None,
            "dense_output": False,
            "events": None,
            "vectorized": False,
            "args": (self.m, self.dampForce, self.springForce, ts),
        }
        kwargs = {**defaultArgs, **kwargs}

        # r = solve_ivp(self.sdof_grad, **kwargs)
        r = solve_ivp(self.sdof_grad, **kwargs)
        acc = 1/self.m*(ts.get_y(r.t) - self.dampForce(r.t, r.y[0], r.y[1]) - self.springForce(r.t, r.y[0], r.y[1]))
        return SdofResponseTimeSeries(r, acc)

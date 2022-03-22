import numpy as np
from scipy.integrate import solve_ivp


class Sdof:
    """
    Class for single-degree-of-freedom system
    Parameters :
    You should provide at least wn or T or (m and k).
        Input:
        -------
            m : scalar
                mass of the system
            c : scalar
                damping constant
            k : scalar
                stiffness
            xi : scalar
                 damping ratio
            wn : scalar
                 Natural Frequency (rad/s)
            T : scalar
                Natural Period
    """

    def __init__(self, m=None, c=0.0, k=None, xi=0.0, wn=None, T=None):
        if T:
            wn = 2*np.pi/T
            m = 1.0
            k = m*wn**2
            c = 2*xi*m*wn
        elif wn:
            T = 2*np.pi/wn
            m = 1.0
            k = m*wn**2
            c = 2*xi*m*wn
        elif (k and m):
            wn = np.sqrt(k/m)
            T = 2*np.pi/wn
            xi = c/(2*m*wn)
        else:
            raise Exception("Incorrect parameters")

        self.m, self.c, self.k = m, c, k
        self.xi, self.wn, self.T = xi, wn, T

    def sdof_grad(self, t, y, tv, f):
        m, c, k = self.m, self.c, self.k
        ft = np.interp(t, tv, f)

        A = np.array([
            [0, 1],
            [-k/m, -c/m]
            ])
        x = np.array(y)
        f = np.array([
            [0.0],
            [ft/m]
            ])
        dy = np.matmul(A, x) + f

        return dy

    def get_response(self, ts, tsType="baseExcitation", **kwargs):
        """
        Wrapper around solve_ivp module from scipy.integrate. It supports all
        the arguments supported by solve_ivp.
        Input :
        ts (timeseries object): timeseries defining loading/base excitation
        tsType (string): "baseExcitation" or "force"
        **kwargs: arguments acceptable to scipy solve_ivp module
        By default the solution will be obtained for duration = 2 * (ts.t duration).
        This can be changed using t_span argument. Default method : BDF.
        """
        if tsType == "baseExcitation":
            f = -self.m * ts.y
        elif tsType == "force":
            f = ts.y
        else:
            raise Exception("Incorrect timeseries type given")

        defaultArgs = {
            "t_span": (ts.t[0], ts.t[-1]*2),
            "y0": [0.0, 0.0],
            "method": "BDF",
            "t_eval": None,
            "dense_output": False,
            "events": None,
            "vectorized": True,
            "args": (ts.t, f),
            "jac": np.array([
                [0, 1],
                [-self.k/self.m, -self.c/self.m]
                ])
            }
        kwargs = {**defaultArgs, **kwargs}

        r = solve_ivp(self.sdof_grad, **kwargs)
        return r

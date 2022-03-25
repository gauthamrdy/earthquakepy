#!/usr/bin/env python3

import numpy as np
from scipy.linalg import eigh
from scipy.integrate import solve_ivp


class Mdof:
    """
    CLass for MDOF system.
    """
    def __init__(self, M=None, C=None, K=None):
        """
        Defines MDOF system object using M, C and K matrices.
        """
        if C is None:
            C = np.zeros_like(M)
        if (M is None) or (K is None):
            raise Exception("Incprrect input parameters. You must provide M and K both.")

        self.M = M
        self.K = K
        self.C = C

        wn, v = eigh(K, M)
        self.Wn = wn
        self.Phi = v

    def __repr__(self):
        return "MDOF class instance"

    def mdof_grad(self, t, y, tv, f, R):
        M, C, K = self.M, self.C, self.K
        ft = np.array([np.interp(t, tv, f[:, i]) for i in range(np.shape(f)[1])]).reshape((6, 1))

        n = np.shape(M)[0]
        Minv = np.linalg.inv(M)
        A = np.zeros((2*n, 2*n))
        A[0:n, n:2*n] = np.eye(n)
        A[n:2*n, 0:n] = -Minv*K
        A[n:2*n, n:2*n] = -Minv*C
        x = np.array(y)

        nInputEqs = 3
        for i in range(nInputEqs):
            R[n+i::nInputEqs, i] = 1

        dy = np.matmul(A, x) + np.matmul(R, ft)
        # print(np.matmul(A, x))
        # print(np.matmul(R, ft))
        # print("A={}\nx={}\nR={}\nft={}\ndy={}".format(A, x, R, ft, dy))

        return dy

    def get_response(self, Ax=None, Ay=None, Az=None, r=None, **kwargs):
        """
        Wrapper around solve_ivp module from scipy.integrate. It supports all
        the arguments supported by solve_ivp.
        Input :
        Ax, Ay, Az (timeseries objects): timeseries defining loading/base acceleration (default) in X, Y and Z direction, respectively.
        r (2D array): Influence coefficient matrix os shape (n*m) where n=Total DOFs, m=DOFs per floor (6). DOFs must be ordered as [x, y, z, thetaX, thetaY, thetaZ].
        **kwargs: arguments acceptable to scipy solve_ivp module
        By default the solution will be obtained for duration = 2 * (ts.t duration).
        This can be changed using t_span argument. Default method : BDF.
        """
        # # Total DOFs
        # n = np.shape(self.M)[0]
        # # DOFs per floor
        # m = 6

        # # Create input acceleration matrix of size (len(Ax.t), 6)
        # Xg = np.zeros((len(Ax.t), m))

        # # Find earthquakes applied direction
        # if Ax is not None:
        #     Xg[:, 0] = Ax.t
        # if Ay is not None:
        #     Xg[:, 1] = Ay.t
        # if Az is not None:
        #     Xg[:, 2] = Az.t

        # if (Ax is None) and (Ay is None) and (Az is None):
        #     raise Exception("At least one earthquake input must be specified.")

        # # Create influence coefficient matrix R from r
        # R = np.zeros((2*n, m))
        # R[n::, 0:n] = r

        # defaultArgs = {
        #     "t_span": (Ax.t[0], Ax.t[-1]*2),
        #     "y0": np.zeros(m),
        #     "method": "BDF",
        #     "t_eval": None,
        #     "dense_output": False,
        #     "events": None,
        #     "vectorized": True,
        #     "args": (Ax.t, Xg, R),
        #     #"jac": np.array([
        #     #    [0, 1],
        #     #    [-self.k/self.m, -self.c/self.m]
        #     #    ])
        #     }
        # kwargs = {**defaultArgs, **kwargs}

        # res = solve_ivp(self.mdof_grad, **kwargs)
        # return res
        print("Not implemented yet!")

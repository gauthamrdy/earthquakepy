#!/usr/bin/env python3

from .multidof import Mdof


def mdof(**kwargs):
    """
    Defines a MDOF system object.

    Parameters
    ----------
    M: (2-D array) Mass matrix
    C: (2-D array) Damping matrix
    K: (2-D array) Stiffness matrix

    Returns
    -------
    Mdof object
    """
    return Mdof(**kwargs)

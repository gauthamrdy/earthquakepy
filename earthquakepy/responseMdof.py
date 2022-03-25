#!/usr/bin/env python3

from .multidof import Mdof


def mdof(**kwargs):
    """
    Defines a MDOF system object.
    Inputs:
        M, C, K (2-D Arrays): Mass, damping and stiffness matrices
    """
    return Mdof(**kwargs)

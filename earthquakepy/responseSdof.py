from earthquakepy import singledof


def sdof(**kwargs):
    """
    Creates and returns sdof system object.

    Parameters
    ----------
    You should provide at least wn or T or (m and k).
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

    Returns
    -------
    Sdof object
    """
    return singledof.Sdof(**kwargs)


def sdofNL(**kwargs):
    """
    Creates and returns nonlinear sdof system object.

    Parameters
    ----------
    m (optional): mass of the system. Default=1.0
    dampForce (optional): function or timeseries object representing damping force.
        The function should have arguments (t, u, v, p) where t: independent variable,
        u: first state variable, v: second state variable, p: Array/list of parameters
    springForce (optional): function or timeseries object representing spring force.
        The function should have arguments (t, u, v, p) where t: independent variable,
        u: first state variable, v: second state variable, p: Array/list of parameters

    Returns
    -------
    SdofNL object
    """
    return singledof.SdofNL(**kwargs)

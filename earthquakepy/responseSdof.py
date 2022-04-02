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

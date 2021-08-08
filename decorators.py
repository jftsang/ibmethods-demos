import numpy as np


def Periodicise(xmin, xmax):
    """Function decorator that extends a function by making it periodic.
    """
    def xshift(x):
        y = np.remainder(x-xmin, (xmax-xmin)) + xmin
        return y

    def periodicised(f):
        return lambda x: f(xshift(x))

    return periodicised


def Reflect(sig, about=0):
    """Function decorator that extends a function by reflecting it."""
    if sig not in (1, -1, "even", "odd"):
        raise ValueError
    if sig == "even": sig = 1
    if sig == "odd": sig = -1

    def inner(f, x):
        return np.where(
            x > about,
            f(x),
            np.where(
                x == about,
                f(x) if sig == 1 else 0,
                sig * f(2 * about - x)
            )
        )

    def reflected(f):
        return lambda x: inner(f, x)

    return reflected


def Clipping(maxamp=1):
    """Function decorator that returns a clipped version of a function.
    """
    def inner(f, x):
        y = f(x)
        return max(-1, min(1, y))

    def clipped(f):
        return lambda x: inner(f, x)

    return lambda f: clipped(f)


def CompactSupport(xmin, xmax):
    """Function decorator that returns a version of the function that is
    zero outside of specified interval.
    """
    def compact_f(f):
        return lambda x: 0 if (x > xmax or x < xmin) else f(x)

    return compact_f


def Vectorize(f):
    """Wrapper around np.vectorize for floats."""
    return np.vectorize(f, otypes=[float])

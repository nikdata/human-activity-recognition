"""
Compute transformations of the data which are not suitable
for use as features, but may be useful to other features.
These are motivated by the lack of gyroscope data; since we
do not have a consistent rotational frame, small changes in
acceleration components are unreliable. The magnitude of the
acceleration is reliable, and the average behavior over long
times is reliable. Hence, we look at only a filtered version
of the direction vector while using an unfiltered acceleration
norm.

Alden Bradford, January 27 2022
"""

import numpy as np
import pandas as pd
from scipy import signal


def direction(acceleration, frequency_cutoff=0.2, filter_order=8):
    """
    First filter out sharp changes in direction, then normalize to put all directions on the unit sphere.
    Parameters for the filter are available as parameters.

    Parameters
    ----------
    acceleration: DataFrame
        the components of acceleration.

    frequency_cutoff: float, optional
        the frequency (in Hertz) at which we filter. Motions faster than this will not be captured.

    filter_order: int
        the order of the Butterworth filter applied to the signal. A higher order will
        have a bigger effect, but also introduce more artifacts such as ringing.

    Returns
    -------
    direction: DataFrame
        an estimate of the direction of greatest acceleration at each time.
        This is normalized, so x^2+y^2+z^2=1.
    """
    t = len(acceleration.groupby("milliseconds"))
    direction = pd.DataFrame(
        signal.sosfiltfilt(
            sos=signal.butter(filter_order, frequency_cutoff, fs=25, output="sos"),
            x=acceleration.to_numpy().reshape([-1, t, 3]),
            axis=1,
        ).reshape([-1, 3]),
        index=acceleration.index,
        columns=acceleration.columns,
    )
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return direction


def magnitude(acceleration):
    """
    Find the magnitude of the acceleration.

    Parameters
    ----------
    acceleration: DataFrame
        the components of acceleration.

    Returns
    -------
    Series
        the magnitude of the acceleration.

    Example
    -------
    >>> import pandas as pd
    >>> a = pd.DataFrame([[3, 4, 0],[1, 2, 2]])
    >>> magnitude(a)
    0    5.0
    1    3.0
    Name: acceleration magnitude, dtype: float64
    """
    return pd.Series(
        np.linalg.norm(acceleration, axis=1),
        index=acceleration.index,
        name="acceleration magnitude",
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod()

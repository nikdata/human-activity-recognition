"""
Generate real-number features.

The list of features is automatically generated using function decorators.

To make a new feature, simply write a function which produces a data frame,
with the columns as labeled features and the rows as incidents indexed by id.
Then give it the decorator @feature, and if it makes sense to apply the feature
to a window, give it the decorator @window. Every feature must be able to accept
every part of the data, even those which it does not use -- the easiest way to
do this is to give it the argument **kwargs as a catch-all.

Alden Bradford, January 30 2022
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.signal import welch
from .transformations import magnitude, direction
from .load_data import load_data


# this will be populated by the decorator
feature_generators = []
window_features = []


def make_features(*args, **kwargs):
    """make every feature available as a data frame. This may take a while to run.

    Those features which make sense to apply to windows have their column name prepended with
    the window over which they were applied (the endpoints are given, in milliseconds).
    
    Parameters are passed to load_data.
    """
    i, a = load_data(*args, **kwargs)
    m = magnitude(a)
    d = direction(a)
    feat = [
        f(magnitude=m, direction=d, incidents=i, acceleration=a)
        for f in feature_generators
    ]
    return pd.concat(feat, axis="columns")


def feature(func):
    feature_generators.append(func)
    return func


def window(func):
    window_features.append(func)
    return func


@window
@feature
def simple_stats(magnitude, **kwargs):
    """
    All of these features treat the data as an unordered collection.
    They are standard statistical features.

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> index = pd.MultiIndex.from_product([range(4), range(10)], names = ['incident_id', 'milliseconds'])
    >>> m = pd.Series(np.linspace(0.8, 1.2, 40)**2, index = index)
    >>> print(simple_stats(m))
                  maximum   minimum      mean  variance      skew  kurtosis
    incident_id                                                            
    0            0.796213  0.640000  0.716844  0.002762  0.048027 -1.197775
    1            0.989770  0.814622  0.900934  0.003472  0.042838 -1.198230
    2            1.204366  1.010283  1.106062  0.004264  0.038660 -1.198558
    3            1.440000  1.226982  1.332229  0.005136  0.035225 -1.198803

    """
    group = magnitude.groupby("incident_id")
    return pd.DataFrame(
        {
            "maximum": group.max(),
            "minimum": group.min(),
            "mean": group.mean(),
            "variance": group.var(),
            "skew": group.skew(),
            "kurtosis": group.apply(lambda df: df.kurtosis()),
        }
    )


@window
@feature
def average_direction(acceleration, **kwargs):
    """
    This is just the average values of x, y, and z. Since we are doing an average over many samples,
    it makes more sense to compute this directly than to use the direction transfomration
    (the noise will be reduced).
    """
    return (acceleration
        .groupby('incident_id')
        .mean()
        .rename(columns = lambda s: f'mean {s}')
    )
    


@feature
def stillness(magnitude, rest_time=1000, **kwargs):
    """
    Is there any period over which the motion is very small?
    This is a quantification of the movement during the stillest portion
    of the incident. A lower value indicates the device was more still.
    By thresholding this value, we can pick out incidents with a horizontal
    portion on their graphs.

    Parameters
    ----------
    rest_time: float or int, optional
        the width of the window over which we look for stillness.

    Returns
    -------
    stillness
        a lower value indicates that, during the window when the motion was
        most still, there was very little movement. A higher value indicates that
        the motion was consistently high.

    middle of stillness
        this is the center point of the interval, in seconds, where the given stillness
        was achieved.

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> index = pd.MultiIndex.from_product([[0], 40*np.arange(50)], names = ['incident_id', 'milliseconds'])
    >>> m = pd.Series(np.linspace(0, 1)**2, index = index)
    >>> print(stillness(m))
                 stillness  middle of stillness
    incident_id                                
    0             0.076165                  0.5
    >>> m.iloc[25:] = 1
    >>> print(stillness(m))
                 stillness  middle of stillness
    incident_id                                
    0                  0.0                  1.5
    """
    rolling_var = (
        magnitude.groupby("incident_id")
        .rolling(rest_time // 40)
        .std()
        .reset_index(0, drop=True)
        .dropna()
        .groupby("incident_id")
    )
    return pd.DataFrame(
        {
            "stillness": rolling_var.min(),
            "middle of stillness": (
                rolling_var.apply(lambda df: df.argmin()) * 40 + rest_time / 2
            )
            / 1000,
        }
    )


@window
@feature
def angle_path(direction, **kwargs):
    """
    These are two measures of the path the direction takes along the unit sphere.
    Treating the orientation as a vector, it is constrained to the sphere so the
    reasonable way to treat distance is as the angel between points.

    Returns
    -------
    angular path length
        For the given interval, what is the length of the path the direction took?
        This is computed using a simple secant-line (or rather, secant arc) sum.
        Since the path is slow-moving (the direction was filtered to remove fast movements)
        this will always be a good approximation.

    biggest angle difference
        All the points are compared to find the two which are farthest apart. Their
        angular distance is returned. This is a measure of how far the direction wanders.
        This is the slowest-to-compute of the features, though I am hopeful that there
        is a better algorithm to compute this. Even though it is the slowest, this is
        still computed quickly.

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> index = pd.MultiIndex.from_product([[0], range(3)], names = ['incident_id', 'milliseconds'])
    >>> d = pd.DataFrame(np.eye(3), index = index)
    >>> print(angle_path(d))
                 angular path length  biggest angle difference
    incident_id                                               
    0                       3.141593                  1.570796
    """
    index = direction.groupby("incident_id").first().index
    d = direction.to_numpy().reshape([len(index), -1, 3])

    inner = (d[:, 1:, :] * d[:, :-1, :]).sum(axis=2)
    path_length = pd.Series(np.arccos(inner).sum(axis=1), index=index)

    biggest_angle = pd.Series(
        np.arccos(1 - np.array([pdist(vecs, "cosine").max() for vecs in d])),
        index=index,
    )

    return pd.DataFrame(
        {"angular path length": path_length, "biggest angle difference": biggest_angle}
    )


@feature
def angle_between_incident_and_vertical(acceleration, direction, **kwargs):
    """
    We can compare the angle between the triggering jolt (the central peak
    of the acceleration curve) and the direction of vertical at that time.
    This is given in radians.
    """
    a = acceleration.xs(key=7520, level=1)
    d = direction.xs(key=7520, level=1)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    angle = np.arccos((a * d).sum(axis=1))
    return pd.DataFrame({"angle between incident and vertical": angle})


@window
@feature
def spectral_power(magnitude, low_threshold=1, high_threshold=3, **kwargs):
    """
    Using Welch's method, compute the power density in three parts of the spectrum.
    This follows the discussion in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7378757/.

    The thresholds define where we should cut off each pfrequency band, and are measured in hertz.
    """
    t = len(magnitude.groupby("milliseconds"))
    m = magnitude.to_numpy().reshape([-1, t])
    # This replicates the default behavior, but by making it explicit we eliminate a warning:
    nperseg = min(t, 256)
    frequency, power = welch(m, fs=15, scaling="spectrum", nperseg=nperseg)
    low = power[:, frequency < low_threshold].sum(axis=1)
    medium = power[:, (low_threshold <= frequency) & (frequency < high_threshold)].sum(
        axis=1
    )
    high = power[:, high_threshold <= frequency].sum(axis=1)
    return pd.DataFrame(
        {
            "low frequency power": low,
            "medium frequency power": medium,
            "high frequency power": high,
        },
        index=magnitude.groupby("incident_id").first().index,
    )


@feature
def windows(magnitude, direction, acceleration, n=5, overlap=True, **kwargs):
    """
    Repeat all the marked features on each of `n` evenly spaced windows.
    If overlap, they are overlapped evenly, like so:

        |-------|-------|-------|
            |-------|-------|

    The resulting frames are tagged by their end points, with the lower
    left endtime noninclusive. This matches the convention that the first
    measurement is given time 40 milliseconds.
    """
    milliseconds = magnitude.index.to_frame()["milliseconds"]
    if overlap:
        step = 15000 // (n + 1)
        size = step * 2
    else:
        step = 15000 // n
        size = step
    endpoints = [(a := step * i, a + size) for i in range(n)]
    times = [
        (milliseconds > tmin) & (milliseconds <= tmax) for (tmin, tmax) in endpoints
    ]
    frames = [
        pd.concat(
            [
                f(
                    magnitude=magnitude[t],
                    direction=direction[t],
                    acceleration=acceleration[t],
                )
                for f in window_features
            ],
            axis="columns",
        )
        for t in times
    ]
    for (tmin, tmax), frame in zip(endpoints, frames):
        frame.rename(
            columns=lambda name: f"window {tmin}:{tmax} {name}",
            inplace=True,
        )
    return pd.concat(frames, axis="columns")

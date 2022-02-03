"""
Load the Makusafe data into a table.

Alden Bradford, January 27 2022
"""


import pandas as pd
from .transformations import batch

dtype = {
    "incident_id": int,
    "sample_number": int,
    "hash_id": "category",
    "motion": "category",
    "miliseconds": int,
    "seconds": int,
    "x": float,
    "y": float,
    "z": float,
}

dates = ["occurrence_ts", "confirmation_ts"]

default_filename = (
    "/depot/tdm-musafe/data/human-activity-recognition/raw-data/har_raw.gz"
)


def load_data(filename=default_filename, drop_batches=True, drop_early=True):
    """Load the makusafe data and make them available as DataFrames.

    In order to keep the data in third normal form (without redundant columns),
    the data is split into two frames. By loading the data from scratch whenever
    it is needed, we ensure that the data cannot be corrupted by other processes;
    however, within a process, it is better to keep the frame in memory
    (it should be stored as a variable)

    Parameters
    ----------
    filename: str, optional
        The name of the file from which to load the data.
        
    drop_batches: bool, optional
        if True, give a reduced data set with only those points whose classification
        was not part of a batch.
        
    drop_early: bool, optional
        if True, give a reduced data set with only those points which occurred after
        November 30, 2020. Data points before this time have a distinctly different character,
        as employers were still learning how the system worked at that time.

    Returns
    -------
    incidents: DataFrame
    acceleration: DataFrame

    Examples
    --------
    >>> incidents, acceleration = load_data()
    >>> list(incidents)
    ['hash_id', 'motion', 'occurrence_ts', 'confirmation_ts']
    >>> list(acceleration)
    ['x', 'y', 'z']
    >>> list(acceleration.index.names)
    ['incident_id', 'milliseconds']
    """
    raw_data = pd.read_csv(filename, dtype=dtype, parse_dates=dates)
    incidents = (
        raw_data[
            ["hash_id", "motion", "incident_id", "occurrence_ts", "confirmation_ts"]
        ]
        .groupby("incident_id")
        .first()
    )
    acceleration = raw_data[["incident_id", "milliseconds", "x", "y", "z"]].set_index(
        ["incident_id", "milliseconds"]
    )
    
    # get a true array to use as a mask
    mask = incidents['hash_id'].apply(lambda x: True)
    if drop_early:
        mask &= incidents['occurrence_ts'] > pd.Timestamp('November 30 2020', tz=0)
    if drop_batches:
        mask &= batch(incidents) == -1
        
    # we may have lost all the representatives of some categories, so we had better
    # remove those categories
    incidents = incidents[mask]
    for col in 'hash_id', "motion":
        incidents[col] = incidents[col].cat.remove_unused_categories()
    return incidents, acceleration.loc[mask[mask].index]
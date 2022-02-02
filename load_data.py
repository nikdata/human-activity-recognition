"""
Load the Makusafe data into a table.

Alden Bradford, January 27 2022
"""


import pandas as pd

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


def load_data(filename=default_filename):
    """Load the makusafe data and make them available as DataFrames.

    In order to keep the data in third normal form (without redundant columns),
    the data is split into two frames. By loading the data from scratch whenever
    it is needed, we ensure that the data cannot be corrupted by other processes;
    however, within a process, it is better to keep the frame in memory
    (it should be stored as a variable)

    Parameters
    ----------
    filename : str, optional
        The name of the file from which to load the data.

    Returns
    -------
    incidents : DataFrame
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
    return incidents, acceleration


if __name__ == "__main__":
    import doctest

    doctest.testmod()

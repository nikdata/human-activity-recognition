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
    >>> incidents
                                          hash_id  ...           confirmation_ts
    incident_id                                    ...                          
    729353       0c96025713b01a04beff5193cbf7d76d  ... 2020-10-29 08:28:48+00:00
    729389       0c96025713b01a04beff5193cbf7d76d  ... 2020-10-29 08:33:32+00:00
    729405       0c96025713b01a04beff5193cbf7d76d  ... 2020-10-28 23:11:05+00:00
    730067       0c96025713b01a04beff5193cbf7d76d  ... 2020-10-29 22:45:21+00:00
    730071       0c96025713b01a04beff5193cbf7d76d  ... 2020-10-30 08:33:57+00:00
    ...                                       ...  ...                       ...
    10333264     a2845cf621b2c5d99603722839247eb6  ... 2022-01-20 13:35:46+00:00
    10333546     5677e1b83984ef28263641675407c914  ... 2022-01-20 13:47:15+00:00
    10333611     a2845cf621b2c5d99603722839247eb6  ... 2022-01-20 13:35:07+00:00
    10344286     3c2f5925dd72cb7982b6fa1efe062dcc  ... 2022-01-20 17:00:31+00:00
    10356133     0c96025713b01a04beff5193cbf7d76d  ... 2022-01-21 13:34:11+00:00
    <BLANKLINE>
    [2770 rows x 4 columns]
    >>> acceleration
                                 x     y     z
    incident_id milliseconds                  
    7543315     40           -0.21  0.68  0.41
                80           -0.10  0.55  0.41
                120          -0.06  0.55  0.46
                160          -0.15  0.73  0.54
                200          -0.09  0.75  0.60
    ...                        ...   ...   ...
    7585893     14840        -0.48  0.36  0.73
                14880        -0.48  0.41  0.75
                14920        -0.45  0.42  0.80
                14960        -0.45  0.45  0.80
                15000        -0.39  0.39  0.82
    <BLANKLINE>
    [1038750 rows x 3 columns]
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

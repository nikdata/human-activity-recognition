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

default_filename = "~/depot/tdm-musafe/data/human-activity-recognition/raw-data/2022_makusafe-purdue-dataset-raw.gz"


def load_data(filename=default_filename):
    """Load the makusafe data and make them available as DataFrames.

    In order to keep the data in third normal form (without redundant columns),
    the data is split into two frames. By loading the data from scratch whenever
    it is needed, we ensure that the data cannot be corrupted by other processes;
    however, within a process, it is better to keep the frame in memory
    (it should be stored as a variable)

    The default filename uses a symlink which should be in your home directory.
    You can set that up by running the following block of code in a Jupyter notebook:

    %%bash
    ln -s /depot $HOME/depot

    Parameters
    ----------
    filename : str, optional
        The name of the file from which to load the data.

    Returns
    -------
    incidents : DataFrame
    acceleration: DataFrame

    Raises
    ------
    FileNotFoundError
        If the data file is inaccessible. Gives a special message if the filename is the default, since this
        probably indicates the symlink was not configured correctly.

    Examples
    --------
    >>> incidents, acceleration = load_data()
    >>> incidents
                                          hash_id motion
    incident_id
    729353       0c96025713b01a04beff5193cbf7d76d  other
    729389       0c96025713b01a04beff5193cbf7d76d  other
    729405       0c96025713b01a04beff5193cbf7d76d  other
    730067       0c96025713b01a04beff5193cbf7d76d  other
    730071       0c96025713b01a04beff5193cbf7d76d  other
    ...                                       ...    ...
    9873102      75acb015aa2cb4071f26f610e528aa80  other
    9875834      3c2f5925dd72cb7982b6fa1efe062dcc   trip
    9876909      3c2f5925dd72cb7982b6fa1efe062dcc   trip
    9887324      5677e1b83984ef28263641675407c914  other
    9889142      5677e1b83984ef28263641675407c914  other
    <BLANKLINE>
    [2630 rows x 2 columns]
    >>> acceleration
                                 x     y     z
    incident_id milliseconds
    9289528     40            0.13  1.04 -0.17
                80           -0.70  0.98  0.05
                120           0.05  0.98  0.35
                160           0.22  0.95  0.21
                200          -0.28  0.95  0.02
    ...                        ...   ...   ...
    6974639     14840        -0.82  0.27  0.52
                14880        -0.78  0.26  0.59
                14920        -0.89  0.16  0.59
                14960        -0.94  0.09  0.61
                15000        -0.94  0.11  0.61
    <BLANKLINE>
    [986250 rows x 3 columns]
    """
    try:
        raw_data = pd.read_csv(filename, dtype=dtype)
    except FileNotFoundError as e:
        if filename == default_filename:
            raise FileNotFoundError(
                "It looks like you don't have the symbolic link to the file directory configured correctly. You need to run the following command to set that up: \nln -s /depot $HOME/depot"
            )
        else:
            raise e
    incidents = (
        raw_data[["hash_id", "motion", "incident_id"]].groupby("incident_id").first()
    )
    acceleration = raw_data[["incident_id", "milliseconds", "x", "y", "z"]].set_index(
        ["incident_id", "milliseconds"]
    )
    return incidents, acceleration


if __name__ == "__main__":
    import doctest

    doctest.testmod()

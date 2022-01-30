"""
A module to create features based on the Makusafe data set.
Basic usage would be:

from make_features import make_features
f = make_features()
f.to_csv('features.csv')

The module is spolit into three submodules:
load_data
    Acquire the data set, with correct formatting. This may be helpful
    for other parts of the project as well.
transformations
    Holds functions which do not themselves create features, but rather
    present useful indirect representations of the original data.
features
    Define the features and make them available as a data frame.

To contribute, use git to clone the repository into your own directory
and then use git push to share your commits.

Alden Bradford, January 27 2022
"""
from .load_data import load_data
from .features import make_features

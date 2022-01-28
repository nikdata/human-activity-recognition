"""
A module to create features based on the Makusafe data set.
Basic usage would be:

from make_features import make_features
f = make_features()
f.to_csv('features.csv')

Alden Bradford, January 27 2022
"""
from .load_data import load_data
from .features import make_features

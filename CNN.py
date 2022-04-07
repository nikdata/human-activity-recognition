"""
Convolutional Neural Network code 

4 April 2022
"""

"""
Load relevant libraries

"""

#allow loading make_features from anywhere:
import sys
sys.path.append('/depot/tdm-musafe/apps')

from make_features import load_data
from make_features import make_features
# from make_features import make_undirectey

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import OneSidedSelection
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import torch.nn.functional as F
import torch

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
from matplotlib import pyplot as plt

#import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
#from keras.utils import to_categorical
from keras.utils.np_utils import to_categorical
#from keras.utils import to_categorical
import plotly.graph_objects as go
from matplotlib import pyplot
import random
from numpy import mean
from numpy import std



"""
Load data function

input value: (Name/Type/Definition)
use_features    | True/False  | False if only raw dataset needed, True if all features needed
window_count    | int         | the number of overlapping windows in the 15 sec data


Output values:

X | DataFrame | All columns except the target
Y | DataFrame | Target column
"""

def cnn_load_data(use_features = False, window_count = 100):
    X = []
    Y = []
    
    dfRawData = load_data(drop_batches = False, drop_early = True)
    dfClassData = dfRawData[0]    
    
    
    if use_features:
        dfFeatures = make_features(drop_batches = False, drop_early = True, feature_parameters = {'n':window_count})

        dfFeatures = dfFeatures.iloc[:,23:]

        X = dfFeatures

    else:
        dfAccData = dfRawData[1]

        X = dfAccData

    
    Y = dfClassData
    
    return X, Y

"""
Process data function

input value: (Name/Type/Definition)
X            |             | 
Y            |             | 
use_features | True/False  | False if only raw dataset needed, True if all features needed
window_count | int         | the number of overlapping windows in the 15 sec data


Output values:

X | DataFrame | All columns except the target
Y | DataFrame | Target column

"""

def cnn_process_data(X, Y, use_features = False, window_count=100):
    
    if use_features:

        X = X.to_numpy().reshape(X.shape[0], window_count, 17)

    else:

        X = X.to_numpy().reshape(Y.shape[0], (int)(X.shape[0]/Y.shape[0]), 3)

        
    Y = pd.DataFrame(Y["motion"]) 
    Y['motion'] = Y['motion'].map({'trip':1, 'slip':2, 'fall':3, 'other':0})
    Y = Y.reset_index(drop=True)   
    Y = to_categorical(Y)
    
    #print(dfRawData)
    
    return X, Y


"""
Validation data splitter function


"""

def validation_data_spliter(X, Y, validation_ratio, is_advance = True, index_arr = None):

    X_other = X[Y[:,0]==1]
    X_trip = X[Y[:,1]==1]
    X_slip = X[Y[:,2]==1]
    X_fall = X[Y[:,3]==1]
    
    validation_ratiop = 1-validation_ratio

    other_index = np.random.choice(range(len(X_other)), 
                                       size=int(len(X_other)*validation_ratio), 
                                       replace=False)
    trip_index = np.random.choice(range(len(X_trip)), 
                                       size=int(len(X_trip)*validation_ratio), 
                                       replace=False)
    slip_index = np.random.choice(range(len(X_slip)), 
                                       size=int(len(X_slip)*validation_ratio), 
                                       replace=False)
    fall_index = np.random.choice(range(len(X_fall)), 
                                       size=int(len(X_fall)*validation_ratio), 
                                       replace=False)
    valid_index_arr = []
    train_index_arr = []
    
    if index_arr is not None:
        other_index_arr = index_arr[Y[:,0]==1]
        trip_index_arr = index_arr[Y[:,1]==1]
        slip_index_arr = index_arr[Y[:,2]==1]
        fall_index_arr = index_arr[Y[:,3]==1]
        
        other_valid_index_arr = other_index_arr[other_index]
        trip_valid_index_arr = trip_index_arr[trip_index]
        slip_valid_index_arr = slip_index_arr[slip_index]
        fall_valid_index_arr = fall_index_arr[fall_index]
        
        valid_index_arr = np.concatenate([other_valid_index_arr,
                                         trip_valid_index_arr,
                                         slip_valid_index_arr,
                                         fall_valid_index_arr])
        
        train_index_arr = np.concatenate([np.delete(other_index_arr, other_index, axis=0),
                                         np.delete(trip_index_arr, trip_index, axis=0),
                                         np.delete(slip_index_arr, slip_index, axis=0),
                                         np.delete(fall_index_arr, fall_index, axis=0)])
                                       
        
    X_other_valid = X_other[other_index]
    X_trip_valid = X_trip[trip_index]
    X_slip_valid = X_slip[slip_index]
    X_fall_valid = X_fall[fall_index]
    
    
    Y_train = np.array([1,0,0,0] * len(X_other_valid) + 
                       [0,1,0,0] * len(X_trip_valid) + 
                       [0,0,1,0] * len(X_slip_valid) + 
                       [0,0,0,1] * len(X_fall_valid))
    Y_train =Y_train.reshape(int(len(Y_train)/4),4)
    X_train = np.concatenate([X_other_valid, X_trip_valid, X_slip_valid, X_fall_valid])

    
    Y_valid = np.array([1,0,0,0] * (len(X_other)-len(X_other_valid)) + 
                       [0,1,0,0] * (len(X_trip)-len(X_trip_valid)) + 
                       [0,0,1,0] * (len(X_slip)-len(X_slip_valid)) +
                       [0,0,0,1] * (len(X_fall)-len(X_fall_valid)))
    Y_valid =Y_valid.reshape(int(len(Y_valid)/4),4)
    X_valid = np.concatenate([np.delete(X_other, other_index, axis=0), 
                              np.delete(X_trip, trip_index, axis=0), 
                              np.delete(X_slip, slip_index, axis=0), 
                              np.delete(X_fall, fall_index, axis=0)])
    
    if index_arr is None:
        return X_train, Y_train, X_valid, Y_valid
    else:
        return X_train, Y_train, X_valid, Y_valid, train_index_arr, valid_index_arr


"""
Evaluate model funtion


"""

def evaluate_model(model, test_X, test_Y, batch_size=32):
    _, accuracy = model.evaluate(test_X, test_Y, batch_size=batch_size, verbose=0)
    result = model.predict(test_X, batch_size=batch_size, verbose=0)
    return accuracy, result

"""
Calling the functions

"""

X,Y = cnn_load_data()

index_arr = Y.index[:]
X,Y = cnn_process_data(X, Y)

trainX, trainY, validX, validY = validation_data_spliter(X, Y, 0.2)

model = train_model(X, Y)
score, result = evaluate_model(model, X, Y)

score
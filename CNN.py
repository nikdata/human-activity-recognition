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


pip list

"""
Load data function

This function uses load_data() provided with maker_features module;
It provides the option to choose between using raw data and using features as our model input;
If use_features is True, the function will process the data into segamented windows, with amount of windows specified by window_count;
If use_features is False, the function will simply return raw data;


input value: (Name/Type/Definition)
use_features    | True/False  | False if only raw dataset needed, True if all features needed
window_count    | int         | (Optional) the number of overlapping windows in the 15 sec data. Use when use_feature == True


Output values:

X | DataFrame | All columns except the target
Y | DataFrame | Target column
"""

def cnn_load_data(use_features = False, window_count = 100):
    #two containers for the results
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

This function convert and reshape data from cnn_load_data() into form accepted by our CNN model;


input value: (Name/Type/Definition)
X            | DataFrame   | the X dataframe returned by cnn_load_data()
Y            | DataFrame   | the Y dataframe returned by cnn_load_data()
use_features | True/False  | False if only raw dataset needed, True if all features needed
window_count | int         | the number of overlapping windows in the 15 sec data


Output values:

X | numpy array | All columns except the target
Y | numpy array | Target column

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

This function splits X, Y data from cnn_process_data() into validation set and training set.
The ratio between the two set can be adjusted by validation_ratio.
By using is_advance, the split will take place in 4 classes respectively, leading to a validation dataset with the same class ratio as the original dataset.
If an index array is provided, the split operation will also be performed on the index array so one can keep track of the indices of all incidents.

input value: (Name/Type/Definition)
X                | numpy array | the X array returned by cnn_process_data()
Y                | numpy array | the Y array returned by cnn_process_data()
validation_ratio | float       | the desired ratio between training and validation data; (i.e. a 0.1 ratio produce 1 part validation and 9 part training)
is_advance       | True/False  | the number of overlapping windows in the 15 sec data
index_arr        | numpy array | (Optional) When a index array is provided, the validation_data_spliter will keep track 

Output values:

X_train         | numpy array | the array containing feature values for the training set
Y_train         | numpy array | the array containing classifications for the training set
X_valid         | numpy array | the array containing feature values for the validation set
Y_valid         | numpy array | the array containing classifications for the validation set
train_index_arr | numpy array | (Optional) the array containing index information for the training set; returned when index_arr is provided.
valid_index_arr | numpy array | (Optional) the array containing index information for the validation set; returned when index_arr is provided.

model | keras model | the model after training that is capable of prediction;


NOTE: Do not specifiy index_arr. The functionality is reserved for possible future needs in identifying classification mistakes.
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


'''
Tom Tang 4/4/2022

input value: (Name/Type/Definition)
trainX          | numpy array | the parameters used for training model; numpy array shaped as 3d arrays
trainy          | numpy array | the true labels used for training model; processed by to_categorical()
new_filters     | int         | the amount of filters will be used in the model;
new_Kernel_size | int         | the size of each filter in the model;
new_strides     | int         | the distance filter will be moved each time;


Output values:

model | keras model | the model after training that is capable of prediction;

'''

def train_model(trainX, trainy, new_filters=64, new_Kernel_size=17, new_strides=8):
    verbose, epochs, batch_size = 0, 10, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    
    #model consist of 2 convolution layers
    model.add(Conv1D(filters=new_filters, kernel_size=new_Kernel_size, strides= new_strides,activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=new_filters, kernel_size=new_Kernel_size, strides= new_strides,activation='relu'))
    
    #followed by a dropout layer
    #model.add(Dropout(0.5))
    
    #and at last a maxpooling layer
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    return model


'''
Tom Tang 4/14/2022

input value: (Name/Type/Definition)
model           | keras model | the model to be evaluated
test_X          | numpy array | the parameters used for evaluation run; numpy array shaped as 3d arrays
test_Y          | numpy array | the true labels used for evaluation run; processed by to_categorical()
batch_size      | int         | the amount of data (incidents) being processed at a time; adjustable parameter; default 32;


Output values:

accuracy        | float       | the accuracy in prediction
result          | numpy array | the prediction result; 2D array, result[i,j], provides prediction for motion j of incident i;

'''

def evaluate_model(model, test_X, test_Y, batch_size=32):
    _, accuracy = model.evaluate(test_X, test_Y, batch_size=batch_size, verbose=0)
    result = model.predict(test_X, batch_size=batch_size, verbose=0)
    return accuracy, result




def y_converter(Y):
    
    new_df = pd.DataFrame(Y, columns= ['other', 'trip', 'slip', 'fall'])
    #new_df2 = pd.DataFrame(y_testDf, columns= ['other', 'trip', 'slip', 'fall'])

    result = new_df.idxmax(axis=1)
    #truth = new_df2.idxmax(axis=1)
    #prediction['index_id'] = index_arr

    #fail_num = compare_false(prediction,truth)
    
    #print(prediction)
    
    return result


def confusion_matrix_generator(Y_true, Y_pred):
    cm = confusion_matrix(Y_true, Y_pred, 
                          labels=['other', 'trip', 'slip', 'fall']) 

    cm_df = pd.DataFrame(cm,
                     index   = ['other', 'trip', 'slip', 'fall'], 
                     columns = ['other', 'trip', 'slip', 'fall'])

    
    plt.figure(figsize=(10,10))
    sns.heatmap(cm_df, annot=True, fmt="d", linewidths=0.5, 
                cmap='Blues', cbar=False, annot_kws={'size':14}, square=True)
    plt.title('Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(Y_true, Y_pred)))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    return cm_df






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
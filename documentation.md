# Introduction

What follows is a summary of what was learned by the Makusafe team during the Spring 2022 semester of the Purdue Data Mine Corporate Partners program. In addition, this serves as an index for the repository we created which stores the code we created. 

In brief, here is a table comparing a few of the models we applied as well as some performance indicators we computed.

method | number of variations attempted | accuracy | ppv
-------|--------------------------------|----------|-----
MLP    |                                |          |
CNN    |                                |          |
SVM    |                                |          |

## Terminology

The original data is classified into four motion categories, "slip", "trip", "fall", and "other". Most of the models we considered were focused on separating the "hazards" (slip, trip, and fall) from the others. We considered the hazards as the "positive" results, and the others as "negative".

From the original acceleration data, we computed numerical properties. Each of those properties is a floating point number, referred to as a "feature". The tools used to attempt classification are referret to as "models"

## Sources consulted

We based most of our strategies on two papers, [1] and [5] in the references below. These applied support vector machines and neural nets to classify lab-generated accelerometer data. We chose these papers because they managed to use only accelerometer data (no gyroscope or magnetometer data) to reliably separate falls from normal activities (activities of daily living, ADL). Source [5] led us to datasets [3], [4], and [6] which allowed us to compare the Makusafe data with existing public datasets.

# Exploration techniques
## Random Forest Classifier (RFC):
- uses a large number of decision trees that operate as an ensemble
- these trees are each responsible for a singular 'vote' on the classification of each motion, and whatever outcome receives the most votes is chosen
- This yielded a ranking of all the 109 features based on their diagnosed importance by Mean Decrease in Impurity (MDI)
- Path: `human-activity-recognition/exploration_and_visualization/Random_Forest.ipynb`
## PyCaret library:
- built-in methods to run the features dataset through many models
- revealed some useful models, but the results were riddled with false positives
- Path: `human-activity-recognition/exploration_and_visualization/pycaret_model_runthrough.ipynb`

## Graphical techniques

We made 3-dimensional animations of the raw acceleration data, available in `exploration_and_visualization/animations/`. This was helpful in two ways:

 - demonstrating that some of the accelerometers were mounted upside down.
 - discouraging any kinematics-based feature engineering. We found that the numerical integral of the acceleration data was highly sensitive to small rotations, which led to small drifts overwhelming any signals.

Another graphical technique we applied was a so-called "pair plot", which takes a list of one-dimensional features and compares them pairwise in a scatter plot. You can see an example of this in `exploration_and_visualization/other_data_comparison/`.

For our poster we produced an image with every acceleration curve, colored according to its motion label. This looks pretty, and communicates the scope of the problem domain. It can be found in `exploration_and_visualization/poster_backdrop/`.

## Exploratory factor analysis

This is a technique for identifying correlated features, which is commonly used in psychology to simplify data models for qualitative/survey data. "Factors" are computed, and each factor has a certain "loading" on each input variable. A higher loading indicates that a given feature contributes more toward a given factor. By highlighting the https://www.overleaf.com/project/625826a3a530b4d365a33325highest-loaded variables, we can get a qualitative description of each factor, and assign it a physical meaning.

This was used to identify redundant features. We had initially considered using overlapping feature windows, but we mostly abandoned that idea after seeing that the resulting features were heavily correlated. This is described in `exploration_and_visualization/feature reduction by factor analysis/`, and some code there was reused in `exploration_and_visualization/other_data_comparison/`.

## Clustering

Clustering techniques we tried were mostly fruitless, with one exception -- this was how we established that a significant number of incidents had a wearable upside-down, and that many of our early successes were identifying this property and nothing else. This was not recorded, but occurs reliably when kMeans is applied to the unprocessed acceleration data with k=2.

## Classification date analysis

This was simply a scatter plot of the incident occurrence time, against the classification time. The main thing this showed us was that several employers were collecting data for several weeks and then classifying it in batches, which show up on the scatter plot as horizontal lines. This is available in `exploration_and_visualization/time_to_confirmation/`.

# Feature generation

We based most of the features on the papers [1] and [5]. Every feature we computed is described briefly where it is defined, and includes a reference for where we got the inspiration to try it out. These are in `data_processing/make_features/features.py`, with the descriptions appearing in the associated docstrings. A brief description is also provided here.

## Transformations applied

Following the discussion in [5], we elected to compute most of our features based on the magnitude of the acceleration on the grounds that this should not be affected by how the accelerometer is worn. This is simply the Euclidean norm, sqrt(x²+y²+z²).

A few features use another transformation, which as far as I know is completely original. By putting the acceleration data through a low-pass filter, we should be able to remove variations which are faster than human-scale motion. The idea behind this is that it should give a more reliable indication of the orientation of the sensor, which is not so affected by short-duration jerks. The details of how this is computed are all in `data_processing/make_features/transformations.py`. We are only interested in recovering the direction from this transformation, so we normalize the resulting acceleration vectors to have magnitude one. In this way, we get a truly three-dimensional representation of our data -- one dimension for magnitude, and two dimensions for direction (since the unit sphere lying within three dimensions is a two-dimensional surface).

When other data sets were compared, two transformations were applied. First, for any data sets with a higher sampling rate, the data were resampled using Fourier theory to have the same sample rate as the Makusafe data. Then, all data (including the Makusafe data) were truncated to a eight-second window. This matches the strategy described in [5]. Also, for the other-data comparison we omitted features based on direction information -- hence the function `make_undirected`, which collects only those features which do not rely on direction information.

## Features computed

### simple statistics

many of the features are based on treating the data as an unordered collection of numbers, then applying standard summary statistics such as mean, variance, and standard deviation. These all use their standard definitions. they may be uninteresting on their own, but hopefully illuminate something when applied to data windows (see below).

### total variation

Inspired by [5] we compute the total variation which is just the gross change in acceleration.

### unprocessed acceleration features

This is from the raw data, without any transformation applied. `average_direction` is the time-averaged values of x, y, and z. This ended up being mostly unhelpful, as the noise from having sensors in multiple orientations dominated any signal. `peak_direction` is just the raw acceleration values during the central peak measurement.

### stillness

we noticed several incidents where, for a significant portion of the incident, the acceleration was not changing. We measured how still it got by taking a rolling variance, and at what time index this occurred.

### filtered angle based features

We have two measures of how much the angle of the sensor changed during the incident: one is measuring the path length along the unit sphere (using cosine similarity), and the other is finding the biggest difference in angle between any two times if the incident.

We also have a feature looking at the angle between the central peak acceleration value and "vertical", where in this case by vertical we mean the low-pass-filtered acceleration direction.

### spectral features

This is entirely based on the discussion in [1]. These use the so-called Welch method to measure how much of the signal's "power" is in each frequency band. In experiments, this mostly correlated with the signal's variance.

### window features

Many of the features were computed over windows (also called data segments), indicating a before, during, and after. We also tried computing features on overlapping windows, but this turned out to be a poor choice (see the discussion in exploratory factor analysis above). The window-based features were computed automatically and indiscriminately, computing every possible combination. The idea behind that was that it would be easy to drop an unused feature from a table at a later date. Windows are labeled according to the time period they cover.

## Brief Exploration of Time-Series Features

Since the features from [1] and [5] did not allow for high-PPV models to be build from them, a very quick exploration of the time series-based features generated by the Kats library (released by Facebook's Infrastructure Data Science) was done to see if there would be any improvement in PPV values. This feature set was ran through PyCaret's "compare models" function to assess any viable improvements in the wide range of classifier models it supports, but nothing meaningful was observed (sadly...). This exploration was done in the span of like 3 hours, so it is possible that there was some error in its implementation. The data may have been significantly altered when the time variable was changed from a "float" to a "datetime" type (necessary in order for Kats tsfeatures() function to work properly). Other time-series feature generating library are available and should be explored in the future.

# Models Attempted

## Multilayer Perceptron (MLP)
A Feedfoward Neural Network model, or MLP model, was attempted because of its ability in pattern finding in large quantity of data. Pytorch linear module is used to create a single feed-forward layer, which is repeated to create hidden and output layers for the MLP model. The sklearn package was used to implement improvement methods. We used K-Fold on validation data set to improve accuracy. To help with the imbalance in data we used SMOTE oversampling to generate more 'others' classification data. This did not produce any particularly impressive results, but it still gave some noticeable improvement. For understanding the package we referred to the official sklearn documentation. 

## Convolution Neural Network (CNN)
A convolutional neural network model or CNN model for short was the next step after the basic feed-forward model. This was chosen for its proficiency in processing data which has adjacency properties, meaning the data is continuous and that nearby points contain information about each other. For the CNN model we used the keras.layers.convolutional package which, similar to the MLP model, allowed us to construct the model layer by layer. The model specifically uses 2 convolution layers and a max-pooling layer, which selects the maximum values from the region covered by the layer. We used 'relu' for the activation function since the data contains only positive values.

### Imbalance Learning
The cleaned data contained at least a thousand values, with 'others' taking up almost 90\% of the values. The rest of the data could only add up to around 100 values, with one classification containing only 8 values. One way to solve this problem was by addressing the dataset directly, so we applied oversampling and undersampling. The SMOTE oversampling implementation failed because we have timeseries data, since the algorithm does not take into account the chronology of data, it may generate random data that is unreasonable given adjacent time data. Undersampling was attempted using sklearn.undersampling.NearMiss. This algorithm reduces majority dataset by taking looking at adjacent values, which makes it viable in timeseries data. Ultimately however, we are losing data values by using undersampling, so we could not substantially improve accuracy. Other undersampling methods like NeighbourhoodCleaningRule and OneSidedSelection were researched but unimplemented. The future plans are to combine undersampling with oversampling to reach a 1:1:1:1 ratio for all classifications. 

### K-fold Validation
To tackle the problem of high test error, we moved to k-fold cross validation. We used 10 folds, i.e. we made 10 parts of the training set and tested our algorithm on every part individually in 10 iterations while training the model on the other 9 parts. Since we were also dealing with imbalanced data, in every fold we used SMOTE to oversample the minority class and then trained the model. At the end we averaged the metrics to get the final performance of the model.

## Support Vector Machines (SVM)

An SVM model was attempted because of its simplicity and easy implementation. Furthermore, a study conducted by Seketa et al. [5] had relative success in implementing a SVM for an accelerometer-based detection algorithm, so it was appropriate to at least attempt a model to see what results it could bear. The sklearn.svm.LinearSVC package was used to build the linear (kernel) SVM. Only the regularization parameter "C" was adjusted to change much the hard-margin was reinforced (i.e. no data points are allowed to be within ±1 of the dividing hyperplane). The SVMs were only trained to classify two types of classes: (1) Events of interest like slip, trips, or falls, and (2) other events not contained in the aforementioned class. The hope was to build a SVM model that was really good at identifying 'other' events so that the events of interest (EOI) could be passed onto a different model that could then determine if those EOIs were slip, trips, or falls (i.e. an ensemble model).

Forward feature selection (FFS) was also used alongside the linear SVM to decide which features should be included in the model to reduce the chance of under- or over-fitting. In this case, the best feature set was selected to maximize the model's positive predictive value (PPV) - features stopped being added to the set once a dip in the PPV was observed. This could give the team an intuition as to what "types" of features would best differentiate between separate classes, and it would provide insight into what direction should be taken for future feature generation (i.e. if variance showed good promise, new features could be generated based on it). Overall, using FFS with an SVM was unsuccessful in building a good feature set to train with because the maximum PPV was observed when only one or two features were being used to train it. In retrospect, this does make sense based on Alden's work, which found that the features used in [1] and [5] (which the team based their feature set on) could not separate Makusafe's data set as well as the external data sets ([3],[4],[6]). The current feature set was incapable of differentiating classes, so of course the SVM would not be able to train a good classifier.

The SVM classifier implemented in the PyCaret library was also attempted to confirm/validate that this finding was consistent with a  well-established, high-level model building package. In the previous FFS SVM workflow (written entirely by the team), only z-score normalization was used to scale and center each feature to make them comparable with each other. The data was split into a training and validation set, and no synthetic data was created using SMOTE (all "real" data points). This yielded a PPV ~50% when only one feature was used [Need to fact check this ~ Chadwick]. However, a z-score normalization may not have been valid if the features did not follow a Gaussian distribution; also, the training and validation sets built may not have fully addressed the class imbalance present in the data (i.e. significantly more 'other' events than 'slip, trips, or falls'). As such, when an SVM model was being built in PyCaret, the following parameters were passed into the model initialization: (1) solving class imbalance via SMOTE, (2) make the features look 'normally distributed' using a Yeo-Johnson transformation, and (3) feature centering and scaling using z-score normalization. When these techniques were applied to both the data and model, the resulting PPV was ~20%. [Need to fact check this ~ Chadwick], which is lower compared to the previous FFS SVM workflow. This could be due to the fact that all of the features when used to train the PyCaret SVM model - there was no ability for the PyCaret workflow to arbitrarily remove a large amount of the feature set.

A very quick exploration of SVM and the time-series features from the Kats library showed no significant improvement to the PPV. In addition, based on the 'compare models' function implemented in PyCaret, an SVM may not have been the best model to use with this specific feature set. The only hyperparameter that had not be changed throughout this exploration was the 'kernel function' used to build the hyperplane between the two classes. This could be a potential area to explore in the future, but that could be a very deep rabbit hole to jump into.

# Conclusions

From the exploration we accomplished, it is clear that a straightforward application of existing state-of-the-art fall detection methods is insufficient for the type of data we were given. Despite this, we were able to extract some useful facts from the data set:

 - Many employers have used a batch classification strategy, where events are sometimes classified well after they occurred. This means that it may not be reasonable to expect the labels to reliably represent the actions performed.
 - Wearables have been placed in inconsistent locations. Since the same action can lead to very different accelerations on different parts of the body, this makes it hard to compare acceleration profiles fairly. Without more information on wearable placement, we would need geometrically-more data to be reasonably certain that we have captured the range of possible motions and the range of possible placements, since the possibilities would multiply.
 - With only acceleration data, we cannot reconstruct the 3-dimensional motion profile of the accelerometer. If we had more sensors, we may be able to produce a more easily-interpreted model, perhaps showing the exact path the accelerometer took though space. Conversely, most fall-detection schemes we compared relied on a significantly shorter time duration with more samples per second -- it could be that a 15 second window is overkill, giving us information which is not needed for classifying the motion.
 - None of the models attempted had an accuracy better than guessing, when we take into account the advantage they have with an imbalanced data set: a model will be right most of the time if it labels every incident as "other". Perhaps paradoxically, the PPV score can be misleading in this case as well -- when very few samples are classified as positive, the numbers are so small that we should have very little confidence that they will generalize to any data collected in the future.

# Recommendations

The following recommendations are based off our literature review, our exploration of the data provided, and observations of our attempted classification models. We cannot recommend the implementation of any of the models we considered, as none of them demonstrated any performance better than guessing. We uncovered some fundamental issues with how the data were collected, which need to be addressed in some way before a successful classification model is possible.

We recommend that MakuSafe review the choice of data collected, and investigate whether it would be feasible to change data collection protocols. Based on the literature we reviewed surrounding fall detection in accelerometry, there are a few ways that the MakuSafe data is significantly different from other data sets which have been successfully applied by others to classify motions. These differences are summarized in the following table. It is possible that, by gathering different data, a more sophisticated or discerning approach could be used, potentially giving reliable results without needing so much training data. In particular, real-time motion tracking requires a record of six degrees of freedom at minimum (usually accomplished with accelerometer + gyroscope) due to the physical laws involved, with more data preferred to account for sampling error. A magnetometer is often used in conjunction with accelerometry data, the resulting measurements being combined using a Kalman filter to get a reliable reconstruction of a motion profile. If other data are available, such as barometer, temperature, ambient light, or ambient sound level, these may also help to make a more robust model.

Attribute | MakuSafe | Other data sets
----------|----------|-----------------
sample rate | 25 Hz | 25-400 Hz
precision | single precision floating point (about 10 digits) | 3-4 digits of precision
smoothing | unsmoothed, raw data | sample rate chosen based off ADC input characteristics to prevent aliasing
duration | 15 seconds | 4-8 seconds
features | accelerometer components (3DOF) | accelerometer, gyroscope, and magnetometer components (9DOF)

We recommend that MakuSafe establish clear and unambiguous definitions for each motion label (slip, trip, fall) and ensure that the people generating labels are trained to use those definitions. The benefits of this are twofold. First, this will make it easier to compare labels generated by different employers. We noticed different prevalence rates for each motion between different employers, but based on the data we have it is impossible to say whether that is due to actual differences in what device wearers were doing, or whether this was only due to differences in how the events were classified. Second, by having clear definitions we can build a classification model which is context-aware, treating data according to the physical laws which are most relevant for each motion. This reduces the amount of training data required, since we would need to learn fewer of the physical laws from the training data.

We recommend that MakuSafe either enforce rules about where the wearable is to be worn on the body, or create some record of where the wearable was actually worn. This could be as simple as turning off fall detection if the employee chooses to wear the device in an unconventional place or orientation. By knowing where the device is worn, we significantly reduce the amount of information which a classification model would have to learn from the data, which would lead to reduced training data requirements and a more robust final model.

We recommend that MakuSafe arrange to gather some sample data in a controlled environment. In general, the characteristics of an accelerometry signal vary considerably from device to device, due to differences in device mass, moments of inertia, and ADC characteristics. This means that the best way to know the specific characteristics of the wearable would be to take measurements with the same wearable. Even if there are not enough measurements in a controlled environment to build an entire classification model, it would be very useful to have a few samples where we do not have to guess at the events which led to the measurement -- this would allow us to immediately rule out many methods if they cannot distinguish between controlled measurements.

If these recommendations are followed then it may be feasible to construct a reliable motion classification model for the MakuSafe wearables.

# References

 - [1] Turke Althobaiti, Stamos Katsigiannis, and Naeem Ramzan. “Triaxial accelerometer-
based falls and activities of daily life detection using machine learning”. In:
*Sensors* 20.13 (2020), p. 3777.

 - [2] Makusafe. *Makusafe Human Activity Recognition dataset.* 2022. url: https:
//github.com/nikdata/human-activity-recognition.

 - [3] Ahmet Turan Ozdemir and Billur Barshan. “Detecting falls with wear-
able sensors using machine learning techniques”. In: *Sensors* 14.6 (2014),
pp. 10691–10708.

 - [4] Majd SALEH and Regine LE BOUQUIN JEANNES. *FallAllD: A Compre-
hensive Dataset of Human Falls and Activities of Daily Living.* 2020. doi:
10.21227/bnya-mn34. url: https://dx.doi.org/10.21227/bnya-mn34.

 - [5] Goran Seketa et al. “Event-centered data segmentation in accelerometer-
based fall detection algorithms”. In: *Sensors* 21.13 (2021), p. 4335.

 - [6] Angela Sucerquia, Jose David Lopez, and Jesus Francisco Vargas-Bonilla.
“SisFall: A fall and movement dataset”. In: *Sensors* 17.1 (2017), p. 198.


<!-- README.md is generated from README.Rmd. Please edit that file -->

# Human Activity Recognition

<!-- badges: start -->

[![MIT
License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
<!-- badges: end -->

This repository contains raw accelerometer data for a wearable that is
worn by a user and determines if a user has experienced ‘harmful’
motion:

-   slip
-   trip
-   fall
-   other

## About the Data

The dataset is available as a compressed `.gz` file and consists of 9
variables (i.e., columns) with 986,250 rows.

|   Variable    | Data Type | Description                                                                                                                                                                               |
|:-------------:|:---------:|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|    hash_id    |  string   | anonymized id that could indicate the wearer of the device                                                                                                                                |
|  incident_id  |  double   | ID number of an incident. Every time the wearable sends data about a potential motion, this instance is known as an incident.                                                             |
|    motion     |  string   | type of motion that wearer experienced (verified by the wearer)                                                                                                                           |
| sample_number |  double   | sample number of the recorded data from accelerometer. Each incident consists of 375 samples                                                                                              |
| milliseconds  |  double   | derived column that describes the ‘time’ elapsed since the last data point. Each sample is recorded at 40 ms. The sampling frequency is 25 Hz (i.e., 25 samples are collected per second) |
|    seconds    |  double   | derived column that describes which ‘second’ this data belongs to. Each incident consists of 15 seconds of data.                                                                          |
|       x       |  double   | the acceleration (in G’s) of the wearable in the X-axis                                                                                                                                   |
|       y       |  double   | the acceleration (in G’s) of the wearable in the Y-axis                                                                                                                                   |
|       z       |  double   | the acceleration (in G’s) of the wearable in the Z-axis                                                                                                                                   |

## FAQ

**What is the sampling frequency?**

The sampling frequency of the wearable worn was 25 Hz - meaning that 25
samples (or readings) were collected per second from the accelerometer

**Where does the 15 seconds come from?**

While the wearable’s accelerometer is continuously ‘listening’ to the
motion of the wearer, it was designed to ‘start’ recording once 2Gs of
acceleration was exceeded in either the X, Y, or Z axes. The moment of
time that when this threshold is exceeded is at the mid-point. And the
wearable ‘stores’ the previous 7.5 seconds of information and the
following 7.5 seconds. Hence, the 15 seconds.

**What is an incident?**

An incident is when the wearable’s accelerometer exceeds 2Gs of
acceleration in either the X, Y, or Z axes. It is at this moment in time
when the wearable will record the previous 7.5 seconds and the following
7.5 seconds. This 15 second data-stream is referred to as an incident
and classified as either a slip, trip, fall, or other.

**Where was the wearable worn?**

This information was not recorded for every wearer. However, all wearers
were instructed to wear the wearable on their upper arm (around their
biceps) on either their right or left arms.

**Are timestamps available for each incident?**

Sorry, no.

## Getting Help

The best way to have questions answered about this dataset or repository
will be to open an
[issue](https://github.com/nikdata/human-activity-recognition/issues).

## Code of Conduct

Please note that this dataset and repository is released with a
[Contributor Code of
Conduct](https://github.com/nikdata/human-activity-recognition/blob/main/CODE_OF_CONDUCT.md).
By contributing to this project, you agree to abide by its terms.

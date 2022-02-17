This folder contains two things:
- `make_features`, the code which we are using to generate features for our models.
- `baseline_model`, which demonstrates how to use the `make_features` module, and demonstrates a few simple (baseline) models.

The models we demonstrate are:
- K Neighbors, which simply classifies new data according to distance from the training data in te features space.
- Decision tree
- Gaussian Naive Bayes, which assumes each class is distributed according to a Gaussian distribution in the features space.
All three use the same pre-scaled version of the data, where the means and veariances of each feature are set to 0 and 1 respectively.

The models were compared using stratified 5-fold cross validation. We used a stratified version in order to eliminate the chance of a training set having very few positive samples. Evaluating them based on the precision (the positive predictive value), we found that Gaussian Naive Bayes was most consistent, giving a precision around 10%. The K Neighbors method did the best on average, with a precision of at least 20% in each of the 5 trials. The decision tree method had highly variable behavior; in one trial it achieved 45% precision and in another it achieved only 7%.
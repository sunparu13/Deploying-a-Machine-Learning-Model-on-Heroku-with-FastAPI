# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Random forest classifier was used for salary prediction. Default configuration were used for training except the parameter n_estimators, which is set to 300.

## Intended Use

This model should be used to predict the salary category of the person with it's status attributes.

## Training Data

The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). The original data set has 32561 rows, and a 80-20 split was used to break this into a train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

## Evaluation Data

The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). The above-mentioned 20% splitted data was used for evaluation/testing.

## Metrics
The overall performance metrics on the test data set are:

Precision: 0.7397
Recall: 0.6294
F1 Score: 0.6802

## Ethical Considerations

The training data and the trained model were not analysed for the possible bias and discrimination.

## Caveats and Recommendations

None
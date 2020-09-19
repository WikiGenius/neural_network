#!python

########################################################
# data preperation for training network file
# Author: Muhammed El-Yamani
# muhammedelyamani92@gmail.com
# September 2020
########################################################

import numpy as np
import pandas as pd


def main():
    # get prepared data
    train_validation_test = prepare(
        csv_file='data/data.csv', fields=['x1', 'x2'], label='y',
        split=True, validation_split=True, dummy_var=None, debug=True)
    return train_validation_test


def process(csv_file, fields, dummy_var=None):
    data = pd.read_csv(csv_file)
    # Make dummy variables for rank
    if dummy_var:
        data = pd.concat([data, pd.get_dummies(
            data[dummy_var], prefix=dummy_var)], axis=1)
        data = data.drop(dummy_var, axis=1)
    # process data
    for field in fields:
        mean, std = data[field].mean(), data[field].std()
        data[field] = (data[field] - mean)/std
    return data


def prepare(csv_file, fields, label, split=True, validation_split=True, dummy_var=None, debug=True, random_seed=42):
    if debug:
        np.random.seed(random_seed)

    data = process(csv_file, fields, dummy_var)

    if split:
        # Split off random 10% of the data for testing
        train_data, test_data = split_data(data, ratio=0.9)
        # split training data into training and validation sets to tune the hyper parameter epochs
        if validation_split:
            train_data, validation_data = split_data(train_data, ratio=0.9)

        # Split into features and targets for train data
        features_train, targets_train = get_features_targets(train_data, label)
        # Split into features and targets for validation data
        if validation_split:
            features_validation, targets_validation = get_features_targets(
                validation_data, label)
        else:
            features_validation, targets_validation = features_train, targets_train
        # Split into features and targets for test data
        features_test, targets_test = get_features_targets(test_data, label)

    else:
        # Split into features and targets
        features, targets = get_features_targets(data, label)

        features_train, targets_train = features, targets
        features_validation, targets_validation = features, targets
        features_test, targets_test = features, targets

    return features_train, targets_train, features_validation, targets_validation, features_test, targets_test, data


def split_data(data, ratio):
    """
    data: pd.DataFrame and here for split
    ratio: random select some of indeces has ratio of the data for data1 and the other for data2
    Split data into two parts with ratio
    """
    sample = np.random.choice(
        data.index, size=int(len(data)*ratio), replace=False)

    data1, data2 = data.loc[sample], data.drop(sample)
    return data1, data2


def get_features_targets(data, label):
    """
    data: pd.DataFrame and here for split
    Split data into features and targets
    """
    features, targets = data.drop(label, axis=1), data[label]
    return features, targets


train_validation_test = main()
# get train data
features_train, targets_train = train_validation_test[0:2]
# get validation data
features_validation, targets_validation = train_validation_test[2:4]
# get test data
features_test, targets_test = train_validation_test[4:6]
# get the whole data
data = train_validation_test[6]

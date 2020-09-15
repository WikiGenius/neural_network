import numpy as np
import pandas as pd


def main():
    # get prepared data
    train_validation_test = prepare(
        csv_file='data.csv', fields=['x1', 'x2'], split=True)
    
    return train_validation_test

def process(csv_file, fields):
    data = pd.read_csv(csv_file)
    # process data
    for field in fields:
        mean, std = data[field].mean(), data[field].std()
        data[field] = (data[field] - mean)/std
    return data


def prepare(csv_file, fields, split=True):
    data = process(csv_file, fields)

    if split:
        np.random.seed(42)
        # Split off random 10% of the data for testing
        train_data, test_data = split_data(data, ratio=0.9)
        # split training data into training and validation sets to tune the hyper parameter epochs
        train_data, validation_data = split_data(train_data, ratio=0.8)

        # Split into features and targets for train data
        features_train, targets_train = get_features_targets(train_data)
        # Split into features and targets for validation data
        features_validation, targets_validation = get_features_targets(
            validation_data)
        # Split into features and targets for test data
        features_test, targets_test = get_features_targets(test_data)
    else:
        # Split into features and targets
        features, targets = get_features_targets(data)

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


def get_features_targets(data):
    """
    data: pd.DataFrame and here for split
    Split data into features and targets
    """
    features, targets = data.drop('y', axis=1), data['y']
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
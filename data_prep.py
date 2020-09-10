import numpy as np
import pandas as pd

split_data = True

data = pd.read_csv('data.csv')
# process data
for field in ['x1', 'x2']:
    mean, std = data[field].mean(), data[field].std()
    data[field] = (data[field] - mean)/std

if split_data:
    # Split off random 10% of the data for testing
    # np.random.seed(42)
    sample = np.random.choice(
        data.index, size=int(len(data)*0.9), replace=False)
    data, test_data = data.iloc[sample], data.drop(sample)

    # Split into features and targets
    features, targets = data.drop('y', axis=1), data['y']
    features_test, targets_test = test_data.drop(
        'y', axis=1), test_data['y']
else:
    # Split into features and targets
    features, targets = data.drop('y', axis=1), data['y']
    features_test, targets_test = features, targets


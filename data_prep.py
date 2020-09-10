import numpy as np
import pandas as pd

data = pd.read_csv('xor.csv')

# process data
# data['bias'] = 1
for field in ['x1', 'x2']:
    mean, std = data[field].mean(), data[field].std()
    data[field] = (data[field] - mean)/std
# Split into features and targets
features, targets = data.drop('y', axis=1), data['y']

features, targets = np.array(features), np.array(targets)

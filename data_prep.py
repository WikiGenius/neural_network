import numpy as np
import pandas as pd

data = pd.read_csv('xor.csv')

# Split into features and targets
features, targets = data.drop('y', axis=1), data['y']
features, targets = np.array(features), np.array(targets)

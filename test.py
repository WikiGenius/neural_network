from scipy import stats
import numpy as np
from data_prep import features_train
# l = [0, 1, 2, -4, -3]
# x = np.array(l)

x = np.array(features_train)
print(x)
print(np.maximum(0,x))

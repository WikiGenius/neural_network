from scipy import stats
import numpy as np
from data_prep import features_train
# l = [0, 1, 2, -4, -3]
# x = np.array(l)

cc = np.array([
    [0.120, 0.34, np.power(10, 100)]
], np.float64)

print(1/(1+np.exp(-cc)))
print(1e-04 <= 1.2279896491044095e-06)

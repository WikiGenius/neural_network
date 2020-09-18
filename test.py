from scipy import stats
import numpy as np
from data_prep import features_train
# l = [0, 1, 2, -4, -3]
# x = np.array(l)

x = np.array(features_train)
print(x)
a = 1
y = np.where(x > 0, x, a * (np.exp(x)-1))
print(y)

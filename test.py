from data_prep import features
import numpy as np
try:
    print(0*np.log(0))
except RuntimeWarning as e:
    print(e)
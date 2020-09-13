# from data_prep import features
# import numpy as np
# try:
#     print(0*np.log(0))
# except RuntimeWarning as e:
#     print(e)
import numpy as np

rows = 10
x = np.outer(np.arange(1,5),np.linspace(-2,2,5))
print(x)
print(x[:,1])

import matplotlib.pyplot as plt

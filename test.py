#!python
import numpy as np
import pandas as pd

from rank_nullspace import rank, nullspace
from data_prep import features_train
from scipy.linalg import lu
# data = pd.DataFrame(np.random.randint(
#     4, 25, size=(4, 2)), columns=['x1', 'x2'])
# data['x3'] = data['x1'] + data['x2']
# print(data)

# print(f"rank user defined: {rank(data)}")
# print(f"rank np.linalg.rank: {np.linalg.matrix_rank(data)}")
# print(nullspace(data))

# A = np.array([[1, 2, 3], [2, 4, 2]])     # example for testing
# # A = np.array(data)
# print(f"A:\n{A}")
# U = lu(A)[2]
# print(f"U:\n{U}")
# lin_indep_columns = [np.flatnonzero(U[i, :])[0] for i in range(U.shape[0])]
# print(f"lin_indep_columns:\n{lin_indep_columns}")

x = np.array(np.random.randn(3,3))
x = np.array(np.arange(-3,6).reshape(3,3))

print(x)
print(0 in x)
print(np.round(x,decimals = 2))

# y = np.power(x, 2)
# print(y)
# print(np.sum(y))
# z = np.abs(x)
# print(z)
# print(np.sum(z))

from data_prep import features
import numpy as np
np.random.seed(42)
m, n = 2,3
x = np.random.normal(loc=0, scale=np.power(n, -0.5),
                   size=2)
y = np.random.normal(loc=0, scale=np.power(n, -0.5),
                     size=(4,2))
print(x)
print(np.r_[x, np.ones(1)])
# print(x[:-1,:])
print(np.ones(1))
if x.ndim == 1:
    m = 1
else: 
    m = x.shape[0]
x = np.r_[x, np.ones(m)]
if y.ndim == 1:
    m = 1
else: 
    m = y.shape[0]
y = np.c_[y, np.ones(m)]

print(x)
print(y)

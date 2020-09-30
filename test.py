from scipy import stats
import numpy as np
from data_prep import features_train

# Momentum
x = -6
v = 0
learning_rate = 0.5
lambd = 0.9
epochs = 1000
last = None
for i in range(epochs):
    gradient = 0.1 * (2*x*np.cos(x) - x**2 * np.sin(x) - 1)
    v = lambd * v + (1 - lambd) * gradient
    x = x - learning_rate * v

    print(f"v = {v} , x = {x}")


#  if last and gradient > last:
#         learning_rate /= 2
#         # if lambd >= 0.5:
#         #     lambd -= 0.1
#         print("increase")
#     else:
#         # if lambd <= 1:
#         #     lambd += 0.1
#         # learning_rate *= 2

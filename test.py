# import random
import numpy as np
# np.random.seed(42)


def decision_dropout(probability, num):
    return np.random.choice([0, 1], size=num, p=[probability, 1 - probability])


# x = decision_dropout(0.2, 10)
l = list()
for i in range(100):
    l.append(decision_dropout(0.2, 10))
l = np.array(l)
print(l.sum(axis=1).sum()/10)

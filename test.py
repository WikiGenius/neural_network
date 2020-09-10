from data_prep import targets
import numpy as np
# _, n_classes = targets.shape
# print(n_classes)

error_terms = [None for _ in range(2)]
# Calculate the first error term related to the output
error_terms[-1] = 0.5
print(error_terms)
x = [5]
print(np.issubdtype(type(x), np.int64))
print(np.issubdtype(type(x), np.object_))
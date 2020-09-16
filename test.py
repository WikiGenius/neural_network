import numpy as np
prob = 0
epsilon = 10**-10
# result = np.where(prob > 10**-10, np.log10(prob+epsilon), -10)
result = np.log(1+epsilon)
print(result)

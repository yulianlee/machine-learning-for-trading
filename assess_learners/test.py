import pandas as pd
import numpy as np

np.random.seed(42)

test = np.array([1,2,-3,-4])
print(test[[i > 0 for i in test]])

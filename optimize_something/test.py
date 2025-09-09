import numpy as np
import pandas as pd


dr = pd.Series([0.2,0.21, 0.201])
print(dr.shift(1))
print(dr/dr.shift(1) - 1)

print(np.sum([1.75353543e-01, 4.01581674e-01, 4.23064783e-01, 5.20417043e-18, 0.00000000e+00]))
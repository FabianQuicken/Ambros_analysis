import numpy as np


a = [np.nan, 2, 4]

a = np.asarray(a)

print(np.nanmean(a))
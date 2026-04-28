import numpy as np
a = [([1, 2, 3], [4, 5, 6]), ([7, 8, 9], [10, 11, 12]), ([13, 14, 15], [16, 17, 18])]


xs = []

for t in a:
    end = len(t[0])
    counter = 0
    while counter < end:
        xs.append(t[0][counter])
        counter += 1
    xs.append(np.nan)

print(xs)
print(len(xs))
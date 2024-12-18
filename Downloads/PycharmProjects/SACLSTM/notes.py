import numpy as np
import matplotlib.pyplot as plt
X = np.random.rand(10000)
plt.hist(X, bins=100)

for i in range(100):
    print(i)

bins = np.linspace(min(X) - 1e-6, max(X) + 1e-6, 100 + 1)
bins
for i in range(100):
    if i == 0:
        X[(X > bins[0]) & (X < bins[1])] = 0
    elif i ==100 - 1:
        X[(X > bins[100 - 1]) & (X < bins[100])] = 100 - 1
    else:
        X[(X > bins[i]) & (X < bins[i + 1])] = i

X = X.astype(int)
X

discretized_path = np.digitize(X, bins, right=False) - 1

count_matrix = np.zeros((100, 100))

len_paths = int(len(X) - 20)
for i in range(len_paths):
    path = X[i: i + 20]

    '''transitions = path[:self.tau - lag_time], path[lag_time:]  # Two slices for transitions

    # Count the transitions
    for start, end in zip(*transitions):
        count_matrix[start, end] += M[i]'''

    count_matrix[path[0], path[20 - 1]] += 1

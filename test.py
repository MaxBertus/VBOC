import numpy as np

min_values = np.array([-1, -1, -1, 1, 1, 1])
max_values = np.array([-5, -10, -15, 5, 10, 15])
Q = len(min_values)
N = 5

# Generate the NxQ matrix
random_matrix = np.array([np.random.uniform(min_values[:3], max_values[:3]) for _ in range(N)])
print(random_matrix)

random_matrix = np.array([np.random.uniform(min_values[3:], max_values[3:]) for _ in range(N)])
print(random_matrix)
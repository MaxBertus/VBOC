import numpy as np
import matplotlib.pyplot as plt

def main():

    n = 10000
    d = np.empty((n,6)) * np.nan

    for i in range(n):
        d[i,:] = np.array([np.random.normal() for _ in range(6)])

    for j in range(d.shape[1]):
        plt.figure()
        plt.grid(True, which='both')
        plt.hist(d[:,j], bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.title(f'Histogram x[{j}]')
        plt.show(block=True) 

if __name__ == '__main__':
    main()
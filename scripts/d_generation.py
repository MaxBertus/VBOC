import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
import time

def generate_direction(n):
    # np.random.seed(n)
    d = np.array([np.random.normal() for _ in range(6)])
    d /= np.linalg.norm(d)
    time.sleep(0.1)  # Simulate some delay
    return d


def main():

    n = 200000
    sub_batch = 100
    d = np.empty((n,6)) * np.nan

    for i in range(n):
        np.random.seed(i)
        d[i,:] = np.array([np.random.normal() for _ in range(6)])
        d[i,:] /= np.linalg.norm(d[i,:])

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Directions generated with for loop")
    axes = axes.flatten()

    for i in range(d.shape[1]):
        axes[i].set_visible(True)
        axes[i].hist(d[:, i], bins=50, edgecolor='black', alpha=0.7)
        axes[i].set_title(f"Dimension {i+1}")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")
        axes[i].grid(True, which='both', alpha=0.75)

    n_batch = n // sub_batch

    for nb in tqdm(range(n_batch)):
        with Pool(26) as pool:
            d_pool = pool.map(generate_direction, range(nb * sub_batch, (nb + 1) * sub_batch))

    d_pool = np.array(d_pool)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Directions generated with Pool")
    axes = axes.flatten()

    for i in range(d_pool.shape[1]):
        axes[i].set_visible(True)
        axes[i].hist(d_pool[:, i], bins=50, edgecolor='black', alpha=0.7)
        axes[i].set_title(f"Dimension {i+1}")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")
        axes[i].grid(True, which='both', alpha=0.75)
    plt.show()



if __name__ == '__main__':
    main()
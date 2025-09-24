import numpy as np
import matplotlib.pyplot as plt
import os

def plot_histogram(data, title="Histogram", xlabel="Value", ylabel="Frequency", bins=30, saving_dir="debug/"):
    """
    Plots a histogram of the given data as subplot.

    Parameters:
    - data: array-like, the data to plot
    - title: str, title of the histogram
    - xlabel: str, label for the x-axis
    - ylabel: str, label for the y-axis
    - bins: int, number of bins in the histogram
    - saving_dir: str, directory to save the histogram image
    """

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(title)
    axes = axes.flatten()
    for i in range(len(axes)):
        axes[i].set_visible(False)  # Hide all subplots initially

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    for i in range(data.shape[1]):
        axes[i].set_visible(True)
        axes[i].hist(data[:, i], bins=bins, edgecolor='black', alpha=0.7)
        axes[i].set_title(f"Dimension {i+1}")
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel(ylabel)
        axes[i].grid(True, which='both', alpha=0.75)
    plt.show(block=False)

    plt.savefig( os.path.join(debug_dir, title + ".png"))

# Example usage:
if __name__ == "__main__":
    
    # Generate paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(script_dir, '..')
    data_dir = os.path.join(root_dir, 'extraData/')
    debug_dir = os.path.join(root_dir, 'debug/')

    x_data = np.load(data_dir + 'sth_x_vboc.npy')
    b_data = np.load(data_dir + 'sth_b_vboc.npy')
    b_all_data = np.load(data_dir + 'sth_b_all_vboc.npy')
    d_data = np.load(data_dir + 'sth_d_vboc.npy')
    status_data = np.load(data_dir + 'sth_status_vboc.npy')

    plot_histogram(x_data[:,:6], title="x[0:6]", xlabel="Value", ylabel="Frequency", bins=50, saving_dir=debug_dir)
    plot_histogram(x_data[:,6:], title="x[6:12]", xlabel="Value", ylabel="Frequency", bins=50, saving_dir=debug_dir)
    plot_histogram(b_data, title="b", xlabel="Value", ylabel="Frequency", bins=50, saving_dir=debug_dir)
    plot_histogram(b_all_data, title="b_all", xlabel="Value", ylabel="Frequency", bins=50, saving_dir=debug_dir)
    plot_histogram(-d_data, title="d", xlabel="Value", ylabel="Frequency", bins=50, saving_dir=debug_dir)
    plot_histogram(status_data, title="status", xlabel="Value", ylabel="Frequency", bins=10, saving_dir=debug_dir)
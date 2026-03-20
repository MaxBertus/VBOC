import os
import sys
import numpy as np

# Disable NumPy's automatic line wrapping when printing arrays
np.set_printoptions(linewidth=np.inf) 

if __name__ == "__main__":
    
    # --- Check input path
    if len(sys.argv) != 2:
        print("Usage: python3 analyze_npy.py <path_to_npy_file>")
        sys.exit(1)
    
    # --- Load data ---
    file_path = sys.argv[1]
    x_data = np.load(f'{file_path}/sth_x_vboc.npy')
    b_data = np.load(f'{file_path}/sth_b_vboc.npy')

    # --- Print data samples ---
    for i in range(len(x_data)):
        if i % 1000 == 0:
            print(f"X_DATA[{i}]: {x_data[i]} --- B_DATA[{i}]: {b_data[i]}")


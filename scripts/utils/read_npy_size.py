import numpy as np
import os
import sys

def get_npy_file_size(file_path):
    """
    Get the size information of a .npy file
    
    Args:
        file_path (str): Path to the .npy file
    
    Returns:
        dict: Dictionary containing size information
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file size on disk (in bytes)
        file_size_bytes = os.path.getsize(file_path)
        
        # Load the numpy array to get array information
        array = np.load(file_path)
        
        # Get array properties
        array_shape = array.shape
        array_dtype = array.dtype
        array_size = array.size  # Total number of elements
        array_memory_size = array.nbytes  # Memory size in bytes
        
        return {
            'file_path': file_path,
            'file_size_bytes': file_size_bytes,
            'file_size_mb': file_size_bytes / (1024 * 1024),
            'array_shape': array_shape,
            'array_dtype': array_dtype,
            'array_size': array_size,
            'array_memory_size_bytes': array_memory_size,
            'array_memory_size_mb': array_memory_size / (1024 * 1024)
        }
    
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def print_size_info(size_info):
    """Print formatted size information"""
    if size_info is None:
        return
    
    print(f"File: {size_info['file_path']}")
    print(f"File size on disk: {size_info['file_size_bytes']:,} bytes ({size_info['file_size_mb']:.2f} MB)")
    print(f"Array shape: {size_info['array_shape']}")
    print(f"Array dtype: {size_info['array_dtype']}")
    print(f"Array elements: {size_info['array_size']:,}")
    print(f"Array memory size: {size_info['array_memory_size_bytes']:,} bytes ({size_info['array_memory_size_mb']:.2f} MB)")

if __name__ == "__main__":
    # Check if file path is provided as command line argument
    if len(sys.argv) != 2:
        print("Usage: python read_npy_size.py <path_to_npy_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    size_info = get_npy_file_size(file_path)
    print_size_info(size_info)
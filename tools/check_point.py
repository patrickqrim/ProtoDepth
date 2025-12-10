import cv2
import numpy as np
import matplotlib.pyplot as plt

# This script visualizes the point distribution of sparse depth data

def read_paths_from_file(file_path):
    with open(file_path, 'r') as f:
        paths = f.readlines()
    return [path.strip() for path in paths]

def calculate_valid_points(sparse_depth_path):

    sparse_depth = cv2.imread(sparse_depth_path, cv2.IMREAD_UNCHANGED)
    validity_map = np.where(sparse_depth > 0, 1.0, 0.0)
    valid_points = np.sum(validity_map)
    
    return valid_points

def main(sparse_depth_paths_file, points=50000):

    sparse_depth_paths = read_paths_from_file(sparse_depth_paths_file)[:50000]
    valid_points_counts = []

    for sparse_depth_path in sparse_depth_paths:
        valid_points = calculate_valid_points(sparse_depth_path)
        valid_points_counts.append(valid_points)
    
    plt.hist(valid_points_counts, bins=50, edgecolor='black')
    plt.xlabel('Number of Valid Points')
    plt.ylabel('Frequency')
    plt.title('Histogram of Valid Points in Sparse Depth Images')
    plt.savefig("histogram.png")
    plt.close()

# Usage:
sparse_depth_paths_file = 'path/to/sparse_depth.txt'

main(sparse_depth_paths_file)

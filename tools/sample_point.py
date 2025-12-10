import cv2
import numpy as np
import random

# This script samples up to x points in each sparse depth image 

def read_paths_from_file(file_path):
    with open(file_path, 'r') as f:
        paths = f.readlines()
    return [path.strip() for path in paths]

def calculate_valid_points(sparse_depth_path):

    sparse_depth = cv2.imread(sparse_depth_path, cv2.IMREAD_UNCHANGED)
    validity_map = np.where(sparse_depth > 0, 1, 0)
    valid_points = np.sum(validity_map)
    
    return valid_points, sparse_depth

def sample_and_save(sparse_depth_path):

    valid_points, sparse_depth = calculate_valid_points(sparse_depth_path)

    if valid_points > 1500:
        
        # Sample 1500 points
        valid_indices = np.where(sparse_depth > 0)
        num_valid_indices = len(valid_indices[0])
        sample_indices = random.sample(range(num_valid_indices), 1500)
        
        mask = np.zeros_like(sparse_depth)
        mask[valid_indices[0][sample_indices], valid_indices[1][sample_indices]] = 1
        
        sparse_depth = sparse_depth * mask
        
        cv2.imwrite(sparse_depth_path, sparse_depth)

        valid_points_after_save, _ = calculate_valid_points(sparse_depth_path)

def main(sparse_depth_paths_file):
    sparse_depth_paths = read_paths_from_file(sparse_depth_paths_file)
    
    for sparse_depth_path in sparse_depth_paths:
        sample_and_save(sparse_depth_path)

# Example usage:
sparse_depth_paths_file = 'path/t0/sparse_depth.txt'

main(sparse_depth_paths_file)

import numpy as np
from plyfile import PlyData, PlyElement
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ply_path', type=str, required=True, help='path for ply file')
    args = parser.parse_args()
    
    plydata = PlyData.read(args.ply_path)
    vertices = plydata['vertex']
    
    condition = (vertices['x'] >= -0.6) & (vertices['x'] <= 0.6) & \
                (vertices['y'] >= -1.95) & (vertices['y'] <= 2.0) & \
                (vertices['z'] >= -2) & (vertices['z'] <= 2)
    
    vertices_in = vertices.data[condition]
    vertices_out = vertices.data[~condition]
    
    output_path_in = os.path.join(os.path.dirname(args.ply_path), '0_point_cloud.ply')
    vertices_in_element = PlyElement.describe(np.array(vertices_in), 'vertex')
    PlyData([vertices_in_element]).write(output_path_in)
    
    output_path_out = os.path.join(os.path.dirname(args.ply_path), '1_point_cloud.ply')
    vertices_out_element = PlyElement.describe(np.array(vertices_out), 'vertex')
    PlyData([vertices_out_element]).write(output_path_out)

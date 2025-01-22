import trimesh
import numpy as np
import open3d as o3d
import tetgen
import pymeshlab
from tqdm import trange

def match_sequence(input_path, output_path, scale=100.0, total=400):
    for i in trange(1, total + 1):
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(input_path + f"frame_{i:04d}.obj")
        vertices = ms.current_mesh().vertex_matrix() 
        faces = ms.current_mesh().face_matrix() 
        vertices *= scale
        vertices[:, 1:] *= -1
        vertices[:, 1] += 0.72
        with open(output_path + f"frame_{i:04d}.obj", "w") as file:
            for row in vertices: file.write("v " + " ".join(map(str, row)) + "\n")
            for row in faces: file.write("f " + " ".join(map(str, row+1)) + "\n")

match_sequence("./dance_siyu/blender_sequence/", './dance_siyu/sequence/', 100.0, 740)
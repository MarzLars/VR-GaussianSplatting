import open3d as o3d
import numpy as np

mesh = o3d.io.read_triangle_mesh("./dance_siyu/sequence/frame_0001.obj")

if not mesh.has_vertex_colors():
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.zeros((len(mesh.vertices), 3)))

num_vertices = np.asarray(mesh.vertices).shape[0]
colors = np.zeros((num_vertices, 3))
for i in range(num_vertices):
    colors[i] = [i / num_vertices, 0, 1 - i / num_vertices]

mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([mesh])

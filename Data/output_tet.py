import trimesh
import numpy as np
import open3d as o3d
import tetgen
import pymeshlab
import os

def get_tet(folder, output_filenames):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(folder + "output/0_remesh.obj")
    
    tet = tetgen.TetGen(ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix())
    tet.make_manifold()
    nodes, elems = tet.tetrahedralize(order=1, mindihedral=20.0, minratio=6.0)

    mesh_raw = {}
    mesh_raw[0] = nodes
    mesh_raw[3] = elems

    for nm in output_filenames:
        with open(folder + nm, 'w') as file:
            file.write(f"{nodes.shape[0]} {elems.shape[0]}\n")
            for row in nodes: file.write(" ".join(map(str, row)) + "\n")
            for row in elems: file.write(" ".join(map(str, row)) + "\n")

    return mesh_raw

get_tet('.\\bear\\', [f'{i}_tetgen.txt' for i in range(0, 1)])
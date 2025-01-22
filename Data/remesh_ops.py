import trimesh
import numpy as np
import open3d as o3d
import tetgen
import pymeshlab

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def check_properties(name, mesh):
    mesh.compute_vertex_normals()
    edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
    vertex_manifold = mesh.is_vertex_manifold()
    self_intersecting = mesh.is_self_intersecting()
    watertight = mesh.is_watertight()
    orientable = mesh.is_orientable()

    print(name)
    print(f"  edge_manifold:          {edge_manifold}")
    print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
    print(f"  vertex_manifold:        {vertex_manifold}")
    print(f"  self_intersecting:      {self_intersecting}")
    print(f"  watertight:             {watertight}")
    print(f"  orientable:             {orientable}")

    geoms = [mesh]
    # o3d.visualization.draw_geometries(geoms, mesh_show_back_face=True)

def adjust_pt_coord(pcd, scale):
    points = np.array(pcd.points)
    normals = np.array(pcd.normals)
    pcd.points = o3d.utility.Vector3dVector(points + (scale-1) * normals)
    return pcd

def get_sample_pts(pts, sample_num=4096, scale=1.0):
    # iterative farthest point sampling method
    mask = np.zeros(pts.shape[0], dtype=np.bool_)
    min_dist = np.linalg.norm(pts - pts[0, np.newaxis], axis=1)
    mask[0] = True
    
    for i in range(sample_num-1):
        new_pt = np.argmax(min_dist)
        min_dist = np.minimum(min_dist, np.linalg.norm(pts - pts[new_pt, np.newaxis], axis=1))
        mask[new_pt] = True

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts[mask])
    point_cloud.estimate_normals()
    return point_cloud

def get_mesh_from_pcd(input_filename, output_filename, scale=1.0, alpha=1.0):
    points = trimesh.load(input_filename).vertices
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.estimate_normals()
    point_cloud = adjust_pt_coord(point_cloud, scale)

    voxel_down_pcd = point_cloud.voxel_down_sample(voxel_size=0.02)
    uni_down_pcd = point_cloud.uniform_down_sample(every_k_points=5)

    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=5,
                                                        std_ratio=2.0)
    point_cloud = voxel_down_pcd.select_by_index(ind)
    point_cloud = get_sample_pts(np.asarray(point_cloud.points), 4096)
    # o3d.visualization.draw_geometries([point_cloud])

    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(point_cloud)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha, tetra_mesh, pt_map)

    mesh.compute_vertex_normals()
    mesh.remove_degenerate_triangles()

    # check_properties("tmp", mesh)
    o3d.io.write_triangle_mesh("./output/tmp.obj", mesh)

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh("./output/tmp.obj")
    ''''
    ms.apply_filter("meshing_isotropic_explicit_remeshing", 
                    iterations=5, 
                    adaptive=True, 
                    targetlen=pymeshlab.Percentage(10.0),
                    maxsurfdist=pymeshlab.Percentage(10.0))
    '''
    ms.apply_filter("meshing_isotropic_explicit_remeshing", 
                    iterations=100, 
                    adaptive=True, 
                    targetlen=pymeshlab.Percentage(5.0),
                    maxsurfdist=pymeshlab.Percentage(10.0))
    ms.save_current_mesh("./output/final.obj")

    tet = tetgen.TetGen(ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix())
    tet.make_manifold()
    nodes, elems = tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5, coarsen_percent=50, coarsen=100)

    mesh_raw = {}
    mesh_raw[0] = nodes
    mesh_raw[3] = elems

    with open(output_filename, 'w') as file:
        file.write(f"{nodes.shape[0]} {elems.shape[0]}\n")
        for row in nodes: file.write(" ".join(map(str, row)) + "\n")
        for row in elems: file.write(" ".join(map(str, row)) + "\n")

    return mesh_raw

get_mesh_from_pcd("./basket_daytime/5_point_cloud.ply", './basket_daytime/5_tetgen.txt', 1.001, 0.75)
# get_mesh_from_pcd("1_point_cloud.ply", '1_tetgen.txt', 1.075, 0.9)
# get_mesh_from_pcd("2_point_cloud.ply", '2_tetgen.txt', 1.075, 0.075)
# get_mesh_from_pcd("3_point_cloud.ply", '3_tetgen.txt', 1.075, 0.3)
import os.path as osp
import copy

import numpy as np
import trimesh

import open3d as o3d

from scipy.spatial.transform import Rotation as R

from ndf_robot.utils import path_util, util
from ndf_robot.utils.plotly_save import multiplot
from ndf_robot.eval.query_points import QueryPoints

import plotly.express as px
import plotly.graph_objects as go



# Most of these are extracted from the open 3d global registration tutorial
# http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(10000000, 0.99999))
    return result

def refine_registration(source, target, source_fpfh,
                        target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    radius_normal = voxel_size * 2
    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def crop_with_query(query_pts, target_pcd):
    """
    Crop a target point cloud using the bounding box of query points

    Args:
        query_pts (o3d pointcloud): Query points to crop with
        target_pcd (o3d pointcloud): Target points to crop
    """
    query_bb = query_pts.get_oriented_bounding_box()
    target_pcd_cropped = target_pcd.crop(query_bb)
    return target_pcd_cropped


# see the demo object descriptions folder for other object models you can try
# obj_model = osp.join(path_util.get_ndf_demo_obj_descriptions(), 'mug_centered_obj_normalized/28f1e7bc572a633cb9946438ed40eeb9/models/model_normalized.obj')
# obj_model = osp.join(path_util.get_ndf_obj_descriptions(), 'bottle_std_centered_obj_normalized/f4851a2835228377e101b7546e3ee8a7/models/model_normalized.obj')
# obj_model = osp.join(path_util.get_ndf_obj_descriptions(), 'bottle_centered_obj_normalized/e593aa021f3fa324530647fc03dd20dc/models/model_normalized.obj')

mug_std = osp.join(path_util.get_ndf_obj_descriptions(), 'mug_std_centered_obj_normalized/28f1e7bc572a633cb9946438ed40eeb9/models/model_normalized.obj')
mug2_std = osp.join(path_util.get_ndf_obj_descriptions(), 'mug_std_centered_obj_normalized/7a8ea24474846c5c2f23d8349a133d2b/models/model_normalized.obj')

# Set up rotations
rot = np.eye(4)
rot[:3, :3] = util.make_rotation_matrix('x', np.pi / 2)  # Make upright

arbitrary_rot = np.eye(4)
arbitrary_rot[:3, :3] = util.make_rotation_matrix('z', 1.5)  # Make odd angle

# Load objects
scale = 1.0
mesh = trimesh.load(mug_std, process=False)
mesh.apply_scale(scale)
mesh.apply_transform(rot)

rotated_mesh = trimesh.load(mug2_std, process=False)
rotated_mesh.apply_scale(scale)
rotated_mesh.apply_transform(rot)
rotated_mesh.apply_transform(arbitrary_rot)

# Convert to Open3D pointcloud
# http://www.open3d.org/docs/release/tutorial/geometry/working_with_numpy.html
radius_normal = 0.010
target_pcd_np = np.array(mesh.sample(1000))
target_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_pcd_np))
target_pcd.points = o3d.utility.Vector3dVector(target_pcd_np)
target_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

ori_pcd_np = np.array(rotated_mesh.sample(1000))
ori_pcd = o3d.geometry.PointCloud()
ori_pcd.points = o3d.utility.Vector3dVector(ori_pcd_np)
ori_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

query_points = QueryPoints.generate_rect(500, 0.05, 0.05, 0.1, 0.3)
query_pcd = o3d.geometry.PointCloud()
query_pcd.points = o3d.utility.Vector3dVector(query_points)
query_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

target_pcd = crop_with_query(query_pts=query_pcd, target_pcd=target_pcd)

# Compute geometric features
voxel_size = 0.005
target_pcd_down, target_pcd_fpfh = preprocess_point_cloud(target_pcd, voxel_size)
ori_pcd_down, ori_pcd_fpfh = preprocess_point_cloud(ori_pcd, voxel_size)

# Compute ransac for global geometric alignment
result_ransac = execute_global_registration(ori_pcd_down, target_pcd_down,
                                            ori_pcd_fpfh, target_pcd_fpfh,
                                            voxel_size)
# Run ICP to get closer
result_icp = refine_registration(ori_pcd, target_pcd, ori_pcd_fpfh, target_pcd_fpfh,
    voxel_size, result_ransac)

# Apply result transformation
transformation = result_ransac.transformation
ori_pcd_ransac= copy.deepcopy(ori_pcd_down)
ori_pcd_ransac.transform(transformation)

transformation = result_icp.transformation
ori_pcd_icp = copy.deepcopy(ori_pcd_down)
ori_pcd_icp.transform(transformation)

# Plot
ori_down_np = np.asarray(ori_pcd_down.points)
ori_down_ransac_np = np.asarray(ori_pcd_ransac.points)
ori_down_icp_np = np.asarray(ori_pcd_icp.points)
target_down_np = np.asarray(target_pcd_down.points)

print('trans: ', np.asarray(transformation))

fname = osp.join(path_util.get_ndf_eval(), 'debug_viz', 'test_o3d.html')
multiplot([ori_down_np, ori_down_ransac_np, ori_down_icp_np, target_down_np], fname)







# Back to numpy
# xyz_load = np.asarray(pcd_load.points)

print(target_pcd)
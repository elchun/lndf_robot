from ndf_robot.eval.evaluate_general import EvaluateNetwork, \
    EvaluateNetworkSetup, QueryPoints, DemoIO
from ndf_robot.opt.optimizer_lite import OccNetOptimizer
import plotly.express as px
import numpy as np
from numpy.lib.npyio import NpzFile
import os
import os.path as osp
import trimesh
from ndf_robot.utils import eval_gen_utils, util, path_util


def multiplot(point_list: 'list[np.ndarray]', fname='debug.html'):
    """
    Plot each group of points in {point_list} in a different color on the same
    graph and saves to {fname}.

    Args:
        point_list (list[np.ndarray]): List of pointclouds in the form
            of (n_i x 3)
        fname (str, optional): Name of file to save result to.
            Defaults to 'debug.html'.

    Returns:
        plotly plot: Plot that is produced.
    """

    plot_pts = np.vstack(point_list)

    color = np.ones(plot_pts.shape[0])

    idx = 0
    for i, pts in enumerate(point_list):
        next_idx = idx + pts.shape[0]
        color[idx:next_idx] *= i
        idx = next_idx

    fig = px.scatter_3d(
        x=plot_pts[:, 0], y=plot_pts[:, 1], z=plot_pts[:, 2], color=color)

    fig.write_html(fname)

    return fig


if __name__ == '__main__':
    # -- Load demo to get query points -- #
    demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos', 'bowl',
    'grasp_rim_anywhere_place_shelf_all_methods_multi_instance_converted')

    demo_fnames = os.listdir(demo_load_dir)
    grasp_demo_filenames = [osp.join(demo_load_dir, fn) for fn in
        demo_fnames if 'grasp_demo' in fn]

    place_demo_filenames = [osp.join(demo_load_dir, fn) for fn in
        demo_fnames if 'place_demo' in fn]

    points_list = []
    for i in range(min(len(grasp_demo_filenames), len(place_demo_filenames))):
        grasp_demo_fname = grasp_demo_filenames[0]
        place_demo_fname = place_demo_filenames[0]

        grasp_demo: NpzFile = np.load(grasp_demo_fname)
        place_demo: NpzFile = np.load(place_demo_fname)

        for f in grasp_demo.files:
            print(f)
        print('---')
        for f in place_demo.files:
            print(f)

        gripper_query_points = grasp_demo['gripper_pts_uniform']
        shelf_query_points = place_demo['shelf_pointcloud_uniform']
        rack_query_points = place_demo['rack_pointcloud_uniform']

        print('Gripper query points: ', len(gripper_query_points))

        break

    points_list.append(gripper_query_points)
    points_list.append(shelf_query_points)
    points_list.append(rack_query_points)

    gripper_query_args = {
        'n_pts': 1000,
        'x': 0.08,
        'y': 0.04,
        'z1': 0.05,
        'z2': 0.02,
    }

    gripper_standard = QueryPoints.generate_rect(**gripper_query_args)
    points_list.append(gripper_standard)

    rack_query_args = {
        'n_pts': 1000,
        'radius': 0.05,
        'height': 0.04,
        'y_rot_rad': 0.68,
        'x_trans': 0.055,
        'y_trans': 0,
        'z_trans': 0.19,
    }

    rack_standard = QueryPoints.generate_rack_arm(**rack_query_args)
    points_list.append(rack_standard)

    shelf_query_args = {
        'n_pts': 1000,
        'radius': 0.1,
        'height': 0.1,
        'y_rot_rad': 0.0,
        'x_trans': 0.0,
        'y_trans': 0.07,
        'z_trans': 0.11,
    }

    shelf_standard = QueryPoints.generate_rack_arm(**shelf_query_args)
    points_list.append(shelf_standard)

    gripper_loaded = QueryPoints.generate_ndf_gripper(1000)
    points_list.append(gripper_loaded)

    multiplot(points_list, 'query_points.html')

    # np.savez('reference_query_points.npz',
    #     gripper=gripper_query_points,
    #     shelf=shelf_query_points,
    #     rack=rack_query_points)


    # print(grasp_demo['table_urdf'])

    # raise ValueError('hi')













    # -- Other stuff --#

#     # config_fname = 'GENERAL_debug.yml'

#     # setup = EvaluateGraspSetup()
#     # setup.load_config(config_fname)
#     # demo_load_dir = setup.get_demo_load_dir(obj_class='mug')

#     demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos', 'bowl',
#     'grasp_rim_anywhere_place_shelf_all_methods_multi_instance_converted')

#     demo_fnames = os.listdir(demo_load_dir)

#     grasp_demo_filenames_orig = [osp.join(demo_load_dir, fn) for fn in
#         demo_fnames if 'grasp_demo' in fn]

#     place_demo_fnames = []
#     grasp_demo_fnames = []
#     for i, fname in enumerate(grasp_demo_filenames_orig):
#         shapenet_id_npz = fname.split('/')[-1].split('grasp_demo_')[-1]
#         place_fname = osp.join('/'.join(fname.split('/')[:-1]), 'place_demo_' + shapenet_id_npz)
#         if osp.exists(place_fname):
#             grasp_demo_fnames.append(fname)
#             place_demo_fnames.append(place_fname)
#             # log_warn('Could not find corresponding placement demo: %s, skipping ' % place_fname)

#     # -- Place data keys -- #
#         # shapenet_id
#         # ee_pose_world
#         # robot_joints
#         # obj_pose_world
#         # obj_pose_camera
#         # object_pointcloud
#         # depth
#         # seg
#         # camera_poses
#         # obj_model_file
#         # obj_model_file_dec
#         # gripper_pts
#         # rack_pointcloud_observed
#         # rack_pointcloud_gt
#         # rack_pointcloud_gaussian
#         # rack_pointcloud_uniform
#         # rack_pose_world
#         # rack_contact_pose
#         # shelf_pose_world
#         # shelf_pointcloud_observed
#         # shelf_pointcloud_uniform
#         # shelf_pointcloud_gt
#         # table_urdf
#         # obj_pcd_ori

#     place_demo_fn = place_demo_fnames[4]
#     print(f'Loading demo from fname: {place_demo_fn}')
#     place_data = np.load(place_demo_fn, allow_pickle=True)
#     # files = place_data.files
#     # print('type: ', type(place_data))
#     # for f in files:
#     #     print(f)

#     grasp_demo_fn = grasp_demo_fnames[4]
#     print(f'Loading demo from fname: {grasp_demo_fn}')
#     grasp_data = np.load(grasp_demo_fn, allow_pickle=True)

#     rack_pcd = place_data['rack_pointcloud_gt']
#     shelf_pcd = place_data['shelf_pointcloud_gt']
#     object_pcd = place_data['object_pointcloud']

#     original_object_pcd = grasp_data['object_pointcloud']

#     # # multiplot([original_object_pcd, object_pcd])
#     # rack_pose = place_data['rack_pose_world']
#     # rack_pcd_posed = OccNetOptimizer._apply_pose_numpy(rack_pcd, rack_pose)
#     # multiplot([rack_pcd, rack_pcd_posed, object_pcd, original_object_pcd])

#     place_demo = DemoIO.process_shelf_place_data(place_data)
#     # grasp_demo = DemoIO.process_grasp_data(grasp_data)

#     shelf_query_pts = QueryPoints.generate_shelf(1000, 0.04, 0.03)

#     shelf_mesh_file = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/shelf_back.stl')
#     shelf_mesh = trimesh.load_mesh(shelf_mesh_file)
#     shelf_pts_gt_fresh = shelf_mesh.sample(500)
#     shelf_pts_transformed = util.apply_pose_numpy(shelf_pts_gt_fresh, place_demo.query_pose_world)

#     print('Pose: ', place_demo.query_pose_world)

#     # -- Ground truth -- #
#     # target_grip, target_rack, shapenet_id = eval_gen_utils.process_demo_data_rack(grasp_data, place_data, None)
#     target_grip, target_shelf, shapenet_id = eval_gen_utils.process_demo_data_shelf(grasp_data, place_data, None)
#     # true_query_pts = target_shelf['demo_query_pts']
#     true_query_pts = target_shelf['demo_query_pts_real_shape']
#     true_obj_pts = target_shelf['demo_obj_pts']

#     # true_grasp_obj_pts = target_grip['demo_obj_pts']
#     # true_grasp_query_pts = target_grip['demo_query_pts']

#     # demo_obj_pts = grasp_demo.obj_pts
#     # demo_obj_pts = util.apply_pose_numpy(demo_obj_pts, grasp_demo.obj_pose_world)

#     # demo_query_pts = grasp_demo.query_pts
#     # demo_query_pts = util.apply_pose_numpy(demo_query_pts, grasp_demo.query_pose_world)

#     demo_shelf_pcd = util.apply_pose_numpy(place_demo.query_pts, place_demo.query_pose_world)
#     # demo_place_obj = util.apply_pose_numpy(grasp_demo.obj_pts, place_demo.obj_pose_world)

#     demo_shelf_query_pcd = util.apply_pose_numpy(shelf_query_pts, place_demo.query_pose_world)


#     # # multiplot([grasp_data['object_pointcloud'], grasp_data['obj_pcd_ori'], grasp_demo.obj_pts, demo_obj_pts])
#     # # multiplot([grasp_data['object_pointcloud'], grasp_demo.obj_pts, demo_obj_pts])
#     # # multiplot([true_grasp_obj_pts, true_grasp_query_pts, demo_obj_pts, demo_query_pts, grasp_data['']])
#     # multiplot([true_grasp_obj_pts, true_grasp_query_pts, demo_obj_pts, demo_query_pts, demo_rack_pcd, demo_place_obj])
#     # multiplot([demo_rack_pcd, demo_place_obj, rack_query_pts, demo_rack_query_pcd])
#     # multiplot([demo_shelf_pcd, demo_place_obj])
#     multiplot([true_query_pts, true_obj_pts, shelf_pts_gt_fresh, shelf_pts_transformed, demo_shelf_query_pcd])


#     # demo_rack_pcd = util.apply_pose_numpy(place_demo.query_pts, place_demo.query_pose_world)

#     # demo_obj_pcd = grasp_demo.obj_pts
#     # demo_obj_pcd = OccNetOptimizer._apply_pose_numpy(demo_obj_pcd, grasp_demo.obj_pose_world)
#     # # demo_obj_pcd = OccNetOptimizer._apply_pose_numpy(demo_obj_pcd, place_demo.obj_pose_world)


#     # multiplot([true_query_pts, true_obj_pts, demo_rack_pcd, demo_obj_pcd])



#     # cylinder_pts = QueryPoints.generate_cylinder(400, 0.02, 0.15, 'z')
#     # transform = np.eye(4)
#     # rot = EvaluateGrasp.make_rotation_matrix('y', 0.68)
#     # trans = np.array([[0.04, 0, 0.17]]).T
#     # transform[:3, :3] = rot
#     # transform[:3, 3:4] = trans
#     # print(transform)

#     # cylinder_pcd = trimesh.PointCloud(cylinder_pts)
#     # cylinder_pcd.apply_transform(transform)
#     # cylinder_pts = np.asarray(cylinder_pcd.vertices)

#     # cylinder_pts = QueryPoints.generate_rack_arm(400)

#     # plot_pts = np.vstack((rack_pcd, cylinder_pts))
#     # color = np.concatenate([np.ones(rack_pcd.shape[0]) * 1, np.ones(cylinder_pts.shape[0]) * 2])

#     # fig = px.scatter_3d(
#     #     x=plot_pts[:, 0], y=plot_pts[:, 1], z=plot_pts[:, 2], color=color)

#     # fig.write_html('debug.html')

#     # print(place_data['rack_pointcloud_uniform'].shape)

#     # grasp_demo_fn = grasp_demo_fnames[0]
#     # print(f'Loading demo from fname: {grasp_demo_fn}')
#     # grasp_data = np.load(grasp_demo_fn, allow_pickle=True)

#     # place_demo = DemoIO.(place_data)
#     # grasp_demo = DemoIO.process_grasp_data(grasp_data)

#     # obj_pts = grasp_demo.obj_pts


#     # obj_pts = OccNetOptimizer._apply_pose_numpy(place_demo.obj_pts, )
#     # obj_pts = OccNetOptimizer._apply_pose_numpy(grasp_demo.obj_pts, place_demo.obj_pose_world)
#     # obj_pts = OccNetOptimizer._apply_pose_numpy(grasp_demo.obj_pts, grasp_demo.obj_pose_world)
#     # obj_pts = OccNetOptimizer._apply_pose_numpy(grasp_demo.obj_pts, grasp_demo.obj_pose_world)
#     # obj_pts = OccNetOptimizer._apply_pose_numpy(obj_pts, place_demo.obj_pose_world)
#     # obj_pts = grasp_demo.obj_pts
#     # query_pts = OccNetOptimizer._apply_pose_numpy(place_demo.query_pts, place_demo.query_pose_world)
#     # print(place_demo.query_pose_world)

#     # plot_pts = np.vstack((obj_pts, query_pts))
#     # color = np.concatenate([np.ones(obj_pts.shape[0]) * 1, np.ones(query_pts.shape[0]) * 2])
#     # fig = px.scatter_3d(
#     #     x=plot_pts[:, 0], y=plot_pts[:, 1], z=plot_pts[:, 2], color=color)
#     # fig.write_html('debug.html')

#     # -- Looking at grasp demos -- #
#     # grasp_demo_fnames = [osp.join(demo_load_dir, fn) for fn in
#     #     demo_fnames if 'grasp_demo' in fn]

#     # grasp_demo_fn = grasp_demo_fnames[0]
#     # print(f'Loading demo from fname: {grasp_demo_fn}')
#     # grasp_data = np.load(grasp_demo_fn, allow_pickle=True)
#     # files = grasp_data.files
#     # print('---')
#     # for f in files:
#     #     print(f)

# # -- Grasp data keys -- #
#     # shapenet_id
#     # ee_pose_world
#     # robot_joints
#     # obj_pose_world
#     # obj_pose_camera
#     # object_pointcloud
#     # depth
#     # seg
#     # camera_poses
#     # obj_model_file
#     # obj_model_file_dec
#     # gripper_pts
#     # gripper_pts_gaussian
#     # gripper_pts_uniform
#     # gripper_contact_pose
#     # table_urdf
#     # obj_pcd_ori
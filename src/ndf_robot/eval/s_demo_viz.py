from typing import Callable, Tuple
from ndf_robot.eval.evaluate_general import EvaluateNetwork, \
    EvaluateNetworkSetup, QueryPoints, DemoIO
from ndf_robot.opt.optimizer_lite import Demo
from ndf_robot.opt.optimizer_lite import OccNetOptimizer
import plotly.express as px
import numpy as np
from numpy.lib.npyio import NpzFile
import os
import os.path as osp
import trimesh
from ndf_robot.utils import eval_gen_utils, util, path_util, plotly_save


def get_demo(demo_load_dir: str, demoio_fn: 'Callable[NpzFile, Demo]',
    demo_prefix: str = 'place_demo', demo_idx: int = 0) -> Tuple[NpzFile, Demo]:
    """
    Get first demo from demo_load_dir with matching prefix.

    Args:
        demo_load_dir (str): Directory where demos are (assuming that working dir
            is path_util.get_ndf_data()/demos)
        demoio_fn (function): Function to process demo with.
        demo_prefix (str, optional): Keyword included in demo. Defaults to 'place_demo'.
        demo_idx (int, optional): Index of demo to get.
    """
    demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos', demo_load_dir)
    raw_demo_fnames = os.listdir(demo_load_dir)
    demo_fnames = [osp.join(demo_load_dir, fn) for fn in
        raw_demo_fnames if demo_prefix in fn]

    demo_fname = demo_fnames[demo_idx]
    data = np.load(demo_fname, allow_pickle=True)
    demo = demoio_fn(data)

    return data, demo


if __name__ == '__main__':
    # -- Plot rack and object to find preplace offset -- #
    # demo_load_dir = 'mug/grasp_rim_hang_handle_gaussian_precise_w_shelf_converted'
    # demoio_fn = DemoIO.process_rack_place_data
    # demo_prefix = 'place_demo'
    # place_data, place_demo = get_demo(demo_load_dir, demoio_fn, demo_prefix)

    # rack_pcd_raw = place_data['rack_pointcloud_gt']
    # obj_pcd = util.apply_pose_numpy(place_demo.obj_pts, place_demo.obj_pose_world)
    # rack_pcd = util.apply_pose_numpy(rack_pcd_raw, place_demo.query_pose_world)

    # PREPLACE_OFFSET_TF = [0.012, -0.042, 0.06, 0, 0, 0, 1]
    # # PREPLACE_OFFSET_TF = [0, -0.084, 0.12, 0, 0, 0, 1]
    # # PREPLACE_OFFSET_TF = [0, -0.042, 0.08, 0, 0, 0, 1]
    # # PREPLACE_OFFSET_TF = [0, -0.084, 0.15, 0, 0, 0, 1]
    # preplace_offset_tf = util.list2pose_stamped(PREPLACE_OFFSET_TF)

    # obj_place_pose = place_demo.obj_pose_world

    # pre_place_ee_pose = util.transform_pose(pose_source=util.list2pose_stamped(obj_place_pose),
    #     pose_transform=preplace_offset_tf)

    # obj_pre_place = util.apply_pose_numpy(place_demo.obj_pts, util.pose_stamped2list(pre_place_ee_pose))

    # plotly_save.multiplot([obj_pcd, obj_pre_place, rack_pcd], 'rack_and_object.html')

    # # -- Plot bottle to figure out offset -- #
    # demo_load_dir = 'bottle/grasp_side_place_shelf'
    # demoio_fn = DemoIO.process_grasp_data
    # demo_prefix = 'grasp_demo'
    # grasp_data, grasp_demo = get_demo(demo_load_dir, demoio_fn, demo_prefix, demo_idx=2)

    # # rack_pcd_raw = grasp_data['rack_pointcloud_gt']
    # obj_pcd = util.apply_pose_numpy(grasp_demo.obj_pts, grasp_demo.obj_pose_world)
    # gripper_pcd = util.apply_pose_numpy(grasp_demo.query_pts, grasp_demo.query_pose_world)

    # # PREPLACE_OFFSET_TF = [0.012, -0.042, 0.06, 0, 0, 0, 1]
    # # # PREPLACE_OFFSET_TF = [0, -0.084, 0.12, 0, 0, 0, 1]
    # # # PREPLACE_OFFSET_TF = [0, -0.042, 0.08, 0, 0, 0, 1]
    # # # PREPLACE_OFFSET_TF = [0, -0.084, 0.15, 0, 0, 0, 1]
    # # preplace_offset_tf = util.list2pose_stamped(PREPLACE_OFFSET_TF)

    # plotly_save.multiplot([obj_pcd, gripper_pcd], 'obj_and_gripper.html')


    # -- Plot bottle to figure out offset -- #
    demo_load_dir = 'mug/grasp_rim_hang_handle_gaussian_precise_w_shelf_converted'
    demoio_fn = DemoIO.process_grasp_data
    demo_prefix = 'grasp_demo'
    extents_list = []
    for i in range(5):
        grasp_data, grasp_demo = get_demo(demo_load_dir, demoio_fn, demo_prefix, demo_idx=i)

        pts = grasp_demo.obj_pts
        max_extents = np.max(pts, axis=0)
        min_extents = np.min(pts, axis=0)

        extents = max_extents - min_extents
        extents_list.append(extents)

    print('Extents: ', np.mean(np.vstack(extents_list), axis=0))




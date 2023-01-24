# There is a bug where during grasp, query points are not centered on
# object.  This doesn't matter for bowls, but its a problem for bottles.

# This script allows you to shift the point clouds around so that they are
# in the same spot as in simulation...

# UPDATE: Bug doesn't rly seem to be an issue, this is used for vizualizing
# stuff now

import os
import os.path as osp
import numpy as np

from typing import Callable, Tuple
from numpy.lib.npyio import NpzFile

import trimesh
from ndf_robot.utils import eval_gen_utils, util, path_util

from ndf_robot.utils.plotly_save import multiplot

from ndf_robot.eval.query_points import QueryPoints


def get_demo_list(demo_load_dir: str, demo_prefix: str = 'place_demo') \
    -> Tuple['list[NpzFile]', 'list[str]']:
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

    return ([np.load(demo_fname, allow_pickle=True) for demo_fname in demo_fnames],
        demo_fnames)


def visualize_grasp_demo_pose(demo_npz: NpzFile, viz_fname: str):
    obj_pcd = demo_npz['obj_pcd_ori']
    query_pcd = demo_npz['gripper_pts']
    obj_pose = demo_npz['obj_pose_world']
    query_pose = demo_npz['ee_pose_world']

    new_query_args = {
        'n_pts': 1000,
        # 'x': 0.04,
        'x': 0.06,
        'y': 0.03,
        'z1': 0.04,
        'z2': 0.01,
    }

    # new_query_args = {
    #     'n_pts': 1000,
    #     'radius': 0.04,
    #     'height': 0.12,
    #     'rot_axis': 'x',
    # }

    # new_query_pcd = QueryPoints.generate_cylinder(**new_query_args)

    new_query_pcd = QueryPoints.generate_rect(**new_query_args)

    obj_pcd = util.apply_pose_numpy(obj_pcd, obj_pose)
    query_pcd = util.apply_pose_numpy(query_pcd, query_pose)
    new_query_pcd = util.apply_pose_numpy(new_query_pcd, query_pose)

    multiplot([obj_pcd, query_pcd, new_query_pcd], viz_fname)


def visualize_place_demo_pose(demo_npz: NpzFile, viz_fname: str):
    obj_pcd = demo_npz['obj_pcd_ori']
    query_pcd = demo_npz['gripper_pts']
    obj_pose = demo_npz['obj_pose_world']
    query_pose = demo_npz['ee_pose_world']

    new_query_args = {
        'n_pts': 1000,
        'x': 0.04,
        'y': 0.02,
        'z1': 0.04,
        'z2': 0.02,
    }

    new_query_pcd = QueryPoints.generate_rect(**new_query_args)

    obj_pcd = util.apply_pose_numpy(obj_pcd, obj_pose)
    query_pcd = util.apply_pose_numpy(query_pcd, query_pose)
    new_query_pcd = util.apply_pose_numpy(new_query_pcd, query_pose)

    multiplot([obj_pcd, query_pcd, new_query_pcd], viz_fname)


if __name__ == '__main__':
    demo_load_dir = 'bottle/grasp_side_place_shelf_with_collision'
    viz_dir = osp.join(path_util.get_ndf_eval(), 'debug_viz')
    demo_list, demo_str_list = get_demo_list(demo_load_dir, 'grasp_demo')
    # demo_list, demo_str_list = get_demo_list(demo_load_dir, 'place_demo')

    for fname in demo_str_list:
        print(fname)

    for i in range(len(demo_list)):
        visualize_grasp_demo_pose(demo_list[i], osp.join(viz_dir,
            f'demo_viz_debug_{i}.html'))
#!/usr/bin/env python3

"""
Script to generate query points to align with a portion of another shape.

@author elchun
"""
from sympy import O
from tinydb import Query
from ndf_robot.utils.plotly_save import multiplot
from ndf_robot.eval.query_points import QueryPoints
import os.path as osp
import os
from ndf_robot.utils import path_util, util
import numpy as np


def get_reference_pts_rack() -> np.ndarray:
    """
    Get reference points for rack using demo.

    Returns:
        np.ndarray: (n, 3) array of pts.
    """
    demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos', 'mug',
    'grasp_rim_hang_handle_gaussian_precise_w_shelf_converted')

    demo_fnames = os.listdir(demo_load_dir)
    place_demo_filenames_orig = [osp.join(demo_load_dir, fn) for fn in
        demo_fnames if 'place_demo' in fn]

    place_demo_fn = place_demo_filenames_orig[0]
    place_data = np.load(place_demo_fn, allow_pickle=True)

    ref_pts = place_data['rack_pointcloud_gt']

    return ref_pts


def get_working_pts_rack(**kwargs) -> np.ndarray:
    return QueryPoints.generate_rack_arm(**kwargs)


def get_reference_pts_shelf() -> np.ndarray:
    """
    Get reference points for rack using demo.

    Returns:
        np.ndarray: (n, 3) array of pts.
    """
    demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos', 'bowl',
    'grasp_rim_anywhere_place_shelf_all_methods_multi_instance_converted')

    demo_fnames = os.listdir(demo_load_dir)
    place_demo_filenames_orig = [osp.join(demo_load_dir, fn) for fn in
        demo_fnames if 'place_demo' in fn]

    place_demo_fn = place_demo_filenames_orig[0]
    place_data = np.load(place_demo_fn, allow_pickle=True)

    ref_pts = place_data['shelf_pointcloud_gt']

    return ref_pts


def get_working_pts_shelf(**kwargs) -> np.ndarray:
    return QueryPoints.generate_shelf(**kwargs)


def print_result_yml(input_args: dict):
    """
    Print dict in format that can be pasted into yaml file.

    Args:
        input_args (dict): Dict to print.
    """
    for k, v in input_args.items():
        print(f'{k}: {v}')


def print_result_dict(input_args: dict):
    """
    Print dict in format that can be pasted into a dict.

    Args:
        input_args (dict): Dict to print.
    """
    for k, v in input_args.items():
        print(f"'{k}': {v},")


if __name__ == '__main__':
    # -- Reference points in the chart -- #
    # reference_pts = get_reference_pts_rack()
    reference_pts = get_reference_pts_shelf()

    # -- Arguments used to generate working points -- #
    working_args = {
        # 'n_pts': 1000,
        # 'radius': 0.05,
        # 'height': 0.08,

        # 'y_rot_rad': 0.68,
        # 'x_trans': 0.04,
        # 'y_trans': 0,
        # 'z_trans': 0.17

        # 'n_pts': 1000,
        # 'radius': 0.05,
        # 'height': 0.04,
        # 'y_rot_rad': 0.68,
        # 'x_trans': 0.055,
        # 'y_trans': 0.0,
        # 'z_trans': 0.19,

        'n_pts': 1000,
        'radius': 0.08,
        'height': 0.03,
        'y_rot_rad': 0.0,
        'x_trans': 0.0,
        'y_trans': 0.07,
        'z_trans': 0.11,
    }

    def working_pts_generator(**kwargs):
        # return QueryPoints.generate_rack_arm(**kwargs)
        return QueryPoints.generate_shelf(**kwargs)

    print('Choose param to tune, type "SAVE" to print in yml format. \n' \
        + 'Press Enter to modify same parameter.\n' \
        + 'Type "exit" to exit.')
    previous_key_to_modify = None
    while True:
        working_pts = working_pts_generator(**working_args)
        multiplot([reference_pts, working_pts], 'query_tune.html')

        key_to_modify = input('Enter param to tune: ')
        if key_to_modify == 'SAVE':
            print_result_yml(working_args)
            print('---')
            print_result_dict(working_args)
            continue
        elif key_to_modify == '' and previous_key_to_modify is not None:
            key_to_modify = previous_key_to_modify
        elif key_to_modify == 'exit':
            print_result_yml(working_args)
            print('---')
            print_result_dict(working_args)
            break
        elif key_to_modify not in working_args.keys():
            print(f'Invalid input: select from {working_args.keys()}')
            continue

        print(f'Current value: {working_args[key_to_modify]}')
        current_type = type(working_args[key_to_modify])
        new_value = input('Enter new value: ')
        working_args[key_to_modify] = current_type(new_value)
        previous_key_to_modify = key_to_modify


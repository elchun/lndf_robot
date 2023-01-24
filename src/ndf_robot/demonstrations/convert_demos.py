"""
Add obj_pcd_ori field to all demos.  Requires loading both demos, then
shifting the pcd of the object, then saving.

@author elchun
"""

import os
import os.path as osp
from ndf_robot.utils import path_util, util
import numpy as np

if __name__ == '__main__':
    # -- Set load and save dirs -- #
    # demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos', 'mug',
    # 'grasp_rim_hang_handle_gaussian_precise_w_shelf')

    # demo_save_dir = osp.join(path_util.get_ndf_data(), 'demos', 'mug',
    # 'grasp_rim_hang_handle_gaussian_precise_w_shelf_converted')

    # demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos', 'bowl',
    # 'grasp_rim_anywhere_place_shelf_all_methods_multi_instance')

    # demo_save_dir = osp.join(path_util.get_ndf_data(), 'demos', 'bowl',
    # 'grasp_rim_anywhere_place_shelf_all_methods_multi_instance_converted')

    # demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos', 'bottle',
    # 'grasp_side_place_shelf_start_upright_all_methods_multi_instance')

    # demo_save_dir = osp.join(path_util.get_ndf_data(), 'demos', 'bottle',
    # 'grasp_side_place_shelf_start_upright_all_methods_multi_instance_converted')

    demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos', 'mug',
    'mug_handle')

    demo_save_dir = osp.join(path_util.get_ndf_data(), 'demos', 'mug',
    'mug_handle_converted')

    assert demo_load_dir != demo_save_dir, 'Must have different load and save dir'

    convert_place_demos = False

    # -- Make save directory if it doesn't already exist -- #
    util.safe_makedirs(demo_save_dir)

    # -- Get grasp demo fnames -- #
    demo_fnames = os.listdir(demo_load_dir)
    grasp_demo_filenames_orig = [osp.join(demo_load_dir, fn) for fn in
        demo_fnames if 'grasp_demo' in fn]

    # -- Find matching place demo (must have pair to convert) -- #
    place_demo_fnames = []
    grasp_demo_fnames = []
    for i, fname in enumerate(grasp_demo_filenames_orig):
        shapenet_id_npz = fname.split('/')[-1].split('grasp_demo_')[-1]
        place_fname = osp.join('/'.join(fname.split('/')[:-1]), 'place_demo_' + shapenet_id_npz)
        if osp.exists(place_fname):
            grasp_demo_fnames.append(fname)
            place_demo_fnames.append(place_fname)

    # -- Get obj pcd, then shift to original object location -- #
    for i in range(len(grasp_demo_fnames)):
        grasp_demo_fn = grasp_demo_fnames[i]
        grasp_data = np.load(grasp_demo_fn, allow_pickle=True)

        if convert_place_demos:
            place_demo_fn = place_demo_fnames[i]
            place_data = np.load(place_demo_fn, allow_pickle=True)

        obj_pts = grasp_data['object_pointcloud']
        grasp_obj_pose = grasp_data['obj_pose_world']
        # inv_grasp_obj_pose_world = np.hstack((-grasp_obj_pose[:3],
        #     util.quat_inverse(grasp_obj_pose[3:])))

        inv_grasp_obj_pose = util.get_inverse_pose(grasp_obj_pose)
        obj_pcd_ori = util.apply_pose_numpy(obj_pts, inv_grasp_obj_pose)

        # print('two poses: ', util.pose_stamped2list(inv_grasp_obj_pose))

        # obj_pcd_ori = util.apply_pose_numpy(obj_pts, inv_grasp_obj_pose)

        # https://stackoverflow.com/questions/61996146/how-to-append-an-array-to-an-existing-npz-file
        grasp_data = dict(grasp_data)
        grasp_data['obj_pcd_ori'] = obj_pcd_ori
        grasp_demo_save_fn = osp.join(demo_save_dir, grasp_demo_fn.split('/')[-1])
        print(f'Saving {grasp_demo_save_fn}')
        np.savez(grasp_demo_save_fn, **grasp_data)

        if convert_place_demos:
            place_data = dict(place_data)
            place_data['obj_pcd_ori'] = obj_pcd_ori
            place_demo_save_fn = osp.join(demo_save_dir, place_demo_fn.split('/')[-1])
            print(f'Saving {place_demo_save_fn}')
            np.savez(place_demo_save_fn, **place_data)


import argparse
import random
import time
from datetime import datetime
import yaml
# from typing import Callable

import os
import os.path as osp

import numpy as np
import torch
import pybullet as p

from scipy.spatial.transform import Rotation as R

from airobot import Robot
from airobot import log_info, log_warn, log_debug, set_log_level
from airobot.utils import common
from airobot.utils.common import euler2quat

import ndf_robot.model.vnn_occupancy_net.vnn_occupancy_net_pointnet_dgcnn \
    as vnn_occupancy_network
import ndf_robot.model.conv_occupancy_net.conv_occupancy_net \
    as conv_occupancy_network

from ndf_robot.opt.optimizer_lite import OccNetOptimizer
from ndf_robot.robot.multicam import MultiCams
from ndf_robot.utils.franka_ik import FrankaIK

from ndf_robot.utils import util, path_util
from ndf_robot.config.default_cam_cfg import get_default_cam_cfg
from ndf_robot.utils.eval_gen_utils import (
    soft_grasp_close, constraint_grasp_close, constraint_obj_world, constraint_grasp_open,
    safeCollisionFilterPair, object_is_still_grasped, get_ee_offset, post_process_grasp_point,
    process_demo_data_rack, process_demo_data_shelf, process_xq_data, process_xq_rs_data, safeRemoveConstraint,
    object_is_intersecting
)
from ndf_robot.eval.evaluate_general_types import (ExperimentTypes, ModelTypes,
    QueryPointTypes, TrialResults, RobotIDs, SimConstants, TrialData)
from ndf_robot.eval.query_points import QueryPoints
from ndf_robot.eval.demo_io import DemoIO


class EvaluateNetwork():
    """
    Parent class for running evaluations on robot arm
    """
    def __init__(self, seed: int, shapenet_obj_dir: str, eval_save_dir: str,
        demo_load_dir: str, test_obj_class: str, pybullet_viz: bool = False, num_trials: int = 200,
        include_avoid_obj: bool = True, any_pose: bool = True):

        self.robot_args = {
            'robot_name': 'franka',
            'pb_cfg': {'gui': pybullet_viz},
            'arm_cfg': {'self_collision': False, 'seed': seed}
        }

        self.robot = Robot(**self.robot_args)

        # self.robot = Robot('franka',
        #                    pb_cfg={'gui': pybullet_viz},
        #                    arm_cfg={'self_collision': False, 'seed': seed})
        self.ik_helper = FrankaIK(gui=False)

        self.shapenet_obj_dir = shapenet_obj_dir
        self.eval_save_dir = eval_save_dir
        self.demo_load_dir = demo_load_dir

        self.eval_grasp_imgs_dir = osp.join(eval_save_dir, 'grasp_imgs')
        self.global_summary_fname = osp.join(eval_save_dir, 'global_summary.txt')
        self.shapenet_id_list_fname = osp.join(eval_save_dir, 'shapenet_id_list.txt')

        util.safe_makedirs(self.eval_grasp_imgs_dir)

        self.num_trials = num_trials
        self.avoid_shapenet_ids = set()
        if not include_avoid_obj:
            self.avoid_shapenet_ids.update(SimConstants.MUG_AVOID_SHAPENET_IDS)
            self.avoid_shapenet_ids.update(SimConstants.BOWL_AVOID_SHAPENET_IDS)
            self.avoid_shapenet_ids.update(SimConstants.BOTTLE_AVOID_SHAPENET_IDS)

        self.train_shapenet_ids = set()
        self.train_shapenet_ids.update(SimConstants.MUG_TRAIN_SHAPENET_IDS)
        self.train_shapenet_ids.update(SimConstants.BOWL_TRAIN_SHAPENET_IDS)
        self.train_shapenet_ids.update(SimConstants.BOTTLE_TRAIN_SHAPENET_IDS)

        self.test_shapenet_ids_all = set()
        self.test_shapenet_ids_all.update(SimConstants.MUG_TEST_SHAPENET_IDS)
        self.test_shapenet_ids_all.update(SimConstants.BOWL_TEST_SHAPENET_IDS)
        self.test_shapenet_ids_all.update(SimConstants.BOTTLE_TEST_SHAPENET_IDS)
        self.test_shapenet_ids_all.update(SimConstants.BOWL_HANDLE_TEST_SHAPENET_IDS)
        self.test_shapenet_ids_all.update(SimConstants.BOTTLE_HANDLE_TEST_SHAPENET_IDS)
        self.test_shapenet_ids_all.update(self.avoid_shapenet_ids)

        self.any_pose = any_pose

        self.experiment_type = None

        self.obj_sample_x_low_high = SimConstants.OBJ_SAMPLE_X_LOW_HIGH
        self.obj_sample_y_low_high = SimConstants.OBJ_SAMPLE_Y_LOW_HIGH

        self.scale_low = SimConstants.MESH_SCALE_LOW
        self.scale_high = SimConstants.MESH_SCALE_HIGH
        self.scale_default = SimConstants.MESH_SCALE_DEFAULT

        self.test_obj_class = test_obj_class

        self.table_urdf_fname = None  # Set in trial

        self.n_demos = 10

    def load_demos(self):
        """
        Load demos relevant to optimizers used.

        Raises:
            NotImplementedError: Implement this!
        """
        raise NotImplementedError

    def configure_sim(self):
        """
        Configure simulation with relevant objects.

        Raises:
            NotImplementedError: Implement this!
        """
        raise NotImplementedError

    def run_trial(self):
        """
        Run a single trial for given experiment

        Raises:
            NotImplementedError: Implement this!
        """
        raise NotImplementedError

    def run_experiment(self, start_idx=0):
        """
        Run experiment of length specified in config.

        Raises:
            NotImplementedError: Implement this!
        """
        raise NotImplementedError

    def _compute_ik_cascade(self, pose: list):
        """
        Solve ik with three different ik solvers, using the next one if the
        previous one fails.

        Args:
            pose (list): [x, y, z, o_x, o_y, o_z, w].

        Returns:
            jnt_positions, TrialResults | None: Tuple where first arg
            describes the joint of panda, while second arg is None if ik was
            successful, else it is a TrialResult error code.
        """
        jnt_pos = None
        ik_found = False
        result = None
        # Try to compute ik in three different ways
        if not ik_found:
            jnt_pos = self.ik_helper.get_feasible_ik(pose)
            ik_found = jnt_pos is not None

        # Try this if the previous ik solver fails
        if not ik_found:
            result = TrialResults.GET_FEASIBLE_IK_FAILED
            jnt_pos = self.ik_helper.get_ik(pose)
            ik_found = jnt_pos is not None

        if not ik_found:
            result = TrialResults.GET_IK_FAILED
            jnt_pos = self.robot.arm.compute_ik(
                pose[:3], pose[3:])

            ik_found = jnt_pos is not None

        if not ik_found:
            print('compute_ik failed')
            result = TrialResults.COMPUTE_IK_FAILED
        else:
            result = None

        return jnt_pos, result

    def _take_image(self, fname):
        """
        Take image of robot in current  position.

        Args:
            fname (filename): Filename of image to save.
        """
        grasp_rgb = self.robot.cam.get_images(get_rgb=True)[0]
        # print(grasp_rgb.shape)
        # print('Cam shape: ', len(self.robot.cam.get_images(get_rgb=True)))
        util.np2img(grasp_rgb.astype(np.uint8), fname)

    # def _take_image_2(self, fname):
    #     for i, cam in enumerate(self.cams.cams):
    #         im_fname = ''.join(fname.split('.')[:-1]) + str(i) + '.' + fname.split('.')[-1]
    #         im_rgb = cam.get_images(get_rgb)[0]
    #         print('Cam shape: ', len(im_rgb))
    #         util.np2img(im_rgb.astype(np.uint8), im_fname)


    @classmethod
    def _compute_anyrot_pose(cls, x_min: float, x_max: float, y_min: float,
        y_max: float, r: float = 0.1) -> 'tuple(list)':
        """
        Compute placement of mug for anyrot trials.  Makes most of
        the mugs physically possible to grab.  The goal is for the open
        end of the mug to be facing the inside of a sphere of radius {r}.
        The sphere is centered at a random point with x and y coordinates
        within [x_min, x_max] and [y_min, y_max], respectively. Since the center
        of the sphere is at the table height, any positions below the table are
        shifted up to table height + a small random shift.  Computed as follows:

        1. Get random orientation for mug.
        2. Transform vector [0, -r, 0] with orientation of mug to get position.
            of mug. The vector has -r in the y position because the mug starts
            with the opening facing the positive y direction.
        3. Compute random shift in x and y

        Args:
            x_min (float): min x position.
            x_max (float): max x position.
            y_min (float): min y position.
            y_max (float): max y position.
            r (float, optional): radius of sphere to place mugs on.  Defaults to
                0.1.

        Returns:
            tuple(list): (pos, ori) where pos is an xyz pose of dim (3, )
                and ori is a quaternion of dim (4, )
        """

        # Mugs init with the opening in the +y direction
        # Reference frame is same as robot
        #     If looking forward at robot, +y is to the right, +x is away,
        #     from robot, +z is up from ground.

        # To debug, use EvaluateGrasp.make_rotation_matrix

        ori_rot = R.random()

        pos_sphere = np.array([0, -r, 0])
        pos_sphere = ori_rot.apply(pos_sphere)

        # So that there is some variation in min z height
        z_center = SimConstants.TABLE_Z + random.random() * 0.05 \
            + SimConstants.OBJ_SAMPLE_Z_OFFSET

        x_offset = random.random() * (x_max - x_min) + x_min
        y_offset = random.random() * (y_max - y_min) + y_min
        pos = [
            pos_sphere[0] + x_offset,
            pos_sphere[1] + y_offset,
            max(z_center, pos_sphere[2] + z_center),
        ]

        ori = ori_rot.as_quat().tolist()

        return pos, ori

    def _insert_object(self, obj_shapenet_id: str, obj_scale: float,
        any_pose: bool, run_decomp_override: bool = False,
        no_gravity: bool = False, spherical_place: bool = True,
        friction=4.0) -> tuple:
        """
        Insert object described by {obj_shapenet_id} at calculated pose.
        Scales input mesh by amount defined in SimConstants.  This amount is
        constant if rand_mesh_scale is False, otherwise it varies based on
        SimConstants.  Inserts object upright if any_pose is False, otherwise
        inserts objects at random rotations and shifted according to
        compute_anyrot_pose().

        Args:
            obj_shapenet_id (str): shapenet id of object
            rand_mesh_scale (bool): True to use random scale for object
            any_pose (bool): True to pose at random orientation and somewhat
                random position.
            run_decomp_override (bool): Force insertion to run vhacd decomposition
                on all objects. (For debugging mostly)
            no_gravity (bool): Prevent objects from falling when inserted.
            spherical_place (bool): True to place anyrot object on exterior of
                sphere, False otherwise. This can help with grasping the rim
                of mugs with the arm.

        Returns:
            tuple: (obj simulation id, object constraint id,
                object pose, object orientation)
        """

        # So that any_pose object doesn't immediately fall
        no_gravity = any_pose or no_gravity
        if no_gravity:
            self.robot.pb_client.set_step_sim(True)

        upright_orientation = common.euler2quat([np.pi / 2, 0, 0]).tolist()

        obj_fname = osp.join(self.shapenet_obj_dir, obj_shapenet_id,
            'models/model_normalized.obj')

        obj_file_dec = obj_fname.split('.obj')[0] + '_dec.obj'

        if obj_scale != -1:
            mesh_scale = [obj_scale * (self.scale_low - self.scale_high)
                + self.scale_low] * 3
        else:
            mesh_scale = [self.scale_default] * 3

        # x_low, x_high = SimConstants.OBJ_SAMPLE_X_LOW_HIGH
        # y_low, y_high = SimConstants.OBJ_SAMPLE_Y_LOW_HIGH

        x_low, x_high = self.obj_sample_x_low_high
        y_low, y_high = self.obj_sample_y_low_high

        r = SimConstants.OBJ_SAMPLE_R if spherical_place else 0

        if any_pose:
            pos, ori = self._compute_anyrot_pose(x_low, x_high, y_low, y_high, r)

        else:
            pos = [np.random.random() * (x_high - x_low) + x_low,
                np.random.random() * (y_high - y_low) + y_low,
                SimConstants.TABLE_Z]
            pose = util.list2pose_stamped(pos + upright_orientation)
            rand_yaw_T = util.rand_body_yaw_transform(pos, min_theta=-np.pi,
                max_theta=np.pi)
            pose_w_yaw = util.transform_pose(pose,
                util.pose_from_matrix(rand_yaw_T))
            pos = util.pose_stamped2list(pose_w_yaw)[:3]
            ori = util.pose_stamped2list(pose_w_yaw)[3:]

        # convert mesh with vhacd
        run_decomp = not osp.exists(obj_file_dec) or run_decomp_override
        if run_decomp:
            p.vhacd(
                obj_fname,
                obj_file_dec,
                'log.txt',
                concavity=0.0025,
                alpha=0.04,
                beta=0.05,
                gamma=0.00125,
                minVolumePerCH=0.0001,
                # resolution=10000000,
                resolution=1000000,
                depth=20,
                planeDownsampling=4,
                # convexhullDownsampling=4,
                # convexhullDownsampling=8,
                convexhullDownsampling=16,
                pca=0,
                mode=0,
                convexhullApproximation=1
            )

        if no_gravity:
            self.robot.pb_client.set_step_sim(True)

        # load object
        obj_id = self.robot.pb_client.load_geom(
            'mesh',
            mass=0.01,
            mesh_scale=mesh_scale,
            visualfile=obj_file_dec,
            collifile=obj_file_dec,
            base_pos=pos,
            base_ori=ori
        )

        # p.changeDynamics(obj_id, -1, lateralFriction=0.5)
        # p.changeDynamics(obj_id, -1, lateralFriction=1.0, linearDamping=5, angularDamping=5)
        p.changeDynamics(obj_id, -1, lateralFriction=friction, linearDamping=5, angularDamping=5)

        o_cid = None
        if no_gravity:
            o_cid = constraint_obj_world(obj_id, pos, ori)
            self.robot.pb_client.set_step_sim(False)

        return obj_id, o_cid, pos, ori

    def _get_pcd(self, obj_id: int) -> np.ndarray:
        """
        Use cameras to get point cloud.

        Args:
            obj_id (int): id of object in simulation.

        Returns:
            ndarray: Point cloud representing observed object.
        """
        depth_imgs = []
        seg_idxs = []
        obj_pcd_pts = []
        table_pcd_pts = []

        for i, cam in enumerate(self.cams.cams):
            # get image and raw point cloud
            rgb, depth, seg = cam.get_images(get_rgb=True, get_depth=True,
                get_seg=True)
            pts_raw, _ = cam.get_pcd(in_world=True, rgb_image=rgb,
                depth_image=depth, depth_min=0.0, depth_max=np.inf)

            # flatten and find corresponding pixels in segmentation mask
            flat_seg = seg.flatten()
            flat_depth = depth.flatten()
            obj_inds = np.where(flat_seg == obj_id)
            table_inds = np.where(flat_seg == self.table_id)
            seg_depth = flat_depth[obj_inds[0]]

            obj_pts = pts_raw[obj_inds[0], :]
            obj_pcd_pts.append(util.crop_pcd(obj_pts))
            table_pts = pts_raw[table_inds[0], :][::int(table_inds[0].shape[0] / 500)]
            table_pcd_pts.append(table_pts)

            depth_imgs.append(seg_depth)
            seg_idxs.append(obj_inds)

        # object shape point cloud
        target_obj_pcd_obs = np.concatenate(obj_pcd_pts, axis=0)
        target_pts_mean = np.mean(target_obj_pcd_obs, axis=0)
        inliers = np.where(np.linalg.norm(
            target_obj_pcd_obs - target_pts_mean, 2, 1) < 0.2)[0]
        target_obj_pcd_obs = target_obj_pcd_obs[inliers]

        return target_obj_pcd_obs

    def _get_test_object_ids(self, demo_shapenet_ids: 'set[str]') -> 'list[str]':
        """
        Find all object that we can test on.

        Args:
            demo_shapenet_ids (set[str]): Set of ids of objects used in demos
                (to be excluded from test)

        Returns:
            list[str]: List of objects to test on.
        """
        test_object_ids = []
        shapenet_id_list = [fn.split('_')[0]
            for fn in os.listdir(self.shapenet_obj_dir)]

        for s_id in shapenet_id_list:
            valid = s_id not in demo_shapenet_ids \
                and s_id not in self.avoid_shapenet_ids \
                and s_id not in self.train_shapenet_ids \
                and s_id in self.test_shapenet_ids_all

            if valid:
                test_object_ids.append(s_id)

        return test_object_ids

    def _set_up_cameras(self):
        """
        Quick helper to put cameras in standard place.
        """
        self.robot.cam.setup_camera(
            focus_pt=[0.4, 0.0, SimConstants.CAMERA_FOCAL_Z],
            dist=0.9,
            yaw=45,
            pitch=-25,
            roll=0)

        cam_cfg = get_default_cam_cfg()

        self.cams = MultiCams(cam_cfg, self.robot.pb_client,
                         n_cams=SimConstants.N_CAMERAS)
        cam_info = {}
        cam_info['pose_world'] = []
        for cam in self.cams.cams:
            cam_info['pose_world'].append(util.pose_from_matrix(cam.cam_ext_mat))
        return cam_info

    def _get_xyz_transform(self, x: float, y: float, z: float):
        return [x, y, z, 0, 0, 0, 1]

    def _step_n_steps(self, n: int):
        """
        If in step simulation, step n times
        """
        for i in range(n):
            p.stepSimulation()

    def reset_sim(self):
        """
        Reset the arm, then set up cameras and place table.  Increases the
        chance that pybullet will detect collisions correctly.  Call at the start
        of each trial.
        """
        assert self.table_urdf_fname is not None, 'Define self.table_urdf_fname' \
            + 'Before running sim.'

        self.robot.arm.reset(force_reset=True)
        self._set_up_cameras()


        table_ori = euler2quat([0, 0, np.pi / 2])
        # Get raw table urdf
        # table_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table.urdf')
        # table_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_rack.urdf')
        with open(self.table_urdf_fname, 'r', encoding='utf-8') as f:
            self.table_urdf = f.read()

        # this is the URDF that was used in the demos -- make sure we load an identical one
        tmp_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_rack_tmp.urdf')
        open(tmp_urdf_fname, 'w').write(self.table_urdf)
        self.table_id = self.robot.pb_client.load_urdf(tmp_urdf_fname,
            SimConstants.TABLE_POS,
            table_ori,
            scaling=SimConstants.TABLE_SCALING)

    def _get_figure_img(self, fname: str, view_mat: list = None, proj_mat: list = None,
        front_view: bool = True):
        """
        Take centered image of simulation.  Set view_mat and proj_mat to set
        your own camera, otherwise default will be used.

        Args:
            fname (str): Filename to save to.  Must include path.
            view_mat (list, optional): length 16 view mat. Generate with
                p.computeViewMatrix.  Defaults to None.
            proj_matrix (list, optional): length 16 proj mat. Generate with
                p.computeProjectionMatrixFOV.  Defaults to None.
        """
        width = 1080
        height = 1080

        fov = 60
        aspect = width / height
        near = 0.02
        far = 4

        # p.computeViewMatrix(cameraEyePosition, cameraTargetPosition, cameraUpVector)
        if view_mat is None:
            if front_view:
                view_mat = p.computeViewMatrix([1.5, 0, 1.6], [0, 0, 1.2], [0, 0, 1])
            else:
                view_mat = p.computeViewMatrix([0.4, 1.1, 1.6], [0.4, 0, 1.2], [0, 0, 1])
        if proj_mat is None:
            proj_mat = p.computeProjectionMatrixFOV(fov, aspect, near, far)

        img = np.array(p.getCameraImage(width, height, view_mat, proj_mat)[2]).reshape(height, width, 4)
        # img = np.array(p.getCameraImage(1080, 1080)[2])
        util.np2img(img.astype(np.uint8), fname)
"""
New evaluate procedure to evaluate grasp ability of networks

Options:
Load different types of networks
Load different types of evaluation procedures

Structure:
Parser:
    Read config file
    Pass appropriate arguments to evaluator

Evaluator:
    Use configs to generate appropriate network
    Use configs to generate appropriate evaluator
    Copy configs to file evaluation folder
"""
import argparse
from multiprocessing.sharedctypes import Value
import numpy as np
import os
import os.path as osp
import yaml
import random
from datetime import datetime
import time
from enum import Enum

import torch
import trimesh

import pybullet as p

from airobot import Robot
from airobot import log_info, log_warn, set_log_level
from airobot.utils import common
from airobot.utils.common import euler2quat
from ndf_robot.config.default_cam_cfg import get_default_cam_cfg
from ndf_robot.config.default_eval_cfg import get_eval_cfg_defaults
from ndf_robot.config.default_obj_cfg import get_obj_cfg_defaults

from ndf_robot.utils import path_util, util

import ndf_robot.model.vnn_occupancy_net.vnn_occupancy_net_pointnet_dgcnn \
    as vnn_occupancy_network
import ndf_robot.model.conv_occupancy_net.conv_occupancy_net \
    as conv_occupancy_network

from ndf_robot.opt.optimizer import OccNetOptimizer
from ndf_robot.robot.multicam import MultiCams
from ndf_robot.utils.franka_ik import FrankaIK

from scipy.spatial.transform import Rotation as R

from ndf_robot.utils.eval_gen_utils import (
    soft_grasp_close, constraint_grasp_close, constraint_obj_world, constraint_grasp_open,
    safeCollisionFilterPair, object_is_still_grasped, get_ee_offset, post_process_grasp_point,
    process_demo_data_rack, process_demo_data_shelf, process_xq_data, process_xq_rs_data, safeRemoveConstraint,
)


ModelTypes = {
    'CONV_OCC',
    'VNN_NDF',
}

QueryPointTypes = {
    'SPHERE',
    'RECT',
}


class TrialResults(Enum):
    SUCCESS = 0
    UNKNOWN_FAILURE = 1
    BAD_GRASP_POS = 2
    NO_FEASIBLE_IK = 3
    INTERSECTING_EE = 4
    GET_FEASIBLE_IK_FAILED = 5
    GET_IK_FAILED = 6
    COMPUTE_IK_FAILED = 7
    POST_PROCESS_FAILED = 8


class RobotIDs:
    finger_joint_id = 9
    left_pad_id = 9
    right_pad_id = 10


class SimConstants:
    # General configs
    N_CAMERAS = 4

    PREGRASP_OFFSET_TF = [0, 0, 0.25, 0, 0, 0, 1]

    # placement of table
    TABLE_POS = [0.5, 0.0, 0.4]
    TABLE_SCALING = 0.9
    # TABLE_Z = 1.05  # Was 1.15
    TABLE_Z = 1.15

    # Different from table so that more stuff is in focus
    CAMERA_FOCAL_Z = 1.15

    # placement of object
    # x is forward / back when facing robot
    # +y is right when facing robot

    OBJ_SAMPLE_X_LOW_HIGH = [0.4, 0.5]
    OBJ_SAMPLE_Y_LOW_HIGH = [-0.2, 0.2]
    OBJ_SAMPLE_Z_OFFSET = 0.0  # was 0.1
    OBJ_SAMPLE_R = 0.2  # was 0.2

    # Object scales
    MESH_SCALE_DEFAULT = 0.5
    MESH_SCALE_HIGH = 0.6
    MESH_SCALE_LOW = 0.4

    # MESH_SCALE_DEFAULT = 0.3
    # MESH_SCALE_HIGH = 0.35
    # MESH_SCALE_LOW = 0.175

    # Avoid Mugs
    MUG_AVOID_SHAPENET_IDS = [
        '32e197b8118b3ff6a4bd4f46ba404890',
        '7374ea7fee07f94c86032c4825f3450',
        '9196f53a0d4be2806ffeedd41ba624d6',
        'b9004dcda66abf95b99d2a3bbaea842a',
        '9ff8400080c77feac2ad6fd1941624c3',
        '4f9f31db3c3873692a6f53dd95fd4468',
        '1c3fccb84f1eeb97a3d0a41d6c77ec7c',
        'cc5b14ef71e87e9165ba97214ebde03',
        '159e56c18906830278d8f8c02c47cde0',
        'c6b24bf0a011b100d536be1f5e11c560',
        '9880097f723c98a9bd8c6965c4665b41',
        'e71102b6da1d63f3a363b55cbd344baa',
        '27119d9b2167080ec190cb14324769d',
        '89bd0dff1b386ebed6b30d74fff98ffd',
        '127944b6dabee1c9e20e92c5b8147e4a',
        '513c3410e8354e5211c7f3807925346a',
        '1bc5d303ff4d6e7e1113901b72a68e7c',
        'b98fa11a567f644344b25d683fe71de',
        'a3cd44bbd3ba5b019a4cbf5d3b79df06',
        'b815d7e084a5a75b8d543d7713b96a41',
        '645b0e2ef3b95979204df312eabf367f',
        '599e604a8265cc0a98765d8aa3638e70',
        '2997f21fa426e18a6ab1a25d0e8f3590',
        'c34718bd10e378186c6c61abcbd83e5a',
        'b7841572364fd9ce1249ffc39a0c3c0b',
        '604fcae9d93201d9d7f470ee20dce9e0',
        'e16a895052da87277f58c33b328479f4',
        '659192a6ba300f1f4293529704725d98',
        '3093367916fb5216823323ed0e090a6f',
        'c7f8d39c406fee941050b055aafa6fb8',
        '64a9d9f6774973ebc598d38a6a69ad2',
        '24b17537bce40695b3207096ecd79542',
        'a1d293f5cc20d01ad7f470ee20dce9e0',
        '6661c0b9b9b8450c4ee002d643e7b29e',
        '85d5e7be548357baee0fa456543b8166',
        'c2eacc521dd65bf7a1c742bb4ffef210',
        'bf2b5e941b43d030138af902bc222a59',
        '127944b6dabee1c9e20e92c5b8147e4a',
        'c2e411ed6061a25ef06800d5696e457f',
        '275729fcdc9bf1488afafc80c93c27a9',
        '642eb7c42ebedabd223d193f5a188983',
        '3a7439cfaa9af51faf1af397e14a566d',
        '642eb7c42ebedabd223d193f5a188983',
        '1038e4eac0e18dcce02ae6d2a21d494a',
        '7223820f07fd6b55e453535335057818',
        '141f1db25095b16dcfb3760e4293e310',
        '4815b8a6406494662a96924bce6ef687',
        '24651c3767aa5089e19f4cee87249aca',
        '5ef0c4f8c0884a24762241154bf230ce',
        '5310945bb21d74a41fabf3cbd0fc77bc',
        '6e884701bfddd1f71e1138649f4c219',
        '345d3e7252156db8d44ee24d6b5498e1',
        'a3cd44bbd3ba5b019a4cbf5d3b79df06',
        '24651c3767aa5089e19f4cee87249aca',
        'b7841572364fd9ce1249ffc39a0c3c0b',
        '1be6b2c84cdab826c043c2d07bb83fc8',
        '604fcae9d93201d9d7f470ee20dce9e0',
        '35ce7ede92198be2b759f7fb0032e59',
        'e71102b6da1d63f3a363b55cbd344baa',
        'dfa8a3a0c8a552b62bc8a44b22fcb3b9',
        'dfa8a3a0c8a552b62bc8a44b22fcb3b9',
        '4f9f31db3c3873692a6f53dd95fd4468',
        '10c2b3eac377b9084b3c42e318f3affc',
        '162201dfe14b73f0281365259d1cf342',
        '1a1c0a8d4bad82169f0594e65f756cf5',
        '3a7439cfaa9af51faf1af397e14a566d',
        '1f035aa5fc6da0983ecac81e09b15ea9',
        '83b41d719ea5af3f4dcd1df0d0a62a93',
        '3d3e993f7baa4d7ef1ff24a8b1564a36',
        '3c0467f96e26b8c6a93445a1757adf6',
        '414772162ef70ec29109ad7f9c200d62',
        '3093367916fb5216823323ed0e090a6f',
        '68f4428c0b38ae0e2469963e6d044dfe',
        'd0a3fdd33c7e1eb040bc4e38b9ba163e',
        'c7ddd93b15e30faae180a52fd2be32',
        '3c0467f96e26b8c6a93445a1757adf6',
        '89bd0dff1b386ebed6b30d74fff98ffd',
        '1dd8290a154f4b1534a8988fdcee4fde',
        '1ae1ba5dfb2a085247df6165146d5bbd',
        '9426e7aa67c83a4c3b51ab46b2f98f30',
        '35ce7ede92198be2b759f7fb0032e59',
        'bcb6be8f0ff4a51872e526c4f21dfca4',
        '43f94ba24d2f075c4d32a65fb7bf4ebc',
        'b9004dcda66abf95b99d2a3bbaea842a',
        '159e56c18906830278d8f8c02c47cde0',
        '275729fcdc9bf1488afafc80c93c27a9',
        '9196f53a0d4be2806ffeedd41ba624d6',
        '64a9d9f6774973ebc598d38a6a69ad2',
        '9880097f723c98a9bd8c6965c4665b41',
        '1dd8290a154f4b1534a8988fdcee4fde',
        '2037531c43448c3016329cbc378d2a2',
        '43f94ba24d2f075c4d32a65fb7bf4ebc',
        'b9f9f5b48ab1153626829c11d9aba173',
        '5582a89be131867846ebf4f1147c3f0f',
        '71ca4fc9c8c29fa8d5abaf84513415a2',
        'd32cd77c6630b77de47c0353c18d58e',
        '1ea9ea99ac8ed233bf355ac8109b9988',
        'c6b24bf0a011b100d536be1f5e11c560',
        'b98fa11a567f644344b25d683fe71de',
        'c82b9f1b98f044fc15cf6e5ad80f2da',
        '5b0c679eb8a2156c4314179664d18101',
        '546648204a20b712dfb0e477a80dcc95',
        'd309d5f8038df4121198791a5b8655c',
        '6c04c2eac973936523c841f9d5051936',
        '71ca4fc9c8c29fa8d5abaf84513415a2',
        '46955fddcc83a50f79b586547e543694',
        '659192a6ba300f1f4293529704725d98',
        'b9be7cfe653740eb7633a2dd89cec754',
        '9fc96d41ec7a66a6a159545213d74ea',
        '5582a89be131867846ebf4f1147c3f0f',
        'c2e411ed6061a25ef06800d5696e457f',
        '8aed972ea2b4a9019c3814eae0b8d399',
        'e363fa24c824d20ca363b55cbd344baa',
        '9426e7aa67c83a4c3b51ab46b2f98f30',
        '6661c0b9b9b8450c4ee002d643e7b29e',
        '8aed972ea2b4a9019c3814eae0b8d399',
        'c39fb75015184c2a0c7f097b1a1f7a5',
        '24b17537bce40695b3207096ecd79542',
        '83b41d719ea5af3f4dcd1df0d0a62a93',
        'c7ddd93b15e30faae180a52fd2be32',
        '46955fddcc83a50f79b586547e543694',
        'c82b9f1b98f044fc15cf6e5ad80f2da',
        'd32cd77c6630b77de47c0353c18d58e',
        '2037531c43448c3016329cbc378d2a2',
        '6500ccc65e210b14d829190312080ea3',
        '6c5ec193434326fd6fa82390eb12174f',
        '1bc5d303ff4d6e7e1113901b72a68e7c',
        '6d2657c640e97c4dd4c0c1a5a5d9a6b8',
        '6c5ec193434326fd6fa82390eb12174f',
        'f3a7f8198cc50c225f5e789acd4d1122',
        'f23a544c04e2f5ccb50d0c6a0c254040',
        'f42a9784d165ad2f5e723252788c3d6e',
        'ea33ad442b032208d778b73d04298f62',
        'ef24c302911bcde6ea6ff2182dd34668',
        'f99e19b8c4a729353deb88581ea8417a',
        'fd1f9e8add1fcbd123c841f9d5051936',
        'f626192a5930d6c712f0124e8fa3930b',
        'ea127b5b9ba0696967699ff4ba91a25',
        'f1866a48c2fc17f85b2ecd212557fda0',
        'ea95f7b57ff1573b9469314c979caef4',
        'b88bcf33f25c6cb15b4f129f868dedb'
    ]


class TrialData():
    """
    Named container class for trial specific information

    Args:
        grasp_success (bool): True if trial was successful
        trial_result (TrialResults): What the outcome of the trial was
            (including) failure modes
        obj_shapenet_id (str): Shapenet id of object used in trial
    """
    grasp_success = False
    trial_result = TrialResults.UNKNOWN_FAILURE
    obj_shapenet_id = None
    best_idx = -1


class EvaluateGrasp():
    """
    Class for running evaluation on robot arm

    Attributes:
        optimizer (OccNetOptimizer): Optimizer instance.
        seed (int): Random seed.
        robot (Robot): Robot to use in simulation.
        ik_helper (FrankaIK): Ik generator.
        obj_class (str): Objects used in test.
        shapenet_obj_dir (str): Dir of test shapenet objects
        eval_save_dir (str): Directory where all info about the trial is saved.
        demo_load_dir (str): Directory where demos are loaded from.
        eval_grasp_imgs_dir (str): Directory where grasp images are saved to.
        global_summary_fname (str): Filename of main summary document where
            trial results are written.
        shapenet_id_list_fname (str): Filename of text list of shapenet ids.
            Used to reproduce exact objects used in trial.
        test_shapenet_ids: list of possible test object ids.  May include ids
            from avoid_shapenet_ids.
        num_trials (int): Number of trials to run.
        avoid_shapenet_ids (list): Shapenet ids to not load.
            Empty if include_avoid_obj is True
        any_pose (bool): True to use any pose for objects.
        cams (list): List of depth cameras used.
        table_id (str or int?): Id that table is assigned when loaded into pb
            simulation.

    """

    def __init__(self, optimizer: OccNetOptimizer, seed: int,
                 shapenet_obj_dir: str, eval_save_dir: str, demo_load_dir: str,
                 pybullet_viz: bool = False, obj_class: str = 'mug',
                 num_trials: int = 200, include_avoid_obj: bool = True,
                 any_pose: bool = True):
        """
        Initialize training class

        Args:
            optimizer (OccNetOptimizer): instance of OccNetOptimizer initialized
                prior to calling this.
            seed (int): Int to serve as random seed.
            shapenet_obj_dir (str): Directory of test shapenet objects
            eval_save_dir (str): Directory to save trial.  Must have been created
                first
            demo_load_dir (str): Directory where the demos are located.
            pybullet_viz (bool, optional): True to show simulation in window.
                Will not work on remote server.  Defaults to False.
            obj_class (str, optional): Type of object.  Must match object class
                of shapenet_obj_dir. Defaults to 'mug'.
            num_trials (int, optional): Number of trials to run.  200 seems
                to provide a reasonable estimate. Defaults to 200.
            include_avoid_obj (bool, optional): True to include objects
                on the avoid object list (funky shapes and whatnot). Defaults
                to True.
            any_pose (bool, optional): True to use the anypose function to
                randomly rotate the mug, then translate it graspable positions.
                Defaults to True.
        """
        self.optimizer = optimizer
        self.seed = seed

        self.robot = Robot('franka',
                           pb_cfg={'gui': pybullet_viz},
                           arm_cfg={'self_collision': False, 'seed': seed})
        self.ik_helper = FrankaIK(gui=False)
        self.obj_class = obj_class

        self.shapenet_obj_dir = shapenet_obj_dir
        self.eval_save_dir = eval_save_dir
        self.demo_load_dir = demo_load_dir
        self.eval_grasp_imgs_dir = osp.join(eval_save_dir, 'grasp_imgs')
        self.global_summary_fname = osp.join(eval_save_dir, 'global_summary.txt')
        self.shapenet_id_list_fname = osp.join(eval_save_dir, 'shapenet_id_list.txt')

        self.test_shapenet_ids = np.loadtxt(osp.join(path_util.get_ndf_share(),
            '%s_test_object_split.txt' % obj_class), dtype=str).tolist()

        self.num_trials = num_trials
        if include_avoid_obj:
            self.avoid_shapenet_ids = []
        else:
            self.avoid_shapenet_ids = SimConstants.MUG_AVOID_SHAPENET_IDS

        self.any_pose = any_pose

        util.safe_makedirs(self.eval_grasp_imgs_dir)

    def configure_sim(self):
        """
        Run after demos are loaded
        """
        set_log_level('info')

        p.changeDynamics(self.robot.arm.robot_id, RobotIDs.left_pad_id,
            lateralFriction=1.0)
        p.changeDynamics(self.robot.arm.robot_id, RobotIDs.right_pad_id,
            lateralFriction=1.0)

        self.robot.arm.reset(force_reset=True)
        self.robot.cam.setup_camera(
            focus_pt=[0.4, 0.0, SimConstants.CAMERA_FOCAL_Z],
            dist=0.9,
            yaw=45,
            pitch=-25,
            roll=0)

        # Set up cameras
        cam_cfg = get_default_cam_cfg()

        self.cams = MultiCams(cam_cfg, self.robot.pb_client,
                         n_cams=SimConstants.N_CAMERAS)
        cam_info = {}
        cam_info['pose_world'] = []
        for cam in self.cams.cams:
            cam_info['pose_world'].append(util.pose_from_matrix(cam.cam_ext_mat))

        # put table at right spot
        table_ori = euler2quat([0, 0, np.pi / 2])

        # Get raw table urdf
        table_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table.urdf')
        with open(table_urdf_fname, 'r', encoding='utf-8') as f:
            self.table_urdf = f.read()

        # this is the URDF that was used in the demos -- make sure we load an identical one
        tmp_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_rack_tmp.urdf')
        open(tmp_urdf_fname, 'w').write(self.table_urdf)
        self.table_id = self.robot.pb_client.load_urdf(tmp_urdf_fname,
            SimConstants.TABLE_POS,
            table_ori,
            scaling=SimConstants.TABLE_SCALING)

    def load_demos(self):
        """
        Load demos from self.demo_load_dir.  Add demo data to optimizer
        and save test_object_ids to self.test_object_ids
        """
        demo_fnames = os.listdir(self.demo_load_dir)
        assert len(demo_fnames), 'No demonstrations found in path: %s!' \
            % self.demo_load_dir

        grasp_demo_fnames = [osp.join(self.demo_load_dir, fn) for fn in
            demo_fnames if 'grasp_demo' in fn]

        # Can add selection of less demos here

        grasp_data_list = []
        demo_target_info_list = []
        demo_shapenet_ids = []

        # Iterate through all demos, extract relevant information and
        # prepare to pass into optimizer
        for grasp_demo_fn in grasp_demo_fnames:
            print('Loading demo from fname: %s' % grasp_demo_fn)
            grasp_data = np.load(grasp_demo_fn, allow_pickle=True)
            grasp_data_list.append(grasp_data)

            # -- Get object points -- #
            # observed shape point cloud at start
            demo_obj_pts = grasp_data['object_pointcloud']
            demo_pts_mean = np.mean(demo_obj_pts, axis=0)
            inliers = np.where(
                np.linalg.norm(demo_obj_pts - demo_pts_mean, 2, 1) < 0.2)[0]
            demo_obj_pts = demo_obj_pts[inliers]

            # -- Get query pts -- #
            demo_gripper_pts = self.optimizer.query_pts_origin
            demo_gripper_pcd = trimesh.PointCloud(demo_gripper_pts)

            # end-effector pose before grasping
            demo_ee_mat = util.matrix_from_pose(
                    util.list2pose_stamped(grasp_data['ee_pose_world']))
            demo_gripper_pcd.apply_transform(demo_ee_mat)

            # points we use to represent the gripper at their canonical pose
            # position shown in the demonstration
            demo_gripper_pts = np.asarray(demo_gripper_pcd.vertices)

            target_info = dict(
                demo_query_pts=demo_gripper_pts,
                demo_query_pts_real_shape=demo_gripper_pts,
                demo_obj_pts=demo_obj_pts,
                demo_ee_pose_world=grasp_data['ee_pose_world'],
                demo_query_pt_pose=grasp_data['gripper_contact_pose'],
                demo_obj_rel_transform=np.eye(4)
            )

            # -- Get shapenet id -- #
            shapenet_id = grasp_data['shapenet_id'].item()

            demo_target_info_list.append(target_info)
            demo_shapenet_ids.append(shapenet_id)

            # # -- Get table urdf -- #
            # Used to get same urdf as used in demos (i.e. with rack)
            # self.table_urdf = grasp_data['table_urdf'].item()

        # -- Set demos -- #
        self.optimizer.set_demo_info(demo_target_info_list)

        # -- Get test objects -- #
        self.test_object_ids = []
        if self.obj_class == 'mug':
            shapenet_id_list = [fn.split('_')[0]
                for fn in os.listdir(self.shapenet_obj_dir)]
        else:
            shapenet_id_list = os.listdir(self.shapenet_obj_dir)

        for s_id in shapenet_id_list:
            valid = s_id not in demo_shapenet_ids \
                and s_id not in self.avoid_shapenet_ids

            if valid:
                self.test_object_ids.append(s_id)

    def run_trial(self, iteration: int = 0, rand_mesh_scale: bool = True,
                  any_pose: bool = True, thin_feature: bool = True,
                  grasp_viz: bool = False,
                  grasp_dist_thresh: float = 0.0025) -> TrialData:
        """
        Run single trial instance.

        Args:
            iteration (int, optional): What iteration the trial is. Defaults to 0.
            rand_mesh_scale (bool, optional): True to randomly scale mesh.
                Defaults to True.
            any_pose (bool, optional): True to use anypose function to pose mug.
                Defaults to True.
            thin_feature (bool, optional): True to treat object as thin feature
                in grasp post process. Defaults to True.
            grasp_viz (bool, optional): True to show image of grasp before trial
                runs. Only works when pybullet_viz is enabled. Defaults to False.
            grasp_dist_thresh (float, optional): Threshold to detect successful
                grasp. Defaults to 0.0025.

        Returns:
            TrialData: Class for storing relevant info about the trial
        """

        # For save purposes
        trial_data = TrialData()

        eval_iter_dir = osp.join(self.eval_save_dir, 'trial_%s' % str(iteration).zfill(3))
        util.safe_makedirs(eval_iter_dir)

        # -- Get and orient object -- #
        obj_shapenet_id = random.sample(self.test_object_ids, 1)[0]
        trial_data.obj_shapenet_id = obj_shapenet_id

        # Write at start so id is recorded regardless of any later bugs
        with open(self.shapenet_id_list_fname, 'a') as f:
            f.write(f'{trial_data.obj_shapenet_id}\n')

        upright_orientation = common.euler2quat([np.pi / 2, 0, 0]).tolist()

        obj_fname = osp.join(self.shapenet_obj_dir, obj_shapenet_id,
            'models/model_normalized.obj')

        obj_file_dec = obj_fname.split('.obj')[0] + '_dec.obj'

        scale_low = SimConstants.MESH_SCALE_LOW
        scale_high = SimConstants.MESH_SCALE_HIGH
        scale_default = SimConstants.MESH_SCALE_DEFAULT
        if rand_mesh_scale:
            mesh_scale = [np.random.random() * (scale_low - scale_high)
                + scale_low] * 3
        else:
            mesh_scale = [scale_default] * 3

        x_low, x_high = SimConstants.OBJ_SAMPLE_X_LOW_HIGH
        y_low, y_high = SimConstants.OBJ_SAMPLE_Y_LOW_HIGH

        # min_r, max_r = SimConstants.OBJ_SAMPLE_R_MIN_MAX
        r = SimConstants.OBJ_SAMPLE_R
        # x_offset = SimConstants.OBJ_SAMPLE_X_OFFSET
        # y_offset = SimConstants.OBJ_SAMPLE_Y_OFFSET

        if any_pose:
            pos, ori = self.compute_anyrot_pose(x_low, x_high, y_low, y_high, r)

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
        if not osp.exists(obj_file_dec):
            p.vhacd(
                obj_fname,
                obj_file_dec,
                'log.txt',
                concavity=0.0025,
                alpha=0.04,
                beta=0.05,
                gamma=0.00125,
                minVolumePerCH=0.0001,
                resolution=1000000,
                depth=20,
                planeDownsampling=4,
                convexhullDownsampling=4,
                pca=0,
                mode=0,
                convexhullApproximation=1
            )

        # -- Run robot -- #
        self.robot.arm.go_home(ignore_physics=True)
        self.robot.arm.move_ee_xyz([0, 0, 0.2])

        if any_pose:
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

        p.changeDynamics(obj_id, -1, lateralFriction=0.5)

        o_cid = None
        if any_pose:
            o_cid = constraint_obj_world(obj_id, pos, ori)
            self.robot.pb_client.set_step_sim(False)
        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
        p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
        time.sleep(1.5)

        # -- Get object point cloud -- #
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

        # -- Get grasp position -- #
        opt_viz_path = osp.join(eval_iter_dir, 'visualize')
        pre_grasp_ee_pose_mats, best_idx = self.optimizer.optimize_transform_implicit(
            target_obj_pcd_obs, ee=True, viz_path=opt_viz_path)
        pre_grasp_ee_pose = util.pose_stamped2list(util.pose_from_matrix(
            pre_grasp_ee_pose_mats[best_idx]))
        trial_data.best_idx = best_idx

        # -- Post process grasp position -- #
        # print('pre_grasp_ee_pose: ', pre_grasp_ee_pose)
        try:
            # When there are no nearby grasp points, this throws an index
            # error.  The try catch allows us to run more trials after the error.
            new_grasp_pt = post_process_grasp_point(
                pre_grasp_ee_pose,
                target_obj_pcd_obs,
                thin_feature=thin_feature,
                grasp_viz=grasp_viz,
                grasp_dist_thresh=grasp_dist_thresh)
        except IndexError:
            trial_data.trial_result = TrialResults.POST_PROCESS_FAILED
            self.robot.pb_client.remove_body(obj_id)
            return trial_data

        pre_grasp_ee_pose[:3] = new_grasp_pt
        pregrasp_offset_tf = get_ee_offset(ee_pose=pre_grasp_ee_pose)
        pre_pre_grasp_ee_pose = util.pose_stamped2list(
            util.transform_pose(
                pose_source=util.list2pose_stamped(pre_grasp_ee_pose),
                pose_transform=util.list2pose_stamped(pregrasp_offset_tf)))

        # -- Attempt grasp -- #
        jnt_pos = grasp_jnt_pos = grasp_plan = None
        grasp_success = False
        for g_idx in range(2):
            # reset everything
            self.robot.pb_client.set_step_sim(False)
            safeCollisionFilterPair(obj_id, self.table_id, -1, -1,
                enableCollision=True)

            if any_pose:
                # Set to step mode (presumably so object doesn't fall when
                # o_cid constraint is removed)
                self.robot.pb_client.set_step_sim(True)
            safeRemoveConstraint(o_cid)
            p.resetBasePositionAndOrientation(obj_id, pos, ori)
            print(p.getBasePositionAndOrientation(obj_id))
            time.sleep(0.5)

            # Freeze object in space and return to realtime
            if any_pose:
                o_cid = constraint_obj_world(obj_id, pos, ori)
                self.robot.pb_client.set_step_sim(False)

            # turn OFF collisions between robot and object / table, and move to pre-grasp pose
            for i in range(p.getNumJoints(self.robot.arm.robot_id)):
                safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id,
                    bodyUniqueIdB=self.table_id,
                    linkIndexA=i,
                    linkIndexB=-1,
                    enableCollision=False,
                    physicsClientId=self.robot.pb_client.get_client_id())

                safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id,
                    bodyUniqueIdB=obj_id,
                    linkIndexA=i,
                    linkIndexB=-1,
                    enableCollision=False,
                    physicsClientId=self.robot.pb_client.get_client_id())

            self.robot.arm.eetool.open()

            # -- Get ik -- #
            ik_found = jnt_pos is not None and grasp_jnt_pos is not None
            # Try to compute ik in three different ways
            if not ik_found:
                jnt_pos = self.ik_helper.get_feasible_ik(pre_pre_grasp_ee_pose)
                grasp_jnt_pos = self.ik_helper.get_feasible_ik(pre_grasp_ee_pose)
                ik_found = jnt_pos is not None and grasp_jnt_pos is not None
                if not ik_found:
                    print('get_feasible_ik failed')
                    trial_data.trial_result = TrialResults.GET_FEASIBLE_IK_FAILED

            # What does this do?
            if not ik_found:
                jnt_pos = self.ik_helper.get_ik(pre_pre_grasp_ee_pose)
                grasp_jnt_pos = self.ik_helper.get_ik(pre_grasp_ee_pose)
                ik_found = jnt_pos is not None and grasp_jnt_pos is not None
                if not ik_found:
                    print('get_ik failed')
                    trial_data.trial_result = TrialResults.GET_IK_FAILED

            if not ik_found:
                jnt_pos = self.robot.arm.compute_ik(
                    pre_pre_grasp_ee_pose[:3], pre_pre_grasp_ee_pose[3:])
                # this is the pose that's at the grasp,
                # where we just need to close the fingers
                grasp_jnt_pos = self.robot.arm.compute_ik(
                    pre_grasp_ee_pose[:3], pre_grasp_ee_pose[3:])

                ik_found = jnt_pos is not None and grasp_jnt_pos is not None
                if not ik_found:
                    print('compute_ik failed')
                    trial_data.trial_result = TrialResults.COMPUTE_IK_FAILED

            if not ik_found:
                continue

            # Get grasp image
            if g_idx == 0:
                self.robot.pb_client.set_step_sim(True)
                self.robot.arm.set_jpos(grasp_jnt_pos, ignore_physics=True)
                self.robot.arm.eetool.close(ignore_physics=True)
                time.sleep(0.2)
                grasp_rgb = self.robot.cam.get_images(get_rgb=True)[0]
                grasp_img_fname = osp.join(self.eval_grasp_imgs_dir,
                    '%s_pose.png' % str(iteration).zfill(3))
                util.np2img(grasp_rgb.astype(np.uint8), grasp_img_fname)

                self.robot.arm.go_home(ignore_physics=True)
                continue

            # -- Get grasp plan -- #
            home_jnt_pos = self.robot.arm.get_jpos()

            # Get to pre grasp location
            plan1 = self.ik_helper.plan_joint_motion(home_jnt_pos, jnt_pos)

            # Get to grasp location
            plan2 = self.ik_helper.plan_joint_motion(jnt_pos, grasp_jnt_pos)

            # Return to home location (for checking if grasp was valid)
            plan3 = self.ik_helper.plan_joint_motion(grasp_jnt_pos, home_jnt_pos)

            if plan1 is not None and plan2 is not None:
                # First move to clearance distance, then try to grasp
                grasp_plan = plan1 + plan2

                self.robot.arm.eetool.open()
                # Go to clearance location (linearly away from grasp area)
                for jnt in plan1:
                    self.robot.arm.set_jpos(jnt, wait=False)
                    time.sleep(0.025)
                self.robot.arm.set_jpos(plan1[-1], wait=False)

                # Used to be below plan2 part

                # turn ON collisions between robot and object
                for i in range(p.getNumJoints(self.robot.arm.robot_id)):
                    safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id,
                        bodyUniqueIdB=obj_id,
                        linkIndexA=i,
                        linkIndexB=-1,
                        enableCollision=True,
                        physicsClientId=self.robot.pb_client.get_client_id())

                # Go to grasp location
                for jnt in plan2:
                    self.robot.arm.set_jpos(jnt, wait=False)
                    time.sleep(0.04)
                self.robot.arm.set_jpos(plan2[-1], wait=False)

                # get pose that's straight up
                # offset_pose = util.transform_pose(
                #     pose_source=util.list2pose_stamped(
                #         np.concatenate(self.robot.arm.get_ee_pose()[:2]).tolist()),
                #     pose_transform=util.list2pose_stamped([0, 0, 0.15, 0, 0, 0, 1])
                # )

                # offset_pose_list = util.pose_stamped2list(offset_pose)
                # offset_jnts = self.ik_helper.get_feasible_ik(offset_pose_list)

                time.sleep(0.8)
                # obj_pos_before_grasp = p.getBasePositionAndOrientation(obj_id)[0]
                # jnt_pos_before_grasp = self.robot.arm.get_jpos()

                grasp_rgb = self.robot.cam.get_images(get_rgb=True)[0]
                grasp_img_fname = osp.join(self.eval_grasp_imgs_dir,
                    f'{str(iteration).zfill(3)}_grasp.png')
                util.np2img(grasp_rgb.astype(np.uint8), grasp_img_fname)

                soft_grasp_close(self.robot, RobotIDs.finger_joint_id, force=50)
                safeRemoveConstraint(o_cid)
                time.sleep(0.8)
                safeCollisionFilterPair(obj_id, self.table_id, -1, -1,
                    enableCollision=False)
                time.sleep(0.8)

                for jnt in plan3:
                    self.robot.arm.set_jpos(jnt, wait=False)
                    time.sleep(0.025)
                self.robot.arm.set_jpos(plan3[-1], wait=False)
                time.sleep(1)

                # self.robot.arm.go_home(ignore_physics=False)
                # time.sleep(0.8)

                if g_idx == 1:
                    original_grasp_success = object_is_still_grasped(self.robot,
                        obj_id, RobotIDs.right_pad_id, RobotIDs.left_pad_id)

                    time.sleep(0.5)

                    # If the ee was intersecting the mug, original_grasp_success
                    # would be true after the table disappears.  However, an
                    # intersection is generally a false grasp When the ee is
                    # opened again, a good grasp should fall down while an
                    # intersecting grasp would stay in contact.

                    # -- Take image of grasp at clearance height -- #
                    grasp_rgb = self.robot.cam.get_images(get_rgb=True)[0]
                    grasp_img_fname = osp.join(self.eval_grasp_imgs_dir,
                        '%s_clearance.png' % str(iteration).zfill(3))
                    util.np2img(grasp_rgb.astype(np.uint8), grasp_img_fname)

                    self.robot.arm.eetool.open()
                    time.sleep(1)
                    ee_intersecting_mug = object_is_still_grasped(
                        self.robot, obj_id, RobotIDs.right_pad_id,
                        RobotIDs.left_pad_id)

                    grasp_success = original_grasp_success and not ee_intersecting_mug

                    if ee_intersecting_mug:
                        print('Intersecting grasp detected')
                        trial_data.trial_result = TrialResults.INTERSECTING_EE
                    else:
                        if not grasp_success:
                            trial_data.trial_result = TrialResults.BAD_GRASP_POS

                    log_info(f'Grasp success: {grasp_success}')

        if grasp_success:
            trial_data.trial_result = TrialResults.SUCCESS

        trial_data.grasp_success = grasp_success
        self.robot.pb_client.remove_body(obj_id)

        return trial_data

    def run_experiment(self, rand_mesh_scale: bool = True):
        """
        Run experiment for {self.num_trials}
        """
        num_success = 0
        for it in range(self.num_trials):
            trial_data = experiment.run_trial(iteration=it,
                rand_mesh_scale=rand_mesh_scale, any_pose=self.any_pose)

            grasp_success = trial_data.grasp_success
            obj_shapenet_id = trial_data.obj_shapenet_id
            trial_result = trial_data.trial_result

            num_success += grasp_success
            log_info(f'Trial result: {trial_result}')
            log_str = f'Successes: {num_success} | Trials {it + 1} | ' \
                + f'Success Rate: {num_success / (it + 1)}'
            log_info(log_str)

            with open(self.global_summary_fname, 'a') as f:
                f.write(f'Trial number: {it}\n')
                f.write(f'Grasp success: {grasp_success}\n')
                f.write(f'Trial result: {trial_result}\n')
                f.write(f'Success Rate: {num_success / (it + 1)}\n')
                f.write(f'Shapenet id: {obj_shapenet_id}\n')
                f.write(f'Best idx: {trial_data.best_idx}\n')
                f.write('\n')

    @classmethod
    def hide_link(cls, obj_id, link_id):
        if link_id is not None:
            p.changeVisualShape(obj_id, link_id, rgbaColor=[0, 0, 0, 0])

    @classmethod
    def show_link(cls, obj_id, link_id, color):
        if link_id is not None:
            p.changeVisualShape(obj_id, link_id, rgbaColor=color)

    @classmethod
    def compute_anyrot_pose(cls, x_min: float, x_max: float, y_min: float,
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

    @classmethod
    def compute_anyrot_pose_legacy(cls, x_high: float, x_low: float,
        y_high: float, y_low: float) -> 'tuple(list)':
        """
        Previous code for anyrot pos calculation.  May be buggy?  Only here
        for comparison to new code.

        Args:
            x_high (float): max x value.
            x_low (float): min x value.
            y_high (float): max y value.
            y_low (float): min y value.

        Returns:
            tuple(list): (pos, ori) where pos is an xyz pose of dim (3, )
                and ori is a quaternion of dim (4, )
        """
        rp = np.random.rand(3) * (2 * np.pi / 3) - (np.pi / 3)
        ori = common.euler2quat([rp[0], rp[1], rp[2]]).tolist()

        pos = [
            np.random.random() * (x_high - x_low) + x_low,
            np.random.random() * (y_high - y_low) + y_low,
            SimConstants.TABLE_Z
        ]

        pose = pos + ori  # concat of pos and orientation
        rand_yaw_T = util.rand_body_yaw_transform(pos, min_theta=-np.pi,
            max_theta=np.pi)
        pose_w_yaw = util.transform_pose(util.list2pose_stamped(pose),
            util.pose_from_matrix(rand_yaw_T))
        pos, ori = util.pose_stamped2list(pose_w_yaw)[:3], \
            util.pose_stamped2list(pose_w_yaw)[3:]

        return pos, ori

    @staticmethod
    def make_rotation_matrix(axis: str, theta: float):
        """
        Make rotation matrix about {axis} with angle {theta}

        Args:
            axis (str): {'x', 'y', 'z'}
            theta (float): angle in radians
        """

        s = np.sin(theta)
        c = np.cos(theta)

        if axis == 'x':
            r = [[1, 0, 0],
                 [0, c, -s],
                 [0, s, c]]

        elif axis == 'y':
            r = [[c, 0, s],
                 [0, 1, 0],
                 [-s, 0, c]]

        elif axis == 'z':
            r = [[c, -s, 0],
                 [s, c, 0],
                 [0, 0, 1]]

        else:
            raise ValueError('Unexpected axis')

        return r


class EvaluateGraspSetup():
    """
    Set up experiment from config file
    """
    def __init__(self):
        self.config_dir = osp.join(path_util.get_ndf_eval(), 'eval_configs')

        self.evaluator_dict = None
        self.model_dict = None
        self.optimizer_dict = None
        self.query_pts_dict = None

        self.seed = None

    def load_config(self, fname: str):
        """
        Load config from yaml file with following fields:
            evaluator:
                ...

            model:
                model_type: VNN_NDF or CONV_OCC
                model_args:
                    ...

            optimizer:
                optimizer_args:
                    ...

            query_pts:
                query_pts_type: SPHERE (later add GRIPPER)
                query_pts_args:
                    ...

        Args:
            fname (str): Name of config file.  Assumes config file is in
                'eval_configs' in 'eval' folder.  Name does not include any
                path prefixes (e.g. 'default_config.yml' is fine)

        """
        config_path = osp.join(self.config_dir, fname)
        with open(config_path, "r") as stream:
            try:
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.config_dict = config_dict
        self.evaluator_dict = config_dict['evaluator']
        self.model_dict = config_dict['model']
        self.optimizer_dict = config_dict['optimizer']
        self.query_pts_dict = config_dict['query_pts']
        self.setup_args = config_dict['setup_args']

        self.seed = self.setup_args['seed']
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        print(config_dict)

    def create_model(self) -> torch.nn.Module:
        """
        Create torch model from given configs

        Returns:
            torch.nn.Module: Either ConvOccNetwork or VNNOccNet
        """
        model_type = self.model_dict['type']
        model_args = self.model_dict['args']
        model_checkpoint = osp.join(path_util.get_ndf_model_weights(),
                                    self.model_dict['checkpoint'])

        assert model_type in ModelTypes, 'Invalid model type'

        if model_type == 'CONV_OCC':
            model = conv_occupancy_network.ConvolutionalOccupancyNetwork(
                **model_args)
            print('Using CONV OCC')

        elif model_type == 'VNN_NDF':
            model = vnn_occupancy_network.VNNOccNet(**model_args)
            print('USING NDF')

        model.load_state_dict(torch.load(model_checkpoint))

        print('---MODEL---\n', model)
        return model

    def create_optimizer(self, model: torch.nn.Module,
                         query_pts: np.ndarray, eval_save_dir=None) -> OccNetOptimizer:
        """
        Create OccNetOptimizer from given config

        Args:
            model (torch.nn.Module): Model to use in the optimizer
            query_pts (np.ndarray): Query points to use in optimizer

        Returns:
            OccNetOptimizer: Optimizer to find best grasp position
        """
        optimizer_args = self.optimizer_dict['args']
        if eval_save_dir is not None:
            opt_viz_path = osp.join(eval_save_dir, 'visualization')
        else:
            opt_viz_path = 'visualization'

        optimizer = OccNetOptimizer(model, query_pts, viz_path=opt_viz_path,
            **optimizer_args)
        return optimizer

    def create_query_pts(self) -> np.ndarray:
        """
        Create query points from given config

        Returns:
            np.ndarray: Query point as ndarray
        """

        query_pts_type = self.query_pts_dict['type']
        query_pts_args = self.query_pts_dict['args']

        assert query_pts_type in QueryPointTypes, 'Invalid query point type'

        if query_pts_type == 'SPHERE':
            query_pts = QueryPoints.generate_sphere(**query_pts_args)
        elif query_pts_type == 'RECT':
            query_pts = QueryPoints.generate_rect(**query_pts_args)

        return query_pts

    def create_eval_dir(self) -> str:
        """
        Create eval save dir as concatenation of current time
        and 'exp_desc'.

        Args:
            exp_desc (str, optional): Description of experiment. Defaults to ''.

        Returns:
            str: eval_save_dir.  Gives access to eval save directory
        """
        if 'exp_dir_suffix' in self.setup_args:
            exp_desc = self.setup_args['exp_dir_suffix']
        else:
            exp_desc = ''
        experiment_class = 'eval_grasp'
        t = datetime.now()
        time_str = t.strftime('%Y-%m-%d_%HH%MM%SS_%a')
        if exp_desc != '':
            experiment_name = time_str + '_' + exp_desc
        else:
            experiment_name = time_str + exp_desc

        eval_save_dir = osp.join(path_util.get_ndf_eval_data(),
                                 experiment_class,
                                 experiment_name)

        util.safe_makedirs(eval_save_dir)

        config_fname_yml = osp.join(eval_save_dir, 'config.yml')
        config_fname_txt = osp.join(eval_save_dir, 'config.txt')
        with open(config_fname_yml, 'w') as f:
            yaml.dump(self.config_dict, f)

        with open(config_fname_txt, 'w') as f:
            yaml.dump(self.config_dict, f)

        return eval_save_dir

    def get_demo_load_dir(self, obj_class: str='mug',
        demo_exp: str='grasp_rim_hang_handle_gaussian_precise_w_shelf') -> str:
        """
        Get directory of demos

        Args:
            obj_class (str, optional): Object class. Defaults to 'mug'.
            demo_exp (str, optional): Demo experiment name. Defaults to
                'grasp_rim_hang_handle_gaussian_precise_w_shelf'.

        Returns:
            str: Path to demo load dir
        """
        if 'demo_exp' in self.setup_args:
            demo_exp = self.setup_args['demo_exp']
        demo_load_dir = osp.join(path_util.get_ndf_data(),
                                 'demos', obj_class, demo_exp)

        return demo_load_dir

    def get_shapenet_obj_dir(self) -> str:
        """
        Get object dir of obj_class

        Args:
            obj_class (str, optional): Class of object (mug, bottle, bowl).
                Defaults to 'mug'.

        Returns:
            str: path to object dir
        """
        obj_class = self.evaluator_dict['obj_class']
        shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(),
                                    obj_class + '_centered_obj_normalized')

        return shapenet_obj_dir

    def get_evaluator_args(self) -> dict:
        return self.evaluator_dict

    def get_seed(self) -> int:
        return self.seed


class QueryPoints():
    @staticmethod
    def generate_sphere(n_pts: int, radius: float=0.05) -> np.ndarray:
        """
        Sample points inside sphere centered at origin with radius {radius}

        Args:
            n_pts (int): Number of point to sample.
            radius (float, optional): Radius of sphere to sample.
                Defaults to 0.05.

        Returns:
            np.ndarray: (n_pts x 3) array of query points
        """
        # http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        u = 2 * np.random.rand(n_pts, 1) - 1
        phi = 2 * np.pi * np.random.rand(n_pts, 1)
        r = radius * (np.random.rand(n_pts, 1)**(1 / 3.))
        x = r * np.cos(phi) * (1 - u**2)**0.5
        y = r * np.sin(phi) * (1 - u**2)**0.5
        z = r * u

        sphere_points = np.hstack((x, y, z))
        return sphere_points

    @staticmethod
    def generate_rect(n_pts: int, x: float, y: float, z1: float, z2: float) \
        -> np.ndarray:
        """
        Create rectangle of query points with center of grip at 'o' and
        dimensions as shown.  'o' is centered in x and y.  All dims should be
        positive.

        With this system, z1 is closest to the gripper body while y is
        along the direction that the gripper fingers move.  It seems that
        a high z1 and lower z2 work well.

                   _________
                 /          /|
            ___ /_________ / |
             |  |         |  |
             |  |         |  |
             |  |         |  |
             z2 |         |  |
             |  |         |  |
             |  |     o   |  |  __
            -+- | - -/    |  /  /
             z1 |         | /  y
            _|_ |_________|/ _/_

                |----x----|

        Args:
            n_pts (int): Number of point to sample.
            x (float): x dim.
            y (float): y dim.
            z1 (float): z1 dim.
            z2 (float): z2 dim.

        Returns:
            np.ndarray: (n_pts x 3) array of query points
        """

        rect_points = np.random.rand(n_pts, 3)
        scale_mat = np.array(
            [[x, 0, 0],
             [0, y, 0],
             [0, 0, z1 + z2]]
        )
        offset_mat = np.array(
            [[x/2, y/2, z1]]
        )
        rect_points = rect_points @ scale_mat - offset_mat
        return rect_points

    @staticmethod
    def generate_cylinder(n_pts: int, radius: float, height: float, rot_axis='z') \
        -> np.ndarray:
        """
        Generate np array of {n_pts} 3d points with radius {radius} and
        height {height} in the shape of a cylinder with axis of rotation about
        {rot_axis} and points along the positive rot_axis.

        Args:
            n_pts (int): Number of points to generate.
            radius (float): Radius of cylinder.
            height (float): height of cylinder.
            rot_axis (str, optional): Choose (x, y, z). Defaults to 'z'.

        Returns:
            np.ndarray: (n_pts, 3) array of points.
        """

        U_TH = np.random.rand(n_pts, 1)
        U_R = np.random.rand(n_pts, 1)
        U_Z = np.random.rand(n_pts, 1)
        X = radius * np.sqrt(U_R) * np.cos(2 * np.pi * U_TH)
        Y = radius * np.sqrt(U_R) * np.sin(2 * np.pi * U_TH)
        Z = height * U_Z
        # Z = np.zeros((n_pts, 1))

        points = np.hstack([X, Y, Z])
        rotate = np.eye(3)
        if rot_axis == 'x':
            rotate[[0, 2]] = rotate[[2, 0]]
        elif rot_axis == 'y':
            rotate[[1, 2]] = rotate[[2, 1]]

        points = points @ rotate
        return points


if __name__ == '__main__':
    # config_fname = 'debug_config.yml'
    # config_fname = 'debug_config_ndf.yml'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fname', type=str, default='debug_config.yml',
                        help='Filename of experiment config yml')

    args = parser.parse_args()
    config_fname = args.config_fname

    setup = EvaluateGraspSetup()
    setup.load_config(config_fname)
    model = setup.create_model()
    query_pts = setup.create_query_pts()
    shapenet_obj_dir = setup.get_shapenet_obj_dir()
    eval_save_dir = setup.create_eval_dir()
    demo_load_dir = setup.get_demo_load_dir(obj_class='mug')
    optimizer = setup.create_optimizer(model, query_pts, eval_save_dir=eval_save_dir)
    evaluator_args = setup.get_evaluator_args()

    experiment = EvaluateGrasp(optimizer=optimizer, seed=setup.get_seed(),
        shapenet_obj_dir=shapenet_obj_dir, eval_save_dir=eval_save_dir,
        demo_load_dir=demo_load_dir, **evaluator_args)

    experiment.load_demos()
    experiment.configure_sim()
    experiment.run_experiment()
    # num_success = 0
    # for i in range(200):
    #     num_success += experiment.run_trial(iteration=i, rand_mesh_scale=True, any_pose=True)
    #     print('---')
    #     print(f'Successes: {num_success} | Trials {i + 1} | '
    #         + f'Success Rate: {num_success / (i + 1)}')

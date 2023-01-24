import random
import time

import os
import os.path as osp

import numpy as np
import pybullet as p

from scipy.spatial.transform import Rotation as R

from airobot import Robot
from airobot import log_info, log_warn, log_debug, set_log_level
from airobot.utils import common
from airobot.utils.common import euler2quat

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
from ndf_robot.eval.demo_io import DemoIO

from ndf_robot.eval.experiments.evaluate_network import EvaluateNetwork


class EvaluateShelfPlaceTeleport(EvaluateNetwork):
    def __init__(self, place_optimizer: OccNetOptimizer,
                 seed: int, shapenet_obj_dir: str, eval_save_dir: str,
                 demo_load_dir: str, pybullet_viz: bool = False,
                 test_obj_class: str = 'mug', num_trials: int = 200,
                 include_avoid_obj: bool = True, any_pose: bool = True):

        super().__init__(seed, shapenet_obj_dir, eval_save_dir,
            demo_load_dir, test_obj_class, pybullet_viz, num_trials,
            include_avoid_obj, any_pose)

        self.place_optimizer = place_optimizer
        self.experiment_type = ExperimentTypes.SHELF_PLACE_TELEPORT

        self.scale_low = SimConstants.MESH_SCALE_LOW
        self.scale_high = 0.5

    def load_demos(self):
        """
        Load demos from self.demo_load_dir.  Add demo data to optimizer
        and save test_object_ids to self.test_object_ids
        """
        demo_fnames = os.listdir(self.demo_load_dir)
        assert len(demo_fnames), 'No demonstrations found in path: %s!' \
            % self.demo_load_dir

        place_demo_fnames = [osp.join(self.demo_load_dir, fn) for fn in
            demo_fnames if 'place_demo' in fn]

        # Can add selection of less demos here
        demo_shapenet_ids = []

        random.shuffle(place_demo_fnames)
        for place_demo_fn in place_demo_fnames[:self.n_demos]:
            print('Loading place demo from fname: %s' % place_demo_fn)
            place_data = np.load(place_demo_fn, allow_pickle=True)

            demo = DemoIO.process_shelf_place_data(place_data)

            self.place_optimizer.add_demo(demo)
            demo_shapenet_ids.append(demo.obj_shapenet_id)

        self.place_optimizer.process_demos()

        # -- Get urdf -- #
        place_data = np.load(place_demo_fnames[0], allow_pickle=True)
        self.table_urdf = DemoIO.get_table_urdf(place_data)
        self.shelf_pose = DemoIO.get_shelf_pose(place_data)

        # -- Get test objects -- #
        self.test_object_ids = self._get_test_object_ids(demo_shapenet_ids)

    def configure_sim(self):
        """
        Run after demos are loaded
        """
        set_log_level('debug')

        p.changeDynamics(self.robot.arm.robot_id, RobotIDs.left_pad_id,
            lateralFriction=1.0)
        p.changeDynamics(self.robot.arm.robot_id, RobotIDs.right_pad_id,
            lateralFriction=1.0)

        self.robot.arm.reset(force_reset=True)
        self._set_up_cameras()

        # put table at right spot
        table_ori = euler2quat([0, 0, np.pi / 2])

        # table_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_shelf.urdf')
        # with open(table_urdf_fname, 'r', encoding='utf-8') as f:
        #     self.table_urdf = f.read()

        # Write urdf from demo to temp file.
        tmp_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_rack_tmp.urdf')
        open(tmp_urdf_fname, 'w').write(self.table_urdf)
        self.table_id = self.robot.pb_client.load_urdf(tmp_urdf_fname,
            SimConstants.TABLE_POS,
            table_ori,
            scaling=SimConstants.TABLE_SCALING)

    def run_trial(self, iteration: int = 0, obj_scale: float = -1,
                  any_pose: bool = True, thin_feature: bool = True,
                  grasp_viz: bool = False,
                  grasp_dist_thresh: float = 0.0025,
                  obj_shapenet_id: 'str | None' = None) -> TrialData:
        trial_data = TrialData()
        trial_data.aux_data = {
            'place_opt_idx': None,
        }

        # -- Get and orient object -- #
        if obj_shapenet_id is None:
            obj_shapenet_id = random.sample(self.test_object_ids, 1)[0]
            log_info('Generate random obj id.')
        else:
            log_info('Using predefined obj id.')
        trial_data.obj_shapenet_id = obj_shapenet_id

        # Write at start so id is recorded regardless of any later bugs
        with open(self.shapenet_id_list_fname, 'a') as f:
            f.write(f'{trial_data.obj_shapenet_id}\n')

        # -- Home Robot -- #
        self.robot.arm.go_home(ignore_physics=True)
        self.robot.arm.move_ee_xyz([0, 0, 0.2])

        # -- load object -- #
        obj_id, o_cid, pos, ori = self._insert_object(obj_shapenet_id,
            obj_scale, any_pose)

        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
        p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
        time.sleep(1.5)

        img_fname = osp.join(self.eval_grasp_imgs_dir,
            f'{str(iteration).zfill(3)}_00ori.png')
        self._take_image(img_fname)

        # -- Get object point cloud from cameras -- #
        target_obj_pcd_obs = self._get_pcd(obj_id)

        eval_iter_dir = osp.join(self.eval_save_dir, 'trial_%s' % str(iteration).zfill(3))
        util.safe_makedirs(eval_iter_dir)

        if target_obj_pcd_obs is None or target_obj_pcd_obs.shape[0] == 0:
            trial_data.trial_result = TrialResults.GET_PCD_FAILED
            self.robot.pb_client.remove_body(obj_id)
            return trial_data

        # -- Get place position -- #
        opt_viz_path = osp.join(eval_iter_dir, 'visualize')
        pose_mats, best_opt_idx = self.place_optimizer.optimize_transform_implicit(
            target_obj_pcd_obs, ee=False, viz_path=opt_viz_path)
        trial_data.aux_data['place_opt_idx'] = best_opt_idx
        relative_pose = util.transform_pose(
            util.pose_from_matrix(pose_mats[best_opt_idx]), util.list2pose_stamped(self.shelf_pose))

        # -- Try place teleport -- #
        obj_pose_world = p.getBasePositionAndOrientation(obj_id)
        obj_pose_world = util.list2pose_stamped(list(obj_pose_world[0]) + list(obj_pose_world[1]))
        obj_end_pose = util.transform_pose(obj_pose_world, relative_pose)
        obj_end_pose = util.pose_stamped2list(obj_end_pose)
        placement_link_id = 0

        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=False)
        safeCollisionFilterPair(obj_id, self.table_id, -1, placement_link_id, enableCollision=False)
        self.robot.pb_client.set_step_sim(True)
        safeRemoveConstraint(o_cid)
        self.robot.pb_client.reset_body(obj_id, obj_end_pose[:3], obj_end_pose[3:])

        # First image suspends object in air, second is when constraints are
        # removed.
        time.sleep(1.0)
        teleport_img_fname = osp.join(self.eval_grasp_imgs_dir,
            f'{str(iteration).zfill(3)}_teleport_place_1.png')
        self._take_image(teleport_img_fname)
        safeCollisionFilterPair(obj_id, self.table_id, -1, placement_link_id, enableCollision=True)
        self.robot.pb_client.set_step_sim(False)
        time.sleep(1.0)
        teleport_img_fname = osp.join(self.eval_grasp_imgs_dir,
            f'{str(iteration).zfill(3)}_teleport_place_2.png')
        self._take_image(teleport_img_fname)

        # -- Check teleport was successful -- #
        obj_surf_contacts = p.getContactPoints(obj_id, self.table_id, -1, placement_link_id)
        touching_surf = len(obj_surf_contacts) > 0
        place_success_teleport = touching_surf
        if place_success_teleport:
            trial_data.trial_result = TrialResults.SUCCESS
        else:
            trial_data.trial_result = TrialResults.BAD_OPT_POS

        self.robot.pb_client.remove_body(obj_id)
        return trial_data

    def run_experiment(self, rand_mesh_scale: bool = True, start_idx: int = 0):
        """
        Run experiment for {self.num_trials}
        """
        num_success = 0

        obj_shapenet_id_list = random.choices(self.test_object_ids, k=self.num_trials)

        if rand_mesh_scale:
            obj_scale_list = np.random.random(self.num_trials).tolist()
        else:
            obj_scale_list = -1 * np.ones(self.num_trials)
            obj_scale_list = obj_scale_list.tolist()

        for it in range(start_idx, self.num_trials):
            obj_shapenet_id = obj_shapenet_id_list[it]
            obj_scale = obj_scale_list[it]
            trial_data: TrialData = self.run_trial(iteration=it,
                obj_scale=obj_scale, any_pose=self.any_pose,
                obj_shapenet_id=obj_shapenet_id)

            trial_result = trial_data.trial_result
            obj_shapenet_id = trial_data.obj_shapenet_id
            best_opt_idx = trial_data.aux_data['place_opt_idx']

            if trial_result == TrialResults.SUCCESS:
                num_success += 1

            log_info(f'Experiment: {self.experiment_type}')
            log_info(f'Trial result: {trial_result}')
            log_info(f'Shapenet id: {obj_shapenet_id}')
            log_str = f'Successes: {num_success} | Trials {it + 1} | ' \
                + f'Success Rate: {num_success / (it + 1):0.3f}'
            log_info(log_str)

            with open(self.global_summary_fname, 'a') as f:
                f.write(f'Trial number: {it}\n')
                f.write(f'Trial result: {trial_result}\n')
                f.write(f'Place teleport Success Rate: {num_success / (it + 1): 0.3f}\n')
                f.write(f'Shapenet id: {obj_shapenet_id}\n')
                f.write(f'Best Grasp idx: {best_opt_idx}\n')
                f.write('\n')

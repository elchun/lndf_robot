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


class EvaluateGraspTeleport(EvaluateNetwork):
    def __init__(self, grasp_optimizer: OccNetOptimizer,
                 seed: int, shapenet_obj_dir: str, eval_save_dir: str,
                 demo_load_dir: str, obj_scale_low: float,
                 obj_scale_high: float, obj_scale_default: float,
                 pybullet_viz: bool = False,
                 test_obj_class: str = 'mug', num_trials: int = 200,
                 include_avoid_obj: bool = True, any_pose: bool = True):

        super().__init__(seed, shapenet_obj_dir, eval_save_dir,
            demo_load_dir, test_obj_class, pybullet_viz, num_trials,
            include_avoid_obj, any_pose)

        print(f'avoid obj: {include_avoid_obj}')
        self.grasp_optimizer = grasp_optimizer
        self.experiment_type = ExperimentTypes.GRASP
        self.table_urdf_fname = osp.join(path_util.get_ndf_descriptions(),
            'hanging/table/table.urdf')

        # Added to test larger objects
        # self.scale_low = 0.50
        # self.scale_high = 0.6
        # self.scale_default = 0.55
        self.scale_low = obj_scale_low
        self.scale_high = obj_scale_high
        self.scale_default = obj_scale_default

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
        demo_shapenet_ids = []

        # Iterate through all demos, extract relevant information and
        # prepare to pass into optimizer
        random.shuffle(grasp_demo_fnames)
        for grasp_demo_fn in grasp_demo_fnames[:self.n_demos]:
            print('Loading grasp demo from fname: %s' % grasp_demo_fn)
            grasp_data = np.load(grasp_demo_fn, allow_pickle=True)

            demo = DemoIO.process_grasp_data(grasp_data)

            self.grasp_optimizer.add_demo(demo)
            demo_shapenet_ids.append(demo.obj_shapenet_id)

        self.grasp_optimizer.process_demos()

        # -- Get table urdf -- #
        grasp_data = np.load(grasp_demo_fnames[0], allow_pickle=True)
        self.table_urdf = DemoIO.get_table_urdf(grasp_data)

        # -- Get test objects -- #
        self.test_object_ids = self._get_test_object_ids(demo_shapenet_ids)

    def configure_sim(self):
        """
        Run after demos are loaded
        """
        # set_log_level('info')
        set_log_level('debug')

        p.changeDynamics(self.robot.arm.robot_id, RobotIDs.left_pad_id,
            lateralFriction=1.0)
        p.changeDynamics(self.robot.arm.robot_id, RobotIDs.right_pad_id,
            lateralFriction=1.0)

        self.reset_sim()


        # # put table at right spot
        # table_ori = euler2quat([0, 0, np.pi / 2])

        # # Get raw table urdf
        # table_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table.urdf')
        # # table_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_rack.urdf')
        # with open(table_urdf_fname, 'r', encoding='utf-8') as f:
        #     self.table_urdf = f.read()

        # # this is the URDF that was used in the demos -- make sure we load an identical one
        # tmp_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_rack_tmp.urdf')
        # open(tmp_urdf_fname, 'w').write(self.table_urdf)
        # self.table_id = self.robot.pb_client.load_urdf(tmp_urdf_fname,
        #     SimConstants.TABLE_POS,
        #     table_ori,
        #     scaling=SimConstants.TABLE_SCALING)

    def run_trial(self, iteration: int = 0, obj_scale: float = -1,
                  any_pose: bool = True, thin_feature: bool = True,
                  grasp_viz: bool = False,
                  grasp_dist_thresh: float = 0.0025,
                  obj_shapenet_id: 'str | None' = None) -> TrialData:
        """
        Teleport object into robot grasp (instead of moving robot to grab object)
        Check if object was successfully grasped.  This checks that the grasp
        was in the right place without worrying about inverse kinematics.

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
            obj_shapenet_id (str | None, optional): Object id to use.  If none,
                will randomly select id.

        Returns:
            TrialData: Class for storing relevant info about the trial
        """
        trial_data = TrialData()
        trial_data.aux_data = {
            'grasp_opt_idx': None,
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
        self.reset_sim()
        self.robot.arm.go_home(ignore_physics=True)
        self.robot.arm.move_ee_xyz([0, 0, 0.2])

        # -- load object -- #
        obj_id, o_cid, pos, ori = self._insert_object(obj_shapenet_id,
            obj_scale, any_pose, no_gravity=True, friction=2.0)

        img_fname = osp.join(self.eval_grasp_imgs_dir,
            '%s_00ori.png' % str(iteration).zfill(3))
        # util.np2img(grasp_rgb.astype(np.uint8), grasp_img_fname)
        self._take_image(img_fname)

        # -- Camera for figure -- #
        img_fname = osp.join(self.eval_grasp_imgs_dir,
            '%s_handle_grasp_figure_img_start.png' % str(iteration).zfill(3))

        self._get_figure_img(img_fname)

        # safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
        p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
        time.sleep(1.5)

        # -- Get object point cloud from cameras -- #
        target_obj_pcd_obs = self._get_pcd(obj_id)

        eval_iter_dir = osp.join(self.eval_save_dir, 'trial_%s' % str(iteration).zfill(3))
        util.safe_makedirs(eval_iter_dir)

        if target_obj_pcd_obs is None or target_obj_pcd_obs.shape[0] == 0:
            trial_data.trial_result = TrialResults.GET_PCD_FAILED
            self.robot.pb_client.remove_body(obj_id)
            return trial_data

        # -- Get grasp position -- #
        log_debug('Getting grasp position.')
        opt_viz_path = osp.join(eval_iter_dir, 'visualize')
        grasp_ee_pose_mats, best_grasp_idx = self.grasp_optimizer.optimize_transform_implicit(
            target_obj_pcd_obs, ee=True, viz_path=opt_viz_path)
        grasp_ee_pose = util.pose_stamped2list(util.pose_from_matrix(
            grasp_ee_pose_mats[best_grasp_idx]))
        trial_data.aux_data['grasp_opt_idx'] = best_grasp_idx

        # # -- Post process grasp position -- #
        # try:
        #     # When there are no nearby grasp points, this throws an index
        #     # error.  The try catch allows us to run more trials after the error.
        #     new_grasp_pt = post_process_grasp_point(
        #         grasp_ee_pose,
        #         target_obj_pcd_obs,
        #         thin_feature=thin_feature,
        #         grasp_viz=grasp_viz,
        #         grasp_dist_thresh=grasp_dist_thresh)
        # except IndexError:
        #     trial_data.trial_result = TrialResults.POST_PROCESS_FAILED
        #     self.robot.pb_client.remove_body(obj_id)
        #     return trial_data

        # grasp_ee_pose[:3] = new_grasp_pt

        # -- Figure out where to teleport object -- #
        ee_pose_world = np.concatenate(self.robot.arm.get_ee_pose()[:2]).tolist()
        # print('ee pose world: ', ee_pose_world)

        # obj_pose_ee = util.convert_reference_frame( pose_source=obj_pose_world,
        #     pose_frame_target=ee_pose_world,
        #     pose_frame_source=util.unit_pose()
        # )

        ee_relative_pose = util.get_transform(
            util.list2pose_stamped(ee_pose_world),
            util.list2pose_stamped(grasp_ee_pose)
        )

        obj_pose_world = p.getBasePositionAndOrientation(obj_id)
        obj_pose_world = util.list2pose_stamped(list(obj_pose_world[0]) + list(obj_pose_world[1]))
        obj_end_pose = util.transform_pose(obj_pose_world, ee_relative_pose)
        obj_end_pose = util.pose_stamped2list(obj_end_pose)

        # -- Teleport object into ee tool -- #
        self.robot.pb_client.set_step_sim(True)
        self.robot.arm.eetool.open()
        self._step_n_steps(10)
        safeRemoveConstraint(o_cid)
        self.robot.pb_client.reset_body(obj_id, obj_end_pose[:3], obj_end_pose[3:])
        o_cid = constraint_obj_world(obj_id, obj_end_pose[:3], obj_end_pose[3:])

        img_fname = osp.join(self.eval_grasp_imgs_dir,
            '%s_01pose.png' % str(iteration).zfill(3))
        # util.np2img(grasp_rgb.astype(np.uint8), grasp_img_fname)
        self._take_image(img_fname)

        # Turn on collisions between object and robot
        for i in range(p.getNumJoints(self.robot.arm.robot_id)):
            safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id,
                bodyUniqueIdB=obj_id,
                linkIndexA=i,
                linkIndexB=-1,
                enableCollision=True,
                physicsClientId=self.robot.pb_client.get_client_id())

        n_grasp_trials = 5
        obj_pos_before_grasp, obj_ori_before_grasp = p.getBasePositionAndOrientation(obj_id)

        for i in range(n_grasp_trials):
            soft_grasp_close(self.robot, RobotIDs.finger_joint_id, force=50)
            self._step_n_steps(100)

            img_fname = osp.join(self.eval_grasp_imgs_dir,
                f'{str(iteration).zfill(3)}_02{i}grasp.png')
            self._take_image(img_fname)

            safeRemoveConstraint(o_cid)
            self._step_n_steps(240)

            # -- Determine if grasp was successful -- #
            contact_grasp_success = object_is_still_grasped(self.robot,
                obj_id, RobotIDs.right_pad_id, RobotIDs.left_pad_id)

            if contact_grasp_success:
                break
            elif i != n_grasp_trials - 1:
                p.resetBasePositionAndOrientation(obj_id, obj_pos_before_grasp, obj_ori_before_grasp)
                o_cid = constraint_obj_world(obj_id, obj_pos_before_grasp, obj_ori_before_grasp)
                self.robot.arm.eetool.open()
                self._step_n_steps(240)

        # If the ee was intersecting the mug, original_grasp_success
        # would be true after the table disappears.  However, an
        # intersection is generally a false grasp When the ee is
        # opened again, a good grasp should fall down while an
        # intersecting grasp would stay in contact.

        # -- Take image of grasp at clearance height -- #
        # grasp_rgb = self.robot.cam.get_images(get_rgb=True)[0]
        # util.np2img(grasp_rgb.astype(np.uint8), grasp_img_fname)
        img_fname = osp.join(self.eval_grasp_imgs_dir,
            '%s_03clearance.png' % str(iteration).zfill(3))
        self._take_image(img_fname)

        img_fname = osp.join(self.eval_grasp_imgs_dir,
            '%s_handle_grasp_figure_img_end.png' % str(iteration).zfill(3))
        self._get_figure_img(img_fname)

        # ee_intersecting_mug = object_is_intersecting(obj_id, self.robot.arm.robot_id, -1, -1)

        self.robot.arm.eetool.open()
        self._step_n_steps(240)
        # Sometimes object collides with gripper body and gets stuck there...
        soft_grasp_close(self.robot, RobotIDs.finger_joint_id, force=50)
        self._step_n_steps(240)
        ee_intersecting_mug = object_is_still_grasped(
            self.robot, obj_id, RobotIDs.right_pad_id,
            RobotIDs.left_pad_id)

        obj_table_pts = p.getClosestPoints(bodyA=obj_id, bodyB=self.table_id, distance=0.002,
                                            linkIndexA=-1, linkIndexB=-1)
        print('DEBUG CLOSEST PTS', ee_intersecting_mug, len(obj_table_pts))

        img_fname = osp.join(self.eval_grasp_imgs_dir,
            '%s_04release.png' % str(iteration).zfill(3))
        self._take_image(img_fname)

        grasp_success = contact_grasp_success and not ee_intersecting_mug

        if ee_intersecting_mug:
            print('Intersecting grasp detected')
            trial_data.trial_result = TrialResults.INTERSECTING_EE
        else:
            if not grasp_success:
                trial_data.trial_result = TrialResults.BAD_OPT_POS

        log_info(f'Grasp success: {grasp_success}')

        if grasp_success:
            trial_data.trial_result = TrialResults.SUCCESS

        self.robot.pb_client.remove_body(obj_id)
        self._step_n_steps(1)
        self.robot.pb_client.set_step_sim(False)
        return trial_data

    def run_experiment(self, rand_mesh_scale: bool = True, start_idx: bool = 0):
        """
        Run experiment for {self.num_trials}
        """
        num_success = 0

        start_time = time.time()

        obj_shapenet_id_list = random.choices(self.test_object_ids, k=self.num_trials)

        if self.test_obj_class == 'bottle':
            thin_feature = False
        else:
            thin_feature = True

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
                obj_shapenet_id=obj_shapenet_id, thin_feature=thin_feature)

            trial_result = trial_data.trial_result
            obj_shapenet_id = trial_data.obj_shapenet_id
            best_opt_idx = trial_data.aux_data['grasp_opt_idx']

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
                f.write(f'Grasp Success Rate: {num_success / (it + 1): 0.3f}\n')
                f.write(f'Shapenet id: {obj_shapenet_id}\n')
                f.write(f'Best Grasp idx: {best_opt_idx}\n')
                f.write(f'Time elapsed: {time.time() - start_time}\n')
                f.write('\n')


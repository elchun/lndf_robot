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


class EvaluateShelfPlaceGrasp(EvaluateNetwork):
    def __init__(self, grasp_optimizer: OccNetOptimizer,
                 place_optimizer: OccNetOptimizer, seed: int,
                 shapenet_obj_dir: str, eval_save_dir: str,
                 demo_load_dir: str, pybullet_viz: bool = False,
                 test_obj_class: str = 'mug', num_trials: int = 200,
                 include_avoid_obj: bool = True, any_pose: bool = True):

        super().__init__(seed, shapenet_obj_dir, eval_save_dir,
            demo_load_dir, test_obj_class, pybullet_viz, num_trials,
            include_avoid_obj, any_pose)

        self.grasp_optimizer = grasp_optimizer
        self.place_optimizer = place_optimizer
        self.experiment_type = ExperimentTypes.SHELF_PLACE_GRASP
        # self.obj_sample_x_low_high = SimConstants.OBJ_SAMPLE_X_LOW_HIGH
        self.obj_sample_x_low_high = [0.4, 0.45]
        # self.obj_sample_y_low_high = [-0.1, 0.1]
        self.obj_sample_y_low_high = [-0.3, -0.1]

        # # NEW
        # self.scale_low = 0.35
        # self.scale_high = 0.5
        # self.scale_default = 0.45

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
        place_demo_fnames = [osp.join(self.demo_load_dir, fn) for fn in
            demo_fnames if 'place_demo' in fn]

        demo_shapenet_ids = set()
        random.shuffle(grasp_demo_fnames)
        for grasp_demo_fn in grasp_demo_fnames[:self.n_demos]:
            print('Loading grasp demo from fname: %s' % grasp_demo_fn)
            grasp_data = np.load(grasp_demo_fn, allow_pickle=True)

            demo = DemoIO.process_grasp_data(grasp_data)

            self.grasp_optimizer.add_demo(demo)
            demo_shapenet_ids.add(demo.obj_shapenet_id)

        random.shuffle(place_demo_fnames)
        for place_demo_fn in place_demo_fnames[:self.n_demos]:
            print('Loading place demo from fname: %s' % place_demo_fn)
            place_data = np.load(place_demo_fn, allow_pickle=True)

            demo = DemoIO.process_shelf_place_data(place_data)

            self.place_optimizer.add_demo(demo)
            demo_shapenet_ids.add(demo.obj_shapenet_id)

        self.grasp_optimizer.process_demos()
        self.place_optimizer.process_demos()

        # -- Get table urdf -- #
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
        # p.changeDynamics(self.robot.arm.robot_id, RobotIDs.left_pad_id,
        #     lateralFriction=1.0)
        # p.changeDynamics(self.robot.arm.robot_id, RobotIDs.right_pad_id,
        #     lateralFriction=1.0)

        p.changeDynamics(self.robot.arm.robot_id, RobotIDs.left_pad_id,
            lateralFriction=3.0)
        p.changeDynamics(self.robot.arm.robot_id, RobotIDs.right_pad_id,
            lateralFriction=3.0)

        self.robot.arm.reset(force_reset=True)

        self._set_up_cameras()

        # put table at right spot
        table_ori = euler2quat([0, 0, np.pi / 2])

        # Get raw table urdf
        # table_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_shelf.urdf')
        # with open(table_urdf_fname, 'r', encoding='utf-8') as f:
        #     self.table_urdf = f.read()

        # this is the URDF that was used in the demos -- make sure we load an identical one
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
        """
        Run trial where we try to grab object.

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
            'grasp_success': False,
            'place_success': False,
            'grasp_opt_idx': None,
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

        # -- Post process grasp position -- #
        try:
            # When there are no nearby grasp points, this throws an index
            # error.  The try catch allows us to run more trials after the error.
            new_grasp_pt = post_process_grasp_point(
                grasp_ee_pose,
                target_obj_pcd_obs,
                thin_feature=thin_feature,
                grasp_viz=grasp_viz,
                grasp_dist_thresh=grasp_dist_thresh)
        except IndexError:
            trial_data.trial_result = TrialResults.POST_PROCESS_FAILED
            self.robot.pb_client.remove_body(obj_id)
            return trial_data

        grasp_ee_pose[:3] = new_grasp_pt

        # -- Create pose which offsets gripper from object -- #
        pregrasp_offset_tf = get_ee_offset(ee_pose=grasp_ee_pose)
        pre_grasp_ee_pose = util.pose_stamped2list(
            util.transform_pose(
                pose_source=util.list2pose_stamped(grasp_ee_pose),
                pose_transform=util.list2pose_stamped(pregrasp_offset_tf)))

        post_grasp_offset_tf = util.list2pose_stamped(SimConstants.SHELF_GRASP_CLEARANCE_OFFSET)
        post_grasp_pos = util.pose_stamped2list(
            util.transform_pose(util.list2pose_stamped(grasp_ee_pose), post_grasp_offset_tf)
        )

        # -- Get place position -- #
        opt_viz_path = osp.join(eval_iter_dir, 'visualize')
        rack_pose_mats, best_place_idx = self.place_optimizer.optimize_transform_implicit(
            target_obj_pcd_obs, ee=False, viz_path=opt_viz_path)
        trial_data.aux_data['place_opt_idx'] = best_place_idx
        rack_relative_pose = util.transform_pose(
            util.pose_from_matrix(rack_pose_mats[best_place_idx]), util.list2pose_stamped(self.shelf_pose))

        place_ee_pose = util.transform_pose(util.list2pose_stamped(grasp_ee_pose),
            pose_transform=rack_relative_pose)

        # preplace_offset_far_tf = util.list2pose_stamped(SimConstants.PREPLACE_OFFSET_FAR_TF)
        # preplace_offset_far_tf = util.list2pose_stamped(SimConstants.PREPLACE_HORIZONTAL_OFFSET_TF)
        # preplace_offset_close_tf = util.list2pose_stamped(SimConstants.PREPLACE_OFFSET_CLOSE_TF)
        preplace_offset_tf = util.list2pose_stamped(SimConstants.SHELF_PREPLACE_OFFSET)
        preplace_pose = util.transform_pose(place_ee_pose, preplace_offset_tf)

        place_ee_pose = util.pose_stamped2list(place_ee_pose)
        preplace_pose = util.pose_stamped2list(preplace_pose)

        # -- Get ik -- #
        log_debug('Getting ik.')
        pre_grasp_jnt_pos = grasp_jnt_pos = None

        ik_status = []
        pre_grasp_jnt_pos, ik_res = self._compute_ik_cascade(pre_grasp_ee_pose)
        ik_status.append(ik_res)
        grasp_jnt_pos, ik_res = self._compute_ik_cascade(grasp_ee_pose)
        ik_status.append(ik_res)
        post_grasp_pos, ik_res = self._compute_ik_cascade(post_grasp_pos)
        ik_status.append(ik_res)
        place_jnt_pose, ik_res = self._compute_ik_cascade(place_ee_pose)
        ik_status.append(ik_res)
        preplace_jnt_pose, ik_res = self._compute_ik_cascade(preplace_pose)
        ik_status.append(ik_res)

        for ik_res in ik_status:
            if ik_res is not None:
                trial_data.trial_result = ik_res
                self.robot.pb_client.remove_body(obj_id)
                return trial_data

        # -- Prep for grasp -- #
        log_debug('Attempting grasp.')

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

        home_jnt_pos = self.robot.arm.get_jpos()
        self.robot.arm.eetool.open()

        # -- Get grasp image -- #
        self.robot.pb_client.set_step_sim(True)
        self.robot.arm.set_jpos(grasp_jnt_pos, ignore_physics=True)
        self.robot.arm.eetool.close(ignore_physics=True)
        time.sleep(0.2)
        # grasp_rgb = self.robot.cam.get_images(get_rgb=True)[0]
        grasp_img_fname = osp.join(self.eval_grasp_imgs_dir,
            '%s_01pose.png' % str(iteration).zfill(3))
        # util.np2img(grasp_rgb.astype(np.uint8), grasp_img_fname)
        self._take_image(grasp_img_fname)
        self.robot.arm.eetool.open(ignore_physics=True)
        self.robot.arm.go_home(ignore_physics=True)
        time.sleep(0.5)
        self.robot.pb_client.set_step_sim(False)

        # -- Plan grasp -- #
        # Get to pre grasp location
        plan1 = self.ik_helper.plan_joint_motion(home_jnt_pos, pre_grasp_jnt_pos)

        # Get to grasp location
        plan2 = self.ik_helper.plan_joint_motion(pre_grasp_jnt_pos, grasp_jnt_pos)

        # Move upwards to check if grasp was valid
        #TODO
        # plan3 = self.ik_helper.plan_joint_motion(grasp_jnt_pos, home_jnt_pos)
        plan3 = self.ik_helper.plan_joint_motion(grasp_jnt_pos, post_grasp_pos)

        # Return to home location (in preparation to place)
        # plan4 = self.ik_helper.plan_joint_motion(post_grasp_pos, home_jnt_pos)

        if None in [plan1, plan2, plan3]:
            trial_data.trial_result = TrialResults.JOINT_PLAN_FAILED
            self.robot.pb_client.remove_body(obj_id)
            return trial_data

        # -- Move for grasp -- #
        self.robot.arm.eetool.open()
        time.sleep(0.5)
        # Go to pre grasp location (linearly away from grasp area)
        for jnt in plan1:
            self.robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.025)
        self.robot.arm.set_jpos(plan1[-1], wait=False)

        # # turn ON collisions between robot and object
        # for i in range(p.getNumJoints(self.robot.arm.robot_id)):
        #     safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id,
        #         bodyUniqueIdB=obj_id,
        #         linkIndexA=i,
        #         linkIndexB=-1,
        #         enableCollision=True,
        #         physicsClientId=self.robot.pb_client.get_client_id())

        # Go to grasp location
        for jnt in plan2:
            self.robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.04)
        self.robot.arm.set_jpos(plan2[-1], wait=False)

        # turn ON collisions between robot and object
        for i in range(p.getNumJoints(self.robot.arm.robot_id)):
            safeCollisionFilterPair(bodyUniqueIdA=self.robot.arm.robot_id,
                bodyUniqueIdB=obj_id,
                linkIndexA=i,
                linkIndexB=-1,
                enableCollision=True,
                physicsClientId=self.robot.pb_client.get_client_id())

        time.sleep(0.8)

        obj_pos_before_grasp = p.getBasePositionAndOrientation(obj_id)[0]
        jnt_pos_before_grasp = self.robot.arm.get_jpos()

        # Testing with different close methods.
        # self.robot.arm.eetool.close()
        soft_grasp_close(self.robot, RobotIDs.finger_joint_id, force=40)
        time.sleep(0.8)

        grasp_img_fname = osp.join(self.eval_grasp_imgs_dir,
            f'{str(iteration).zfill(3)}_02grasp.png')
        self._take_image(grasp_img_fname)

        safeRemoveConstraint(o_cid)
        safeCollisionFilterPair(obj_id, self.table_id, -1, -1,
            enableCollision=False)
        time.sleep(0.8)

        # Move to clearance location
        for jnt in plan3:
            self.robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.025)
        self.robot.arm.set_jpos(plan3[-1], wait=False)
        time.sleep(1)

        grasp_success = object_is_still_grasped(self.robot,
            obj_id, RobotIDs.right_pad_id, RobotIDs.left_pad_id)
        time.sleep(0.5)

        # If the ee was intersecting the mug, original_grasp_success
        # would be true after the table disappears.  However, an
        # intersection is generally a false grasp When the ee is
        # opened again, a good grasp should fall down while an
        # intersecting grasp would stay in contact.

        # ^ This is not currently implemented cuz I don't wanna debug the mug moving
        # when its reset to that position.  If this error occurs, the place will
        # be false so its fine...

        # -- Take image of grasp at clearance height -- #
        grasp_img_fname = osp.join(self.eval_grasp_imgs_dir,
            '%s_03clearance.png' % str(iteration).zfill(3))
        self._take_image(grasp_img_fname)

        if grasp_success:
            trial_data.trial_result = TrialResults.GRASP_SUCCESS
            trial_data.aux_data['grasp_success'] = True

            # Get firm grasp on object, then move back to clearance
            safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
            self.robot.arm.eetool.open()
            time.sleep(0.5)
            p.resetBasePositionAndOrientation(obj_id, obj_pos_before_grasp, ori)
            self.robot.arm.set_jpos(jnt_pos_before_grasp, ignore_physics=True)
            soft_grasp_close(self.robot, RobotIDs.finger_joint_id, force=40)
            grasp_cid = constraint_grasp_close(self.robot, obj_id)

            for jnt in plan3:
                self.robot.arm.set_jpos(jnt, wait=False)
                time.sleep(0.025)
            self.robot.arm.set_jpos(plan3[-1], wait=False)
            time.sleep(1)

            safeCollisionFilterPair(obj_id, self.table_id, -1, -1,
                enableCollision=False)

        else:
            trial_data.trial_result = TrialResults.BAD_OPT_POS
            self.robot.pb_client.remove_body(obj_id)
            return trial_data

        # for jnt in plan4:
        #     self.robot.arm.set_jpos(jnt, wait=False)
        #     time.sleep(0.025)
        # self.robot.arm.set_jpos(plan3[-1], wait=False)
        time.sleep(1)

        # -- Set up for place -- #
        placement_link_id = 0

        # turn OFF collisions between object / table and object / rack, and move to pre-place pose
        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=False)
        safeCollisionFilterPair(obj_id, self.table_id, -1, placement_link_id, enableCollision=False)
        time.sleep(1.0)

        img_fname = osp.join(self.eval_grasp_imgs_dir,
            '%s_04clearance_place.png' % str(iteration).zfill(3))
        self._take_image(img_fname)

        # Go to far approach position
        #TODO
        plan1 = self.ik_helper.plan_joint_motion(post_grasp_pos, home_jnt_pos)

        plan2 = self.ik_helper.plan_joint_motion(home_jnt_pos, preplace_jnt_pose)
        # plan2 = self.ik_helper.plan_joint_motion(post_grasp_pos, preplace_jnt_pose)

        # Go to close approach position
        plan3 = self.ik_helper.plan_joint_motion(preplace_jnt_pose, place_jnt_pose)

        # Return to home position
        # plan4 = self.ik_helper.plan_joint_motion(place_jnt_pose, place_far_jnt_pose)
        # plan5 = self.ik_helper.plan_joint_motion(place_far_jnt_pose, home_jnt_pos)

        if None in [plan1, plan2, plan3]:
            trial_data.trial_result = TrialResults.JOINT_PLAN_FAILED
            self.robot.pb_client.remove_body(obj_id)
            return trial_data

        for jnt in plan1:
            self.robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.04)
        self.robot.arm.set_jpos(plan1[-1], wait=False)

        for jnt in plan2:
            self.robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.04)
        self.robot.arm.set_jpos(plan2[-1], wait=False)

        for jnt in plan3:
            self.robot.arm.set_jpos(jnt, wait=False)
            time.sleep(0.04)
        self.robot.arm.set_jpos(plan3[-1], wait=False)

        img_fname = osp.join(self.eval_grasp_imgs_dir,
            f'{str(iteration).zfill(3)}_05pre_place.png')
        self._take_image(img_fname)

        # turn ON collisions between object and rack, and open fingers
        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=True)
        safeCollisionFilterPair(obj_id, self.table_id, -1, placement_link_id, enableCollision=True)
        time.sleep(0.5)

        img_fname = osp.join(self.eval_grasp_imgs_dir,
            f'{str(iteration).zfill(3)}_06place.png')
        self._take_image(img_fname)

        # p.changeDynamics(self.robot.arm.robot_id, RobotIDs.left_pad_id,
        #     lateralFriction=4.0)
        # p.changeDynamics(self.robot.arm.robot_id, RobotIDs.right_pad_id,
        #     lateralFriction=4.0)

        p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
        self.robot.arm.eetool.open()
        constraint_grasp_open(grasp_cid)

        time.sleep(0.3)
        img_fname = osp.join(self.eval_grasp_imgs_dir,
            f'{str(iteration).zfill(3)}_07place_release.png')
        self._take_image(img_fname)

        self.robot.arm.go_home(ignore_physics=True)
        time.sleep(0.3)

        safeCollisionFilterPair(obj_id, self.table_id, -1, -1, enableCollision=False)
        img_fname = osp.join(self.eval_grasp_imgs_dir,
            f'{str(iteration).zfill(3)}_08place_release_home.png')
        self._take_image(img_fname)

        # -- Check place was successful -- #

        placement_link_id = 0
        obj_surf_contacts = p.getContactPoints(obj_id, self.table_id, -1, placement_link_id)
        touching_surf = len(obj_surf_contacts) > 0
        place_success = touching_surf
        if place_success:
            trial_data.trial_result = TrialResults.SUCCESS
            trial_data.aux_data['place_success'] = True

        self.robot.pb_client.remove_body(obj_id)
        return trial_data

    def run_experiment(self, rand_mesh_scale: bool = True, start_idx: int = 0):
        """
        Run experiment for {self.num_trials}
        """
        num_success = 0

        obj_shapenet_id_list = random.choices(self.test_object_ids, k=self.num_trials)

        # # TODO
        # obj_shapenet_id_list = ['2c1df84ec01cea4e525b133235812833-h']

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
            best_grasp_idx = trial_data.aux_data['grasp_opt_idx']
            best_place_idx = trial_data.aux_data['place_opt_idx']
            grasp_success = trial_data.aux_data['grasp_success']
            place_success = trial_data.aux_data['place_success']

            if trial_result == TrialResults.SUCCESS:
                num_success += 1

            log_info(f'Experiment: {self.experiment_type}')
            log_info(f'Trial result: {trial_result}')
            log_info(f'Shapenet id: {obj_shapenet_id}')
            log_info(f'Grasp Success: {grasp_success} | Place Success: {place_success}')
            log_str = f'Successes: {num_success} | Trials {it + 1} | ' \
                + f'Success Rate: {num_success / (it + 1):0.3f}'
            log_info(log_str)

            with open(self.global_summary_fname, 'a') as f:
                f.write(f'Trial number: {it}\n')
                f.write(f'Trial result: {trial_result}\n')
                f.write(f'Grasp Success Rate: {num_success / (it + 1): 0.3f}\n')
                f.write(f'Shapenet id: {obj_shapenet_id}\n')
                f.write(f'Best Grasp idx: {best_grasp_idx}\n')
                f.write(f'Best Place idx: {best_place_idx}\n')
                f.write('\n')

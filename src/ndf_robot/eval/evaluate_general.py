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
from ndf_robot.opt.optimizer_geom import GeomOptimizer
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

from ndf_robot.eval.experiments.evaluate_network import EvaluateNetwork

from ndf_robot.eval.experiments.evaluate_grasp import EvaluateGrasp
from ndf_robot.eval.experiments.evaluate_grasp_teleport import EvaluateGraspTeleport

from ndf_robot.eval.experiments.evaluate_rack_place_teleport import EvaluateRackPlaceTeleport
from ndf_robot.eval.experiments.evaluate_rack_place_grasp import EvaluateRackPlaceGrasp
from ndf_robot.eval.experiments.evaluate_rack_place_grasp_ideal import EvaluateRackPlaceGraspIdeal
from ndf_robot.eval.experiments.evaluate_rack_place_grasp_ideal_step import EvaluateRackPlaceGraspIdealStep

from ndf_robot.eval.experiments.evaluate_shelf_place_teleport import EvaluateShelfPlaceTeleport
from ndf_robot.eval.experiments.evaluate_shelf_place_grasp import EvaluateShelfPlaceGrasp
from ndf_robot.eval.experiments.evaluate_shelf_place_grasp_ideal import EvaluateShelfPlaceGraspIdeal


class EvaluateNetworkSetup():
    """
    Set up experiment from config file
    """
    def __init__(self):
        self.config_dir = osp.join(path_util.get_ndf_eval(), 'eval_configs')
        self.config_dict = None
        self.seed = None

    def set_up_network(self, fname: str) -> EvaluateNetwork:
        """
        Create an instance of EvaluateNetwork by loading a config file.

        Args:
            fname (str): Filename of config file to use.  Must be a yaml file.

        Raises:
            ValueError: config file must have correct evaluator type.

        Returns:
            EvaluateNetwork: Network Evaluator with parameters set by config file.
        """
        config_path = osp.join(self.config_dir, fname)
        with open(config_path, "r") as stream:
            try:
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.config_dict = config_dict
        setup_dict = self.config_dict['setup_args']
        self.seed = setup_dict['seed']

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        print(config_dict)

        evaluator_type: str = setup_dict['evaluator_type']

        if evaluator_type == 'GRASP':
            return self._grasp_setup()
        elif evaluator_type == 'RACK_PLACE_TELEPORT':
            return self._rack_place_teleport_setup()
        elif evaluator_type == 'SHELF_PLACE_TELEPORT':
            return self._shelf_place_teleport_setup()
        elif evaluator_type == 'RACK_PLACE_GRASP':
            return self._rack_place_grasp_setup()
        elif evaluator_type == 'SHELF_PLACE_GRASP':
            return self._shelf_place_grasp_setup()
        elif evaluator_type == 'RACK_PLACE_GRASP_IDEAL':
            return self._rack_place_grasp_ideal_setup()
        elif evaluator_type == 'SHELF_PLACE_GRASP_IDEAL':
            return self._shelf_place_grasp_ideal_setup()
        else:
            raise ValueError('Invalid evaluator type.')

    def _grasp_setup(self) -> EvaluateNetwork:
        """
        Set up grasp experiment.  Must have 'grasp_optimizer' in config file.

        Returns:
            EvaluateNetwork
        """
        setup_config = self.config_dict['setup_args']
        obj_class = self.config_dict['evaluator']['test_obj_class']

        model = self._create_model(self.config_dict['model'])
        gripper_query_pts = self._create_query_pts(self.config_dict['gripper_query_pts'])
        eval_save_dir = self._create_eval_dir(setup_config)

        evaluator_config = self.config_dict['evaluator']
        shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(),
            obj_class + '_centered_obj_normalized')

        demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos',
            setup_config['demo_exp'])

        grasp_optimizer = self._create_optimizer(self.config_dict['grasp_optimizer'],
            model, gripper_query_pts, eval_save_dir)

        # experiment = EvaluateGrasp(grasp_optimizer=grasp_optimizer,
        #     seed=self.seed, shapenet_obj_dir=shapenet_obj_dir,
        #     eval_save_dir=eval_save_dir, demo_load_dir=demo_load_dir,
        #     **evaluator_config)

        # This experiment teleports the object into the gripper handle
        # This allows us to test the pose without worrying about kinematics
        experiment = EvaluateGraspTeleport(grasp_optimizer=grasp_optimizer,
            seed=self.seed, shapenet_obj_dir=shapenet_obj_dir,
            eval_save_dir=eval_save_dir, demo_load_dir=demo_load_dir,
            **evaluator_config)

        return experiment

    def _rack_place_teleport_setup(self) -> EvaluateNetwork:
        """
        Not commonly used --> Legacy for teleporting objects onto the rack

        Returns:
            EvaluateNetwork: experiment object
        """
        setup_config = self.config_dict['setup_args']
        obj_class = self.config_dict['evaluator']['test_obj_class']

        model = self._create_model(self.config_dict['model'])
        rack_query_pts = self._create_query_pts(self.config_dict['rack_query_pts'])
        eval_save_dir = self._create_eval_dir(setup_config)

        evaluator_config = self.config_dict['evaluator']
        shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(),
            obj_class + '_centered_obj_normalized')

        demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos',
            setup_config['demo_exp'])

        place_optimizer = self._create_optimizer(self.config_dict['place_optimizer'],
            model, rack_query_pts, eval_save_dir)

        experiment = EvaluateRackPlaceTeleport(place_optimizer=place_optimizer,
            seed=self.seed, shapenet_obj_dir=shapenet_obj_dir,
            eval_save_dir=eval_save_dir, demo_load_dir=demo_load_dir,
            **evaluator_config)

        return experiment

    def _shelf_place_teleport_setup(self) -> EvaluateNetwork:
        """
        Not commonly used --> Legacy for teleporting objects onto shelf

        Returns:
            EvaluateNetwork: experiment object.
        """
        setup_config = self.config_dict['setup_args']
        obj_class = self.config_dict['evaluator']['test_obj_class']

        model = self._create_model(self.config_dict['model'])
        shelf_query_pts = self._create_query_pts(self.config_dict['shelf_query_pts'])
        eval_save_dir = self._create_eval_dir(setup_config)

        evaluator_config = self.config_dict['evaluator']
        shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(),
            obj_class + '_centered_obj_normalized')

        demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos',
            setup_config['demo_exp'])

        place_optimizer = self._create_optimizer(self.config_dict['place_optimizer'],
            model, shelf_query_pts, eval_save_dir)

        experiment = EvaluateShelfPlaceTeleport(place_optimizer=place_optimizer,
            seed=self.seed, shapenet_obj_dir=shapenet_obj_dir,
            eval_save_dir=eval_save_dir, demo_load_dir=demo_load_dir,
            **evaluator_config)

        return experiment

    def _rack_place_grasp_setup(self) -> EvaluateNetwork:
        """
        Raw rack place experiment.  Uses arm to move object to rack.

        Returns:
            EvaluateNetwork: experiment object.
        """
        setup_config = self.config_dict['setup_args']
        obj_class = self.config_dict['evaluator']['test_obj_class']

        model = self._create_model(self.config_dict['model'])
        gripper_query_pts = self._create_query_pts(self.config_dict['gripper_query_pts'])
        rack_query_pts = self._create_query_pts(self.config_dict['rack_query_pts'])
        eval_save_dir = self._create_eval_dir(setup_config)

        evaluator_config = self.config_dict['evaluator']
        shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(),
            obj_class + '_centered_obj_normalized')

        demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos',
            setup_config['demo_exp'])

        grasp_optimizer = self._create_optimizer(self.config_dict['grasp_optimizer'],
            model, gripper_query_pts, eval_save_dir)
        place_optimizer = self._create_optimizer(self.config_dict['place_optimizer'],
            model, rack_query_pts, eval_save_dir)

        experiment = EvaluateRackPlaceGrasp(grasp_optimizer=grasp_optimizer,
            place_optimizer=place_optimizer,
            seed=self.seed, shapenet_obj_dir=shapenet_obj_dir,
            eval_save_dir=eval_save_dir, demo_load_dir=demo_load_dir,
            **evaluator_config)

        return experiment

    def _rack_place_grasp_ideal_setup(self) -> EvaluateNetwork:
        """
        Idealized rack place experiment.  Attempts to grab object with gripper,
        but uses teleport to place object on rack

        Returns:
            EvaluateNetwork: experiment object.
        """
        setup_config = self.config_dict['setup_args']
        obj_class = self.config_dict['evaluator']['test_obj_class']

        model = self._create_model(self.config_dict['model'])
        gripper_query_pts = self._create_query_pts(self.config_dict['gripper_query_pts'])
        rack_query_pts = self._create_query_pts(self.config_dict['rack_query_pts'])
        eval_save_dir = self._create_eval_dir(setup_config)

        evaluator_config = self.config_dict['evaluator']
        shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(),
            obj_class + '_centered_obj_normalized')

        demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos',
            setup_config['demo_exp'])

        grasp_optimizer = self._create_optimizer(self.config_dict['grasp_optimizer'],
            model, gripper_query_pts, eval_save_dir)
        place_optimizer = self._create_optimizer(self.config_dict['place_optimizer'],
            model, rack_query_pts, eval_save_dir)

        experiment = EvaluateRackPlaceGraspIdealStep(grasp_optimizer=grasp_optimizer,
            place_optimizer=place_optimizer,
            seed=self.seed, shapenet_obj_dir=shapenet_obj_dir,
            eval_save_dir=eval_save_dir, demo_load_dir=demo_load_dir,
            **evaluator_config)

        return experiment

    def _shelf_place_grasp_setup(self) -> EvaluateNetwork:
        """
        Raw shelf place experiment.  Uses arm to move object to shelf.

        Returns:
            EvaluateNetwork: experiment object.
        """
        setup_config = self.config_dict['setup_args']
        obj_class = self.config_dict['evaluator']['test_obj_class']

        model = self._create_model(self.config_dict['model'])
        gripper_query_pts = self._create_query_pts(self.config_dict['gripper_query_pts'])
        shelf_query_pts = self._create_query_pts(self.config_dict['shelf_query_pts'])
        eval_save_dir = self._create_eval_dir(setup_config)

        evaluator_config = self.config_dict['evaluator']
        shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(),
            obj_class + '_centered_obj_normalized')

        demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos',
            setup_config['demo_exp'])

        grasp_optimizer = self._create_optimizer(self.config_dict['grasp_optimizer'],
            model, gripper_query_pts, eval_save_dir)
        place_optimizer = self._create_optimizer(self.config_dict['place_optimizer'],
            model, shelf_query_pts, eval_save_dir)

        experiment = EvaluateShelfPlaceGrasp(grasp_optimizer=grasp_optimizer,
            place_optimizer=place_optimizer,
            seed=self.seed, shapenet_obj_dir=shapenet_obj_dir,
            eval_save_dir=eval_save_dir, demo_load_dir=demo_load_dir,
            **evaluator_config)

        return experiment

    def _shelf_place_grasp_ideal_setup(self) -> EvaluateNetwork:
        """
        Idealized shelf place experiment.  Attempts to grab object with gripper,
        but uses teleport to place object on shelf

        Returns:
            EvaluateNetwork: experiment object.
        """
        setup_config = self.config_dict['setup_args']
        obj_class = self.config_dict['evaluator']['test_obj_class']

        model = self._create_model(self.config_dict['model'])
        gripper_query_pts = self._create_query_pts(self.config_dict['gripper_query_pts'])
        shelf_query_pts = self._create_query_pts(self.config_dict['shelf_query_pts'])
        eval_save_dir = self._create_eval_dir(setup_config)

        evaluator_config = self.config_dict['evaluator']
        shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(),
            obj_class + '_centered_obj_normalized')

        demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos',
            setup_config['demo_exp'])

        grasp_optimizer = self._create_optimizer(self.config_dict['grasp_optimizer'],
            model, gripper_query_pts, eval_save_dir)
        place_optimizer = self._create_optimizer(self.config_dict['place_optimizer'],
            model, shelf_query_pts, eval_save_dir)

        experiment = EvaluateShelfPlaceGraspIdeal(grasp_optimizer=grasp_optimizer,
            place_optimizer=place_optimizer,
            seed=self.seed, shapenet_obj_dir=shapenet_obj_dir,
            eval_save_dir=eval_save_dir, demo_load_dir=demo_load_dir,
            **evaluator_config)

        return experiment

    def _create_model(self, model_config) -> torch.nn.Module:
        """
        Create torch model from given configs

        Returns:
            torch.nn.Module: Either ConvOccNetwork or VNNOccNet
        """
        model_type = model_config['type']
        model_args = model_config['args']
        model_checkpoint = osp.join(path_util.get_ndf_model_weights(),
                                    model_config['checkpoint'])

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

    def _create_optimizer(self, optimizer_config: dict, model: torch.nn.Module,
            query_pts: np.ndarray, eval_save_dir=None) -> OccNetOptimizer:
        """
        Create OccNetOptimizer from given config

        Args:
            model (torch.nn.Module): Model to use in the optimizer
            query_pts (np.ndarray): Query points to use in optimizer

        Returns:
            OccNetOptimizer: Optimizer to find best grasp position
        """
        if 'opt_type' in optimizer_config:
            optimizer_type = optimizer_config['opt_type']  # LNDF or GEOM
        else:
            optimizer_type = None

        optimizer_config = optimizer_config['args']
        if eval_save_dir is not None:
            opt_viz_path = osp.join(eval_save_dir, 'visualization')
        else:
            opt_viz_path = 'visualization'

        if optimizer_type == 'GEOM':
            print('Using geometric optimizer')
            optimizer = GeomOptimizer(model, query_pts, viz_path=opt_viz_path,
                **optimizer_config)
        else:
            print('Using Occ Net optimizer')
            optimizer = OccNetOptimizer(model, query_pts, viz_path=opt_viz_path,
                **optimizer_config)
        return optimizer

    def _create_query_pts(self, query_pts_config: dict) -> np.ndarray:
        """
        Create query points from given config

        Args:
            query_pts_config(dict): Configs loaded from yaml file.

        Returns:
            np.ndarray: Query point as ndarray
        """

        query_pts_type = query_pts_config['type']
        query_pts_args = query_pts_config['args']

        assert query_pts_type in QueryPointTypes, 'Invalid query point type'

        if query_pts_type == 'SPHERE':
            query_pts = QueryPoints.generate_sphere(**query_pts_args)
        elif query_pts_type == 'RECT':
            query_pts = QueryPoints.generate_rect(**query_pts_args)
        elif query_pts_type == 'CYLINDER':
            query_pts = QueryPoints.generate_cylinder(**query_pts_args)
        elif query_pts_type == 'ARM':
            query_pts = QueryPoints.generate_rack_arm(**query_pts_args)
        elif query_pts_type == 'SHELF':
            query_pts = QueryPoints.generate_shelf(**query_pts_args)
        elif query_pts_type == 'NDF_GRIPPER':
            query_pts = QueryPoints.generate_ndf_gripper(**query_pts_args)
        elif query_pts_type == 'NDF_RACK':
            query_pts = QueryPoints.generate_ndf_rack(**query_pts_args)
        elif query_pts_type == 'NDF_SHELF':
            query_pts = QueryPoints.generate_ndf_shelf(**query_pts_args)

        return query_pts

    def _create_eval_dir(self, setup_config: dict) -> str:
        """
        Create eval save dir as concatenation of current time
        and 'exp_desc'.

        Args:
            exp_desc (str, optional): Description of experiment. Defaults to ''.

        Returns:
            str: eval_save_dir.  Gives access to eval save directory
        """
        if 'exp_dir_suffix' in setup_config:
            exp_desc = setup_config['exp_dir_suffix']
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

    def _get_demo_load_dir(self,
        demo_exp: str='mug/grasp_rim_hang_handle_gaussian_precise_w_shelf') -> str:
        """
        Get directory of demos

        Args:
            obj_class (str, optional): Object class. Defaults to 'mug'.
            demo_exp (str, optional): Demo experiment name. Defaults to
                'grasp_rim_hang_handle_gaussian_precise_w_shelf'.

        Returns:
            str: Path to demo load dir
        """
        # demo_load_dir = osp.join(path_util.get_ndf_data(),
        #                          'demos', obj_class, demo_exp)

        demo_load_dir = osp.join(path_util.get_ndf_data(),
                                 'demos', demo_exp)

        return demo_load_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fname', type=str, default='debug_config.yml',
                        help='Filename of experiment config yml')

    args = parser.parse_args()
    config_fname = args.config_fname

    setup = EvaluateNetworkSetup()
    experiment = setup.set_up_network(config_fname)
    experiment.load_demos()
    experiment.configure_sim()
    experiment.run_experiment()

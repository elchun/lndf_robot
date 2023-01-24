import os, os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import trimesh
import open3d as o3d
from sklearn.manifold import TSNE
import plotly.express as px

from airobot import log_info, log_warn, log_debug, log_critical

from ndf_robot.utils import util, torch_util, trimesh_util, torch3d_util
from ndf_robot.utils.eval_gen_utils import object_is_still_grasped
from ndf_robot.utils.plotly_save import plot3d, multiplot


class Demo:
    """
    Container class for a demo.
    """
    def __init__(self, obj_pts: np.ndarray, query_pts: np.ndarray,
                 obj_pose_world: np.ndarray, query_pose_world: np.ndarray,
                 obj_shapenet_id: str):
        """
        Create instance of demo

        Args:
            obj_pts (np.ndarray): Mean centered points of object.
            query_pts (np.ndarray): Mean centered query points.
            obj_pose_world (np.ndarray): Pose of object points in world coords.
            query_pose_world (np.ndarray): Pose of query points in world coords.
        """
        self.obj_pts = obj_pts
        self.query_pts = query_pts
        self.obj_pose_world = obj_pose_world
        self.query_pose_world = query_pose_world
        self.obj_shapenet_id = obj_shapenet_id


class GeomOptimizer:
    def __init__(self, model, query_pts=None, query_pts_real_shape=None, opt_iterations=250,
                 noise_scale=0.0, noise_decay=0.5, single_object=False,
                 rand_translate=False, viz_path='visualization', use_tsne=False,
                 M_override: 'bool | int' = None, query_pts_override=True,
                 opt_fname_prefix: str = 'ee_pose_optimized',
                 save_all_opt: bool = False, cos_loss: bool = False):

        self.n_obj_points = 2000
        self.n_query_points = 1500

        self.model = model
        self.model_type = self.model.model_type

        self.query_pts = query_pts

        # Use user defined query points.  Make sure they are at the right
        # location
        self.query_pts_override = query_pts_override
        if self.query_pts_override:
            assert self.query_pts is not None
        # else:
            # assert self.query_pts is None, 'Query points will be set by demos' \
            #     + 'if override not in use'

        self.demos: list[Demo] = []
        self.target_act_hat = None

        self.cos_loss = cos_loss
        if cos_loss:
            def loss_fn(output, target):
                # print(output.shape)
                return -F.cosine_similarity(output, target, dim=1).mean()
            self.loss_fn = loss_fn
        else:
            self.loss_fn = torch.nn.L1Loss()
        if torch.cuda.is_available():
            self.dev = torch.device('cuda:0')
        else:
            self.dev = torch.device('cpu')

        if self.model is not None:
            self.model = self.model.to(self.dev)
            self.model.eval()

        self.opt_iterations = opt_iterations

        self.noise_scale = noise_scale
        self.noise_decay = noise_decay

        self.demo_info = None

        self.debug_viz_path = 'debug_viz'
        self.viz_path = viz_path
        util.safe_makedirs(self.debug_viz_path)
        util.safe_makedirs(self.viz_path)
        self.viz_files = []

        self.rot_grid = util.generate_healpix_grid(size=1e6)

        # Translate query point init location randomly within bounding box of pcd
        self.rand_translate = rand_translate

        # For saving tsne visualization
        self.use_tsne = use_tsne

        # For overriding number of initializations in opt
        self.M_override = M_override

        self.opt_fname_prefix = opt_fname_prefix

        self.save_all_opt = save_all_opt

        self.target_pcds = []
        self.target_pcds_down = []
        self.target_pcds_fpfh = []
        # self.target_pcd = None
        # self.target_pcd_down = None
        # self.target_pcd_fpfh = None

        self.radius_normal = 0.010
        self.voxel_size = 0.005

        self.use_query_pts = False

    def _scene_dict(self):
        self.scene_dict = {}
        plotly_scene = {
            'xaxis': {'nticks': 16, 'range': [-0.5, 0.5]},
            'yaxis': {'nticks': 16, 'range': [-0.5, 0.5]},
            'zaxis': {'nticks': 16, 'range': [-0.5, 0.5]}
        }
        self.scene_dict['scene'] = plotly_scene

    def _preprocess_point_cloud(self, pcd, voxel_size):
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    def _execute_global_registration(self, source_down, target_down, source_fpfh,
                                    target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(10000000, 0.99999))
        return result

    def _refine_registration(self, source, target, source_fpfh,
                            target_fpfh, voxel_size, result_ransac):
        distance_threshold = voxel_size * 0.05
        radius_normal = voxel_size * 2
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        print(":: Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        print("   distance threshold %.3f." % distance_threshold)
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        return result

    def _crop_with_query(self, query_pcd, target_pcd):
        """
        Crop a target point cloud using the bounding box of query points

        Args:
            query_pts (o3d pointcloud): Query points to crop with
            target_pcd (o3d pointcloud): Target points to crop
        """
        query_bb = query_pcd.get_oriented_bounding_box()
        print('bb: ', query_bb)
        target_pcd_cropped = target_pcd.crop(query_bb)
        return target_pcd_cropped, query_bb.get_box_points()

    def add_demo(self, demo: Demo):
        """
        Store new demo.

        Args:
            demo (Demo): Demo to use.
        """
        self.demos.append(demo)

    def process_demos(self):
        """
        Get target activation from all demos and save to {self.target_pcd,
        self.target_pcd_down, and self.target_pcd_fpfh}

        Must be called before running optimization.
        """
        # demo_acts_list = []
        # demo_latents_list = []

        demo_pcd_list = []
        for i, demo in enumerate(self.demos):
            # Note that obj_pts and query_pts are continuously modified here.
            # I think it makes the code cleaner, but they do change types

            # -- Load pts from demo -- #
            obj_pts = demo.obj_pts
            if self.query_pts_override:
                query_pts = self.query_pts
            else:
                query_pts = demo.query_pts
                if self.query_pts is None:
                    self.query_pts = demo.query_pts

            # -- Transform pts to same position as demo (in world coords) -- #
            obj_pts = util.apply_pose_numpy(obj_pts, demo.obj_pose_world)
            inverse_query_pose = util.get_inverse_pose(demo.query_pose_world)
            obj_pts = util.apply_pose_numpy(obj_pts, inverse_query_pose)

            # query_pts = util.apply_pose_numpy(query_pts, demo.query_pose_world)

            # DEBUG PLOTS
            multiplot([demo.query_pts, self.query_pts], osp.join(self.debug_viz_path,
                f'{self.opt_fname_prefix}_query_compare.html'))
            multiplot([obj_pts, query_pts], osp.join(self.debug_viz_path,
                f'{self.opt_fname_prefix}_demo_geom{i}.html'))

            # # -- Keep relative orientation, but center points on query mean -- #
            # query_pts_mean = query_pts.mean(0)
            # obj_pts = obj_pts - query_pts_mean
            # query_pts = query_pts - query_pts_mean

            # -- Create geometric target feature -- #
            object_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pts))
            query_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query_pts))

            object_pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius_normal, max_nn=30))
            query_pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius_normal, max_nn=30))

            if self.use_query_pts:
                object_pcd_cropped, bb_pts = self._crop_with_query(query_pcd=query_pcd, target_pcd=object_pcd)

                fname = osp.join(self.debug_viz_path, f'{self.opt_fname_prefix}single_pts{i}.html')
                multiplot([np.asarray(object_pcd.points),
                    np.asarray(bb_pts),
                    np.asarray(object_pcd_cropped.points)], fname)

                demo_pcd_list.append(object_pcd_cropped)
            else:
                demo_pcd_list.append(object_pcd)

        target_pcd = demo_pcd_list[0]

        for target_pcd in demo_pcd_list:
            self.target_pcds.append(target_pcd)

            target_pcd_down, target_pcd_fpfh = self._preprocess_point_cloud(target_pcd, self.voxel_size)
            self.target_pcds_down.append(target_pcd_down)
            self.target_pcds_fpfh.append(target_pcd_fpfh)

        # demo_pcd_combined_np = np.asarray(demo_pcd_combined.points)
        # fname = osp.join(self.debug_viz_path, f'{self.opt_fname_prefix}_combo_pts.html')
        # multiplot([demo_pcd_combined_np], fname)

        # fname = osp.join(self.debug_viz_path, 'single_pts.html')
        # multiplot([np.asarray(demo_pcd_list[0].points)], fname)

        # self.target_pcd = target_pcd
        # self.target_pcd_down = target_pcd_down
        # self.target_pcd_fpfh = target_pcd_fpfh
        fname = osp.join(self.debug_viz_path, f'{self.opt_fname_prefix}_combo_pts.html')
        # multiplot([np.asarray(target_pcd_down.points)], fname)
        multiplot([np.asarray(pcd.points) for pcd in self.target_pcds_down], fname)

    def optimize_transform_implicit(self, shape_pts_world_np, viz_path='visualize',
        ee=True, *args, **kwargs):
        """
        Function to optimzie the transformation of our query points, conditioned on
        a set of shape points observed in the world

        Args:
            shape_pts_world (np.ndarray): N x 3 array representing 3D point cloud of the object
                to be manipulated, expressed in the world coordinate system
        """

        # Make temp viz dir
        util.safe_makedirs(viz_path)

        # -- Get number of inits -- #
        # if self.M_override is not None:
        #     assert type(self.M_override) == int, 'Expected int number of M'
        #     M = self.M_override
        # elif 'dgcnn' in self.model_type:
        #     M = 5   # dgcnn can't fit 10 initialization in memory
        # else:
        #     M = 10

        best_idx = 0
        tf_list = []
        losses = []

        object_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(shape_pts_world_np))
        object_pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius_normal, max_nn=30))
        object_pcd_down, object_pcd_fpfh = self._preprocess_point_cloud(object_pcd, self.voxel_size)

        for i in range(len(self.target_pcds_down)):
            # Compute ransac for global geometric alignment
            result_ransac = self._execute_global_registration(object_pcd_down, self.target_pcds_down[i],
                                                        object_pcd_fpfh, self.target_pcds_fpfh[i],
                                                        self.voxel_size)

            result_icp = self._refine_registration(object_pcd, self.target_pcds_down[i],
                object_pcd_fpfh, self.target_pcds_fpfh[i],
                self.voxel_size, result_ransac)

            # inlier_rmse = result_icp.inlier_rmse
            fitness = result_icp.fitness

            transform_mat_np = np.asarray(result_icp.transformation)

            if not ee:  # Opposite of what we do for ndf
                T_mat = transform_mat_np
            else:
                T_mat = np.linalg.inv(transform_mat_np)

            # final_query_pts = util.transform_pcd(self.query_pts, T_mat)
            tf_list.append(T_mat)
            losses.append(-fitness)
            print(T_mat)

        best_idx = np.argmin(losses)
        print('losses: ', losses)

        # Visualize best trial
        for i in range(len(self.target_pcds_down)):
            T_mat = tf_list[i]
            if not ee:
                T_mat = np.linalg.inv(T_mat)

            final_query_pts = util.transform_pcd(self.query_pts, T_mat)
            opt_fname = f'{self.opt_fname_prefix}_{i}.html'
            target_pts = np.asarray(self.target_pcds_down[i].points)
            target_pts = util.transform_pcd(target_pts, T_mat)
            print(opt_fname)
            if self.save_all_opt:
                # self._visualize_pose(shape_pts_world_np, final_query_pts, viz_path, opt_fname)
                multiplot([shape_pts_world_np, final_query_pts, target_pts], osp.join(viz_path, opt_fname))
            elif i == best_idx:
                multiplot([shape_pts_world_np, final_query_pts, target_pts], osp.join(viz_path, opt_fname))
                # self._visualize_pose(shape_pts_world_np, final_query_pts, viz_path, opt_fname)

        return tf_list, best_idx

    def _visualize_reconstruction(self, model_input, viz_path):
        """
        Compute reconstruction of obj using network and save to file in
        {viz_path}.

        Args:
            model_input (dict): Input to network to use.
            viz_path (str): Path to directory to save visualization to.
        """
        jj = 0
        shape_mi = {}
        shape_mi['point_cloud'] = model_input['point_cloud'][jj][None, :, :].detach()
        shape_np = shape_mi['point_cloud'].cpu().numpy().squeeze()
        shape_mean = np.mean(shape_np, axis=0)
        inliers = np.where(np.linalg.norm(shape_np - shape_mean, 2, 1) < 0.2)[0]
        shape_np = shape_np[inliers]
        shape_pcd = trimesh.PointCloud(shape_np)
        bb = shape_pcd.bounding_box
        bb_scene = trimesh.Scene()
        bb_scene.add_geometry([shape_pcd, bb])

        eval_pts = bb.sample_volume(10000)
        shape_mi['coords'] = torch.from_numpy(eval_pts)[None, :, :].float().to(self.dev).detach()
        out = self.model(shape_mi)
        thresh = 0.3
        in_inds = torch.where(out['occ'].squeeze() > thresh)[0].cpu().numpy()

        in_pts = eval_pts[in_inds]
        self._scene_dict()
        plot3d(
            [in_pts, shape_np],
            ['blue', 'black'],
            osp.join(viz_path, 'recon_overlay.html'),
            scene_dict=self.scene_dict,
            z_plane=False)

        if self.use_tsne:
            self._tsne_viz(in_pts, osp.join(viz_path, 'tsne.html'))

    def _visualize_pose(self, obj_pts, final_query_pts, viz_path, opt_fname):
        all_pts = [final_query_pts, obj_pts]
        # opt_fname = 'ee_pose_optimized_%d.html' % j if ee else 'rack_pose_optimized_%d.html' % j
        opt_scene_dict = {
            'scene': {
                'xaxis': {'nticks': 16, 'range': [-1, 1]},
                'yaxis': {'nticks': 16, 'range': [-1, 1]},
                'zaxis': {'nticks': 16, 'range': [0, 2]}
            }
        }
        plot3d(
            all_pts,
            ['black', 'purple'],
            osp.join(viz_path, opt_fname),
            scene_dict=opt_scene_dict,
            z_plane=False)
        self.viz_files.append(osp.join(viz_path, opt_fname))

    def _tsne_viz(self, pcd: np.ndarray, output_fn: str):
        n_query_pts = 500
        n_components = 1

        model_input = {}
        pcd_torch = torch.from_numpy(pcd).float().to(self.dev)

        if pcd.shape[0] <= 0:
            return
        rix = np.random.randint(0, pcd.shape[0], (n_query_pts))
        pcd_small = pcd[rix, :]

        object_pcd_torch = torch.from_numpy(pcd_small).float().to(self.dev)
        object_pcd_torch = object_pcd_torch[None, :, :]  # Query points

        model_input['coords'] = object_pcd_torch[None, :, :]
        model_input['point_cloud'] = pcd_torch[None, :, :]

        latent = self.model.extract_latent(model_input).detach()
        act_torch = self.model.forward_latent(latent, object_pcd_torch).detach()
        act = act_torch.squeeze().cpu().numpy()

        tsne = TSNE(n_components)
        tsne_result = tsne.fit_transform(act)

        fig = px.scatter_3d(x=pcd_small[:, 0], y=pcd_small[:, 1],
                            z=pcd_small[:, 2], color=tsne_result[:, 0])

        fig.write_html(output_fn)



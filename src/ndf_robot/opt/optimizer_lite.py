import os, os.path as osp
from cv2 import transform
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import trimesh
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


class OccNetOptimizer:
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

    def _scene_dict(self):
        self.scene_dict = {}
        plotly_scene = {
            'xaxis': {'nticks': 16, 'range': [-0.5, 0.5]},
            'yaxis': {'nticks': 16, 'range': [-0.5, 0.5]},
            'zaxis': {'nticks': 16, 'range': [-0.5, 0.5]}
        }
        self.scene_dict['scene'] = plotly_scene

    def add_demo(self, demo: Demo):
        """
        Store new demo.

        Args:
            demo (Demo): Demo to use.
        """
        self.demos.append(demo)

    def process_demos(self):
        """
        Get target activation from all demos and save to {self.target_act_hat}

        Must be called before running optimization.
        """
        demo_acts_list = []
        demo_latents_list = []
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
            query_pts = util.apply_pose_numpy(query_pts, demo.query_pose_world)
            # DEBUG PLOTS
            multiplot([demo.query_pts, self.query_pts], osp.join(self.debug_viz_path,
                f'{self.opt_fname_prefix}_query_compare.html'))
            multiplot([obj_pts, query_pts], osp.join(self.debug_viz_path,
                f'{self.opt_fname_prefix}_demo_{i}.html'))

            # -- Keep relative orientation, but center points on obj mean -- #
            obj_pts = torch.from_numpy(obj_pts).float().to(self.dev)
            query_pts = torch.from_numpy(query_pts).float().to(self.dev)

            obj_pts_mean = obj_pts.mean(0)
            obj_pts = obj_pts - obj_pts_mean
            query_pts = query_pts - obj_pts_mean

            # -- Sample points and prepare model input -- #
            rndperm = torch.randperm(obj_pts.size(0))
            demo_model_input = {
                'point_cloud': obj_pts[None, rndperm[:self.n_obj_points], :],
                'coords': query_pts[None, :self.n_query_points, :]
            }

            # -- Get latent, then use it to find activations -- #
            demo_latent = self.model.extract_latent(demo_model_input).detach()
            demo_act_hat = self.model.forward_latent(demo_latent,
                demo_model_input['coords']).detach()

            demo_latents_list.append(demo_latent.squeeze())
            demo_acts_list.append(demo_act_hat.squeeze())

        demo_acts_all = torch.stack(demo_acts_list, 0)
        self.target_act_hat = torch.mean(demo_acts_all, 0)

        # print('target_act_hat: ', self.target_act_hat)

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

        dev = self.dev
        n_pts = 1500
        # opt_pts = 500
        # opt_pts = 1000 # Seems to work well
        opt_pts = 2000
        perturb_scale = self.noise_scale
        perturb_decay = self.noise_decay

        target_act_hat = self.target_act_hat
        assert target_act_hat is not None, 'Did you run process_demos() yet?'

        # -- mean center obj pts -- #
        obj_pts = torch.from_numpy(shape_pts_world_np).float().to(self.dev)
        obj_pts_mean = obj_pts.mean(0)
        obj_pts = obj_pts - obj_pts_mean

        # -- Get query points -- #
        # Centers points so that its easier to translate them to init position
        query_pts = torch.from_numpy(self.query_pts).float().to(self.dev)
        # query_pts_mean = query_pts.mean(0)
        # query_pts = query_pts - query_pts_mean

        # # convert query points to camera frame, and center the query based on the it's shape mean, so that we perform optimization starting with the query at the origin
        # query_pts_world = torch.from_numpy(self.query_pts_origin).float().to(self.dev)
        # query_pts_mean = query_pts_world.mean(0)
        # query_pts_cent = query_pts_world - query_pts_mean

        query_pts_tf = np.eye(4)
        # query_pts_tf[:-1, -1] = query_pts_mean.cpu().numpy()

        # -- Get number of inits -- #
        if self.M_override is not None:
            assert type(self.M_override) == int, 'Expected int number of M'
            M = self.M_override
        elif 'dgcnn' in self.model_type:
            M = 5   # dgcnn can't fit 10 initialization in memory
        else:
            M = 10

        best_loss = np.inf
        best_idx = 0
        # best_tf = np.eye(4)
        tf_list = []
        # M = full_opt

        trans = (torch.rand((M, 3)) * 0.1).float().to(dev)
        rot = torch.rand(M, 3).float().to(dev)
        # rot_idx = np.random.randint(self.rot_grid.shape[0], size=M)
        # rot = torch3d_util.matrix_to_axis_angle(torch.from_numpy(self.rot_grid[rot_idx])).float()

        # rand_rot_init = (torch.rand((M, 3)) * 2*np.pi).float().to(dev)
        rand_rot_idx = np.random.randint(self.rot_grid.shape[0], size=M)
        rand_rot_init = torch3d_util.matrix_to_axis_angle(torch.from_numpy(self.rot_grid[rand_rot_idx])).float()
        rand_mat_init = torch_util.angle_axis_to_rotation_matrix(rand_rot_init)
        rand_mat_init = rand_mat_init.squeeze().float().to(dev)

        # query_pts_cam_cent_rs, query_pts_tf_rs = self._get_query_pts_rs()
        # X_rs = query_pts_cam_cent_rs[:opt_pts][None, :, :].repeat((M, 1, 1))

        # ADDITIONS
        min_coords = torch.min(obj_pts, dim=0).values.reshape(3, 1)
        max_coords = torch.max(obj_pts, dim=0).values.reshape(3, 1)
        coord_range = max_coords - min_coords

        # Create a translation to random point in bounding box of observed pcd
        # for each init
        if self.rand_translate:
            rand_translate = torch.rand(M, 3, 1).to(dev)
            rand_translate = rand_translate * coord_range + min_coords
            # # print('translate: ', rand_translate)

            rand_mat_init[:, :3, 3:4] = rand_translate

        # set up optimization
        X = query_pts[:opt_pts][None, :, :].repeat((M, 1, 1))
        X = torch_util.transform_pcd_torch(X, rand_mat_init)
        # X_rs = torch_util.transform_pcd_torch(X_rs, rand_mat_init)

        # mi is model input
        mi_point_cloud = []
        for ii in range(M):
            rndperm = torch.randperm(obj_pts.size(0))
            mi_point_cloud.append(obj_pts[rndperm[:n_pts]])
        mi_point_cloud = torch.stack(mi_point_cloud, 0)
        mi = dict(point_cloud=mi_point_cloud)
        obj_mean_trans = np.eye(4)
        obj_mean_trans[:-1, -1] = obj_pts_mean.cpu().numpy()
        # shape_pts_world_np = shape_pts_world.cpu().numpy()

        rot.requires_grad_()
        trans.requires_grad_()
        full_opt = torch.optim.Adam([trans, rot], lr=1e-2)
        full_opt.zero_grad()

        loss_values = []

        # set up model input with shape points and the shape latent that will be used throughout
        mi['coords'] = X
        latent = self.model.extract_latent(mi).detach()

        # run optimization
        pcd_traj_list = {}
        for jj in range(M):
            pcd_traj_list[jj] = []

        # -- Visualize reconstruction -- #
        self._visualize_reconstruction(mi, viz_path)

        # -- Run optimization -- #
        for i in range(self.opt_iterations):
            T_mat = torch_util.angle_axis_to_rotation_matrix(rot).squeeze()

            # Generating noise vec takes a lot of cpu!!!
            if perturb_scale > 0:
                noise_vec = (torch.randn(X.size()) * (perturb_scale / ((i+1)**(perturb_decay)))).to(dev)
                X_perturbed = X + noise_vec
            else:
                X_perturbed = X

            X_new = torch_util.transform_pcd_torch(X_perturbed, T_mat) + trans[:, None, :].repeat((1, X.size(1), 1))

            act_hat = self.model.forward_latent(latent, X_new)
            t_size = target_act_hat.size()

            losses = [self.loss_fn(act_hat[ii].view(t_size), target_act_hat) for ii in range(M)]
            loss = torch.mean(torch.stack(losses))
            if (i + 1) % 100 == 0:
                losses_str = ['%f' % val.item() for val in losses]
                loss_str = ', '.join(losses_str)
                log_debug(f'i: {i}, losses: {loss_str}')
            loss_values.append(loss.item())
            full_opt.zero_grad()
            loss.backward()
            full_opt.step()

        # -- Find best index -- #
        best_idx = torch.argmin(torch.stack(losses)).item()
        best_loss = losses[best_idx]
        log_debug('best loss: %f, best_idx: %d' % (best_loss, best_idx))

        for j in range(M):
            # -- Pose query points -- #
            trans_j, rot_j = trans[j], rot[j]

            transform_mat_np = torch_util.angle_axis_to_rotation_matrix(
                rot_j.view(1, -1)).squeeze().detach().cpu().numpy()
            transform_mat_np[:-1, -1] = trans_j.detach().cpu().numpy()

            # Send query points back to where they came from
            # transform_mat_np = rand_mat_init[j].detach().cpu().numpy()
            # transform_mat_np = np.matmul(transform_mat_np, query_pts_tf)
            transform_mat_np = np.matmul(transform_mat_np, rand_mat_init[j].detach().cpu().numpy())
            # transform_mat_np = np.matmul(transform_mat_np, query_pts_tf)
            transform_mat_np = np.matmul(obj_mean_trans, transform_mat_np)

            final_query_pts = util.transform_pcd(self.query_pts, transform_mat_np)
            opt_fname = f'{self.opt_fname_prefix}_{j}.html'

            if self.save_all_opt:
                self._visualize_pose(shape_pts_world_np, final_query_pts, viz_path, opt_fname)
            elif j == best_idx:
                self._visualize_pose(shape_pts_world_np, final_query_pts, viz_path, opt_fname)

            if ee:
                T_mat = transform_mat_np
            else:
                T_mat = np.linalg.inv(transform_mat_np)
            tf_list.append(T_mat)

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



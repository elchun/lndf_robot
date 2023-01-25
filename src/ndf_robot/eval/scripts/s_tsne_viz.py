"""
Script for TSNE visualization of mug with given model weights
"""
import os.path as osp
from cv2 import transform

import torch
import numpy as np
import trimesh
from sklearn.manifold import TSNE
from scipy.spatial.transform import Rotation
import plotly.express as px

import ndf_robot.model.conv_occupancy_net.conv_occupancy_net as conv_occupancy_network
from ndf_robot.utils import path_util, torch3d_util, torch_util

from ndf_robot.share import globals


class TSNEViz:
    """
    Visualize the latent activations of a model on any given
    object

    Attributes:
        self.model (torch.nn.Module): Model to use
        self.dev (torch.Device): Device to run network on
        self.pcd_list (list<np.ndArray>): List of pcds to run visualize on
        self.query_pts_list (list<np.ndarray>): List of query points to run
            visualize on.  pcd of smae index is used as the input pcd to model
            for given query point.
    """

    def __init__(self, model: torch.nn.Module):
        """
        Create TSNEViz Object

        Args:
            model (torch.nn.Module): Model to use
        """
        self.model = model
        self.dev = torch.device('cuda:0') if torch.cuda.is_available() \
            else torch.device('cpu')
        self.pcd_list = []
        self.query_pts_list = []

    def load_object(self, object_fn: str, sample_bb: bool=False):
        """
        Load object model from file, sample mesh to create object point cloud
        and query points.  Save both to TSNEViz Object.

        Args:
            object_fn (str): Path to object to load
        """
        mesh = trimesh.load(object_fn, process=False)
        pcd = mesh.sample(5000)

        if sample_bb:
            # pcd_mean = np.mean(pcd, axis=0)
            # trans = np.eye(4)
            # trans[:3, 3] = pcd_mean
            # print(pcd_mean)
            # print(trans)
            # query_pts = trimesh.primitives.Box(extents=[1, 1, 1])
            # # sphere = trimesh.primitives.Sphere(radius=1, center=pcd_mean)
            # # query_pts = sphere.sample_volume(500)
            # # https://trimsh.org/examples/quick_start.html
            # # query_pts = mesh.bounding_box_oriented.sample_volume(500)

            to_origin, extents = trimesh.bounds.oriented_bounds(pcd)
            max_extent = max(extents)
            # print(max_extent)
            extents_max = np.repeat(max_extent, 3)


            # bb = trimesh.primitives.Box(extents=extents, transform=to_origin)

            bb_scale = 1.5
            query_pts = trimesh.sample.volume_rectangular(extents_max * bb_scale, 500, transform=to_origin)


            # query_pts = trimesh.sample.volume_rectangular(extents, 500, transform=to_origin)
        else:
            query_pts = mesh.sample(500)  # Set to also sample within body

        self.pcd_list.append(pcd)
        self.query_pts_list.append(query_pts)

    def viz_all_objects(self, base_output_fn: str='tsne_viz',
        rand_rotate: bool=False):
        """
        Run TSNE on all objects currently loaded into TSNEViz object.  Save
        all to files with prefix {base_output_fn} followed by '_i.html' where
        i is the index of the object stored.

        Args:
            base_output_fn (str, optional): Filename to prefix all viz html
                files with. Defaults to 'tsne_viz'.
            rand_rotate (bool, optional): True to randomly rotated objects to
                visualize. Defaults to False.
        """
        base_output_fn = base_output_fn.split('.')[0]

        if rand_rotate:
            random_transform = torch.tensor(TSNEViz.__random_rot_transform()).float().to(self.dev)
        else:
            random_transform = None

        i = 0
        for pcd, query_pts in zip(self.pcd_list, self.query_pts_list):
            object_fn = base_output_fn + '_' + str(i) + '.html'
            self.viz_object(pcd, query_pts, object_fn, rand_rotate=False,
            random_transform=random_transform)
            i += 1

    def viz_all_objects_together(self, output_fn: str='tsne_viz.html',
        rand_rotate: bool=False, num_repeats=1):
        """
        Run the single tsne for activations of all objects, plots all in the
        same plot.

        Args:
            output_fn (str, optional): Filename to save visualization to.
                Defaults to 'tsne_viz'.
            rand_rotate (bool, optional): True to randomly rotate all objects by
                a transform. Defaults to False.
        """

        activations_list = []
        transformed_query_pts_list = []

        for i in range(num_repeats):
            if rand_rotate:
                random_transform = torch.tensor(TSNEViz.__random_rot_transform()).float().to(self.dev)
            else:
                random_transform = None

            for pcd, query_pts in zip(self.pcd_list, self.query_pts_list):
                model_input = {}
                query_pts_torch = torch.from_numpy(query_pts).float().to(self.dev)
                pcd_torch = torch.from_numpy(pcd).float().to(self.dev)

                # Transform using random_transform if it exists
                if random_transform is not None:
                    query_pts_torch = torch_util.transform_pcd_torch(query_pts_torch,
                        random_transform).float()
                    pcd_torch = torch_util.transform_pcd_torch(pcd_torch,
                        random_transform).float()

                model_input['coords'] = query_pts_torch[None, :, :]
                model_input['point_cloud'] = pcd_torch[None, :, :]

                latent_torch = self.model.extract_latent(model_input).detach()
                act_torch = self.model.forward_latent(latent_torch, model_input['coords']).detach()
                act = act_torch.squeeze().cpu().numpy()

                activations_list.append(act)
                query_pts_np = query_pts_torch.cpu().numpy()
                transformed_query_pts_list.append(query_pts_np)

        combined_acts = np.vstack(activations_list)
        print('acts shape: ', combined_acts.shape)

        n_components = 1
        tsne = TSNE(n_components)
        tsne_result = tsne.fit_transform(combined_acts)

        # Apply offset to plot so its easier to see all objects
        plot_x_offset = 1
        n = len(transformed_query_pts_list)
        for i in range(n):
            transformed_query_pts_list[i][:, 0] += (i - n / 2) * plot_x_offset

        combined_query_pts = np.vstack(transformed_query_pts_list)
        print('query pts shape: ', combined_query_pts.shape)

        low_range = -(n + 1) * plot_x_offset
        high_range = (n + 1) * plot_x_offset
        fig = px.scatter_3d(x=combined_query_pts[:, 0], y=combined_query_pts[:, 1], z=combined_query_pts[:, 2],
            color=tsne_result[:, 0], range_x=(low_range, high_range),
            range_y=(low_range, high_range), range_z=(low_range, high_range))

        fig.write_html(output_fn)

    def viz_object(self, pcd: np.ndarray, query_pts: np.ndarray,
                   output_fn: str='tnse_viz.html', rand_rotate: bool=False,
                   random_transform: torch.Tensor=None):
        """
        Visualize single object with tsne plot and save to output_fn as html
        file.

        Args:
            pcd (np.ndarray): Input point cloud to condition network with
            query_pts (np.ndarray): Input query points to get activations of
            output_fn (str, optional): output filename.
                Defaults to 'tnse_viz.html'.
            rand_rotate (bool, optional): True to randomly rotate the input pcd.
                prior to running through the network. Will not be used if
                random_transform is given.  Defaults to False.
            random_transform (torch.Tensor, optional): Transform to apply to
                input pcd.  Is used instead of rand_rotate when given.
                Defaults to None.
        """
        n_components = 1

        model_input = {}
        query_pts_torch = torch.from_numpy(query_pts).float().to(self.dev)
        pcd_torch = torch.from_numpy(pcd).float().to(self.dev)

        # Transform using random_transform if it exists
        if random_transform is not None:
            query_pts_torch = torch_util.transform_pcd_torch(query_pts_torch,
                random_transform).float()
            pcd_torch = torch_util.transform_pcd_torch(pcd_torch,
                random_transform).float()

        elif rand_rotate:
            random_transform = torch.tensor(TSNEViz.__random_rot_transform()).float().to(self.dev)
            query_pts_torch = torch_util.transform_pcd_torch(query_pts_torch,
                random_transform).float()
            pcd_torch = torch_util.transform_pcd_torch(pcd_torch,
                random_transform).float()

        model_input['coords'] = query_pts_torch[None, :, :]
        model_input['point_cloud'] = pcd_torch[None, :, :]

        # n_query_pts = query_pts.shape[0]

        # model_input['coords'] = query_pts[None, :, :][:, [1, 0, 2]]
        # model_input['point_cloud'] = pcd[None, :, :]

        print('input: ', model_input['point_cloud'].shape)
        print('query: ', model_input['coords'].shape)

        latent_torch = self.model.extract_latent(model_input).detach()
        act_torch = self.model.forward_latent(latent_torch, model_input['coords']).detach()
        act = act_torch.squeeze().cpu().numpy()
        print('act_np', act.shape)

        tsne = TSNE(n_components)
        tsne_result = tsne.fit_transform(act)

        # plot3d([pcd, query_pts], ['blue', 'red'], 'tsne_plot.html',
        #        auto_scene=True)

        # fig = px.scatter(x=tsne_result[:, 0], y=tsne_result[:, 1],
        #     labels={'x': 'tsne1', 'y': 'tsne2'})

        query_pts_np = query_pts_torch.cpu().numpy()

        fig = px.scatter_3d(x=query_pts_np[:, 0], y=query_pts_np[:, 1], z=query_pts_np[:, 2],
            color=tsne_result[:, 0])

        fig.write_html(output_fn)

    @staticmethod
    def __random_quaternions(n: int):
        """
        Generate random quaternions representing rotations,
        i.e. versors with nonnegative real part.

        Modified from random_quaternions in util.torch3d_util.py

        Args:
            n: Number of quaternions in a batch to return.
            dtype: Type to return.
            device: Desired device of returned tensor. Default:
                uses the current device for the default tensor type.

        Returns:
            Quaternions as tensor of shape (N, 4).
        """
        o = torch.randn((n, 4))
        s = (o * o).sum(1)
        o = o / torch3d_util._copysign(torch.sqrt(s), o[:, 0])[:, None]
        return o

    @staticmethod
    def __random_rot_transform():
        """
        Generate a random rotation transform

        Args:
            translate (bool, optional): True to include translation in
                transform. Defaults to False.

        Raises:
            NotImplementedError: translation not done yet

        Returns:
            Transform with random rotation
        """
        rand_quat = TSNEViz.__random_quaternions(1)

        rand_rot = Rotation.from_quat(rand_quat)
        rand_rot = rand_rot.as_matrix()

        transform = np.eye(4)
        transform[:3, :3] = rand_rot

        return transform


if __name__ == '__main__':

    # CONSTANTS #
    object_fns = [
        osp.join(path_util.get_ndf_demo_obj_descriptions(),
            'mug_centered_obj_normalized/edaf960fb6afdadc4cebc4b5998de5d0/models/model_normalized.obj'),
        # osp.join(path_util.get_ndf_demo_obj_descriptions(),
        #     'mug_centered_obj_normalized/e984fd7e97c2be347eaeab1f0c9120b7/models/model_normalized.obj'),
        # osp.join(path_util.get_ndf_demo_obj_descriptions(),
        #     'mug_centered_obj_normalized/ec846432f3ebedf0a6f32a8797e3b9e9//models/model_normalized.obj'),
        # osp.join(path_util.get_ndf_demo_obj_descriptions(),
        #     'mug_centered_obj_normalized/ff1a44e1c1785d618bca309f2c51966a//models/model_normalized.obj'),
        # osp.join(path_util.get_ndf_demo_obj_descriptions(),
        #     'mug_centered_obj_normalized/f1e439307b834015770a0ff1161fa15a//models/model_normalized.obj'),
        # osp.join(path_util.get_ndf_demo_obj_descriptions(),
        #     'mug_centered_obj_normalized/f7d776fd68b126f23b67070c4a034f08/models/model_normalized.obj'),
        # osp.join(path_util.get_ndf_demo_obj_descriptions(),
        #     'mug_centered_obj_normalized/eecb13f61a93b4048f58d8b19de93f99/models/model_normalized.obj'),
        # osp.join(path_util.get_ndf_demo_obj_descriptions(),
        #     'mug_centered_obj_normalized/e9bd4ee553eb35c1d5ccc40b510e4bd/models/model_normalized.obj'),

        # Weird mugs
        # osp.join(path_util.get_ndf_demo_obj_descriptions(),
        #     'mug_centered_obj_normalized/6d2657c640e97c4dd4c0c1a5a5d9a6b8/models/model_normalized.obj'),
        # osp.join(path_util.get_ndf_demo_obj_descriptions(),
        #     'mug_centered_obj_normalized/5ef0c4f8c0884a24762241154bf230ce/models/model_normalized.obj'),

        # Bowls
        # osp.join(path_util.get_ndf_obj_descriptions(),
        #     'bowl_centered_obj_normalized/1f910faf81555f8e664b3b9b23ddfcbc/models/model_normalized.obj'),
        # osp.join(path_util.get_ndf_obj_descriptions(),
        #     'bowl_centered_obj_normalized/2c1df84ec01cea4e525b133235812833/models/model_normalized.obj'),

        # Rack
        # osp.join(path_util.get_ndf_descriptions(), 'hanging/table/simple_rack.obj')
    ]
    base_output_fn = 'tsne_viz/tsne_viz'
    # output_fn = 'tsne_viz_latent_32.html'

    # LOAD MODEL #
    # model = conv_occupancy_network.ConvolutionalOccupancyNetwork(latent_dim=32,
    #     model_type='pointnet', return_features=True, sigmoid=True,
    #     acts='last').cuda()

    # model = conv_occupancy_network.ConvolutionalOccupancyNetwork(latent_dim=4,
    #     model_type='pointnet', return_features=True, sigmoid=True,
    #     acts='last').cuda()

    # model = conv_occupancy_network.ConvolutionalOccupancyNetwork(latent_dim=8,
    #     model_type='pointnet', return_features=True, sigmoid=True,
    #     acts='last').cuda()

    # model = conv_occupancy_network.ConvolutionalOccupancyNetwork(latent_dim=64,
    #     model_type='pointnet', return_features=True, sigmoid=True,
    #     acts='last').cuda()

    model = conv_occupancy_network.ConvolutionalOccupancyNetwork(latent_dim=128,
        model_type='pointnet', return_features=True, sigmoid=True,
        acts='last').cuda()

    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_latent_adaptive_2/checkpoints/model_epoch_0009_iter_099000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_latent_4_0/checkpoints/model_epoch_0010_iter_130000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_latent_dim4_rotated_triplet_0/checkpoints/model_epoch_0000_iter_006000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_latent_dim4_rotated_triplet_n_margin_10e3_last_acts_1/checkpoints/model_epoch_0000_iter_006000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_latent_dim4_rotated_triplet_n_margin_10e3_last_acts_1/checkpoints/model_epoch_0004_iter_056000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_latent_dim4_rotated_triplet_n_margin_10e3_last_acts_margin_0p001_0p1_0/checkpoints/model_epoch_0004_iter_050000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden4_anyrot_2/checkpoints/model_epoch_0002_iter_029000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_latent_dim4_rotated_triplet_0/checkpoints/model_epoch_0009_iter_113000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden4_anyrot_6/checkpoints/model_epoch_0011_iter_143000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden8_anyrot_0/checkpoints/model_epoch_0011_iter_132000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_train_any_rot_hidden4_rot_similar_0/checkpoints/model_epoch_0001_iter_016000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_latent_log_4_10_8_0/checkpoints/model_epoch_0011_iter_143000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_train_any_rot_hidden4_rot_similar_super_aggressive_1/checkpoints/model_epoch_0003_iter_042000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_train_any_rot_hidden4_rot_similar_log_1/checkpoints/model_epoch_0005_iter_066000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_train_any_rot_hidden4_rot_similar_log_1/checkpoints/model_epoch_0011_iter_143000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_train_any_rot_hidden4_occ_similar_only_0/checkpoints/model_epoch_0006_iter_081000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_train_any_rot_hidden4_occ_similar_only_margin_noneg_10_1/checkpoints/model_epoch_0003_iter_039000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_latent32_triplet_similar_occ_only_all_1/checkpoints/model_epoch_0003_iter_037000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_train_any_rot_hidden4_occ_similar_only_0/checkpoints/model_epoch_0010_iter_124000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden32_anyrot_1/checkpoints/model_epoch_0011_iter_170000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_latent32_triplet_similar_occ_only_0/checkpoints/model_epoch_0009_iter_113000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden32_anyrot_part2_1/checkpoints/model_epoch_0011_iter_179000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden32_anyrot_simocc_0/checkpoints/model_epoch_0011_iter_143000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_train_any_rot_hidden4_occ_similar_only_margin_noneg_10_1/checkpoints/model_epoch_0011_iter_143000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden64_anyrot_0/checkpoints/model_epoch_0023_iter_358000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden32_anyrot_simfull_0/checkpoints/model_epoch_0007_iter_111000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden32_anyrot_simlat_100_0/checkpoints/model_epoch_0005_iter_080000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden32_anyrot_simocc_0/checkpoints/model_final.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden32_anyrot_simocc_10x10_0/checkpoints/model_epoch_0003_iter_051000.pth')

    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_3/checkpoints/model_epoch_0017_iter_349000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden64_latent_margin_0/checkpoints/model_epoch_0014_iter_221000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden64_latent_margin_4/checkpoints/model_epoch_0003_iter_051000.pth')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden64_latent_margin_0/checkpoints/model_epoch_0018_iter_271000.pth')

    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s4_7/checkpoints/model_epoch_0001_iter_060000.pth')
    model_path = osp.join(path_util.get_ndf_model_weights(), 'lndf_refined/2023-01-17_13H53M56S_Tue_conv_hidden_128_with_l2_r0p05_0/checkpoints/model_epoch_0001_iter_060000.pth')

    model.load_state_dict(torch.load(model_path))

    # RUN PLOTTER #
    tsne_plotter = TSNEViz(model)

    for object_fn in object_fns:
        tsne_plotter.load_object(object_fn, sample_bb=True)
        tsne_plotter.load_object(object_fn, sample_bb=False)

    # tsne_plotter.viz_all_objects(base_output_fn=base_output_fn, rand_rotate=False)
    # tsne_plotter.viz_all_objects(base_output_fn=base_output_fn, rand_rotate=True)

    tsne_plotter.viz_all_objects_together(base_output_fn + '.html',
        rand_rotate=True, num_repeats=8)

# # Look at globals
# id_to_check = 'fad118b32085f3f2c2c72e575af174cd'
# print('ID in bad mugs: ', id_to_check in globals.bad_shapenet_mug_ids_list)
# # print(globals.bad_shapenet_mug_ids_list)


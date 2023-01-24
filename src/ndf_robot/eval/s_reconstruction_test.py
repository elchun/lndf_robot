"""
Use select occupancy network to reconstruct mug

For debugging conv occupancy network
"""
import os, os.path as osp
from turtle import shape
from typing import no_type_check_decorator
import torch
import numpy as np
import trimesh
import random
import argparse
import copy
from scipy.spatial.transform import Rotation
import plotly.graph_objects as go
from ndf_robot.utils.plotly_save import plot3d

from ndf_robot.utils import path_util
import ndf_robot.model.vnn_occupancy_net.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
import ndf_robot.model.conv_occupancy_net.conv_occupancy_net as conv_occupancy_network


def make_cam_frame_scene_dict():
    """
    Generate a plotly frame scene dict for
    viewing reconstruction
    """
    cam_frame_scene_dict = {}
    cam_up_vec = [0, 1, 0]
    plotly_camera = {
        'up': {'x': cam_up_vec[0], 'y': cam_up_vec[1],'z': cam_up_vec[2]},
        'center': {'x': 0, 'y': 0, 'z': 0},
        'eye': {'x': -0.6, 'y': -0.6, 'z': 0.4},
    }

    plotly_scene = {
        'xaxis':
            {
                'backgroundcolor': 'rgb(255, 255, 255)',
                'gridcolor': 'white',
                'zerolinecolor': 'white',
                'tickcolor': 'rgb(255, 255, 255)',
                'showticklabels': False,
                'showbackground': False,
                'showaxeslabels': False,
                'visible': False,
                'range': [-0.5, 0.5]},
        'yaxis':
            {
                'backgroundcolor': 'rgb(255, 255, 255)',
                'gridcolor': 'white',
                'zerolinecolor': 'white',
                'tickcolor': 'rgb(255, 255, 255)',
                'showticklabels': False,
                'showbackground': False,
                'showaxeslabels': False,
                'visible': False,
                'range': [-0.5, 0.5]},
        'zaxis':
            {
                'backgroundcolor': 'rgb(255, 255, 255)',
                'gridcolor': 'white',
                'zerolinecolor': 'white',
                'tickcolor': 'rgb(255, 255, 255)',
                'showticklabels': False,
                'showbackground': False,
                'showaxeslabels': False,
                'visible': False,
                'range': [-0.5, 0.5]},
    }
    cam_frame_scene_dict['camera'] = plotly_camera
    cam_frame_scene_dict['scene'] = plotly_scene

    return cam_frame_scene_dict

def plotly_create_local_frame(transform=None, length=0.03):
    """???"""
    if transform is None:
        transform = np.eye(4)

    x_vec = transform[:-1, 0] * length
    y_vec = transform[:-1, 1] * length
    z_vec = transform[:-1, 2] * length

    origin = transform[:-1, -1]

    lw = 8
    x_data = go.Scatter3d(
        x=[origin[0], x_vec[0] + origin[0]], y=[origin[1], x_vec[1] + origin[1]], z=[origin[2], x_vec[2] + origin[2]],
        line=dict(
            color='red',
            width=lw
        ),
        marker=dict(
            size=0.0
        )
    )
    y_data = go.Scatter3d(
        x=[origin[0], y_vec[0] + origin[0]], y=[origin[1], y_vec[1] + origin[1]], z=[origin[2], y_vec[2] + origin[2]],
        line=dict(
            color='green',
            width=lw
        ),
        marker=dict(
            size=0.0
        )
    )
    z_data = go.Scatter3d(
        x=[origin[0], z_vec[0] + origin[0]], y=[origin[1], z_vec[1] + origin[1]], z=[origin[2], z_vec[2] + origin[2]],
        line=dict(
            color='blue',
            width=lw
        ),
        marker=dict(
            size=0.0
        )
    )
    # fig = go.Figure(data=[x_data, y_data, z_data])
    # fig.show()

    data = [x_data, y_data, z_data]
    return data


if __name__ == '__main__':
    ### ARG PARSING ###
    parser  = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show_recon', action='store_true')
    parser.add_argument('--random_rot', action='store_true', help='Apply random rotation to object')
    args = parser.parse_args()


    ### INIT ###
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)


    ### LOAD OBJECTS ###
    # see the demo object descriptions folder for other object models you can try
    # obj_model = osp.join(path_util.get_ndf_demo_obj_descriptions(), 'mug_centered_obj_normalized/28f1e7bc572a633cb9946438ed40eeb9/models/model_normalized.obj')
    # obj_model = osp.join(path_util.get_ndf_demo_obj_descriptions(), 'mug_centered_obj_normalized/1c3fccb84f1eeb97a3d0a41d6c77ec7c/models/model_normalized.obj')
    # obj_model = osp.join(path_util.get_ndf_demo_obj_descriptions(), 'mug_centered_obj_normalized/1c9f9e25c654cbca3c71bf3f4dd78475/models/model_normalized.obj') # May be train data

    # TEST DATA
    # obj_model = osp.join(path_util.get_ndf_demo_obj_descriptions(), 'mug_centered_obj_normalized/d75af64aa166c24eacbe2257d0988c9c/models/model_normalized.obj')
    # obj_model = osp.join(path_util.get_ndf_demo_obj_descriptions(), 'mug_centered_obj_normalized/daee5cf285b8d210eeb8d422649e5f2b/models/model_normalized.obj')
    # obj_model = osp.join(path_util.get_ndf_demo_obj_descriptions(), 'mug_centered_obj_normalized/e984fd7e97c2be347eaeab1f0c9120b7/models/model_normalized.obj')
    obj_model = osp.join(path_util.get_ndf_demo_obj_descriptions(), 'mug_centered_obj_normalized/f7d776fd68b126f23b67070c4a034f08/models/model_normalized.obj')
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_demo_mug_weights.pth')

    # CONV WEIGHTS
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_exp_archive/checkpoints/model_epoch_0010_iter_074720.pth')  # Looks sort of fine
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_exp_archive/checkpoints/model_epoch_0015_iter_112080.pth')  # Looks eh
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_exp_archive/checkpoints/model_epoch_0020_iter_149500.pth')  # Looks sort of fine <-- Lets go with 20 for now
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_exp_archive/checkpoints/model_epoch_0040_iter_298880.pth')  # Looks fine
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_exp_archive/checkpoints/model_epoch_0099_iter_747100.pth')  # Looks fine

    # VNN WEIGHTS
    # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/vnn_occ_exp/checkpoints/model_epoch_0003_iter_026700.pth')




    ### INIT OBJECTS ###
    scale1 = 0.25
    scale2 = 0.4
    obj_mesh = trimesh.load(obj_model, process=False)
    obj_mesh.apply_scale(scale1)



    # apply a random initial rotation to the new shape
    if args.random_rot:
        quat = np.random.random(4)
        quat = quat / np.linalg.norm(quat)
        rot = np.eye(4)
        rot[:-1, :-1] = Rotation.from_quat(quat).as_matrix()
        obj_mesh.apply_transform(rot)



    ### GENERATE POINTCLOUD ###
    obj_pcd = obj_mesh.sample(5000)

    # Mean center pointcloud
    obj_pcd = obj_pcd - np.mean(obj_pcd, axis=0)

    ### INIT MODEL ###
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')


    # model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True, sigmoid=True).cuda()
    # model = conv_occupancy_network.ConvolutionalOccupancyNetwork(latent_dim=32, model_type='pointnet', return_features=False, sigmoid=True).cuda()
    model = conv_occupancy_network.ConvolutionalOccupancyNetwork(latent_dim=32, model_type='pointnet', return_features=True, sigmoid=False).cuda()
    # model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=False, sigmoid=True).cuda()
    # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5])
    model.load_state_dict(torch.load(model_path))


    ### PROCESS POINTS ###
    n_pts = 1500


    ### PREDICT REFERENCE SHAPE OCC ###
    ref_shape_pcd = torch.from_numpy(obj_pcd[:n_pts]).float().to(device)
    ref_pcd = ref_shape_pcd[None, :, :]

    # Get bounding box
    shape_np = obj_pcd
    assert len(shape_np.shape) == 2, 'expected pcd to be have two dimensions'
    assert shape_np.shape[-1] == 3, 'expected points to be 3d'
    pcd_mean = np.mean(shape_np, axis=0)
    inliers = np.where(np.linalg.norm(shape_np - pcd_mean, 2, 1) < 0.2)[0]
    shape_np = shape_np[inliers]

    shape_pcd = trimesh.PointCloud(shape_np)
    ref_bb = shape_pcd.bounding_box

    # Get eval points
    eval_pts = ref_bb.sample_volume(100000)

    shape_mi = {}
    shape_mi['point_cloud'] = ref_pcd
    shape_mi['coords'] = torch.from_numpy(eval_pts)[None, :, :].float().to(device).detach()
    out = model(shape_mi)

    thresh = 0.1
    in_inds = torch.where(out['occ'].squeeze() > thresh)[0].cpu().numpy()
    out_inds = torch.where(out['occ'].squeeze() < thresh)[0].cpu().numpy()

    in_pts = eval_pts[in_inds]
    out_pts = eval_pts[out_inds]


    ### SAVE VISUALIZATION ###
    cam_frame_scene_dict = make_cam_frame_scene_dict()

    viz_path = 'visualization'
    if not osp.exists(viz_path):
        os.makedirs(viz_path)

    viz_fn = osp.join(viz_path, "recon_test.html")
    print(f'Saving visualization to: {viz_fn}')
    plot3d(
        [in_pts, shape_np],
        ['blue', 'black'],
        viz_fn,
        scene_dict=cam_frame_scene_dict,
        z_plane=False,
        pts_label_list=['in_pts', 'shape_np'])

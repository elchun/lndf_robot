import os.path as osp

import random
import numpy as np
import torch
from torch.nn import functional as F
import trimesh

from scipy.spatial.transform import Rotation as R
import plotly.express as px

from ndf_robot.utils import path_util, util
from ndf_robot.utils.plotly_save import multiplot

import ndf_robot.model.vnn_occupancy_net.vnn_occupancy_net_pointnet_dgcnn \
    as vnn_occupancy_network
import ndf_robot.model.conv_occupancy_net.conv_occupancy_net \
    as conv_occupancy_network

import plotly.express as px
import plotly.graph_objects as go

def get_activations(pcd, query, model):
    """
    Get activations of pcd and query points when passed into model.

    Args:
        pcd (np.ndarray): (n, 3)
        query (np.ndarray): (k, 3)

    Returns:
        np.ndarray: (n, z) where z is the length of activations.
    """

    if torch.cuda.is_available():
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    model_input = {}

    query = torch.from_numpy(query).float().to(dev)
    pcd = torch.from_numpy(pcd).float().to(dev)

    model_input['coords'] = query[None, :, :]
    model_input['point_cloud'] = pcd[None, :, :]
    latent = model.extract_latent(model_input).detach()

    act_torch = model.forward_latent(latent, model_input['coords']).detach()
    act = act_torch.squeeze().cpu().numpy()

    return act

def add_plane(fig, normal_ax: str, x_extents: tuple, y_extents: tuple, z_extents: tuple,
    axis_loc: float, color: np.ndarray):
    if normal_ax == 'z':
        x = np.linspace(x_extents[0], x_extents[1], 20)
        y = np.linspace(y_extents[0], y_extents[1], 20)
        z = axis_loc * np.ones((20, 20))
    if normal_ax == 'x':
        x = np.linspace(axis_loc, axis_loc, 20)
        y = np.linspace(y_extents[0], y_extents[1], 20)
        z = np.repeat(np.linspace(z_extents[0], z_extents[1], 20).reshape(1, 20), 20, axis=0)

    if normal_ax == 'y':
        x = np.linspace(x_extents[0], x_extents[1], 20)
        y = np.linspace(axis_loc, axis_loc, 20)
        z = np.repeat(np.linspace(z_extents[0], z_extents[1], 20).reshape(20, 1), 20, axis=1)

    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=color,  showscale=False))

    return fig

def make_voxel(fig, voxel_min_coords, voxel_max_coords, color):
    add_plane(fig, 'z', (voxel_min_coords[0], voxel_max_coords[0]),
        (voxel_min_coords[1], voxel_max_coords[1]),
        (voxel_min_coords[2], voxel_max_coords[2]), voxel_min_coords[2], color)
    add_plane(fig, 'z', (voxel_min_coords[0], voxel_max_coords[0]),
        (voxel_min_coords[1], voxel_max_coords[1]),
        (voxel_min_coords[2], voxel_max_coords[2]), voxel_max_coords[2], color)
    add_plane(fig, 'x', (voxel_min_coords[0], voxel_max_coords[0]),
        (voxel_min_coords[1], voxel_max_coords[1]),
        (voxel_min_coords[2], voxel_max_coords[2]), voxel_min_coords[0], color)
    add_plane(fig, 'x', (voxel_min_coords[0], voxel_max_coords[0]),
        (voxel_min_coords[1], voxel_max_coords[1]),
        (voxel_min_coords[2], voxel_max_coords[2]), voxel_max_coords[0], color)
    add_plane(fig, 'y', (voxel_min_coords[0], voxel_max_coords[0]),
        (voxel_min_coords[1], voxel_max_coords[1]),
        (voxel_min_coords[2], voxel_max_coords[2]), voxel_min_coords[1], color)
    add_plane(fig, 'y', (voxel_min_coords[0], voxel_max_coords[0]),
        (voxel_min_coords[1], voxel_max_coords[1]),
        (voxel_min_coords[2], voxel_max_coords[2]), voxel_max_coords[1], color)
    return fig


def plot_grid(fig, grid_start, grid_spacing, n_voxels, line_color, point_color):
    # color = 'rgba(100, 0, 0, 0.2)'
    n_line_per_side = n_voxels + 1
    # grid range: (3, )
    # grid start: (3, )

    # Plot in z
    for idx1 in range(n_line_per_side):
        for idx2 in range(n_line_per_side):
            x = (grid_start[0] + idx1 * grid_spacing) * np.ones((n_line_per_side,))
            y = (grid_start[1] + idx2 * grid_spacing) * np.ones((n_line_per_side,))
            z = grid_start[2] + np.arange(0, n_line_per_side) * grid_spacing

            # fig.add_trace(go.Scatter3d(x=np.array([0, 0.1]), y=np.array([0, 0.1]), z = np.array([0, 0.1]), mode='lines+markers', line=dict(color=sample_pt_color, width=10)))
            # fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines+markers', line=dict(color=line_color, width=4), marker=dict(color=point_color, size=6)))
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=line_color, width=6), marker=dict(color=point_color, size=6)))

            z = (grid_start[2] + idx1 * grid_spacing) * np.ones((n_line_per_side,))
            x = (grid_start[0] + idx2 * grid_spacing) * np.ones((n_line_per_side,))
            y = grid_start[1] + np.arange(0, n_line_per_side) * grid_spacing

            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=line_color, width=6), marker=dict(color=point_color, size=6)))

            y = (grid_start[1] + idx1 * grid_spacing) * np.ones((n_line_per_side,))
            z = (grid_start[2] + idx2 * grid_spacing) * np.ones((n_line_per_side,))
            x = grid_start[0] + np.arange(0, n_line_per_side) * grid_spacing

            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=line_color, width=6), marker=dict(color=point_color, size=6)))

    return fig


if __name__ == '__main__':

    # seed = 0
    seed = 6
    # seed = 2

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    use_conv = True
    # use_conv = False

    if torch.cuda.is_available():
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    # -- Set up model -- #
    model = conv_occupancy_network.ConvolutionalOccupancyNetwork(latent_dim=128,
        model_type='pointnet', return_features=True, sigmoid=False, acts='last').cuda()
    model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s4_7/checkpoints/model_epoch_0001_iter_060000.pth')
    model.load_state_dict(torch.load(model_path))

    # -- Load and apply demo object -- #
    # see the demo object descriptions folder for other object models you can try
    mug_std = osp.join(path_util.get_ndf_obj_descriptions(), 'mug_std_centered_obj_normalized/28f1e7bc572a633cb9946438ed40eeb9/models/model_normalized.obj')
    mug2_std = osp.join(path_util.get_ndf_obj_descriptions(), 'mug_std_centered_obj_normalized/7a8ea24474846c5c2f23d8349a133d2b/models/model_normalized.obj')
    bottle_std = osp.join(path_util.get_ndf_obj_descriptions(), 'bottle_std_centered_obj_normalized/f4851a2835228377e101b7546e3ee8a7/models/model_normalized.obj')
    bottle_handle_std = osp.join(path_util.get_ndf_obj_descriptions(), 'bottle_handle_std_centered_obj_normalized/e8b48d395d3d8744e53e6e0633163da8-h/models/model_normalized.obj')
    # obj_model = osp.join(path_util.get_ndf_obj_descriptions(), 'bottle_centered_obj_normalized/e593aa021f3fa324530647fc03dd20dc/models/model_normalized.obj')
    bowl_handle_std = osp.join(path_util.get_ndf_obj_descriptions(), 'bowl_handle_std_centered_obj_normalized/34875f8448f98813a2c59a4d90e63212-h/models/model_normalized.obj')
    bowl_handle2_std = osp.join(path_util.get_ndf_obj_descriptions(), 'bowl_handle_std_centered_obj_normalized/2c1df84ec01cea4e525b133235812833-h/models/model_normalized.obj')


    scale = 1.0
    bottle_std = trimesh.load(bottle_std, process=False)
    bottle_std.apply_scale(scale)

    scale = 1.0
    bottle_handle_std = trimesh.load(bottle_handle_std, process=False)
    bottle_handle_std.apply_scale(scale)

    scale = 1.0
    mug_std = trimesh.load(mug_std, process=False)
    mug_std.apply_scale(scale)

    scale = 1.0
    mug2_std = trimesh.load(mug2_std, process=False)
    mug2_std.apply_scale(scale)

    scale = 1.0
    bowl_handle_std = trimesh.load(bowl_handle_std, process=False)
    bowl_handle_std.apply_scale(scale)

    scale = 1.0
    bowl_handle2_std = trimesh.load(bowl_handle2_std, process=False)
    bowl_handle2_std.apply_scale(scale)

    # object_list = [bottle_std, bottle_std, bottle_handle_std, bottle_handle_std]
    # object_list = [mug_std, bowl_handle_std, bottle_handle_std, mug2_std]
    object_list = [mug_std, bowl_handle_std, bowl_handle2_std, mug2_std]

    # sample_pts = mesh1.sample(n_samples)

    # # -- Make input upright -- #
    # for i in range(len(object_list)):
    #     rot1 = np.eye(4)
    #     rot1[:3, :3] = util.make_rotation_matrix('x', np.pi / 2)

    #     object_list[i].apply_transform(rot1)

    upright_transform = np.eye(4)
    upright_transform[:3, :3] = util.make_rotation_matrix('x', np.pi / 2)

    obj_sample_pt_list = [
        np.array([[0.0, 0.08, 0.07]]),
        np.array([[0.00, -0.13, 0.05]]),
        np.array([[0.00, -0.13, 0.04]]),
        # np.array([[-0.02, -0.09, 0.06]]),
        np.array([[0.0, 0.10, 0.06]])
    ]

    # obj_sample_pt_list = [
    #     np.array([[-0.02, 0, 0.08]]),
    #     np.array([[-0.02, 0, 0.08]]),
    #     np.array([[-0.02, 0, 0.08]]),
    #     np.array([[-0.02, 0, 0.08]]),
    # ]

    # For mug
    # sample_pt = np.array([[0.0, 0.36, 0.10]])
    # sample_pt = np.array([[0.0, 0.27, 0.10]])
    # sample_pt = np.array([[0.00, 0.10, 0.05]])

    # For bottle
    # sample_pt = np.array([[-0.02, 0, 0.08]]),

    # For bowl handle
    # sample_pt = np.array([[0.00, -0.13, 0.05]])

    # x is the axis to rotated by

    pcd_list = []
    color_list = []
    sample_pt_list = []

    n_rot = 4
    n_pts = 1000
    rots = [0, np.pi/2, np.pi, 3 * np.pi / 2]
    # rots = [np.pi/2, np.pi/2, np.pi, 3 * np.pi / 2]
    first_ref_act = None
    for i in range(n_rot):
        working_mesh = object_list[i].copy()
        sample_pt = obj_sample_pt_list[i]
        # Want 0 rot to look upright
        angle = rots[i]
        if angle != 0:
            angle += random.random() * np.pi/8
        rot = np.eye(4)
        rot[:3, :3] = util.make_rotation_matrix('x', angle)
        working_mesh.apply_transform(upright_transform)
        working_mesh.apply_transform(rot)
        rot_sample_pt = util.apply_pose_numpy(sample_pt, util.pose_stamped2list(util.pose_from_matrix(rot)))

        pcd_whole = np.array(working_mesh.sample(4 * n_pts))
        pcd = pcd_whole[pcd_whole[:, 0] < 0, :][:n_pts, :]

        # Add offset so we can view side by side
        y_offset = -0.5
        pcd_list.append(pcd + np.array([0, y_offset * i, 0]))
        sample_pt_list.append(rot_sample_pt + np.array([0, y_offset * i, 0]))

        # -- Debug plot distances -- #
        # distances = np.sqrt(((pcd - rot_sample_pt)**2).sum(axis=1))
        # color = 1 / (distances + 0.1)

        # -- Plot activation similarity -- #
        # ref_act = get_activations(pcd_whole, rot_sample_pt, model)
        # ref_act = ref_act[None, :]
        # ref_act = np.repeat(ref_act, n_pts, axis=0)

        if first_ref_act is None:
            ref_act = get_activations(pcd_whole, rot_sample_pt, model)
            ref_act = ref_act[None, :]
            ref_act = np.repeat(ref_act, n_pts, axis=0)
            first_ref_act = ref_act
        else:
            ref_act = first_ref_act

        acts = get_activations(pcd_whole, pcd, model)

        # print(ref_act)
        # print('---\n\n---')
        # print(acts)

        cor = F.cosine_similarity(torch.from_numpy(acts).float().to(dev),
            torch.from_numpy(ref_act).float().to(dev), dim=1)

        cor = cor.cpu().numpy()
        color = cor

        color_list.append(color)

    scale_color_pt = np.array([[00, 5.5, 0]])
    scale_color_color = np.array([1.4])

    # So that we don't have bright yellow.  Could change scale too but this is
    # faster
    pcd_list.append(scale_color_pt)
    color_list.append(scale_color_color)

    pcd = np.vstack(pcd_list)
    color = np.concatenate(color_list, axis=0)

    print(color)

    min_n_voxel = 4

    # sample_pt = np.array([[-0.010, 0.020, 0.117]])
    # sample_pt = np.array([[0.004, 0.2905, 0.2454]])
    # sample_pt = np.array([[-0.20, 0.04, 0.29]])

    # distances = np.sqrt(((pcd - sample_pt)**2).sum(axis=1))
    # color = 1 / (distances + 0.1)

    # --- Plot pcd -- #
    fig = px.scatter_3d(
        x=pcd[:, 0], y=pcd[:, 1], z=pcd[:, 2], color=color, opacity=0.5
    )

    # https://plotly.com/python/3d-axes/
    fig.update_layout(scene_aspectmode='data')

    # min_pts = pcd.min(axis=0)
    # max_pts = pcd.max(axis=0)
    # min_range = min(max_pts - min_pts)
    # voxel_width = min_range / min_n_voxel

    # print(voxel_width)

    # sample_voxel_idx = (sample_pt - min_pts) // voxel_width
    # print(sample_voxel_idx)

    # voxel_min_coords = (sample_voxel_idx * voxel_width + min_pts).flatten()
    # voxel_max_coords = ((sample_voxel_idx + 1) * voxel_width + min_pts).flatten()

    # print(voxel_min_coords, voxel_max_coords)

    color = np.zeros(pcd.shape[0])

    # --- Plot pcd -- #
    # fig = px.scatter_3d(
    #     x=pcd[:, 0], y=pcd[:, 1], z=pcd[:, 2], color=color
    # )
    # fig.update_traces(marker_color='rgba(50, 50, 50, 0.4)', selector=dict(type='scatter3d'))
    # fig.update_traces(marker_color='rgba(50, 50, 50, 0.0)', selector=dict(type='scatter3d'))

    # -- Plot sample pt -- #
    # sample_pt_color = 'rgba(21, 49, 140, 1.0)'
    # sample_pt_color = 'rgba(135, 206, 250, 1.0)'
    # sample_pt_color = 'rgba(248, 230, 216, 1.0)'
    # sample_pt_color = 'rgba(200, 100, 100, 1.0)'

    sample_pt_color = 'rgba(255, 106, 0, 1.0)'
    for sample_pt in sample_pt_list:
        fig.add_trace(go.Scatter3d(x=sample_pt[:, 0], y=sample_pt[:, 1], z =sample_pt[:, 2], mode='markers', marker=dict(color=sample_pt_color, size=20)))

    # # https://stackoverflow.com/questions/62403763/how-to-add-planes-in-a-3d-scatter-plot
    # fig.add_trace(go.Surface())


    # -- Plot voxel -- #
    # bright_blue = [[0, '#7DF9FF'], [1, '#7DF9FF']]
    # bright_pink = [[0, '#FF007F'], [1, '#FF007F']]
    color1 = 'rgba(135, 206, 250, 0.2)'
    # color1 = 'rgba(255, 106, 0, 0.3)'
    light_blue = [[0, color1], [1, color1]]

    # make_voxel(fig, voxel_min_coords, voxel_max_coords, light_blue)

    # -- Plot grid -- #
    n_plot_voxel = 4

    min_pts = pcd.min(axis=0)
    max_pts = pcd.max(axis=0)
    mean_pts = pcd.mean(axis=0).flatten()
    v_range = (max_pts - min_pts).flatten()
    max_range = max(max_pts - min_pts)
    voxel_width = max_range / n_plot_voxel
    voxel_start = mean_pts - max_range / 2

    print(voxel_width)

    # https://stackoverflow.com/questions/70155529/how-to-plot-a-3d-line-using-plotly-graph-objects
    # fig.add_trace(go.Scatter3d(x=np.array([0, 0.1]), y=np.array([0, 0.1]), z = np.array([0, 0.1]), mode='lines+markers', line=dict(color=sample_pt_color, width=10)))

    grid_color = 'rgba(80, 80, 80, 0.5)'
    point_color = 'rgba(135, 206, 250, 1.0)'
    # voxel_pt_color = 'rgba(200, 100, 100, 0.2)'
    voxel_pt_color = 'rgba(135, 206, 250, 0.2)'
    voxel_pt_color = [[0, voxel_pt_color], [1, voxel_pt_color]]
    # 238	205	205

    # 248	230	216
    # plot_grid(fig, voxel_start, voxel_width, n_plot_voxel, grid_color, point_color)

    sample_voxel_idx = (sample_pt - voxel_start) // voxel_width
    print(sample_voxel_idx)

    voxel_min_coords = (sample_voxel_idx * voxel_width + voxel_start).flatten()
    voxel_max_coords = ((sample_voxel_idx + 1) * voxel_width + voxel_start).flatten()

    # make_voxel(fig, voxel_min_coords, voxel_max_coords, voxel_pt_color)



    # -- Reference -- #

    # add_plane(fig, 'z', (voxel_min_coords[0], voxel_max_coords[0]),  (voxel_min_coords[1], voxel_max_coords[1]), (voxel_min_coords[2], voxel_max_coords[2]), voxel_min_coords[2], light_yellow)
    # add_plane(fig, 'z', (voxel_min_coords[0], voxel_max_coords[0]),  (voxel_min_coords[1], voxel_max_coords[1]), (voxel_min_coords[2], voxel_max_coords[2]), voxel_max_coords[2], light_yellow)
    # add_plane(fig, 'x', (voxel_min_coords[0], voxel_max_coords[0]),  (voxel_min_coords[1], voxel_max_coords[1]), (voxel_min_coords[2], voxel_max_coords[2]), voxel_min_coords[0], light_yellow)
    # add_plane(fig, 'x', (voxel_min_coords[0], voxel_max_coords[0]),  (voxel_min_coords[1], voxel_max_coords[1]), (voxel_min_coords[2], voxel_max_coords[2]), voxel_max_coords[0], light_yellow)
    # add_plane(fig, 'y', (voxel_min_coords[0], voxel_max_coords[0]),  (voxel_min_coords[1], voxel_max_coords[1]), (voxel_min_coords[2], voxel_max_coords[2]), voxel_min_coords[1], light_yellow)
    # add_plane(fig, 'y', (voxel_min_coords[0], voxel_max_coords[0]),  (voxel_min_coords[1], voxel_max_coords[1]), (voxel_min_coords[2], voxel_max_coords[2]), voxel_max_coords[1], light_yellow)

    # add_plane(fig, 'z', (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), 0, light_yellow)
    # add_plane(fig, 'x', (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), 0, light_yellow)
    # add_plane(fig, 'y', (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), 0, light_yellow)

    # -- Hide all axis -- #
    # https://plotly.com/python/3d-axes/
    # https://stackoverflow.com/questions/61693014/how-to-hide-plotly-yaxis-title-in-python
    fig.update_layout(scene = dict(
        xaxis = dict(
            gridcolor="white",
            showbackground=False,
            # showticklabels = False,
            visible = False,
            zerolinecolor="white",),
        yaxis = dict(
            gridcolor="white",
            showbackground=False,
            # showticklabels = False,
            visible = False,
            zerolinecolor="white"),
        zaxis = dict(
            # backgroundcolor="rgb(230, 230,200)",
            backgroundcolor='white',
            gridcolor="white",
            showbackground=False,
            # showticklabels = False,
            visible = False,
            zerolinecolor="white",),),
    )


    fname = osp.join(path_util.get_ndf_eval(), 'debug_viz', 'debug_equivar_fig.html')

    fig.write_html(fname)

import os.path as osp

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
    n_samples = 1000

    if torch.cuda.is_available():
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    # see the demo object descriptions folder for other object models you can try
    obj_model = osp.join(path_util.get_ndf_demo_obj_descriptions(), 'mug_centered_obj_normalized/28f1e7bc572a633cb9946438ed40eeb9/models/model_normalized.obj')
    # obj_model = osp.join(path_util.get_ndf_obj_descriptions(), 'bottle_std_centered_obj_normalized/f4851a2835228377e101b7546e3ee8a7/models/model_normalized.obj')
    # obj_model = osp.join(path_util.get_ndf_obj_descriptions(), 'bottle_centered_obj_normalized/e593aa021f3fa324530647fc03dd20dc/models/model_normalized.obj')

    scale = 1.0
    mesh = trimesh.load(obj_model, process=False)
    mesh.apply_scale(scale)

    # sample_pts = mesh1.sample(n_samples)

    # Make mesh 1 upright
    rot1 = np.eye(4)
    rot1[:3, :3] = util.make_rotation_matrix('x', np.pi / 2)
    # rot1 = np.eye(4)
    # rot1[:3, :3] = R.random().as_matrix()
    mesh.apply_transform(rot1)

    rot2 = np.eye(4)
    rot2[:3, :3] = R.random().as_matrix()
    # mesh.apply_transform(rot2)

    min_n_voxel = 4

    pcd = np.array(mesh.sample(1000))
    pcd = pcd[pcd[:, 0] < 0, :]
    # sample_pt = np.array([[-0.010, 0.020, 0.117]])
    # sample_pt = np.array([[0.004, 0.2905, 0.2454]])
    # sample_pt = np.array([[-0.20, 0.04, 0.29]])
    sample_pt = np.array([[-0.20, 0.01, -0.04]])

    distances = np.sqrt(((pcd - sample_pt)**2).sum(axis=1))
    color = 1 / (distances + 0.1)

    # --- Plot pcd -- #
    fig = px.scatter_3d(
        x=pcd[:, 0], y=pcd[:, 1], z=pcd[:, 2], color=color, opacity=0.5
    )
    # fig.update_traces(marker_color='rgba(50, 50, 50, 0.4)', selector=dict(type='scatter3d'))


    print(distances.shape)

    min_pts = pcd.min(axis=0)
    max_pts = pcd.max(axis=0)
    min_range = min(max_pts - min_pts)
    voxel_width = min_range / min_n_voxel

    print(voxel_width)

    sample_voxel_idx = (sample_pt - min_pts) // voxel_width
    print(sample_voxel_idx)

    voxel_min_coords = (sample_voxel_idx * voxel_width + min_pts).flatten()
    voxel_max_coords = ((sample_voxel_idx + 1) * voxel_width + min_pts).flatten()

    print(voxel_min_coords, voxel_max_coords)



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
    sample_pt_color = 'rgba(200, 100, 100, 1.0)'

    sample_pt_color = 'rgba(255, 106, 0, 1.0)'
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


    fname = osp.join(path_util.get_ndf_eval(), 'debug_viz', 'debug_dis_fig.html')

    fig.write_html(fname)

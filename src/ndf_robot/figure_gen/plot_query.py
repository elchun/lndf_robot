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
from ndf_robot.eval.query_points import QueryPoints

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


def get_query_pts(q_type: str, n_pts: int = 1000):
    """
    Get query points of q_type

    Args:
        q_type (str): 'rack', 'shelf', 'gripper'.
        n_pts (int): Number of query points to sample.
    """
    if q_type == 'rack':
        q_args = dict(
            n_pts=n_pts,
            radius=0.05,
            height=0.04,
            y_rot_rad=0.68,
            x_trans=0.055,
            y_trans=0,
            z_trans=0.19,
        )
        return QueryPoints.generate_rack_arm(**q_args)
    elif q_type == 'shelf':
        q_args = dict(
            n_pts=n_pts,
            radius=0.06,
            height=0.10,
            y_rot_rad=0.0,
            x_trans=0.0,
            y_trans=0.07,
            z_trans=0.08,
        )
        return QueryPoints.generate_shelf(**q_args)
    elif q_type == 'gripper':
        q_args = dict(
            n_pts=n_pts,
            x=0.06,
            y=0.04,
            z1=0.05,
            z2=0.02,
        )
        return QueryPoints.generate_rect(**q_args)

if __name__ == '__main__':

    # seed = 0
    seed = 6
    # seed = 2

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    n_pts = 1000

    if torch.cuda.is_available():
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    pcd = get_query_pts('gripper', n_pts)
    color = np.ones(pcd.shape[0])

    # --- Plot pcd -- #
    fig = px.scatter_3d(
        x=pcd[:, 0], y=pcd[:, 1], z=pcd[:, 2], color=color, opacity=0.5
    )
    fig.update_traces(marker_color='rgba(50, 50, 50, 0.4)', selector=dict(type='scatter3d'))

    color1 = 'rgba(135, 206, 250, 0.2)'
    light_blue = [[0, color1], [1, color1]]


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

    fname = osp.join(path_util.get_ndf_eval(), 'debug_viz', 'query_fig.html')

    fig.write_html(fname)

import numpy as np
import trimesh

from ndf_robot.utils import util


class QueryPoints():
    """
    Container class for generating different shaped query points.
    """

    @staticmethod
    def generate_sphere(n_pts: int, radius: float=0.05) -> np.ndarray:
        """
        Sample points inside sphere centered at origin with radius {radius}

        Args:
            n_pts (int): Number of point to sample.
            radius (float, optional): Radius of sphere to sample.
                Defaults to 0.05.

        Returns:
            np.ndarray: (n_pts x 3) array of query points
        """
        # http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        u = 2 * np.random.rand(n_pts, 1) - 1
        phi = 2 * np.pi * np.random.rand(n_pts, 1)
        r = radius * (np.random.rand(n_pts, 1)**(1 / 3.))
        x = r * np.cos(phi) * (1 - u**2)**0.5
        y = r * np.sin(phi) * (1 - u**2)**0.5
        z = r * u

        sphere_points = np.hstack((x, y, z))
        return sphere_points

    @staticmethod
    def generate_rect(n_pts: int, x: float, y: float, z1: float, z2: float) \
        -> np.ndarray:
        """
        Create rectangle of query points with center of grip at 'o' and
        dimensions as shown.  'o' is centered in x and y.  All dims should be
        positive.

        With this system, z1 is closest to the gripper body while y is
        along the direction that the gripper fingers move.  It seems that
        a high z1 and lower z2 work well.

                   _________
                 /          /|
            ___ /_________ / |
             |  |         |  |
             z2 |         |  |
             |  |     o   |  |
            -+- | - -/    |  |
             |  |         |  |
             |  |         |  |  __
             z1 |         |  /  /
             |  |         | /  y
            _|_ |_________|/ _/_

                |----x----|

        Args:
            n_pts (int): Number of point to sample.
            x (float): x dim.
            y (float): y dim.
            z1 (float): z1 dim.
            z2 (float): z2 dim.

        Returns:
            np.ndarray: (n_pts x 3) array of query points
        """

        rect_points = np.random.rand(n_pts, 3)
        scale_mat = np.array(
            [[x, 0, 0],
             [0, y, 0],
             [0, 0, z1 + z2]]
        )
        offset_mat = np.array(
            [[x / 2, y / 2, z1]]
        )
        rect_points = rect_points @ scale_mat - offset_mat
        return rect_points

    @staticmethod
    def generate_cylinder(n_pts: int, radius: float, height: float, rot_axis='z') \
        -> np.ndarray:
        """
        Generate np array of {n_pts} 3d points with radius {radius} and
        height {height} in the shape of a cylinder with axis of rotation about
        {rot_axis} and points along the positive rot_axis.

        Args:
            n_pts (int): Number of points to generate.
            radius (float): Radius of cylinder.
            height (float): heigh tof cylinder.
            rot_axis (str, optional): Choose (x, y, z). Defaults to 'z'.

        Returns:
            np.ndarray: (n_pts, 3) array of points.
        """

        U_TH = np.random.rand(n_pts, 1)
        U_R = np.random.rand(n_pts, 1)
        U_Z = np.random.rand(n_pts, 1)
        X = radius * np.sqrt(U_R) * np.cos(2 * np.pi * U_TH)
        Y = radius * np.sqrt(U_R) * np.sin(2 * np.pi * U_TH)
        Z = height * U_Z
        # Z = np.zeros((n_pts, 1))

        points = np.hstack([X, Y, Z])
        rotate = np.eye(3)
        if rot_axis == 'x':
            rotate[[0, 2]] = rotate[[2, 0]]
        elif rot_axis == 'y':
            rotate[[1, 2]] = rotate[[2, 1]]

        points = points @ rotate
        return points

    @staticmethod
    def generate_rack_arm(n_pts: int, radius: float, height: float,
        y_rot_rad: float = 0.68, x_trans: float = 0.04,
        y_trans: float = 0, z_trans: float = 0.17) -> np.ndarray:
        """
        Generate points that align with arm on demo rack.

        Args:
            n_pts (int): Number of points in pcd.

        Returns:
            np.ndarray: (n_pts, 3).
        """
        # y_rot_rad = 0.68
        # x_trans = 0.04
        # y_trans = 0
        # z_trans = 0.17

        cylinder_pts = QueryPoints.generate_cylinder(n_pts, radius, height, 'z')
        transform = np.eye(4)
        rot = util.make_rotation_matrix('y', y_rot_rad)
        trans = np.array([[x_trans, y_trans, z_trans]]).T
        transform[:3, :3] = rot
        transform[:3, 3:4] = trans
        cylinder_pcd = trimesh.PointCloud(cylinder_pts)
        cylinder_pcd.apply_transform(transform)
        cylinder_pts = np.asarray(cylinder_pcd.vertices)

        return cylinder_pts

    @staticmethod
    def generate_shelf(n_pts: int, radius: float, height: float,
        y_rot_rad: float = 0, x_trans: float = 0,
        y_trans: float = 0.07, z_trans: float = 0.10) -> np.ndarray:
        """
        Generate points that align with demo shelf.

        Args:
            n_pts (int): Number of points in pcd.

        Returns:
            np.ndarray: (n_pts, 3).
        """
        # radius: 0.08
        # height: 0.03
        # y_rot_rad: 0.0
        # x_trans: 0.0
        # y_trans: 0.07
        # z_trans: 0.22

        cylinder_pts = QueryPoints.generate_cylinder(n_pts, radius, height, 'z')
        transform = np.eye(4)
        rot = util.make_rotation_matrix('y', y_rot_rad)
        trans = np.array([[x_trans, y_trans, z_trans]]).T
        transform[:3, :3] = rot
        transform[:3, 3:4] = trans
        cylinder_pcd = trimesh.PointCloud(cylinder_pts)
        cylinder_pcd.apply_transform(transform)
        cylinder_pts = np.asarray(cylinder_pcd.vertices)

        return cylinder_pts

    def generate_ndf_gripper(n_pts: int):
        """
        Get gripper used in original ndf paper (loaded from demo)

        Args:
            n_pts (int): Number of pts in pcd, will be capped at 500.

        Returns:
            np.ndarray: (n_pts, 3)
        """
        ref_fname = 'reference_query_points.npz'
        ref_query_pts = np.load(ref_fname, allow_pickle=True)
        return ref_query_pts['gripper'][:n_pts]

    def generate_ndf_rack(n_pts: int):
        """
        Get gripper used in original ndf paper (loaded from demo)

        Args:
            n_pts (int): Number of pts in pcd, will be capped at 500.

        Returns:
            np.ndarray: (n_pts, 3)
        """
        ref_fname = 'reference_query_points.npz'
        ref_query_pts = np.load(ref_fname, allow_pickle=True)
        return ref_query_pts['rack']

    def generate_ndf_shelf(n_pts: int):
        """
        Get gripper used in original ndf paper (loaded from demo)

        Args:
            n_pts (int): Number of pts in pcd, will be capped at 500.

        Returns:
            np.ndarray: (n_pts, 3)
        """
        ref_fname = 'reference_query_points.npz'
        ref_query_pts = np.load(ref_fname, allow_pickle=True)
        return ref_query_pts['shelf']

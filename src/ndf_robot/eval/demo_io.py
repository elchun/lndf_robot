import numpy as np

from numpy.lib.npyio import NpzFile
from ndf_robot.opt.optimizer_lite import Demo


class DemoIO():
    """Container class for converting data to standard format"""
    @staticmethod
    def process_grasp_data(data: NpzFile) -> Demo:
        """
        Construct {Demo} object from {data} representing grasp demo.

        Args:
            data (NpzFile): Imput data produced by simulation demo.

        Returns:
            Demo: Container for relevant demo information.
        """
        # -- Get obj pts in world coordinate frame -- #
        demo_obj_pts = data['obj_pcd_ori']
        demo_pts_mean = np.mean(demo_obj_pts, axis=0)
        inliers = np.where(
            np.linalg.norm(demo_obj_pts - demo_pts_mean, 2, 1) < 0.2)[0]
        demo_obj_pts = demo_obj_pts[inliers]

        demo = Demo(
            obj_pts=demo_obj_pts,
            query_pts=data['gripper_pts_uniform'],
            obj_pose_world=data['obj_pose_world'],
            # query_pose_world=data['gripper_contact_pose'],
            query_pose_world=data['ee_pose_world'],
            obj_shapenet_id=data['shapenet_id'].item())

        return demo

    @staticmethod
    def process_rack_place_data(data: NpzFile) -> Demo:
        """
        Construct {Demo} object from {data} representing place demo.

        Args:
            data (NpzFile): Imput data produced by simulation demo.

        Returns:
            Demo: Container for relevant demo information.
        """
        # -- Get obj pts -- #
        demo_obj_pts = data['obj_pcd_ori']
        demo_pts_mean = np.mean(demo_obj_pts, axis=0)
        inliers = np.where(
            np.linalg.norm(demo_obj_pts - demo_pts_mean, 2, 1) < 0.2)[0]
        demo_obj_pts = demo_obj_pts[inliers]

        demo = Demo(
            obj_pts=demo_obj_pts,
            query_pts=data['rack_pointcloud_gt'],
            obj_pose_world=data['obj_pose_world'],
            query_pose_world=data['rack_pose_world'],
            obj_shapenet_id=data['shapenet_id'].item())

        return demo

    @staticmethod
    def process_shelf_place_data(data: NpzFile) -> Demo:
        """
        Construct {Demo} object from {data} representing place demo.

        Args:
            data (NpzFile): Imput data produced by simulation demo.

        Returns:
            Demo: Container for relevant demo information.
        """
        # -- Get obj pts -- #
        demo_obj_pts = data['obj_pcd_ori']
        demo_pts_mean = np.mean(demo_obj_pts, axis=0)
        inliers = np.where(
            np.linalg.norm(demo_obj_pts - demo_pts_mean, 2, 1) < 0.2)[0]
        demo_obj_pts = demo_obj_pts[inliers]

        demo = Demo(
            obj_pts=demo_obj_pts,
            query_pts=data['shelf_pointcloud_uniform'],
            obj_pose_world=data['obj_pose_world'],
            query_pose_world=data['shelf_pose_world'],
            obj_shapenet_id=data['shapenet_id'].item())

        return demo

    @staticmethod
    def get_table_urdf(data: NpzFile) -> str:
        """
        Helper method to get table urdf from any demo object.

        Args:
            data (NpzFile): Any result from demo.

        Returns:
            str: Urdf of table.
        """
        return data['table_urdf'].item()

    @staticmethod
    def get_rack_pose(data: NpzFile) -> list:
        """
        Helper method to get rack pose from place demo.

        Args:
            data (NpzFile): Place demo

        Returns:
            list: Pose of rack so we can place in demo.
        """
        return data['rack_pose_world']

    @staticmethod
    def get_shelf_pose(data: NpzFile) -> list:
        """
        Helper method to get shelf pose from place demo.

        Args:
            data (NpzFile): Place demo with shelf pose

        Returns:
            list: Pose of shelf so we can place in demo.
        """
        return data['shelf_pose_world']

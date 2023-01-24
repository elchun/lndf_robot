import numpy as np
import torch
from torch.utils.data import Dataset
import random
import glob
import os.path as osp
from scipy.spatial.transform import Rotation
import pickle

from ndf_robot.utils import path_util, geometry, torch3d_util, torch_util


class JointOccTrainDataset(Dataset):
    def __init__(self, sidelength, depth_aug=False, multiview_aug=False,
        phase='train', obj_class='all', any_rot=False, neg_any_se3=False,
        trans_ratio=0.5):
        """
        Dataloader for object reconstruction

        Args:
            sidelength (_type_): _description_
            depth_aug (bool, optional): _description_. Defaults to False.
            multiview_aug (bool, optional): _description_. Defaults to False.
            phase (str, optional): _description_. Defaults to 'train'.
            obj_class (str, optional): _description_. Defaults to 'all'.
            any_rot (bool, optional): _description_. Defaults to False.
            neg_any_se3 (bool, optional): True to randomly translate objects too. Defaults to False.
        """

        # Path setup (change to folder where your training data is kept)
        #   these are the names of the full dataset folders
        mug_path = osp.join(path_util.get_ndf_data(),
            'training_data/mug_table_all_pose_4_cam_half_occ_full_rand_scale')
        bottle_path = osp.join(path_util.get_ndf_data(),
            'training_data/bottle_table_all_pose_4_cam_half_occ_full_rand_scale')
        bowl_path = osp.join(path_util.get_ndf_data(),
            'training_data/bowl_table_all_pose_4_cam_half_occ_full_rand_scale')
        # bottle_path = osp.join(path_util.get_ndf_data(),
        #     'training/bottle_table_all_pose_4_cam_half_occ_full_rand_scale')
        # bowl_path = osp.join(path_util.get_ndf_data(),
        #     'training/bowl_table_all_pose_4_cam_half_occ_full_rand_scale')

        # these are the names of the mini-dataset folders, to ensure everything
        #   is up and running
        # mug_path = osp.join(path_util.get_ndf_data(), 'training_data/test_mug')
        # bottle_path = osp.join(path_util.get_ndf_data(), 'training_data/test_bottle')
        # bowl_path = osp.join(path_util.get_ndf_data(), 'training_data/test_bowl')

        if obj_class == 'all':
            paths = [mug_path, bottle_path, bowl_path]
        else:
            paths = []
            if 'mug' in obj_class:
                paths.append(mug_path)
            if 'bowl' in obj_class:
                paths.append(bowl_path)
            if 'bottle' in obj_class:
                paths.append(bottle_path)

        print('---- \n Loading from paths: ', paths, '\n----')

        files_total = []
        for path in paths:
            files = list(sorted(glob.glob(path + "/*.npz")))
            n = len(files)
            idx = int(0.9 * n)

            if phase == 'train':
                files = files[:idx]
            else:
                files = files[idx:]

            files_total.extend(files)

        self.files = files_total

        self.sidelength = sidelength
        self.depth_aug = depth_aug
        self.multiview_aug = multiview_aug
        self.any_rot = any_rot
        self.neg_any_se3 = neg_any_se3
        self.trans_ratio = trans_ratio

        block = 128
        bs = 1 / block
        hbs = bs * 0.5
        self.bs = bs
        self.hbs = hbs

        self.shapenet_mug_dict = pickle.load(open(osp.join(path_util.get_ndf_data(),
            'training_data/occ_shapenet_mug.p'), 'rb'))
        self.shapenet_bowl_dict = pickle.load(open(osp.join(path_util.get_ndf_data(),
            'training_data/occ_shapenet_bowl.p'), "rb"))
        self.shapenet_bottle_dict = pickle.load(open(osp.join(path_util.get_ndf_data(),
            'training_data/occ_shapenet_bottle.p'), "rb"))

        self.shapenet_dict = {'03797390': self.shapenet_mug_dict,
            '02880940': self.shapenet_bowl_dict,
            '02876657': self.shapenet_bottle_dict}

        self.projection_mode = "perspective"

        self.cache_file = None
        self.count = 0

        print("files length ", len(self.files))

    def __len__(self):
        return len(self.files)

    def get_item(self, index):
        """
        Args:
            index (_type_): _description_

        Returns:
            res = {'point_cloud': point_cloud.float(),
                   'coords': coord.float(),
                   'intrinsics': intrinsics.float(),
                   'cam_poses': np.zeros(1)}  # cam poses not used
            'point_cloud' --> torch.tensor
            'coords' --> torch.tensor
            'intrinsics' --> torch.tensor
            return res, {'occ': torch.from_numpy(labels).float()}
        """
        try:
            data = np.load(self.files[index], allow_pickle=True)

            # legacy naming, used to use pose expressed in camera frame.
            # global reference frame doesn't matter though
            posecam = data['object_pose_cam_frame']

            # What is the numbers of transforms

            idxs = list(range(posecam.shape[0]))
            random.shuffle(idxs)
            select = random.randint(1, 4)

            if self.multiview_aug:
                idxs = idxs[:select]

            poses = []
            quats = []

            # print('idxs: ', idxs)
            for i in idxs:
                pos = posecam[i, :3]
                quat = posecam[i, 3:]

                poses.append(pos)
                quats.append(quat)

            shapenet_id = str(data['shapenet_id'].item())
            category_id = str(data['shapenet_category_id'].item())

            depths = []
            segs = []
            cam_poses = []

            for i in idxs:
                seg = data['object_segmentation'][i, 0]
                depth = data['depth_observation'][i]
                rix = np.random.permutation(depth.shape[0])[:1000]
                seg = seg[rix]
                depth = depth[rix]

                if self.depth_aug:
                    depth = depth + np.random.randn(*depth.shape) * 0.1

                segs.append(seg)
                depths.append(torch.from_numpy(depth))
                cam_poses.append(data['cam_pose_world'][i])

            # change these values depending on the intrinsic parameters of camera used to collect the data. These are what we used in pybullet
            y, x = torch.meshgrid(torch.arange(480), torch.arange(640))

            # Compute native intrinsic matrix
            sensor_half_width = 320
            sensor_half_height = 240

            vert_fov = 60 * np.pi / 180

            vert_f = sensor_half_height / np.tan(vert_fov / 2)
            hor_f = sensor_half_width / (np.tan(vert_fov / 2) * 320 / 240)

            intrinsics = np.array(
                [[hor_f, 0., sensor_half_width, 0.],
                [0., vert_f, sensor_half_height, 0.],
                [0., 0., 1., 0.]]
            )

            # Rescale to new sidelength
            intrinsics = torch.from_numpy(intrinsics)

            # build depth images from data
            dp_nps = []
            for i in range(len(segs)):
                seg_mask = segs[i]
                dp_np = geometry.lift(x.flatten()[seg_mask], y.flatten()[seg_mask], depths[i].flatten(), intrinsics[None, :, :])
                dp_np = torch.cat([dp_np, torch.ones_like(dp_np[..., :1])], dim=-1)
                dp_nps.append(dp_np)

            # load in voxel occupancy data
            voxel_path = osp.join(category_id, shapenet_id, 'models', 'model_normalized_128.mat')
            coord, voxel_bool, _ = self.shapenet_dict[category_id][voxel_path]

            rix = np.random.permutation(coord.shape[0])

            # print('Voxel bool: ', voxel_bool.shape)
            # print('Coord shape: ', coord.shape)
            # print('label_val: ', voxel_bool * 1)
            # l_debug = voxel_bool * 1
            # print('min: ', l_debug.min())
            # print('max: ', l_debug.max())
            # print('l sum: ', (l_debug).sum())

            # -- Coord selection -- #
            # Weight coords so half have positive gt occ, the other have neg
            n_in_pts = 750
            n_out_pts = 750

            flat_label = voxel_bool.flatten() * 1
            non_zero_idx = np.where(flat_label != 0)[0]
            zero_idx = np.where(flat_label == 0)[0]

            non_zero_idx = np.random.permutation(non_zero_idx).repeat(2)[:n_in_pts]
            zero_idx = np.random.permutation(zero_idx).repeat(2)[:n_out_pts]

            idx = np.hstack([non_zero_idx, zero_idx])

            coord = coord[idx]
            label = voxel_bool[idx]

            # print(coord.shape)
            # print(label.shape)

            # -- Old selection code -- #
            # coord = coord[rix[:1500]]
            # label = voxel_bool[rix[:1500]]

            offset = np.random.uniform(-self.hbs, self.hbs, coord.shape)
            coord = coord + offset
            coord = coord * data['mesh_scale']

            coord = torch.from_numpy(coord)

            # transform everything into the same frame
            transforms = []
            for quat, pos in zip(quats, poses):
                quat_list = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
                rotation_matrix = Rotation.from_quat(quat_list)
                rotation_matrix = rotation_matrix.as_matrix()

                transform = np.eye(4)
                transform[:3, :3] = rotation_matrix
                transform[:3, -1] = pos
                transform = torch.from_numpy(transform)
                transforms.append(transform)

            # print('transforms: ', transforms)

            # Why take the first transform?
            transform = transforms[0]
            coord = torch.cat([coord, torch.ones_like(coord[..., :1])], dim=-1)
            coord = torch.sum(transform[None, :, :] * coord[:, None, :], dim=-1)
            coord = coord[..., :3]

            points_world = []

            # print('dp_nps: ', dp_nps)
            for i, dp_np in enumerate(dp_nps):
                point_transform = torch.matmul(transform, torch.inverse(transforms[i]))
                dp_np = torch.sum(point_transform[None, :, :] * dp_np[:, None, :], dim=-1)
                points_world.append(dp_np[..., :3])

            point_cloud = torch.cat(points_world, dim=0)

            rix = torch.randperm(point_cloud.size(0))
            point_cloud = point_cloud[rix[:1000]]

            if point_cloud.size(0) != 1000:
                return self.get_item(index=random.randint(0, self.__len__() - 1))

            label = (label - 0.5) * 2.0

            # Rotate pcd randomly
            if self.any_rot:
                random_transform = torch.tensor(JointOccTrainDataset.__random_rot_transform())

                point_cloud = torch_util.transform_pcd_torch(point_cloud,
                    random_transform)

                coord = torch_util.transform_pcd_torch(coord,
                    random_transform)

            # translate everything to the origin based on the point cloud mean
            center = point_cloud.mean(dim=0)
            coord = coord - center[None, :]
            point_cloud = point_cloud - center[None, :]

            labels = label

            # Generate a new random rotation and create object
            if self.neg_any_se3:
                max_range = point_cloud.max(dim=0)[0] - point_cloud.min(dim=0)[0]
                max_range = max_range.mean().numpy()
                max_trans = max_range * self.trans_ratio
                random_transform = torch.tensor(JointOccTrainDataset.__random_se3_transform(max_trans))
            else:
                random_transform = torch.tensor(JointOccTrainDataset.__random_rot_transform())

            point_cloud_transformed = torch_util.transform_pcd_torch(point_cloud,
                random_transform)

            coords_transformed = torch_util.transform_pcd_torch(coord,
                random_transform)

            # Generate shuffled coordinates.  Simulates random sampling in
            # bounding box for negative triplet loss example
            # shuffler = torch.randperm(coords_transformed.shape[0])
            # coords_transformed_shuffled = coords_transformed[shuffler]

            # Generate random samples within bounding box of rotated
            min_coords = torch.min(coords_transformed, dim=0)[0].numpy()
            max_coords = torch.max(coords_transformed, dim=0)[0].numpy()
            coord_range = max_coords - min_coords

            rand_coords = np.random.random(coords_transformed.shape)
            rand_coords = rand_coords * coord_range.reshape((1, 3))
            rand_coords = torch.from_numpy(rand_coords)

            # # at the end we have 3D point cloud observation from depth images,
            # voxel occupancy values and corresponding voxel coordinates

            res = {'point_cloud': point_cloud.float(),
                   'coords': coord.float(),
                   'intrinsics': intrinsics.float(),
                   'rot_point_cloud': point_cloud_transformed.float(),
                   'rot_coords': coords_transformed.float(),
                #    'rot_coords_shuffled': coords_transformed_shuffled.float(),
                   'rand_coords': rand_coords.float()}
                #    'rot_intrinsics': torch.tensor([])}
                #    'cam_poses': np.zeros(1)}  # cam poses not used
            # print('dataio pcd: ', point_cloud.shape)

            # print(point_cloud.shape) # (1000, 3)
            return res, {'occ': torch.from_numpy(labels).float()}

        except Exception as e:
           print(e)
        #    print(file)
           return self.get_item(index=random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): int index of data to get

        Returns:
            res = {'point_cloud': point_cloud.float(),
                   'coords': coord.float(),
                   'intrinsics': intrinsics.float(),
                   'cam_poses': np.zeros(1)}  # cam poses not used
            return res, {'occ': torch.from_numpy(labels).float()}
        """
        return self.get_item(index)

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
            translate (bool, optional): True to include translation in transform.
                Defaults to False.

        Raises:
            NotImplementedError: translation not done yet

        Returns:
            Transform with random rotation
        """
        rand_quat = JointOccTrainDataset.__random_quaternions(1)

        rand_rot = Rotation.from_quat(rand_quat)
        rand_rot = rand_rot.as_matrix()

        transform = np.eye(4)
        transform[:3, :3] = rand_rot

        return transform

    @staticmethod
    def __random_se3_transform(max_translate=0.05):
        """
        Generate a random SE(3) transform

        Args:
            translate (bool, optional): True to include translation in transform.
                Defaults to False.

        Raises:
            NotImplementedError: translation not done yet

        Returns:
            Transform with random rotation
        """
        rand_quat = JointOccTrainDataset.__random_quaternions(1)
        rand_trans = np.random.random((3,)) * max_translate

        rand_rot = Rotation.from_quat(rand_quat)
        rand_rot = rand_rot.as_matrix()

        transform = np.eye(4)
        transform[:3, :3] = rand_rot
        transform[:3, 3] = rand_trans

        return transform
from ast import mod
from pymunk import PointQueryInfo
from torch.utils.data import DataLoader

from ndf_robot.training import dataio_conv as dataio

train_dataset = dataio.JointOccTrainDataset(128, depth_aug=False, 
    multiview_aug=False, obj_class="all")
val_dataset = dataio.JointOccTrainDataset(128, phase='val', depth_aug=False, 
    multiview_aug=False, obj_class="all")

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True,
    drop_last=True, num_workers=6)

test_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True,
    drop_last=True, num_workers=4)

# print(train_dataloader.__getitem__())

for model_input, gt in train_dataloader:
    print('Keys: ', model_input.keys())
    point_cloud = model_input['point_cloud']
    coords = model_input['coords']
    intrinsics = model_input['intrinsics']

    rot_point_cloud = model_input['rot_point_cloud']
    rot_coords = model_input['rot_coords']

    print('pcd: ', point_cloud.shape)
    print('coords: ', coords.shape)
    print('intrinsics: ', intrinsics.shape)
    print('rot_pcd', rot_point_cloud.shape)
    print('rot_coords', rot_coords.shape)

    # pcd:  torch.Size([16, 1000, 3])
    # coords:  torch.Size([16, 1500, 3])
    # intrinsics:  torch.Size([16, 3, 4])
    # rot_pcd torch.Size([16, 1000, 3]

    print('pcd: ', point_cloud[0, :10, :])
    print('rot_pcd: ', rot_point_cloud[0, :10, :])

    print('coords: ', coords[0, :10, :])
    print('rot_coords: ', rot_coords[0, :10, :])

    occ = gt['occ']
    print('occ: ', occ.shape)
    print('occ', occ[0, :100, :])
    
    break
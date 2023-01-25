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
    latent = model.extract_latent(model_input)

    act_torch = model.forward_latent(latent, model_input['coords']).detach()
    act = act_torch.squeeze().cpu().numpy()

    return act


def get_recon(pcd, query, model, thresh=0.2):

    if torch.cuda.is_available():
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    model_input = {}

    query = torch.from_numpy(query).float().to(dev)
    pcd = torch.from_numpy(pcd).float().to(dev)

    model_input['coords'] = query[None, :, :]
    model_input['point_cloud'] = pcd[None, :, :]

    out = model(model_input)

    in_inds = torch.where(out['occ'].squeeze() > thresh)[0].cpu().numpy()
    out_inds = torch.where(out['occ'].squeeze() < thresh)[0].cpu().numpy()
    all_occ = out['occ']

    in_pts = query[in_inds].cpu().numpy()
    out_pts = query[out_inds].cpu().numpy()
    all_occ = out['occ'].squeeze().cpu().detach().numpy()
    return in_pts, out_pts, all_occ


if __name__ == '__main__':

    # seed = 0
    # seed = 1
    seed = 6  # Main test seed
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
    # obj_model = osp.join(path_util.get_ndf_demo_obj_descriptions(), 'mug_centered_obj_normalized/28f1e7bc572a633cb9946438ed40eeb9/models/model_normalized.obj')
    obj_model = osp.join(path_util.get_ndf_obj_descriptions(), 'bottle_centered_obj_normalized/f4851a2835228377e101b7546e3ee8a7/models/model_normalized.obj')
    # obj_model = osp.join(path_util.get_ndf_obj_descriptions(), 'bottle_centered_obj_normalized/e593aa021f3fa324530647fc03dd20dc/models/model_normalized.obj')

    # One of the models that the L2 networks tend to fail at
    # obj_model = osp.join(path_util.get_ndf_obj_descriptions(), 'mug_centered_obj_normalized/e94e46bc5833f2f5e57b873e4f3ef3a4/models/model_normalized.obj')
    # obj_model = osp.join(path_util.get_ndf_obj_descriptions(), 'mug_centered_obj_normalized/d46b98f63a017578ea456f4bbbc96af9/models/model_normalized.obj')
    # obj_model = osp.join(path_util.get_ndf_obj_descriptions(), 'mug_centered_obj_normalized/f7d776fd68b126f23b67070c4a034f08/models/model_normalized.obj')
    # obj_model = osp.join(path_util.get_ndf_obj_descriptions(), 'bottle_centered_obj_normalized/f4851a2835228377e101b7546e3ee8a7/models/model_normalized.obj')

    if use_conv:
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_high_0/checkpoints/model_epoch_0001_iter_093000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_0/checkpoints/model_epoch_0008_iter_508000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_1/checkpoints/model_epoch_0008_iter_467000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_2/checkpoints/model_epoch_0004_iter_267000.pth')

        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_1/checkpoints/model_epoch_0003_iter_202000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_3/checkpoints/model_epoch_0001_iter_063000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/DEBUG_conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_16/checkpoints/model_epoch_0000_iter_055000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/DEBUG_conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_20/checkpoints/model_epoch_0000_iter_017000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/DEBUG_conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_cos_7/checkpoints/model_epoch_0000_iter_018000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_cos_9/checkpoints/model_epoch_0000_iter_003000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_cos_11/checkpoints/model_epoch_0000_iter_001000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_partial_neg_extreme_0/checkpoints/model_epoch_0000_iter_005000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/DEBUG_conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_cos_2/checkpoints/model_epoch_0000_iter_013000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_cos_ok_1/checkpoints/model_epoch_0000_iter_031000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_cos_ok_3/checkpoints/model_epoch_0000_iter_034000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_cos_r_dif_0/checkpoints/model_epoch_0000_iter_007000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_cos_r_dif_2/checkpoints/model_epoch_0000_iter_019000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_cos_r_dif_5/checkpoints/model_epoch_0000_iter_001000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_cos_r_dif_3/checkpoints/model_epoch_0001_iter_061000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_single_v2_2/checkpoints/model_epoch_0000_iter_052000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_single_v2_scratch_0/checkpoints/model_epoch_0000_iter_052000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_single_v2_max_dif_4/checkpoints/model_epoch_0000_iter_003000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_single_v2_strong_512_0/checkpoints/model_epoch_0000_iter_007000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_single_v2_r_diff_weak_512_1/checkpoints/model_epoch_0000_iter_001000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_single_v2_strong_512x200_1/checkpoints/model_epoch_0001_iter_100000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_0/checkpoints/model_epoch_0000_iter_007000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_1/checkpoints/model_epoch_0002_iter_123000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_2/checkpoints/model_epoch_0000_iter_002000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_5/checkpoints/model_epoch_0000_iter_001000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_DEBUG_0/checkpoints/model_epoch_0000_iter_004000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_DEBUG_1/checkpoints/model_epoch_0000_iter_001000.pth')

        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s2_0/checkpoints/model_epoch_0001_iter_089000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s2_DEBUG_0/checkpoints/model_epoch_0001_iter_083000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s2_DEBUG_1/checkpoints/model_epoch_0000_iter_002000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s2_DEBUG_5/checkpoints/model_epoch_0000_iter_002000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s2_DEBUG_8/checkpoints/model_epoch_0000_iter_002000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s2_DEBUG_9/checkpoints/model_epoch_0000_iter_003000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s2_1/checkpoints/model_epoch_0000_iter_032000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s2_2/checkpoints/model_epoch_0000_iter_030000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s2_3/checkpoints/model_epoch_0000_iter_005000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s2_4/checkpoints/model_epoch_0000_iter_003000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s2_5/checkpoints/model_epoch_0000_iter_008000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s2_6/checkpoints/model_epoch_0000_iter_005000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s2_6/checkpoints/model_epoch_0000_iter_009000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s2_7/checkpoints/model_epoch_0000_iter_010000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s2_8/checkpoints/model_epoch_0002_iter_124000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s2_9/checkpoints/model_epoch_0001_iter_094000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s2_10/checkpoints/model_epoch_0001_iter_083000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s2_11/checkpoints/model_epoch_0000_iter_010000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s2_12/checkpoints/model_epoch_0000_iter_005000.pth')

        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s3_0/checkpoints/model_epoch_0000_iter_001000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s3_5/checkpoints/model_epoch_0002_iter_132000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s3_6/checkpoints/model_epoch_0001_iter_116000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s3_7/checkpoints/model_epoch_0002_iter_132000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/DEBUG_conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s4_31/checkpoints/model_epoch_0000_iter_026000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s4_2/checkpoints/model_epoch_0001_iter_093000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s4_3/checkpoints/model_epoch_0000_iter_048000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s4_4/checkpoints/model_epoch_0000_iter_012000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s4_5/checkpoints/model_epoch_0000_iter_009000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s4_6/checkpoints/model_epoch_0000_iter_011000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s4_6/checkpoints/model_epoch_0000_iter_020000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s4_7/checkpoints/model_epoch_0000_iter_020000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s4_8/checkpoints/model_epoch_0000_iter_008000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s3_all_1/checkpoints/model_epoch_0000_iter_005000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s2_DEBUG_9/checkpoints/model_epoch_0000_iter_031000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/DEBUG_conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s4_22/checkpoints/model_epoch_0000_iter_003000.pth')

        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden64_anyrot_multicategory_latent_sim_occ_neg_se3_s4_1/checkpoints/model_epoch_0000_iter_050000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden64_anyrot_multicategory_latent_sim_occ_neg_se3_s4_1/checkpoints/model_epoch_0002_iter_117000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden64_anyrot_multicategory_latent_sim_occ_neg_se3_s4_1/checkpoints/model_epoch_0005_iter_326000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden32_anyrot_dist_cont_1/checkpoints/model_epoch_0001_iter_060000.pth')
        model_path = osp.join(path_util.get_ndf_model_weights(), 'ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s4_7/checkpoints/model_epoch_0001_iter_060000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'lndf_refined/2023-01-14_16H36M11S_Sat_conv_hidden_128_with_l2_0/checkpoints/model_epoch_0002_iter_120000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'lndf_refined/2023-01-14_16H06M03S_Sat_DEBUG_conv_hidden128_0/checkpoints/model_epoch_0001_iter_060000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'lndf_refined/2023-01-15_15H48M36S_Sun_conv_hidden_128_with_l2_pointy_0/checkpoints/model_epoch_0001_iter_060000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'lndf_refined/2023-01-15_16H07M47S_Sun_conv_hidden_128_with_l2_0/checkpoints/model_epoch_0001_iter_060000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'lndf_refined/2023-01-16_13H28M16S_Mon_conv_hidden_128_with_l2_light_0/checkpoints/model_epoch_0002_iter_120000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'lndf_refined/2023-01-17_01H02M11S_Tue_conv_hidden_128_with_l2_0/checkpoints/model_epoch_0000_iter_040000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'lndf_refined/2023-01-17_13H55M38S_Tue_conv_hidden_128_with_l2_r0p02_0/checkpoints/model_epoch_0002_iter_120000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'lndf_refined/2023-01-17_13H53M56S_Tue_conv_hidden_128_with_l2_r0p05_0/checkpoints/model_epoch_0002_iter_120000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'lndf_refined/2023-01-18_14H21M46S_Wed_conv_hidden_128_with_l2_r0p04_0/checkpoints/model_epoch_0000_iter_030000.pth')
        # model_path = osp.join(path_util.get_ndf_model_weights(), 'lndf_refined/2023-01-21_15H00M07S_Sat_conv_hidden_128_with_l2_r0p1_0/checkpoints/model_epoch_0004_iter_240000.pth')


    else:
        model_path = osp.join(path_util.get_ndf_model_weights(), 'multi_category_weights.pth')

    assert osp.exists(model_path), 'Model weights not found'
    print('Model weights good!')

    scale = 0.25
    mesh1 = trimesh.load(obj_model, process=False)
    mesh1.apply_scale(scale)

    mesh2 = trimesh.load(obj_model, process=False)
    mesh2.apply_scale(scale)

    extents = mesh1.extents
    extents_offset = 0.02
    recon_sample_pts = trimesh.sample.volume_rectangular((extents + 2 * extents_offset) - extents_offset, n_samples, transform=None)
    sample_pts = mesh1.sample(n_samples)
    upright_sample_pts = sample_pts[:, :]
    ref_pt = mesh1.sample(1)

    # Make mesh 1 upright
    rot1 = np.eye(4)
    rot1[:3, :3] = util.make_rotation_matrix('x', np.pi / 2)
    # rot1 = np.eye(4)
    # rot1[:3, :3] = R.random().as_matrix()
    mesh1.apply_transform(rot1)
    ref_pt = util.transform_pcd(ref_pt, rot1)
    upright_sample_pts = util.transform_pcd(upright_sample_pts, rot1)

    rot2 = np.eye(4)
    rot2[:3, :3] = R.random().as_matrix()
    # rot2[:3, 3] = [0, 0.1, 0]  # Test translation
    mesh2.apply_transform(rot2)
    sample_pts = util.transform_pcd(sample_pts, rot2)
    recon_sample_pts = util.transform_pcd(recon_sample_pts, rot2)

    pcd1 = mesh1.sample(5000)
    pcd2 = mesh2.sample(5000)  # point cloud representing different shape

    ref_plot_pt = ref_pt.reshape(1, 3) + np.random.random((20, 3)) * 0.005

    multiplot([pcd1, pcd2, ref_plot_pt, sample_pts], osp.join(path_util.get_ndf_eval(), 'debug_viz', 'debug_correspondance.html'))

    if use_conv:
        # model = conv_occupancy_network.ConvolutionalOccupancyNetwork(latent_dim=128,
        #     model_type='pointnet', return_features=True, sigmoid=False, acts='last').cuda()

        model = conv_occupancy_network.ConvolutionalOccupancyNetwork(latent_dim=128,
            model_type='pointnet', return_features=True, sigmoid=False, acts='last').cuda()

        # model = conv_occupancy_network.ConvolutionalOccupancyNetwork(latent_dim=64,
        #     model_type='pointnet', return_features=True, sigmoid=False, acts='last').cuda()
    else:
        model = vnn_occupancy_network.VNNOccNet(latent_dim=256,
            model_type='pointnet', return_features=True, sigmoid=True).cuda()

    model.load_state_dict(torch.load(model_path))

    # -- Get activations -- #
    ref_act = get_activations(pcd1, ref_pt, model)
    acts = get_activations(pcd2, sample_pts, model)

    ref_act = ref_act[None, :]
    ref_act = np.repeat(ref_act, n_samples, axis=0)

    # print(ref_act)
    # cor = F.l1_loss(torch.from_numpy(acts).float().to(dev),
    #     torch.from_numpy(ref_act).float().to(dev),
    #     reduction='none')

    # With cosine similarity, most similar is 1 and least similar is -1
    cor = F.cosine_similarity(torch.from_numpy(acts).float().to(dev),
        torch.from_numpy(ref_act).float().to(dev), dim=1)

    cor = cor.cpu().numpy()

    # cor = cor.sum(axis=1)

    print(cor.shape)

    # -- Get distances -- #
    # Should be length k
    # print(upright_sample_pts)
    # print(ref_pt)
    distances = np.sqrt(((upright_sample_pts - ref_pt)**2).sum(axis=-1))  # sample_pts is k x 3

    distance_corr = np.stack([distances, cor], axis=0).T
    print(distance_corr.shape)
    np.savetxt('corr.csv', distance_corr, delimiter=',')



    # -- Get reconstruction -- #
    in_pts, out_pts, all_occ = get_recon(pcd2, recon_sample_pts, model)
    fname = osp.join(path_util.get_ndf_eval(), 'debug_viz', 'debug_recon.html')
    multiplot([in_pts, out_pts], fname)
    fig = px.scatter_3d(
        x=recon_sample_pts[:, 0], y=recon_sample_pts[:, 1], z=recon_sample_pts[:, 2],
        color=all_occ
    )
    fname = osp.join(path_util.get_ndf_eval(), 'debug_viz', 'debug_recon2.html')
    fig.write_html(fname)

    plot_pts = sample_pts
    color = cor

    # Cap colors so I can actually see differences in the presence of outliers
    max_color = 1000
    outliers = np.where(color > max_color)
    color[outliers] = max_color

    fig = px.scatter_3d(
        x=plot_pts[:, 0], y=plot_pts[:, 1], z=plot_pts[:, 2], color=color)

    fname = osp.join(path_util.get_ndf_eval(), 'debug_viz', 'debug_correspondance_cor.html')

    fig.write_html(fname)


    fig = px.scatter_3d(
        x=plot_pts[:, 0], y=plot_pts[:, 1], z=plot_pts[:, 2], color=distances)

    # fname = osp.join(path_util.get_ndf_eval(), 'debug_viz', 'DELETE_debug_dist.html')
    # fig.write_html(fname)

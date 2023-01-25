import sys
import os, os.path as osp
from datetime import datetime
from turtle import position
import configargparse
import torch
from torch.utils.data import DataLoader
from torch import nn

import ndf_robot.model.vnn_occupancy_net.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
import ndf_robot.model.conv_occupancy_net.conv_occupancy_net as conv_occupancy_network

# from ndf_robot.training import summaries, losses, training, dataio, config
from ndf_robot.training import summaries, losses, training
from ndf_robot.training import dataio_conv as dataio
# from ndf_robot.training import dataio as dataio

from ndf_robot.utils import path_util
from ndf_robot.training.util import make_unique_path_to_dir


if __name__ == '__main__':
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--logging_root', type=str, default=osp.join(path_util.get_ndf_model_weights(), 'lndf'), help='root for logging')
    # p.add_argument('--logging_root', type=str, default=osp.join(path_util.get_ndf_model_weights(), 'lndf_refined'), help='root for logging')
    p.add_argument('--obj_class', type=str, required=True,
                help='bottle, mug, bowl, all')
    p.add_argument('--experiment_name', type=str, required=True,
                help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

    p.add_argument('--sidelength', type=int, default=128)

    # General training options
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
    # p.add_argument('--lr', type=float, default=1e-3, help='learning rate. default=5e-5')
    p.add_argument('--num_epochs', type=int, default=100,
                help='Number of epochs to train for.')
    # p.add_argument('--num_epochs', type=int, default=40001,
    #                help='Number of epochs to train for.')

    p.add_argument('--epochs_til_ckpt', type=int, default=5,
                help='Time interval in seconds until checkpoint is saved.')
    # p.add_argument('--epochs_til_ckpt', type=int, default=10,
    #                help='Time interval in seconds until checkpoint is saved.')
    p.add_argument('--steps_til_summary', type=int, default=500,
                help='Time interval in seconds until tensorboard summary is saved.')
    p.add_argument('--iters_til_ckpt', type=int, default=10000,
                help='Training steps until save checkpoint')

    p.add_argument('--depth_aug', action='store_true', help='depth_augmentation')
    p.add_argument('--multiview_aug', action='store_true', help='multiview_augmentation')

    p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
    p.add_argument('--dgcnn', action='store_true', help='If you want to use a DGCNN encoder instead of pointnet (requires more GPU memory)')
    # p.add_argument('--conv', action='store_true', help='If you want to train convolutional occ instead of non-convolutional')

    p.add_argument('--triplet_loss', action='store_true', help='Run triplet loss on'
        + ' activations')

    opt = p.parse_args()

    # -- MODEL ARGUMENTS -- #
    latent_dim_32 = {
        'latent_dim': 32,
        'return_features': True,
        'acts': 'last'
    }

    latent_dim_64 = {
        'latent_dim': 64,
        'return_features': True,
        'acts': 'last'
    }

    latent_dim_128 = {
        'latent_dim': 128,
        'return_features': True,
        'acts': 'last'
    }
    # conv_occ_args = latent_dim_32
    # conv_occ_args = latent_dim_64
    conv_occ_args = latent_dim_128

    # -- LOSS FUNCTION ARGS -- #
    no_similarity = {
        'latent_loss_scale': 0
    }
    contrastive_cross_entropy_args = {
        'latent_loss_scale': 0.1,
        'dis_offset': 0.002,
    }

    contrastive_l2_args = {
        'latent_loss_scale': 1,
        'radius': 0.1,
    }

    # loss_fn_args = no_similarity
    loss_fn_args = contrastive_cross_entropy_args
    # loss_fn_args = contrastive_l2_args

    # -- DATALOADER ARGS -- #
    sidelength = 128

    train_dataloader_args = {
        'sidelength': sidelength,
        'depth_aug': opt.depth_aug,
        'multiview_aug': opt.multiview_aug,
        'obj_class': opt.obj_class,
        'any_rot': True,
        'neg_any_se3': True,
        # Ratio (between 0 and 1) of how much to translate object by.  This can
        # help with over-fitting.
        'trans_ratio': 0,
    }

    val_dataloader_args = {
        'sidelength': sidelength,
        'phase': 'val',
        'depth_aug': opt.depth_aug,
        'multiview_aug': opt.multiview_aug,
        'obj_class': opt.obj_class,
        'any_rot': True,
        'neg_any_se3': True,
        'trans_ratio': 0
    }

    # -- CREATE DATALOADERS -- #
    train_dataset = dataio.JointOccTrainDataset(**train_dataloader_args)
    val_dataset = dataio.JointOccTrainDataset(**val_dataloader_args)

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                drop_last=True, num_workers=6)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True,
                                drop_last=True, num_workers=4)

    # -- CREATE MODEL -- #
    model = conv_occupancy_network.ConvolutionalOccupancyNetwork(
        **conv_occ_args).cuda()

    print(model)

    # -- LOAD CHECKPOINT --#
    if opt.checkpoint_path is not None:
        checkpoint_path = osp.join(path_util.get_ndf_model_weights(), opt.checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path))

    # Can use if have multiple gpus.  Currently untested.
    # model_parallel = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5])
    model_parallel = model

    # -- CREATE SAVE UTILS -- #
    summary_fn = summaries.occupancy_net

    t = datetime.now()
    time_str = t.strftime('%Y-%m-%d_%HH%MM%SS_%a')
    experiment_name = time_str + '_' + opt.experiment_name

    root_path = os.path.join(opt.logging_root, experiment_name)

    root_path = make_unique_path_to_dir(root_path)

    # -- CREATE CONFIG -- #
    config = {}
    config['model_args'] = conv_occ_args
    config['argparse_args'] = vars(opt)
    config['loss_fn_args'] = loss_fn_args
    config['train_dataloader_args'] = train_dataloader_args
    config['val_dataloader_args'] = val_dataloader_args

    # -- RUN TRAIN FUNCTION -- #
    # Set loss function here
    loss_fn = val_loss_fn = losses.contrastive_cross_entropy(**loss_fn_args)
    # loss_fn = val_loss_fn = losses.contrastive_l2(**loss_fn_args)

    training.train_conv_cos(model=model_parallel, train_dataloader=train_dataloader,
        val_dataloader=val_dataloader, epochs=opt.num_epochs, lr=opt.lr,
        steps_til_summary=opt.steps_til_summary,
        epochs_til_checkpoint=opt.epochs_til_ckpt,
        model_dir=root_path, loss_fn=loss_fn, iters_til_checkpoint=opt.iters_til_ckpt,
        summary_fn=summary_fn, clip_grad=False, val_loss_fn=val_loss_fn, overwrite=True,
        config_dict=config)
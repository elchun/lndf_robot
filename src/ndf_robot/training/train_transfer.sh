# CUDA_VISIBLE_DEVICES=5 python train_conv_occupancy_net.py --obj_class all --experiment_name  conv_occ_latent_transfer_rand_coords_margin_no_neg_margin_13_lr3 --num_epochs 12 --iters_til_ckpt 1000 --steps_til_summary 100 --batch_size 10 --checkpoint_path ndf_vnn/conv_occ_exp_archive/checkpoints/model_epoch_0099_iter_747100.pth --triplet_loss

# CUDA_VISIBLE_DEVICES=4 python train_conv_occupancy_net.py --obj_class all --experiment_name  conv_occ_latent32_triplet_similar_occ_only_all --num_epochs 12 --iters_til_ckpt 1000 --steps_til_summary 100 --batch_size 10 --checkpoint_path archive/conv_occ_latent_margin_143000.pth --triplet_loss

# CUDA_VISIBLE_DEVICES=4 python train_conv_occupancy_net.py --obj_class all --experiment_name  conv_occ_latent_dim4_rotated_triplet_n_margin_10e3_last_acts_margin_0p001_0p1 --num_epochs 12 --iters_til_ckpt 1000 --steps_til_summary 100 --batch_size 10 --checkpoint_path ndf_vnn/conv_occ_latent_4_0/checkpoints/model_epoch_0010_iter_130000.pth --triplet_loss

# CUDA_VISIBLE_DEVICES=3 python train_conv_occupancy_net.py --obj_class all --experiment_name conv_occ_train_any_rot_hidden4_occ_similar_only_margin_noneg_10 --num_epochs 12 --iters_til_ckpt 1000 --steps_til_summary 100 --batch_size 10 --checkpoint_path ndf_vnn/conv_occ_hidden4_anyrot_6/checkpoints/model_epoch_0011_iter_143000.pth --triplet_loss

CUDA_VISIBLE_DEVICES=4 python train_conv_occupancy_net.py --obj_class all --experiment_name conv_occ_hidden32_anyrot_simocc_10x10 --num_epochs 12 --iters_til_ckpt 1000 --steps_til_summary 100 --batch_size 8 --checkpoint_path ndf_vnn/conv_occ_hidden32_anyrot_part2_1/checkpoints/model_final.pth --triplet_loss

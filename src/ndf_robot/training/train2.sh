# python train_vnn_occupancy_net.py --obj_class all --experiment_name  ndf_training_exp --num_epochs 100
CUDA_VISIBLE_DEVICES=6 python train_conv_occupancy_net.py --obj_class all --experiment_name  conv_occ_latent_custom_triplet_shuffle_coords --num_epochs 12 --iters_til_ckpt 1000 --steps_til_summary 100 --batch_size 12 


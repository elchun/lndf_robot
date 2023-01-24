# python train_vnn_occupancy_net.py --obj_class all --experiment_name  ndf_training_exp --num_epochs 100
# CUDA_VISIBLE_DEVICES=3 python train_conv_occupancy_net.py --obj_class mug --experiment_name  conv_occ_hidden64_anyrot_mug --num_epochs 24 --iters_til_ckpt 1000 --steps_til_summary 100 --batch_size 8 --triplet_loss --checkpoint_path ndf_vnn/

python train_conv_occupancy_net.py \
    --obj_class all \
    --experiment_name conv_hidden_128_with_l2_r0p1\
    --num_epochs 12 \
    --iters_til_ckpt 1000 \
    --steps_til_summary 100 \
    --batch_size 6 \
    --triplet_loss \
    --checkpoint_path ndf_vnn/conv_occ_hidden128_anyrot_multicategory_1/checkpoints/model_epoch_0010_iter_600000.pth
    # --checkpoint_path ndf_vnn/conv_occ_hidden32_anyrot_1/checkpoints/model_final.pth
    # --experiment_name conv_occ_hidden64_anyrot_multicategory_latent_sim_occ_neg_se3_s4\
    # --checkpoint_path ndf_vnn/conv_occ_hidden128_anyrot_multicategory_1/checkpoints/model_epoch_0008_iter_520000.pth
    # --checkpoint_path ndf_vnn/conv_occ_hidden128_anyrot_multicategory_1/checkpoints/model_epoch_0003_iter_202000.pth
    # --checkpoint_path ndf_vnn/conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s3_7/checkpoints/model_epoch_0002_iter_117000.pth
   #  --checkpoint_path ndf_vnn/conv_occ_hidden128_anyrot_multicategory_0/checkpoints/model_epoch_0008_iter_508000.pth
    # --checkpoint_path ndf_vnn/conv_occ_hidden64_anyrot_multicategory_part2_0/checkpoints/model_final.pth

    # --experiment_name conv_occ_hidden128_anyrot_multicategory\
    # --experiment_name conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s2\
    # --experiment_name conv_occ_hidden128_anyrot_multicategory_latent_sim_occ_neg_se3_s2\
    # --checkpoint_path ndf_vnn/conv_occ_hidden64_anyrot_multicategory_part2_0/checkpoints/model_final.pth

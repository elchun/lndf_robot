# python train_vnn_occupancy_net.py --obj_class all --experiment_name  ndf_training_exp --num_epochs 100
# CUDA_VISIBLE_DEVICES=3 python train_conv_occupancy_net.py --obj_class mug --experiment_name  conv_occ_hidden64_anyrot_mug --num_epochs 24 --iters_til_ckpt 1000 --steps_til_summary 100 --batch_size 8 --triplet_loss --checkpoint_path ndf_vnn/

python train_conv_occupancy_net.py \
    --obj_class all \
    --experiment_name test_lndf_weights\
    --num_epochs 12 \
    --iters_til_ckpt 1000 \
    --steps_til_summary 100 \
    --batch_size 6 \
    --triplet_loss \
    --checkpoint_path lndf_no_se3_weights.pth
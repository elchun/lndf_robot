CUDA_VISIBLE_DEVICES=1 python evaluate_ndf.py \
    --demo_exp grasp_rim_hang_handle_gaussian_precise_w_shelf \
    --object_class mug\
    --opt_iterations 1000 \
    --rand_mesh_scale \
    --model_path multi_category_weights \
    --save_vis_per_model \
    --config eval_mug_gen \
    --exp DEBUG\
    --num_iterations 100 \
    --any_pose \
    --only_test_ids \
#   --pybullet_viz \
#   --use_full_hand \
#   --grasp_viz \

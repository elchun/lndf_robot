import os
import os.path as osp
import shutil

from ndf_robot.utils import path_util

if __name__ == '__main__':
    # obj_class = 'bowl'
    obj_class = 'bottle'

    shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(),
        obj_class + '_handle_centered_obj_normalized')

    shapenet_id_list = [fn.split('_')[0] for fn in os.listdir(shapenet_obj_dir)]

    for obj_shapenet_id in shapenet_id_list:
        for idx in [0, 1, 2, 3]:
            obj_input_fname = osp.join(shapenet_obj_dir, obj_shapenet_id,
                f'models/model_normalized v{idx}.obj')

            obj_output_fname = osp.join(shapenet_obj_dir, obj_shapenet_id,
                'models/model_normalized.obj')

            if osp.exists(obj_input_fname):
                shutil.move(obj_input_fname, obj_output_fname)

    print('Done!')



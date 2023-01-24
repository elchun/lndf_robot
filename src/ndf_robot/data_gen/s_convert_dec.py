import os
import os.path as osp
import shutil

import pybullet as p

from ndf_robot.utils import path_util

if __name__ == '__main__':
    # obj_class = 'bowl'
    # obj_class = 'bottle_handle_std'
    # obj_class = 'mug'
    # obj_class = 'mug_std'

    shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(),
        obj_class + '_centered_obj_normalized')

    shapenet_id_list = [fn.split('_')[0] for fn in os.listdir(shapenet_obj_dir)]

    for obj_shapenet_id in shapenet_id_list:
        obj_fname = osp.join(shapenet_obj_dir, obj_shapenet_id,
            'models/model_normalized.obj')

        obj_file_dec = obj_fname.split('.obj')[0] + '_dec.obj'

        # if osp.exists(obj_file_dec):
        #     os.remove(obj_file_dec)
        # if not osp.exists(obj_file_dec):
            # with open(obj_file_dec, 'w') as f:
            #     f.write('')

        p.vhacd(
            obj_fname,
            obj_file_dec,
            'log.txt',
            concavity=0.0025,
            alpha=0.04,
            beta=0.05,
            gamma=0.00125,
            minVolumePerCH=0.0001,
            resolution=2000000,
            # resolution=10000,
            depth=20,
            planeDownsampling=4,
            convexhullDownsampling=4,
            pca=0,
            mode=0,
            convexhullApproximation=1
        )

        print(f'Shapenet id: {obj_shapenet_id}')







import os
import os.path as osp

import psutil

from ndf_robot.eval.evaluate_general_types import SimConstants
from ndf_robot.utils import path_util


if __name__ == '__main__':
    shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(),
    'mug' + '_centered_obj_normalized')

    shapenet_id_list = [fn.split('_')[0]
        for fn in os.listdir(shapenet_obj_dir)]

    # print(shapenet_id_list)

    avoid_mugs = SimConstants.MUG_AVOID_SHAPENET_IDS
    train_mugs = SimConstants.MUG_TRAIN_SHAPENET_IDS
    test_mugs = SimConstants.MUG_TEST_SHAPENET_IDS

    # -- Determine if the three catagories cover all mugs -- #
    # res = []
    # for mug in shapenet_id_list:
    #     if mug not in avoid_mugs and mug not in train_mugs and mug not in test_mugs:
    #         res.append(mug)

    # print(mug)

    # -- Determine if test_mugs = all mugs - avoid_mugs - train_mugs -- #
    # pseudo_test_mugs = []
    # for mug in shapenet_id_list:
    #     if mug not in avoid_mugs and mug not in train_mugs:
    #         pseudo_test_mugs.append(mug)

    # unoverlap = [mug for mug in pseudo_test_mugs if mug not in test_mugs]
    # print(unoverlap)

    # -- Determine if odd test is ok -- #
    new_test = avoid_mugs.union(test_mugs)
    overlap = [mug for mug in new_test if mug in train_mugs]
    print(overlap)



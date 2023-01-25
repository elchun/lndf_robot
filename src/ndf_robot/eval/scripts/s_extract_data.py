import os
import os.path as osp

from ndf_robot.utils import path_util


if __name__ == '__main__':
    eval_dir = '2022-08-11_00H10M49S_Thu_DEBUG_ndf_rack_grasp_ideal_ori_query'

    base_path = osp.join(path_util.get_ndf_eval_data(), 'eval_grasp')
    for eval_dir in os.listdir(base_path):
        print(eval_dir)
        if eval_dir == 'old':
            continue
        data_fname = osp.join(base_path, eval_dir, 'global_summary.txt')
        last_trial_fname = osp.join(base_path, eval_dir, 'last_trial.txt')

        # # Don't need to write again
        # if osp.exists(last_trial_fname):
        #     continue

        # Can't extract data from nothing
        if not osp.exists(data_fname):
            print(data_fname)
            print('no data')
            continue

        print('ok')

        record = False
        res_list = []

        with open(data_fname, 'r') as f:
            for line in f:
                if 'Trial number' in line and '199' in line:
                    record = True
                if record:
                    res_list.append(line)

        with open(last_trial_fname, 'w') as f:
            for line in res_list:
                f.write(line)
                print(line)

    print('Done!')
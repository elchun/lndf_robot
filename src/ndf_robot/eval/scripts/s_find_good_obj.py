import os
import os.path as osp

from ndf_robot.utils import path_util


def get_best_s_id(id_list):
    best_s_id = None
    best_success_rate = 0
    current_n_success = 0
    for s_id, data in id_list.items():
        _, n_success, success_rate = data
        if success_rate > best_success_rate or (success_rate == best_success_rate and n_success > current_n_success):
            best_s_id = s_id
            best_success_rate = success_rate
            current_n_success = n_success
    return best_s_id

if __name__ == '__main__':
    # eval_dir = '2022-09-10_01H40M58S_Sat_EVAL_conv_shelf_upright_bowl_handle'
    # eval_dir = '2022-09-09_18H35M15S_Fri_EVAL_conv_shelf_upright_bottle_handle'
    eval_dir = '2022-09-10_12H10M57S_Sat_EVAL_conv_rack_grasp_upright_bowl_handle'

    base_path = osp.join(path_util.get_ndf_eval_data(), 'eval_grasp')
    data_fname = osp.join(base_path, eval_dir, 'global_summary.txt')
    # Can't extract data from nothing
    if not osp.exists(data_fname):
        print(data_fname)
        print('no data')

    ids = {}
    with open(data_fname, 'r') as f:
        start = False
        success = False
        for line in f:
            if 'Trial number' in line:
                start = True
                success = False
            if start:
                if 'Trial result' in line:
                    success = 'TrialResults.SUCCESS' in line
                if 'Shapenet id' in line:
                    s_id = line.split(' ')[-1].strip()
                    ids.setdefault(s_id, [0, 0, 0])
                    ids[s_id][0] += 1
                    ids[s_id][1] += success

                    start = False
                    success = False

    # Solve for success rate
    for s_id in ids.keys():
        ids[s_id][2] = ids[s_id][1] / ids[s_id][0]

    ids[s_id] = tuple(ids[s_id])

    final_id_list = []
    working_ids = ids.copy()
    n_good_ids = 5
    for i in range(n_good_ids):
        s_id = get_best_s_id(working_ids)
        final_id_list.append(s_id)
        working_ids.pop(s_id)

    for f_id in final_id_list:
        print(f_id, ids[f_id])
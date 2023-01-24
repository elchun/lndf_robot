import os.path as osp

from ndf_robot.utils import path_util


def parse_data(fname: str):
    """
    Parse data file that has lines with 'Grasp Success' '|' and 'Place Success'
    in that order.

    Also looks for 'Success Rate' and prints the most recent success rate

    Args:
        fname (str): Path to data file to parse.
    """

    success_lines = []
    total_success_list = []
    with open(fname, 'r') as f:
        for line in f:
            if 'Grasp Success' in line and 'Place Success' in line:
                success_lines.append(line)
                print(line)
            if 'Success Rate' in line:
                total_success = float(line.split(' ')[-1])
                total_success_list.append(total_success)

    grasp_success_count = 0
    place_success_count = 0
    total_count = 0
    for line in success_lines:
        grasp_success, place_success = [get_success(phrase) for phrase in line.split('|')]
        grasp_success_count += grasp_success
        place_success_count += place_success
        total_count += 1

    grasp_success_ratio = grasp_success_count / total_count
    place_success_ratio = place_success_count / total_count
    print(grasp_success_ratio, place_success_ratio, total_success_list[-1])
    print(f'Grasp success: {grasp_success_ratio}, '
        + f'Place Success: {place_success_ratio}, Total Success: {total_success_list[-1]}')


def get_success(line):
    return 'True' in line


if __name__ == '__main__':
    eval_dir = '2022-08-11_00H10M49S_Thu_DEBUG_ndf_rack_grasp_ideal_ori_query'
    fname = osp.join(path_util.get_ndf_eval_data(),
        'eval_grasp',
        eval_dir,
        'global_summary.txt')
    parse_data(fname)
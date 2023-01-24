import os
import os.path as osp
import trimesh
import random

from ndf_robot.utils import path_util


def apply_scale(input_fname: str, output_fname: str, ref_sum: float,
    random_percent: float = 0.00) -> float:
    """
    Scale all meshes so that they have the same sum of extents
    """
    mesh = trimesh.load_mesh(input_fname, 'obj', process=False)
    extents_sum = sum(mesh.extents) # length width height
    target_sum = ref_sum * (1 + (random_percent * 2) * (random.random() - 0.5))

    scale_factor = target_sum / extents_sum

    print('ori: ', extents_sum)
    mesh.apply_scale(scale_factor)
    print('res: ', sum(mesh.extents))

    res = trimesh.exchange.obj.export_obj(mesh)
    if not osp.exists(output_fname):
        dir_to_make = '/' + osp.join(*output_fname.split('/')[:-1])
        os.makedirs(dir_to_make)

    with open(output_fname, 'w') as f:
        # print(output_fname)
        f.write(res)


def get_ref_extents_sum(fname: str) -> float:
    mesh = trimesh.load_mesh(fname, 'obj', process=False)
    extents_sum = sum(mesh.extents) # length width height
    return extents_sum


if __name__ == '__main__':
    obj_class = 'bowl_handle'
    ref_obj_class = 'bowl_handle'
    # ref_obj_class = 'bottle'
    shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(),
        obj_class + '_centered_obj_normalized')
    output_shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(),
        obj_class + '_std_centered_obj_normalized')

    shapenet_id_list = [fn.split('_')[0] for fn in os.listdir(shapenet_obj_dir)]

    ref_shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(),
        ref_obj_class + '_centered_obj_normalized')
    ref_shapenet_id_list = [fn.split('_')[0] for fn in os.listdir(ref_shapenet_obj_dir)]

    # -- Get target scale if desired -- #
    extents_list = []
    for obj_shapenet_id in ref_shapenet_id_list:
        obj_input_fname = osp.join(ref_shapenet_obj_dir, obj_shapenet_id,
            'models/model_normalized.obj')

        extents_list.append(get_ref_extents_sum(obj_input_fname))

    average_extents_sum = sum(extents_list) / len(extents_list)
    print('Average extents sum: ', average_extents_sum)

    target_extents_sum = average_extents_sum
    # target_extents_sum = average_extents_sum

    # -- Apply scale -- #
    # target_extents_sum = 0.5  # For normal bottles
    target_extents_sum = 0.55

    for obj_shapenet_id in shapenet_id_list:
        obj_input_fname = osp.join(shapenet_obj_dir, obj_shapenet_id,
            'models/model_normalized.obj')

        # obj_final_fname = osp.join(shapenet_obj_dir, obj_shapenet_id,
        #     'models/model_normalized_scaled.obj')

        obj_final_fname = osp.join(output_shapenet_obj_dir, obj_shapenet_id,
            'models/model_normalized.obj')

        # apply_scale(obj_fname, obj_fname, 1.6)
        apply_scale(obj_input_fname, obj_final_fname, target_extents_sum)

        obj_file_dec = obj_final_fname.split('.obj')[0] + '_dec.obj'

        if osp.exists(obj_file_dec):
            os.remove(obj_file_dec)

        # scaled_fname = obj_final_fname.split('.obj')[0] + '_scaled.obj'

        # if osp.exists(scaled_fname):
        #     os.remove(scaled_fname)





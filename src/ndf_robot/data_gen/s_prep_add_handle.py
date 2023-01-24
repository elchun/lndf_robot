import os
import os.path as osp
import shutil

import pybullet as p

from ndf_robot.utils import path_util, util

if __name__ == '__main__':
    # obj_class = 'bowl'
    obj_class = 'bottle'

    # shapenet_id_list = [
    #     'f2ef5e5b49f2bb8340dfb1e6c8f5a333',
    #     '34875f8448f98813a2c59a4d90e63212',
    #     'f0fdca5f5c7a06252dbdfbe028032489',
    #     'a0b34a720dd364d9ccdca257be791a55',
    #     '2d2c419136447fe667964ba700cd97f5',
    #     'f09d5d7ed64b6585eb6db0b349a2b804',
    #     'bd2ba805bf1739cdedd852e9640b8d4',
    #     '64d7f5eb886cfa48ce6101c7990e61d4',
    #     'ea473a79fd2c98e5789eafad9d8a9394',
    #     'ee3b4a98683feab4633d74df68189e22',
    #     'e30e5cbc54a62b023c143af07c12991a',
    #     'a1d26a16a0caa78243f1c519d66bb167',
    #     'faa200741fa93abb47ec7417da5d353d',
    #     'fa23aa60ec51c8e4c40fe5637f0a27e1',
    #     '2e545ccae1cdc69d879b85bd5ada6e71',
    #     '2c1df84ec01cea4e525b133235812833',
    #     'e3e57a94be495771f54e1b6f41fdd78a',
    #     'e816066ac8281e2ecf70f9641eb97702',
    #     'f2cb15fb793e7fa244057c222118625',
    #     '188281000adddc9977981b941eb4f5d1',
    #     '7995c6a5838e12ed447eea2e92abe28f',
    #     'eff9864bfb9920b521374fbf1ea544c',
    #     'fa61e604661d4aa66658ecd96794a1cd',
    #     'f44387d8cb8d2e4ebaedc225f2279ecf',
    #     'ed220bdfa852f00ba2c59a4d90e63212',
    #     'ff7c33db3598df342d88c45db31bc366',
    #     '32f9c710e264388e2150a45ec52bcbd7',
    #     'e3d4d57aea714a88669ff09d7001bab6',
    #     'e4c871d1d5e3c49844b2fa2cac0778f5',
    #     'fc77ad0828db2caa533e44d90297dd6e'
    # ]

    shapenet_id_list = [
        'e4ada697d05ac7acf9907e8bdd53291e',
        'e101cc44ead036294bc79c881a0e818b',
        'e23f62bb4794ee6a7fdd0518ed16e820',
        'e24fb21f7cb7998d94b2f4c4a75fd722',
        'e29e8121d93af93bba803759c05367b0',
        'e4915635a488cbfc4c3a35cee92bb95b',
        'e4ada697d05ac7acf9907e8bdd53291e',
        'e56e77c6eb21d9bdf577ff4de1ac394c',
        'e593aa021f3fa324530647fc03dd20dc',
        'e5a5dac174982cea2dcdfc29ed7a2492',
        'e656d6586d481f41eb69804478f9c547',
        'e824b049f16b29f19ab27ff78a8ea481',
        'e8b48d395d3d8744e53e6e0633163da8',
        'e9371d3abbb3bb7265bca0cae1ecfff5',
        'ed55f39e04668bf9837048966ef3fcb9',
        'ed8aff7768d2cc3e45bcca2603f7a948',
        'edfcffbdd585d00ec41b4a535d52e063',
        'ee007f1aac12fbe549a44197486ae284',
        'ee3ca78e36c6a7b2a367d39c852840d5',
        'ee74f5bfb0d7c8a5bd288303be3d57e7',
        'ee77714471b911baeeff0dee910fd183',
        'f0611ec9ff89209bf10c4513652c1c5e',
        'f2279b29f7d45bbd744c199e33849f2a',
        'f47cbefc9aa5b6a918431871c8e05789',
        'f4851a2835228377e101b7546e3ee8a7',
        'f49d6c4b75f695c44d34bdc365023cf4',
        'f68e99b57596b33d197a35146ee825cd',
        'f6ffca90c543edf9d6438d5cb8c578c6',
        'f83c3b75f637241aebe67d9b32c3ddf8',
        'f853ac62bc288e48e56a63d21fb60ae9',
        'f9f67fe61dcf46d7e19818797f240d91',
        'fa44223c6f785c60e71da2487cb2ee5b',
        'fca70766d88fc7b9e5054d95cb6a63e2',
        'fd0ccd09330865277602de57eab5e08f',
        'fda8d8820e4d166bd7134844380eaeb0',
        'fec05c85454edafc4310636931b68fdb',
        'ff13595434879bba557ef92e2fa0ccb2',
    ]

    shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(),
        obj_class + '_centered_obj_normalized')

    target_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(),
        obj_class + '_handle' + '_centered_obj_normalized_prep')

    for obj_shapenet_id in shapenet_id_list:
        obj_fname = osp.join(shapenet_obj_dir, obj_shapenet_id,
            'models/model_normalized.obj')

        obj_file_dec = obj_fname.split('.obj')[0] + '_dec.obj'

        # convert mesh with vhacd
        if not osp.exists(obj_file_dec):
            p.vhacd(
                obj_fname,
                obj_file_dec,
                'log.txt',
                concavity=0.0025,
                alpha=0.04,
                beta=0.05,
                gamma=0.00125,
                minVolumePerCH=0.0001,
                resolution=1000000,
                # resolution=10000,
                depth=20,
                planeDownsampling=4,
                convexhullDownsampling=4,
                pca=0,
                mode=0,
                convexhullApproximation=1
            )

        print(f'Shapenet id: {obj_shapenet_id}')

        target_dir = osp.join(target_obj_dir, obj_shapenet_id + '-h', 'models')
        # target_dir = osp.join(target_obj_dir, obj_shapenet_id, 'models')
        util.safe_makedirs(target_dir)

        shutil.copy(obj_file_dec, osp.join(target_dir, 'model_normalized.obj'))













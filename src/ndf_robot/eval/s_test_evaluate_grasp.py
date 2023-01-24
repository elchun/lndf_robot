"""
Script for testing evaluate_grasp
"""

from ndf_robot.eval.evaluate_grasp import EvaluateGrasp, EvaluateGraspSetup, QueryPoints
from ndf_robot.opt.optimizer_lite import OccNetOptimizer
import plotly.express as px
import numpy as np

if __name__ == '__main__':
    config_fname = 'GENERAL_debug.yml'

    setup = EvaluateGraspSetup()
    setup.load_config(config_fname)
    model = setup.create_model()
    gripper_query_pts = setup.create_gripper_query_pts()
    rack_query_pts = setup.create_rack_query_pts()

    shapenet_obj_dir = setup.get_shapenet_obj_dir()
    eval_save_dir = setup.create_eval_dir()
    demo_load_dir = setup.get_demo_load_dir(obj_class='mug')

    grasp_optimizer = OccNetOptimizer(model, query_pts = gripper_query_pts, eval_save_dir=eval_save_dir)

    # grasp_optimizer = setup.create_grasp_optimizer(model, gripper_query_pts, eval_save_dir=eval_save_dir)
    # place_optimizer = setup.create_place_optimizer(model, rack_query_pts, eval_save_dir=eval_save_dir)

    # evaluator_args = setup.get_evaluator_args()

    experiment = EvaluateNetwork(grasp_optimizer=grasp_optimizer, place_optimizer=place_optimizer,
        seed=setup.get_seed(),
        shapenet_obj_dir=shapenet_obj_dir, eval_save_dir=eval_save_dir,
        demo_load_dir=demo_load_dir, **evaluator_args)

    experiment.load_demos()
    experiment.configure_sim()
    experiment.run_experiment()





    # ---------------------------- #
    # setup = EvaluateGraspSetup()
    # setup.load_config('debug_config.yml')
    # setup.create_model()
    # setup.create_eval_dir('DEBUG')
    # print(setup.get_demo_load_dir())
    # # print(setup.get_shapenet_obj_dir())
    # rect = QueryPoints.generate_rect(500, 2, 2, 0.5, 4)
    # cyl = QueryPoints.generate_cylinder(5000, 0.4, 4, 'y')
    # fig = px.scatter_3d(x=cyl[:, 0], y=cyl[:, 1], z=cyl[:, 2])
    # fig.write_html('debug_cyl.html')

    # config_fname = 'debug_config.yml'

    # setup = EvaluateGraspSetup()
    # setup.load_config(config_fname)
    # model = setup.create_model()
    # query_pts = setup.create_query_pts()
    # shapenet_obj_dir = setup.get_shapenet_obj_dir()
    # eval_save_dir = setup.create_eval_dir()
    # demo_load_dir = setup.get_demo_load_dir(obj_class='mug')
    # optimizer = setup.create_optimizer(model, query_pts, eval_save_dir=eval_save_dir)
    # evaluator_args = setup.get_evaluator_args()

    # experiment = EvaluateGrasp(optimizer=optimizer, seed=setup.get_seed(),
    #     shapenet_obj_dir=shapenet_obj_dir, eval_save_dir=eval_save_dir,
    #     demo_load_dir=demo_load_dir, **evaluator_args)

    # res = []
    # for i in range(1000):
    #     pos = experiment.compute_anyrot_pose(0.4, 0.4, 0, 0)[0]
    #     res.append(pos)

    # res = np.vstack(res)

    # fig = px.scatter_3d(x=res[:, 0], y=res[:, 1], z=res[:, 2])
    # fig.write_html('debug_anypose.html')


    # x offset was 0.5



    # optimizer_lite took 7:50 for 15 trials
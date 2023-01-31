# Local Neural Descriptor Fields (LNDF)

PyTorch implementation for training continuous convolutional neural fields to represent dense correspondence across objects, and using these descriptor fields to mimic demonstrations of a pick-and-place task on a robotic system.


<p align="center">
<img src="./doc/icra_2023.gif" alt="drawing" width="700"/>
</p>

## Setup
### Clone this repo
```
git clone --recursive git@github.com:elchun/lndf_robot.git
cd lndf_robot
```
### Install dependencies (using a virtual environment is highly recommended):
```
pip install -e .
```

### Setup additional tools (Franka Panda inverse kinematics -- unnecessary if not using simulated robot for evaluation):
```
cd pybullet-planning/pybullet_tools/ikfast/franka_panda
python setup.py
```


### Setup torch scatter
Install with pip or see additional instructions [here](https://github.com/rusty1s/pytorch_scatter#pytorch-140).  Replace `${CUDA}` with the appropriate version (cpu, cu102, cu113, cu116).  Generally, the torch
and cuda version of torch-scatter should match those of your pytorch installation.  E.g. the output of  `torch.__version__`

For example, the authors installed with `pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu102.html`

```
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+${CUDA}.html
```

### Setup environment variables
(this script must be sourced in each new terminal where code from this repository is run)
```
source lndf_env.sh
```


## Quickstart Demo
### Download pretrained weights
```
./scripts/download_demo_weights.sh
```

### Download demos
```
./scripts/download_demos.sh
```

### Download object data
```
./scripts/download_obj_data.sh
```

## Evaluate in Simulation
### Download pretrained weights
```
./scripts/download_demo_weights.sh
```

### Download demos
```
./scripts/download_demos.sh
```

### Download object data
```
./scripts/download_obj_data.sh
```

### Run experiment
```
cd src/ndf_robot/eval
CUDA_VISIBLE_DEVICES=0 python3 evaluate_general.py --config_fname {your_config}
```
The configuration files are located in the `eval_configs` directory.  Include the `.yml` extension when specifying the config file.

For example, evaluating a grasping experiment may use the following command:
```
CUDA_VISIBLE_DEVICES=0 python3 evaluate_general.py --config_fname lndf_grasp.yml
```

You may create additional experiments by viewing the `tutorial.yml` config file
in `src/ndf_robot/eval/eval_configs` and creating an appropriate experiment in
`src/ndf_robot/eval/experiments`.

## Train Models
### Overview
Navigate to the `training` directory inside `ndf_robot`.  Run or modify the
shell script `train.sh` to train the convolutional occupancy network.

Two main scripts control how the model is trained:  `train_conv_occupancy_net.py` and `losses.py`.
The shell script calls `train_conv_occupancy_net.py` which allows the user
to specify the model and loss function to use.  These loss functions are defined
in `losses.py`.  A user may use the given loss functions or test additional loss functions

To evaluate a new set of weights, the `s_vis_correspondance.py` script is
convenient.  By setting the model_path and obj_model variables, you can compare
the latent representation of an unrotated and rotated version of the
same object. This allows you to check how strongly SE(3) equivariance
is enforced.

### Loss functions
Four main loss functions are defined in `losses.py`.  The first
two occupancy_net losses are used for training a standard occupancy network
or a convolutional occupancy network.  The latter two contrastive loss functions
implement the distance based loss function which we describe in our publication.
We find that the `contrastive_cross_entropy` loss performs the best, while the
`contrastive_l2` loss works but produces lower success rates.

Both contrastive loss functions select a sample point, $x_0$ and a set of target points $\{x_i\}$.  To enforce SE(3) equivariance, we compare the latent codes of these
points on a rotated and unrotated object.  More specifically, we calculate the
cosine similarity between the latent code of the sample point on an unrotated object and the latent code of the target points on the rotated object.  We
would like target points closer to the sample point to have more similar
latent codes.

**Contrastive Cross Entropy**

In `contrastive_cross_entropy` loss, we enforce this similarity by
calculating the euclidean distance between the sample and target points
(assuming both are on the same mug).  We then produce a probability distribution
where the probability assigned to a target point is

$$
p_i = \frac{1 / (d_i + \epsilon)}{\sum_j{1/(d_j + \epsilon)}}
$$

where $d_i$ is the distance between the sample and target points and $\epsilon$ is
a constant set prior to training which dictates how sharply $p_i$ increases as the target points get closer to the sample point.
We then consider this loss as a prediction problem where the cosine similarity
between $x_0$ and $x_i$ predicts that $x_i = x_0$ with probability $p_i$.

**Contrastive L2**

In `contrastive_l2` loss, we also calculate the euclidean distance between
the sample and target points.  However, for each target point $x_i$, we instead create a target value $t_i$ defined by

$$
t_i = 2*\exp({-d_i^2 * \frac{\ln{2}}{r^2}}) - 1
$$

Where $d_i$ is the distance between $x_i$ and $x_0$ and $r$ is a constant that defines the steepness of this distribution.  This function enforces an exponential decay as a function of $d_i$ where $r$ is the distance at which $t_i = 0$.  Additionally, this function is bound between 1 and -1.

We calculate then drive the cosine similarity of the latent code of $x_0$ and $x_i$ to this $t_i$ value using L2 loss.




## Add demonstrations
Navigate to the `demonstrations` directory inside `ndf_robot`.  Use the
`label_demos.py` script to log new demonstrations.  The current evaluator uses
a slightly different file format than the output of `label_demos.py` so you
must run the `convert_demos.py` script to convert the demonstrations into the
new file format.

## Advanced Changes

- Add or modify models: Use the `descriptions` directory to modify or add new models.
- Modify optimizer: See both the deep or geometric optimizer classes in the `opt` directory.

## Data Generation
**Download all the object data assets**
```
./scripts/download_obj_data.sh
```

**Run data generation**
```
cd src/ndf_robot/data_gen
python shapenet_pcd_gen.py \
    --total_samples 100 \
    --object_class mug \
    --save_dir test_mug \
    --rand_scale \
    --num_workers 2
```
More information on dataset generation can be found [here](./doc/dataset.md).

## Collect new demonstrations with teleoperated robot in PyBullet
Make sure you have downloaded all the object data assets (see Data Generation section)

**Run teleoperation pipeline**
```
cd src/ndf_robot/demonstrations
python label_demos.py --exp test_bottle --object_class bottle --with_shelf
```
More information on collecting robot demonstrations can be found [here](./doc/demonstrations.md).






------------ TODO: EDIT ALL STUFF BELOW THIS LINE -------------------

This is the reference implementation for our paper:

### Neural Descriptor Fields: SE(3)-Equivariant Object Representations for Manipulation
<p align="center">
<img src="./doc/point.gif" alt="drawing" width="320"/>
<img src="./doc/pose.gif" alt="drawing" width="320"/>
</p>

[PDF](https://arxiv.org/abs/2112.05124) | [Video](https://youtu.be/dXl9xI2LrRw)

[Anthony Simeonov*](https://anthonysimeonov.github.io/), [Yilun Du*](https://yilundu.github.io/), [Andrea Tagliasacchi](https://taiya.github.io/), [Joshua B. Tenenbaum](http://web.mit.edu/cocosci/josh.html), [Alberto Rodriguez](http://meche.mit.edu/people/faculty/ALBERTOR@MIT.EDU), [Pulkit Agrawal**](http://people.csail.mit.edu/pulkitag/), [Vincent Sitzmann**](https://www.vincentsitzmann.com/)
(*Equal contribution, order determined by coin flip. **Equal advising)

---
## Google Colab
If you want a quickstart demo of NDF without installing anything locally, we have written a [Colab](https://colab.research.google.com/drive/16bFIFq_E8mnAVwZ_V2qQiKp4x4D0n1sG#scrollTo=YxiZ-ZE21wIm). It runs the same demo as the Quickstart Demo section below where a local coordinate frame near one object is sampled, and the corresponding local frame near a new object (with a different shape and pose) is recovered via our energy optimization procedure.

---

## Setup
**Clone this repo**
```
git clone --recursive https://github.com/anthonysimeonov/ndf_robot.git
cd ndf_robot
```
**Install dependencies** (using a virtual environment is highly recommended):
```
pip install -e .
```

**Setup additional tools** (Franka Panda inverse kinematics -- unnecessary if not using simulated robot for evaluation):
```
cd pybullet-planning/pybullet_tools/ikfast/franka_panda
python setup.py
```

**Setup environment variables** (this script must be sourced in each new terminal where code from this repository is run)
```
source ndf_env.sh
```

## Quickstart Demo
**Download pretrained weights**
```
./scripts/download_demo_weights.sh
```

**Download data assets**
```
./scripts/download_demo_data.sh
```

**Run example script**
```
cd src/ndf_robot/eval
python ndf_demo.py
```

The code in the `NDFAlignmentCheck` class in the file [`src/ndf_robot/eval/ndf_alignment.py`](src/ndf_robot/eval/ndf_alignment.py) contains a minimal implementation of our SE(3)-pose energy optimization procedure. This is what is used in the Quickstart demo above. For a similar implementation that is integrated with our pick-and-place from demonstrations pipeline, see [`src/ndf_robot/opt/optimizer.py`](src/ndf_robot/opt/optimizer.py)

## Training
**Download all data assets**

If you want the full dataset (~150GB for 3 object classes):
```
./scripts/download_training_data.sh
```
If you want just the mug dataset (~50 GB -- other object class data can be downloaded with the according scripts):
```
./scripts/download_mug_training_data.sh
```

If you want to recreate your own dataset, see Data Generation section

**Run training**
```
cd src/ndf_robot/training
python train_vnn_occupancy_net.py --obj_class all --experiment_name  ndf_training_exp
```
More information on training [here](doc/training.md)

## Evaluation with simulated robot
Make sure you have set up the additional inverse kinematics tools (see Setup section)

**Download all the object data assets**
```
./scripts/download_obj_data.sh
```

**Download pretrained weights**
```
./scripts/download_demo_weights.sh
```

**Download demonstrations**
```
./scripts/download_demo_demonstrations.sh
```

**Run evaluation**

If you are running this command on a remote machine, be sure to remove the `--pybullet_viz` flag!
```
cd src/ndf_robot/eval
CUDA_VISIBLE_DEVICES=0 python evaluate_ndf.py \
        --demo_exp grasp_rim_hang_handle_gaussian_precise_w_shelf \
        --object_class mug \
        --opt_iterations 500 \
        --only_test_ids \
        --rand_mesh_scale \
        --model_path multi_category_weights \
        --save_vis_per_model \
        --config eval_mug_gen \
        --exp test_mug_eval \
        --pybullet_viz
```
More information on experimental evaluation can be found [here](./doc/eval.md).

## Data Generation
**Download all the object data assets**
```
./scripts/download_obj_data.sh
```

**Run data generation**
```
cd src/ndf_robot/data_gen
python shapenet_pcd_gen.py \
    --total_samples 100 \
    --object_class mug \
    --save_dir test_mug \
    --rand_scale \
    --num_workers 2
```
More information on dataset generation can be found [here](./doc/dataset.md).

## Collect new demonstrations with teleoperated robot in PyBullet
Make sure you have downloaded all the object data assets (see Data Generation section)

**Run teleoperation pipeline**
```
cd src/ndf_robot/demonstrations
python label_demos.py --exp test_bottle --object_class bottle --with_shelf
```
More information on collecting robot demonstrations can be found [here](./doc/demonstrations.md).

# Citing
If you find our paper or this code useful in your work, please cite our paper:
```
@article{simeonovdu2021ndf,
  title={Neural Descriptor Fields: SE(3)-Equivariant Object Representations for Manipulation},
  author={Simeonov, Anthony and Du, Yilun and Tagliasacchi, Andrea and Tenenbaum, Joshua B. and Rodriguez, Alberto and Agrawal, Pulkit and Sitzmann, Vincent},
  journal={arXiv preprint arXiv:2112.05124},
  year={2021}
}
```

# Acknowledgements
Parts of this code were built upon the implementations found in the [occupancy networks repo](https://github.com/autonomousvision/occupancy_networks) and the [vector neurons repo](https://github.com/FlyingGiraffe/vnn). Check out their projects as well!

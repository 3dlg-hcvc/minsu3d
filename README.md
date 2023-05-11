# MINSU3D
MINSU3D：**Min**kowskiEngine-powered **S**cene **U**nderstanding in **3D** contains reimplementation of state-of-the-art 3D scene understanding methods on point clouds powered by [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine).  

<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/Lightning-792DE4?style=for-the-badge&logo=pytorch-lightning&logoColor=white"></a>
<a href="https://wandb.ai/site"><img alt="WandB" src="https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white"></a>

We support the following instance segmentation methods:
- [PointGroup](https://github.com/dvlab-research/PointGroup)
- [HAIS](https://github.com/hustvl/HAIS)
- [SoftGroup](https://github.com/thangvubk/SoftGroup)

We also provide bounding boxes predictions based on instance segmentation for 3D object detection.

## Features
- Highly-modularized design enables researchers to easily add different models and datasets.
- Multi-GPU and distributed training support through [PytorchLightning](https://github.com/Lightning-AI/lightning).
- Better logging with [W&B](https://github.com/wandb/wandb), periodic evaluation during training.
- Easy experiment configuration and management with [Hydra](https://github.com/facebookresearch/hydra).
- Unified and optimized C++ and CUDA extensions.

## Changelog
1. MINSU3D v2.0 release, ~1.8 times faster, ~4GB less CPU memory usage and ~400MB less GPU memory usage

## Setup

### Conda (recommended)
We recommend the use of [miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage system dependencies.

```shell
# create and activate the conda environment
conda create -n minsu3d python=3.10
conda activate minsu3d

# install PyTorch 2.0
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia

# install Python libraries
pip install .

# install OpenBLAS
conda install openblas-devel --no-deps -c anaconda

# install MinkowskiEngine
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
--install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"

# install C++ extensions
export CPATH=$CONDA_PREFIX/include:$CPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd minsu3d/common_ops
pip install .
```

### Pip (without conda)
Note: Setting up with pip (no conda) requires [OpenBLAS](https://github.com/xianyi/OpenBLAS) to be pre-installed in your system.

```shell
# create and activate the virtual environment
virtualenv --no-download env
source env/bin/activate

# install PyTorch 2.0
pip3 install torch

# install Python libraries
pip install .

# install OpenBLAS and SparseHash via APT
sudo apt install libopenblas-dev

# install MinkowskiEngine
pip install MinkowskiEngine

# install C++ extensions
cd minsu3d/common_ops
pip install .
```

## Data Preparation

### ScanNet v2 dataset
1. Download the [ScanNet v2](http://www.scan-net.org/) dataset and put it under `minsu3d/data/scannetv2`. To acquire the access to the dataset, please refer to their [instructions](https://github.com/ScanNet/ScanNet#scannet-data). You will get a `download-scannet.py` script after your request is approved:
```shell
# about 10.7GB in total
python download-scannet.py -o data/scannet --type _vh_clean_2.ply
python download-scannet.py -o data/scannet --type _vh_clean.aggregation.json
python download-scannet.py -o data/scannet --type _vh_clean_2.0.010000.segs.json
```

The raw dataset files should be organized as follows:

```shell
minsu3d
├── data
│   ├── scannetv2
│   │   ├── scans
│   │   │   ├── [scene_id]
│   │   │   │   ├── [scene_id]_vh_clean_2.ply
│   │   │   │   ├── [scene_id]_vh_clean_2.0.010000.segs.json
│   │   │   │   ├── [scene_id].aggregation.json
│   │   │   │   ├── [scene_id].txt
```

2. Preprocess the data, it converts original meshes and annotations to `.pth` data:
```shell
python data/scannetv2/preprocess_all_data.py data=scannetv2
```

## Training, Inference and Evaluation
Note: Configuration files are managed by [Hydra](https://hydra.cc/), you can easily add or override any configuration attributes by passing them as arguments.
```shell
# log in to WandB
wandb login

# train a model from scratch
# available model_name: pointgroup, hais, softgroup
# available dataset_name: scannetv2
python train.py model={model_name} data={dataset_name} experiment_name={experiment_name}

# train a model from scratch with 2 GPUs
python train.py model={model_name} data={dataset_name} model.trainer.devices=2

# train a model from a checkpoint
python train.py model={model_name} data={dataset_name} model.ckpt_path={checkpoint_path}

# test a pretrained model
python test.py model={model_name} data={dataset_name} model.ckpt_path={pretrained_model_path}

# evaluate inference results
python eval.py model={model_name} data={dataset_name} experiment_name={experiment_name}

# examples:
# python train.py model=pointgroup data=scannetv2 model.trainer.max_epochs=480
# python test.py model=pointgroup data=scannetv2 model.ckpt_path=PointGroup_best.ckpt
# python eval.py model=hais data=scannetv2 experiment_name=run_1
```

## Pretrained Models

We provide pretrained models for ScanNet v2. The pretrained model, corresponding config file, and performance on ScanNet v2 val set are given below.  Note that all MINSU3D models are trained from scratch. After downloading a pretrained model, run `test.py` to do inference as described in the above section.

### ScanNet v2 val set
| Model      | Code | mean AP | AP 50% | AP 25% | Bbox AP 50% | Bbox AP 25% | Download |
|:-----------|:--------|:--------|:-------|:-------|:------------|:------------|:---------|
| MINSU3D PointGroup | [config](https://github.com/3dlg-hcvc/minsu3d-internal/blob/main/config/model/pointgroup.yaml) \| [model](https://github.com/3dlg-hcvc/minsu3d-internal/blob/main/minsu3d/model/pointgroup.py) | 36.4 | 57.9 | 71.1 | 49.9 | 60.0 | [link](https://aspis.cmpt.sfu.ca/projects/minsu3d/pretrained_models/PointGroup_best.ckpt)|
| [Official PointGroup](https://github.com/dvlab-research/PointGroup) | - | 35.2 | 57.1 | 71.4 | - | - | - |
| MINSU3D HAIS | [config](https://github.com/3dlg-hcvc/minsu3d-internal/blob/main/config/model/hais.yaml) \| [model](https://github.com/3dlg-hcvc/minsu3d-internal/blob/main/minsu3d/model/hais.py)  | 42.6 | 61.9 | 72.6 | 51.4 | 62.9 | [link](https://aspis.cmpt.sfu.ca/projects/minsu3d/pretrained_models/HAIS_best.ckpt) |
| [Official HAIS (retrained)](https://github.com/hustvl/HAIS)  | - | 42.2 | 61.0   | 72.9 | - | - | - |
| [Official HAIS](https://github.com/hustvl/HAIS)  | - | 44.1 | 64.4   | 75.7   | - | - | - |
| MINSU3D SoftGroup | [config](https://github.com/3dlg-hcvc/minsu3d-internal/blob/main/config/model/softgroup.yaml) \| [model](https://github.com/3dlg-hcvc/minsu3d-internal/blob/main/minsu3d/model/softgroup.py)  | 42.3   | 65.1 | 77.8 | 55.8 | 69.3 | [link](https://aspis.cmpt.sfu.ca/projects/minsu3d/pretrained_models/SoftGroup_best.ckpt) |
| [Official SoftGroup](https://github.com/thangvubk/SoftGroup<sup>1</sup>) | - | 46.0 | 67.6   | 78.9 | 59.4 | 71.6 | - |

<sup>1</sup> The official pretrained SoftGroup model was trained with HAIS checkpoint as pretrained backbone.

<sup>2</sup> The MINSU3D HAIS model's scores are 2-3 lower than the official pretrained HAIS's. To investigate, we retrained the official HAIS model using their code, the best scores we can get are 42.2 / 61.0 / 72.9 for mean AP / AP 50% / AP 25%, which match our MINSU3D HAIS model's scores.

## Visualization
We provide scripts to visualize the predicted segmentations and bounding boxes. To use the visualization scripts, place the mesh (ply) file from the Scannet dataset as follows.

```
minsu3d
├── data
│   ├── scannetv2
│   │   ├── scans
│   │   │   ├── [scene_id]
|   |   |   |   ├── [scene_id]_vh_clean_2.ply
```

To visualize the predictions, use `visualize/scannet/generate_ply.py` to generate ply files with vertices colored according to the semantic or instance.
```shell
cd visualize/scannet
python generate_prediction_ply.py --predict_dir {path to the predictions} --split {test/val/train} --bbox --mode {semantic/instance} --output_dir {output directory of ply files}

# example:
# python generate_prediction_ply.py --predict_dir ../../output/ScanNet/PointGroup/test/predictions/instance --split val --bbox --mode semantic --output_dir output_ply
```

The `--mode` option allows you to specify the color mode.  
In the 'semantic' mode, objects with the same semantic prediction will have the same color.  
In the 'instance' mode, each independent object instance will have an unique color, allowing the user to check how well the model performs on instance segmentation.  

The `--bbox` option allows you to generate ply file that uses bounding box to specify the position of objects.

| Semantic Segmentation(color)              | Instance Segmentation(color)           |
|:-----------------------------------:|:-------------------------------:|
| <img src="https://github.com/3dlg-hcvc/minsu3d-internal/blob/main/visualize/example/color_semantic.png" width="400"/> | <img src="https://github.com/3dlg-hcvc/minsu3d-internal/blob/main/visualize/example/color_instance.png" width="400"/> |

| Semantic Segmentation(bbox)              | Instance Segmentation(bbox)           |
|:-----------------------------------:|:-------------------------------:|
| <img src="https://github.com/3dlg-hcvc/minsu3d-internal/blob/main/visualize/example/bbox_semantic.png" width="400"/> | <img src="https://github.com/3dlg-hcvc/minsu3d-internal/blob/main/visualize/example/bbox_instance.png" width="400"/> |

If you find that many bounding boxes are overlapping, you can choose to do non maximum suppression during the inference phase. This can be achieved by adjusting `TEST_NMS_THRESH` in the config file

## Performance

**Test environment**
- CPU: Intel Core i9-9900K @ 3.60GHz × 16
- RAM: 64GB
- GPU: NVIDIA GeForce RTX 2080 Ti 11GB
- System: Ubuntu 22.04.2 LTS

**Training time in total (train set only, without validation)**
| Model      | Epochs | Batch Size | MINSU3D | Official Version |
|:-----------|:--------|:--------|:--------|:-------|
| [PointGroup](https://github.com/dvlab-research/PointGroup) | 450 | 4 | 28hr | 51hr |
| [HAIS](https://github.com/hustvl/HAIS)| 450 | 4 | 38hr | 60hr |
| [SoftGroup](https://github.com/thangvubk/SoftGroup) | 256 | 4 | (to be updated) | 30hr |


**Inference time per scene (avg)**
| Model      | MINSU3D | Official Version |
|:-----------|:--------|:-------|
| [PointGroup](https://github.com/dvlab-research/PointGroup) | (to be updated) | 176ms |
| [HAIS](https://github.com/hustvl/HAIS)| (to be updated) | 165ms |
| [SoftGroup](https://github.com/thangvubk/SoftGroup) | (to be updated) | 204ms |

## Customization
MINSU3D allows for easy additions of custom datasets and models. All code under `minsu3d/data/dataset` and `minsu3d/model` are automatically registered and managed by [Hydra](https://github.com/facebookresearch/hydra) using configuration files under `config/data` and `config/model`, respectively. 

### Implement your own dataset
1. Add a new dataset config file (.yaml) at `config/data/{your_dataset}.yaml`.
2. Add a new dataset processing code at `minsu3d/data/dataset/{your_dataset}.py`, it should inherit the `GeneralDataset()` class from `minsu3d/data/dataset/general_dataset.py`.

### Implement your own model
1. Add a new model config file (.yaml) at `config/model/{your_model}.yaml`.
2. Add a new model code at `minsu3d/model/{your_model}.py`, it should inherit the `GeneralModel()` class from `minsu3d/model/general_model.py`.

## Acknowledgement
This repo is built upon the [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine), [PointGroup](https://github.com/dvlab-research/PointGroup), [HAIS](https://github.com/hustvl/HAIS), and [SoftGroup](https://github.com/thangvubk/SoftGroup).  We train our models on [ScanNet](https://github.com/ScanNet/ScanNet). If you use this repo and the pretrained models, please cite the original papers.

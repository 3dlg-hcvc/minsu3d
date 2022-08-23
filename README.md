# MINSU3D
MINSU3D：**Min**kowskiEngine-powered **S**cene **U**nderstanding in **3D** is a repository that contains reimplementation of state-of-the-art 3D scene understanding methods (see below) powered by [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine).

Supported 3D scene understanding methods:
- [PointGroup](https://github.com/dvlab-research/PointGroup)
- [HAIS](https://github.com/hustvl/HAIS)
- [SoftGroup](https://github.com/thangvubk/SoftGroup)

## Features
- Highly-modularized design enables researchers to easily add different models and datasets.
- Focus on research by letting [PytorchLightning](https://github.com/Lightning-AI/lightning) handle engineering code.
- Easy multi-GPU training.
- Easy experiment configuration and management with [Hydra](https://github.com/facebookresearch/hydra) and [W&B](https://github.com/wandb/wandb).


## Setup

**Environment requirements**
- CUDA 11.X
- Python 3.8

### Conda (recommended)
```shell
# create and activate the conda environment
conda create -n minsu3d python=3.8
conda activate minsu3d

# install PyTorch 1.8.2
conda install pytorch cudatoolkit=11.1 -c pytorch-lts -c nvidia

# install Python libraries
pip install -e .

# install MinkowskiEngine
conda install openblas-devel -c anaconda
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"

# install C++ extensions
conda install -c bioconda google-sparsehash
export CPATH=$CONDA_PREFIX/include:$CPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd minsu3d/common_ops
python setup.py develop
```

### Pip
Note: Setting up via Pip requires [OpenBLAS](https://github.com/xianyi/OpenBLAS) and [SparseHash](https://github.com/sparsehash/sparsehash) pre-installed in your system.
```shell
# install OpenBLAS and SparseHash via APT
sudo apt install libopenblas-dev libsparsehash-dev
```

```shell
# create and activate the virtual environment
virtualenv --no-download env
source env/bin/activate

# install PyTorch
pip3 install torch==1.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

# install Python libraries
pip install -e .

# install MinkowskiEngine
pip install MinkowskiEngine

# install C++ extensions
cd minsu3d/common_ops
python setup.py develop
```

## Data Preparation

### ScanNet v2 dataset
1. Download the [ScanNet v2](http://www.scan-net.org/) dataset.
2. Preprocess the data
```shell
cd data/scannet
python prepare_all_data.py data=scannet +raw_scan_path={PATH_TO_SCANNET_V2}/scans
```

## Training, Inference and Evaluation
Note: Configuration files are managed by [Hydra](https://hydra.cc/), you can easily add or override any configuration attributes by passing them as arguments.
```shell
# train a model from scratch
python train.py model={model_name} data={dataset_name}

# train a model from scratch with 2 GPUs
python train.py model={model_name} data={dataset_name} model.trainer.devices=2

# train a model from a checkpoint
python train.py model={model_name} data={dataset_name} model.ckpt_path={checkpoint_path}

# test a pretrained model
python test.py model={model_name} data={dataset_name} model.ckpt_path={pretrained_model_path}

# evaluate inference results
python eval.py model={model_name} data={dataset_name} model.model.experiment_name={experiment_name}

# examples:
# python train.py model=pointgroup data=scannet model.trainer.max_epochs=480
# python test.py model=pointgroup data=scannet model.ckpt_path=PointGroup_best.ckpt
# python eval.py model=hais data=scannet model.model.experiment_name=run_1
```

## Pretrained Models
Note: All models are trained from scratch. We use hyperparameters listed in default config files (.yaml).

### ScanNet v2 val set
| Model      | Code | mean AP | AP 50% | AP 25% | Bbox AP 50% | Bbox AP 25% | Download |
|:-----------|:--------|:--------|:-------|:-------|:------------|:------------|:---------|
| [PointGroup](https://github.com/dvlab-research/PointGroup) | [config](https://github.com/3dlg-hcvc/minsu3d-internal/blob/main/config/model/pointgroup.yaml) \| [model](https://github.com/3dlg-hcvc/minsu3d-internal/blob/main/minsu3d/model/pointgroup.py) | 36.1 | 57.8 | 71.4 | 50.4 | 61.2 | [link](https://aspis.cmpt.sfu.ca/projects/minsu3d/pretrained_models/PointGroup_best.ckpt)|
| [HAIS](https://github.com/hustvl/HAIS)  | [config](https://github.com/3dlg-hcvc/minsu3d-internal/blob/main/config/model/hais.yaml) \| [model](https://github.com/3dlg-hcvc/minsu3d-internal/blob/main/minsu3d/model/hais.py)  | 42.1 | 62.0 | 73.8 | 52.8 | 62.6 | [link](https://aspis.cmpt.sfu.ca/projects/minsu3d/pretrained_models/HAIS_best.ckpt) |
| [SoftGroup](https://github.com/thangvubk/SoftGroup) | [config](https://github.com/3dlg-hcvc/minsu3d-internal/blob/main/config/model/softgroup.yaml) \| [model](https://github.com/3dlg-hcvc/minsu3d-internal/blob/main/minsu3d/model/softgroup.py)  | 42.2 | 65.5 | 78.0 | 56.0 | 69.5 | [link](https://aspis.cmpt.sfu.ca/projects/minsu3d/pretrained_models/SoftGroup_best.ckpt) |

## Visualization
To visualize predictions as mesh, you need to load scannet's mesh file and mega file of each scans. These dataset files should be organized as follows.

```
minsu3d-internal
├── data
│   ├── scannet
│   │   ├── scans
│   │   │   ├── [scene_id]
|   |   |   |   ├── [scene_id]_vh_clean_2.ply & [scene_id].txt
```

You can generate ply files to visualize the predictions of scannet. Please find the `generate_ply.py` under `visualize/scannet`
```shell
cd visualize/scannet
python generate_ply.py --predict_dir {path to the predictions} --split {test/val/train} --bbox --mode {semantic/instance} --output_dir {output directory of ply files}

# example:
# python generate_ply.py --predict_dir ../../output/ScanNet/PointGroup/test/predictions/instance --split val --bbox --mode semantic --output_dir output_ply
```

The `--mode` option allows you to specify the color mode.  
In the 'semantic' mode, objects with the same semantic prediction will have the same color.  
In the 'instance' mode, each independent object will have unique color, which allows the user to check how well the model performs on instance segmentation.  

The `--bbox` option allows you to generate ply file that uses bounding box to specify the position of objects.

| Semantic Segmentation(color)              | Instance Segmentation(color)           |
|:-----------------------------------:|:-------------------------------:|
| <img src="https://github.com/3dlg-hcvc/minsu3d-internal/blob/main/visualize/example/color_semantic.png" width="400"/> | <img src="https://github.com/3dlg-hcvc/minsu3d-internal/blob/main/visualize/example/color_instance.png" width="400"/> |

| Semantic Segmentation(bbox)              | Instance Segmentation(bbox)           |
|:-----------------------------------:|:-------------------------------:|
| <img src="https://github.com/3dlg-hcvc/minsu3d-internal/blob/main/visualize/example/bbox_semantic.png" width="400"/> | <img src="https://github.com/3dlg-hcvc/minsu3d-internal/blob/main/visualize/example/bbox_instance.png" width="400"/> |

With `--nms`, the program will perform non-maximum suppression before generating the bounding box. This will alleviate the overlapping of the bounding boxes.

## Performance

**Test environment**
- CPU: Intel Core i9-9900K @ 3.60GHz × 16
- RAM: 64GB
- GPU: NVIDIA GeForce RTX 2080 Ti 11GB
- System: Ubuntu 20.04.2 LTS

**Training time per scene**
| Model      | MINSU3D | Official Version |
|:-----------|:--------|:-------|
| [PointGroup](https://github.com/dvlab-research/PointGroup) | 420ms | 383ms |
| [HAIS](https://github.com/hustvl/HAIS)| 475ms | 432ms |
| [SoftGroup](https://github.com/thangvubk/SoftGroup) | 511ms | 357ms |

**Inference time per scene**
| Model      | MINSU3D | Official Version |
|:-----------|:--------|:-------|
| [PointGroup](https://github.com/dvlab-research/PointGroup) | 179ms | 176ms |
| [HAIS](https://github.com/hustvl/HAIS)| 160ms | 165ms |
| [SoftGroup](https://github.com/thangvubk/SoftGroup) | 165ms | 204ms |

**Evaluation scores on ScanNet v2 val set**
| Model      | mean AP | AP 50% | AP 25% | Bbox AP 50% | Bbox AP 25% |
|:-----------|:--------|:-------|:-------|:------------|:------------|
| MINSU3D PointGroup | 36.1 | 57.8 | 71.4 | 50.4 | 61.2 |
| [Official PointGroup](https://github.com/dvlab-research/PointGroup) | 35.2 | 57.1 | 71.4 | - | - |
| MINSU3D HAIS | 42.1 | 62.0 | 73.8 | 52.8 | 62.6 |
| [Official HAIS](https://github.com/hustvl/HAIS)  | 44.1 | 64.4 | 75.7 | - | - | 
| MINSU3D SoftGroup | 42.2 | 65.5 | 78.0 | 56.0 | 69.5 |
| [Official SoftGroup](https://github.com/thangvubk/SoftGroup) | 46.0 | 67.6 | 78.9 | 59.4 | 71.6 |

## Customization
MINSU3D supports custom datasets and models. All code under `minsu3d/data/dataset` and `minsu3d/model` are automatically registered and managed by [Hydra](https://github.com/facebookresearch/hydra) using configuration files under `config/data` and `config/model`, respectively. 

### Use your own dataset
1. Add a new dataset config file (.yaml) at `config/data/{your_dataset}.yaml`.
2. Add a new dataset processing code at `minsu3d/data/dataset/{your_dataset}.py`, it should inherit the `GeneralDataset()` class from `lib/data/dataset/general_dataset.py`.

### Implement your own model
1. Add a new model config file (.yaml) at `config/model/{your_model}.yaml`.
2. Add a new model code at `minsu3d/model/{your_model}.py`.

## Acknowledgement
This repo is built upon [PointGroup](https://github.com/dvlab-research/PointGroup), [HAIS](https://github.com/hustvl/HAIS), [SoftGroup](https://github.com/thangvubk/SoftGroup) and [ScanNet](https://github.com/ScanNet/ScanNet). To use their work, please cite them.

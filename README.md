# minsu3d
**Min**kowskiEngine-powered **S**cene **U**nderstanding in **3D**

The repository contains reimplementation of state-of-the-art 3D scene understanding methods (PointGroup, SoftGroup, HAIS...) powered by [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine).

## Features
- Highly-modularized design enables researchers to easily add different models and datastes.
- Focus on research by letting [PytorchLightning](https://github.com/Lightning-AI/lightning) handle engineering code.
- Easy experiment configuration and management with [Hydra](https://github.com/facebookresearch/hydra) and [W&B](https://github.com/wandb/wandb).

## TODOs
- Models
  - [x] [PointGroup](https://arxiv.org/abs/2004.01658)
  - [ ] [SoftGroup](https://arxiv.org/abs/2203.01509)
  - [ ] [HAIS](https://arxiv.org/abs/2108.02350)
- Data
  - [x] [ScanNet](http://www.scan-net.org/)
  - [ ] MultiScan

## Setup

**Environment requirements**
- Ubuntu 20.04 LTS
- CUDA 11.X
- Python 3.8
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine#requirements)

### Anaconda (recommended)
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

# train a model from a checkpoint
python train.py model={model_name} data={dataset_name} model.ckpt_path={checkpoint_path}

# test and evaluate a pretrained model
python test.py model={model_name} data={dataset_name} model.ckpt_path={pretrained_model_path}

# examples:
# python train.py model=pointgroup data=scannet model.trainer.max_epochs=480
# python test.py model=pointgroup data=scannet model.ckpt_path=pretrained.ckpt
```

## Pretrained Models
...

## Visualization
You can generate ply files to visualize the predictions of scannet. Please find the `generate_ply.py` under `visualize/scannet`
```shell
cd visualize/scannet
python generate_ply.py --predict_dir {path to the predictions} --split {test/val/train} --mode {semantic/instance} --output_dir {output directory of ply files}

# example:
# python generate_ply.py --predict_dir ../../output/ScanNet/PointGroup/test/predictions/instance --split val --mode semantic --output_dir output_ply
```
The 'mode' option allows you to specify the color mode.  
In the 'semantic' mode, objects with the same semantic prediction will have the same color.  
In the 'instance' mode, each independent object will have unique color, which allows the user to check how well the model performs on instance segmentation.  

Example:  
1. Semantic mode  
![image](https://github.com/3dlg-hcvc/minsu3d-internal/blob/main/visualize/example/visualize_example_semantic.png)
2. Instance mode  
![image](https://github.com/3dlg-hcvc/minsu3d-internal/blob/main/visualize/example/visualize_example_instance.png)



## Customization

### Use your own dataset
1. Add a new dataset config file (.yaml) at `config/data/{your_dataset}.yaml`
2. Add a new dataset processing code at `minsu3d/data/dataset/{your_dataset}.py`, it should inherit the `GeneralDataset()` class from `lib/data/dataset/general_dataset.py`

### Implement your own model
1. Add a new model config file (.yaml) at `config/model/{your_model}.yaml`
2. Add a new model code at `minsu3d/model/{your_model}.py`

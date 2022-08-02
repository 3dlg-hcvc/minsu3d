# PointGroup-MinkowskiEngine

## Installation

**Environment requirements**
- CUDA 11.X
- Python 3.9

### Install via Conda (recommended)
```shell
# create and activate the conda environment
conda create -n pointgroup python=3.9
conda activate pointgroup

# install PyTorch
conda install pytorch cudatoolkit=11.3 -c pytorch

# install Python libraries
pip install -e .

# install MinkowskiEngine
conda install openblas-devel -c anaconda
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"

# install C++ extensions
conda install -c bioconda google-sparsehash
export CPATH=$CONDA_PREFIX/include:$CPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd minpg/lib/common_ops
python setup.py develop
```

### Install via Pip
```shell
# create and activate the virtual environment
virtualenv --no-download env
source env/bin/activate

# install PyTorch
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu113

# install Python libraries
pip install -e .

# install MinkowskiEngine
pip install MinkowskiEngine

# install C++ extensions
cd minpg/lib/common_ops
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
### MultiScan dataset
Comming soon ...

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
# python train.py model=hais data=scannet
# python train.py model=pointgroup data=multiscan model.trainer.max_epochs=480
# python test.py model=softgroup data=multiscan model.ckpt_path=pretrained.ckpt
```

## Pretrained Models
...

## Visualization
...

## Customization

### Use your own dataset
1. Add a new dataset config file (.yaml) at `config/data/{your_dataset}.yaml`
2. Add a new dataset processing code at `lib/data/dataset/{your_dataset}.py`, it should inherit the `GeneralDataset()` class from `lib/data/dataset/general_dataset.py`

### Implement your own model
1. Add a new model config file (.yaml) at `config/model/{your_model}.yaml`
2. Add a new model code at `model/{your_model}.py`

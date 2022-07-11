# PointGroup-MinkowskiEngine

## Installation

**Environment requirements**
- CUDA 11.1 or higher
- Python 3.X

### Install via Conda (recommended)
```shell
# create and activate the conda environment
conda create -n pointgroup python=3.8
conda activate pointgroup

# install PyTorch
conda install pytorch cudatoolkit=11.1 -c pytorch-lts -c nvidia

# install Python libraries
pip install -r requirements.txt

# install MinkowskiEngine
conda install openblas-devel -c anaconda
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"

# install C++ extensions
conda install -c bioconda google-sparsehash
export CPATH=$CONDA_PREFIX/include:$CPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd lib/common_ops
python setup.py develop
```

### Install via Pip
```shell
# create and activate the virtual environment
virtualenv --no-download env
source env/bin/activate

# install PyTorch
pip install torch==1.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

# install Python libraries
pip install -r requirements.txt

# install MinkowskiEngine
pip install MinkowskiEngine

# install C++ extensions
cd lib/common_ops
python setup.py develop
```

## Data Preparation

### ScanNet v2 dataset
1. Download the [ScanNet v2](http://www.scan-net.org/) dataset.
2. Preprocess the data
```shell
cd data/scannet
sh prepare_all_data.sh
```

## Training, Inference and Evaluation
Note: Configuration files are managed by [Hydra](https://hydra.cc/), you can easily add or override any configuration attributes by passing them as arguments.
```shell
# train a model from scratch
python train.py --model={model_name} --data={dataset_name}

# train a model from a checkpoint
python train.py --model={model_name} --data={dataset_name} --model.ckpt_path={checkpoint_path}

# test and evaluate a pretrained model
python test.py --model={model_name} --data={dataset_name} --model.ckpt_path={pretrained_model_path}

# examples:
# python train.py --model=pointgroup --dataset=scannet
# python train.py --model=pointgroup --dataset=multiscan --model.trainer.max_epochs=480
# python test.py --model=softgroup --dataset=multiscan --model.ckpt_path=pretrained.ckpt
```

## Pretrained Models
...

## Visualization
...

## Customization

### Use your own dataset
...

### Implement your own model
...

# PointGroup-MinkowskiEngine

## Setup

```shell
$ git clone https://github.com/3dlg-hcvc/pointgroup-minkowski.git
$ cd pointgroup-minkowski
```

### cs-3dlg-09 (Ubuntu 20.04 LTS)
```shell
$ conda create -n pointgroup python=3.8
$ conda activate pointgroup
$ module load LIB/CUDA/11.1
$ conda install pytorch cudatoolkit=11.1 -c pytorch-lts -c nvidia
$ conda install openblas-devel -c anaconda
$ pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
$ conda install -c bioconda google-sparsehash
$ export CPATH=$CONDA_PREFIX/include:$CPATH
$ export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
$ cd lib/common_ops
$ python setup.py develop
$ cd ../..
$ pip install -r requirements.txt
```

### ComputeCanada (Cedar)
```shell
$ module load StdEnv/2020 python/3.8
$ virtualenv --no-download env
$ source env/bin/activate
$ pip install torch==1.8.0 torchvision==0.9.0 --no-index
$ pip install MinkowskiEngine==0.5.2 --no-index 
$ pip install -r requirements_cc.txt
```
```
$ salloc --time=1:0:0 --gres=gpu:p100:1 --nodes=1 --ntasks=4 --mem=32G --account=rrg-msavva
$ module load StdEnv/2020 gcc/9.3.0 cuda/10.2 sparsehash/2.0.4
$ source env/bin/activate
$ cd lib/pointgroup_ops
$ python setup.py develop
```


## Inference on your own dataset
1. Prepare your data like `data/scannet/prepare_scannet.py`. Only the `mesh (xyz+rgb)` is necessary.
2. Prepare your dataset and dataloader like `lib/dataset/scannet.py`. Only the test branch is necessary, including 'locs', 'locs_scaled' and 'feats' keys. 'id' and 'scene_id' are optional.
3. Configure your own YAML file by replacing data-related arguments in `conf/pointgroup_scannet.yaml` with yours.
4. Inference: `python test.py --split [data_split] --config [YOUR_YAML_FILE]`.
5. Outputs are saved under `log/[dataset]/[model]/test/[datetime]/splited_pred`.
6. You may find `visualize` is useful to convert outputs to .ply files.

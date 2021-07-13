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
$ module load LIB/CUDA/11.1 LIB/CUDNN/8.0.5-CUDA11.1
$ conda install -y -c conda-forge -c pytorch pytorch=1.7.1 cudatoolkit=11.0
$ conda install openblas-devel -c anaconda
$ pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
$ conda install -c bioconda google-sparsehash
$ cd lib/pointgroup_ops
$ python setup.py develop
$ cd ../..
$ pip install -r requirements.txt
```

if `conda install google-sparsehash -c bioconda` doesn’t work for installing pointgroup operations, try out `sudo apt-get install libsparsehash-dev`. ([error: google/dense_hash_map: No such file or directory](https://github.com/facebookresearch/SparseConvNet/issues/96))

### Solar Cluster (Ubuntu 18.04.3 LTS)
```shell
$ srun -J "interactive-bash" --gres=gpu:2080_ti:1 --cpus-per-task=4 --pty bash
$ conda create -n pointgroup python=3.8
$ conda activate pointgroup
$ module load LIB/CUDA/10.2 LIB/CUDNN/7.6.5-CUDA10.2
$ conda install pytorch==1.7.1 cudatoolkit=10.2 -c pytorch
$ conda install openblas-devel -c anaconda
$ pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
$ cd lib/pointgroup_ops
$ python setup.py develop
$ cd ../..
$ pip install -r requirements.txt
```

## Inference on your own dataset
1. Prepare your data like `data/scannet/prepare_scannet.py`. Only the `mesh (xyz+rgb)` is necessary.
2. Prepare your dataset and dataloader like `lib/dataset/scannet.py`. Only the test branch is necessary, including 'locs', 'locs_scaled' and 'feats' keys. 'id' and 'scene_id' are optional.
3. Configure your own YAML file by replacing data-related arguments in `conf/pointgroup_scannet.yaml` with yours.
4. Inference: `python test.py --split [data_split] --config [YOUR_YAML_FILE]`.
5. Outputs are saved under `log/[dataset]/[model]/test/[datetime]/splited_pred`.
6. You may find `visualize` is useful to convert outputs to .ply files.
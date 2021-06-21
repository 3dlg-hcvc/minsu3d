# PointGroup-MinkowskiEngine

## Setup

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

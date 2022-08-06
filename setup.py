from setuptools import find_packages, setup

setup(
    name="minsu3d",
    version="1.0",
    author="3dlg-hcvc",
    url="https://github.com/3dlg-hcvc/pointgroup-minkowski.git",
    description="",
    packages=find_packages(include=("lib", "model")),
    install_requires=["plyfile", "tqdm", "trimesh", "pytorch-lightning==1.6.5", "scipy", "open3d", "wandb", "hydra-core", "h5py"]
)

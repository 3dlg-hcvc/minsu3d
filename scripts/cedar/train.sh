#!/bin/bash
#SBATCH --account=rrg-msavva
#SBATCH --nodes=2             # Request 2 node so all resources are in two nodes.
#SBATCH --gres=gpu:p100:2          # Request 2 GPU "generic resources”. You will get 2 per node.
#SBATCH --tasks-per-node=2    # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in parallel.
#SBATCH --cpus-per-task=4 
#SBATCH --mem=50G       # memory per node
#SBATCH --time=2-15:00
#SBATCH --mail-user=qiruiw@sfu.ca
#SBATCH --mail-type=FAIL
#SBATCH --job-name=pointgroup.xyz.rgb.384e
#SBATCH --output=/home/qiruiw/projects/def-angelx/qiruiw/pointgroup-minkowski/output/scannet/pointgroup/refactor/train/%N-%j.out

module load StdEnv/2020 gcc/9.3.0 cuda/10.2
source $PWD/env/bin/activate

nvidia-smi
echo $SLURM_NODELIST

export NCCL_BLOCKING_WAIT=1 #Pytorch Lightning uses the NCCL backend for inter-GPU communication by default. Set this variable to avoid timeout errors.

# PyTorch Lightning will query the environment to figure out if it is running inside a SLURM batch job
# If it is, it expects the user to have requested one task per GPU.
# If you do not ask for 1 task per GPU, and you do not run your script with "srun", your job will fail!

srun python train.py --num_nodes $SLURM_NNODES
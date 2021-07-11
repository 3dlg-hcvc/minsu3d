#!/bin/bash
#SBATCH -J pointgroup     # Name that will show up in squeue
#SBATCH --gres=gpu:2080_ti:1         
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00       # Max job time is 48 hours
#SBATCH --output=%N-%j.out   # Terminal output to file named (hostname)-(jobid).out
#SBATCH --partition=short     # short partition (allows up to 2 days runtime)

# The SBATCH directives above set options similarly to command line arguments to srun
# Run this script with: sbatch my_experiment.sh
# The job will appear on squeue and output will be written to the current directory
# You can do tail -f <output_filename> to track the job.
# You can kill the job using scancel <job_id> where you can find the <job_id> from squeue

# Your experiment setup logic here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pointgroup
# hostname
# echo $CUDA_AVAILABLE_DEVICES

# Note the actual command is run through srun
srun python train.py

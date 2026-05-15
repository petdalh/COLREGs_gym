#!/bin/bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-iv-imt
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --job-name=colregs_training
#SBATCH --output=logs/job_%j.out

set -eo pipefail

source ~/.bashrc
conda activate vessel_planning

# Keep BLAS/Torch thread pools from oversubscribing the CPUs used by env workers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python train.py --config configuration/config.yaml

# Run:
#  - sbatch {job_name}.slurm
# The to check if it's runnign:
#  - squeue -u pettdalh
# And to watch output
#  - tail -f logs/job_{job_id}.out

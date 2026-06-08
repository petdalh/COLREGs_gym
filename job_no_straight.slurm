#!/bin/bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=32G
#SBATCH --job-name=colregs_multispeed
#SBATCH --output=logs/job_%j.out

set -eo pipefail

source ~/.bashrc
conda activate vessel_planning

# Keep BLAS/Torch thread pools from oversubscribing the CPUs used by env workers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python train.py --config configuration/confiug_no_straight.yaml

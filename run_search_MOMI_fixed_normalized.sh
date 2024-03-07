#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./logs/job.out.%j
#SBATCH -e ./logs/job.err.%j
# Initial working directory:
#SBATCH -D /raven/u/beda/Source/TrapDiffusion/
# Job name
#SBATCH -J trap_diffusion
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#
#SBATCH --mail-type=none
#SBATCH --mail-user=david.berger@tum.de
#SBATCH --time=08:00:00

module purge
module load anaconda/3/2023.03
module load pytorch/gpu-cuda-11.6/2.1.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1
export KERAS_BACKEND="torch"

source venv/bin/activate
python run_search.py --quiet --n 10 --method hyperband -o "MOMI_fixed_normalized_hyperband" --dataset_name "Multi-Occupation, Multi Isotope, fixed matrix, normalized"
deactivate

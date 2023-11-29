#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./logs/job.out.%j
#SBATCH -e ./logs/job.err.%j
# Initial working directory:
#SBATCH -D /raven/u/beda/Source/TrapDiffusion/
# Job name
#SBATCH -J create_trap_diffusion_datasets
#
#SBATCH --ntasks=3
#SBATCH --mem=30000
#
#SBATCH --mail-type=none
#SBATCH --mail-user=david.berger@tum.de
#SBATCH --time=01:00:00

module purge
module load anaconda/3/2023.03
module load pytorch/gpu-cuda-11.6/2.1.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

source venv/bin/activate
srun --exclusive --ntasks=1 --mem=10000 python create_dataset.py --preset SOSI_fixed --quiet &
srun --exclusive --ntasks=1 --mem=10000 python create_dataset.py --preset SOSI_random --quiet &
srun --exclusive --ntasks=1 --mem=10000 python create_dataset.py --preset MOMI_fixed --quiet &
wait
deactivate

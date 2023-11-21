#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./jobs/job.out.%j
#SBATCH -e ./jobs/job.err.%j
# Initial working directory:
#SBATCH -D /raven/u/beda/Source/TrapDiffusion/
# Job name
#SBATCH -J create_trap_diffusion_datasets
#
#SBATCH --ntasks=3
#SBATCH --mem=10000
#
#SBATCH --mail-type=none
#SBATCH --mail-user=david.berger@tum.de
#SBATCH --time=01:00:00

module purge
module load anaconda/3/2023.03
module load pytorch/gpu-cuda-11.6/2.1.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source venv/bin/activate
srun --ntasks=1 python create_dataset.py --preset SOSI_fixed --verbose False
srun --ntasks=1 python create_dataset.py --preset SOSI_random --verbose False
srun --ntasks=1 python create_dataset.py --preset MOMI_fixed --verbose False
nvidia-smi
deactivate

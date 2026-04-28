#!/bin/bash
#SBATCH --job-name=EX_CTFM
#SBATCH --output=output_CTFM.txt
#SBATCH --error=error_CTFM.txt
#SBATCH --partition=bb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mail-user=mccoidc@mcmaster.ca
#SBATCH --no-requeue

srun apptainer exec fractures-code_dolfinx.sif python3 EX_CTFM.py --method Parallelogram
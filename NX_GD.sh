#!/bin/bash
#SBATCH --job-name=NX_GD
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --partition=bb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

srun apptainer exec fractures-code_dolfinx.sif python3 NX_GD.py --write
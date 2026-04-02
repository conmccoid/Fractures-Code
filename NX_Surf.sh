#!/bin/bash
#SBATCH --job-name=NX_Surf
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --partition=bb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mccoidc@mcmaster.ca
#SBATCH --no-requeue

srun apptainer exec fractures-code_dolfinx.sif python3 NX_Surf.py --write
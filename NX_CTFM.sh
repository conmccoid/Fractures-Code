#!/bin/bash
#SBATCH --job-name=NX_CTFM
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --partition=bb
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mccoidc@mcmaster.ca

srun apptainer exec fractures-code_dolfinx.sif python3 NX_CTFM.py
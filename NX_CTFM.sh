#!/bin/bash
#SBATCH --job-name=EX_CTFM
#SBATCH --output=output_CTFM.txt
#SBATCH --error=error_CTFM.txt
#SBATCH --partition=bb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mail-user=mccoidc@mcmaster.ca
#SBATCH --no-requeue

srun apptainer exec fractures-code_dolfinx.sif python3 EX_CTFM.py --method AltMin
srun apptainer exec fractures-code_dolfinx.sif python3 EX_CTFM.py --method CubicBacktracking
srun apptainer exec fractures-code_dolfinx.sif python3 EX_CTFM.py --method Parallelogram
srun apptainer exec fractures-code_dolfinx.sif python3 EX_CTFM.py --method Triangle
srun apptainer exec fractures-code_dolfinx.sif python3 EX_CTFM.py --method Tetrahedron
# srun apptainer exec fractures-code_dolfinx.sif python3 NX_CTFM.py
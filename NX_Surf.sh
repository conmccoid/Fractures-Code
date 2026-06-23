#!/bin/bash
#SBATCH --job-name=NX_Surf
#SBATCH --output=output_Surf.txt
#SBATCH --error=error_Surf.txt
#SBATCH --partition=bb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mccoidc@mcmaster.ca
#SBATCH --no-requeue

srun apptainer exec fractures-code_dolfinx.sif python3 EX_Surf.py --method AltMin --write
srun apptainer exec fractures-code_dolfinx.sif python3 EX_Surf.py --method CubicBacktracking --write
srun apptainer exec fractures-code_dolfinx.sif python3 EX_Surf.py --method Parallelogram --write
srun apptainer exec fractures-code_dolfinx.sif python3 EX_Surf.py --method Triangle --write
srun apptainer exec fractures-code_dolfinx.sif python3 EX_Surf.py --method Tetrahedron --write
srun apptainer exec fractures-code_dolfinx.sif python3 NX_Surf.py
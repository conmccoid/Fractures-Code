from EX_GD import main as EX
from Plotters import plotNX
from mpi4py import MPI
import argparse
import sys

def main(id_list=['GD_AltMin','GD_CubicBacktracking','GD_Parallelogram'],en_list=None, WriteSwitch=True):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if WriteSwitch:
        en_list=[None]*len(id_list) # placeholder for energies if writing to file, will be filled in by each process
        for i, id in enumerate(id_list):
            en_i, id_i=EX(id,WriteSwitch=True)
            en_list[i]=en_i
            id_list[i]=id_i
            comm.Barrier() # ensure all processes have finished writing before moving on

    if rank == 0:
        plotNX('GD',id_list, en_list)

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Run the GD problem with specified parameters.')
    parser.add_argument('--write', action='store_true', default=False, help='Write results to file')
    parser.add_argument('--ids', nargs='+', default=['AltMin','CubicBacktracking','Parallelogram'], help='List of identifiers for different runs')
    args = parser.parse_args()
    main(id_list=args.ids, en_list=None, WriteSwitch=args.write)
    sys.exit()
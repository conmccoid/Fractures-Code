from EX_GD import main as EX
from Utilities import plotNX
from mpi4py import MPI
import argparse
import sys

def main(id_list=['GD_AltMin','GD_CubicBacktracking','GD_Parallelogram'],en_list=None, WriteSwitch=True):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if WriteSwitch:
        en1, id1=EX('AltMin',WriteSwitch=True)
        comm.Barrier() # ensure all processes have finished writing before moving on
        en2, id2=EX('CubicBacktracking',WriteSwitch=True)
        comm.Barrier()
        en3, id3=EX('Parallelogram',WriteSwitch=True)
        comm.Barrier()
        if rank == 0:
            id_list=[id1,id2,id3]
            en_list=[en1,en2,en3]

    if rank == 0:
        plotNX('GD',id_list, en_list)

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Run the GD problem with specified parameters.')
    parser.add_argument('--write', action='store_true', default=False, help='Write results to file')
    args = parser.parse_args()
    main(WriteSwitch=args.write)
    sys.exit()
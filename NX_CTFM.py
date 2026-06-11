from EX_CTFM import main as EX
from Plotters import plotNX
from mpi4py import MPI
import argparse
import sys

def main(id_list=['CTFM_AltMin','CTFM_CubicBacktracking','CTFM_Parallelogram','CTFM_Tetrahedron'],en_list=None, WriteSwitch=True):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if WriteSwitch:
        en1, id1=EX('AltMin',WriteSwitch=True)
        comm.Barrier() # ensure all processes have finished writing before moving on
        en2, id2=EX('CubicBacktracking',WriteSwitch=True)
        comm.Barrier()
        en3, id3=EX('Parallelogram',WriteSwitch=True)
        comm.Barrier()
        en4, id4=EX('Tetrahedron',WriteSwitch=True)
        comm.Barrier()
        if rank == 0:
            id_list=[id1,id2,id3,id4]
            en_list=[en1,en2,en3,en4]

    if rank == 0:
        plotNX('CTFM',id_list, en_list)

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Run the CTFM problem with specified parameters.')
    parser.add_argument('--write', action='store_true', default=False, help='Write results to file')
    args = parser.parse_args()
    main(WriteSwitch=args.write)
    sys.exit()
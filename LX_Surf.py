from EX_Surf import main as EX
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

def main(ell_list):
    marker= ['o', 's', 'D', '^', 'v']
    color= ['g', 'r', 'c', 'm', 'y']
    ms=5
    fig1, ax1=plt.subplots(1,3)
    fig2, ax2=plt.subplots(1,3)
    fig3, ax3=plt.subplots(1,1)
    for ell in ell_list:
        energies, _ = EX('AltMin', ell=ell)
        ax1[0].plot(energies[:,0],energies[:,5],label=f"ell={ell}",ls='--',marker=marker[ell_list.index(ell)],color=color[ell_list.index(ell)],ms=ms)
        if energies.shape[1]>5:
            ax1[1].plot(energies[:,0],energies[:,6],label=f"ell={ell}",ls='--',marker=marker[ell_list.index(ell)],color=color[ell_list.index(ell)],ms=ms)
            ax1[2].plot(energies[:,0],energies[:,6]/np.maximum(1,energies[:,5]),label=f"ell={ell}",ls='--',marker=marker[ell_list.index(ell)],color=color[ell_list.index(ell)],ms=ms)

        ax2[0].plot(energies[:,0],energies[:,1],label=f"ell={ell}",ls='--',marker=marker[ell_list.index(ell)],color=color[ell_list.index(ell)],ms=ms)
        ax2[1].plot(energies[:,0],energies[:,2],label=f"ell={ell}",ls='--',marker=marker[ell_list.index(ell)],color=color[ell_list.index(ell)],ms=ms)
        ax2[2].plot(energies[:,0],energies[:,3],label=f"ell={ell}",ls='--',marker=marker[ell_list.index(ell)],color=color[ell_list.index(ell)],ms=ms)
        ax3.plot(energies[:,0],energies[:,4],label=f"ell={ell}",ls='--',marker=marker[ell_list.index(ell)],color=color[ell_list.index(ell)],ms=ms)

    ax1[0].set_xlabel('t')
    ax1[0].set_ylabel('Iterations')
    # ax1[0].legend()
    ax1[0].set_title('Outer iterations')
    ax1[1].set_xlabel('t')
    ax1[1].set_ylabel('Iterations')
    # ax1[1].legend()
    ax1[1].set_title('Inner iterations')
    ax1[2].set_xlabel('t')
    ax1[2].set_ylabel('Ave. inner / outer')
    ax1[2].legend()
    ax1[2].set_title('Ave. inner per outer')
    fig1.savefig(f"output/LFIG_Surf_its.pdf")

    ax2[0].set_xlabel('t')
    ax2[0].set_ylabel('Elastic energy')
    # ax2[0].legend()
    ax2[1].set_xlabel('t')
    ax2[1].set_ylabel('Dissipated energy')
    # ax2[1].legend()
    ax2[2].set_xlabel('t')
    ax2[2].set_ylabel('Total energy')
    ax2[2].legend()
    fig2.savefig(f"output/LFIG_Surf_energy.pdf")

    ax3.set_xlabel('t')
    ax3.set_ylabel('Time elapsed (s) per load step')
    ax3.legend()
    fig3.savefig(f"output/LFIG_Surf_time.pdf")

import argparse
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the EX_Surf example with varying ell values.')
    parser.add_argument('--ell_list', nargs='+', type=float, default=[0.1, 0.5, 1.0], help='List of ell values to run')
    args = parser.parse_args()
    
    main(args.ell_list)
    sys.exit()
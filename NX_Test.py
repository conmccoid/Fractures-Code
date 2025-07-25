from EX_Test import main as EX
from mpi4py import MPI
import matplotlib.pyplot as plt
import csv
import sys

def main(WriteSwitch=False):
    comm = MPI.COMM_WORLD
    rank = comm.rank

    output, energies=EX('AltMin',WriteSwitch=True)
    if WriteSwitch and rank==0:
        with open(f"output/TBL_Test_AltMin_energy.csv",'w') as csv.file:
            writer=csv.writer(csv.file,delimiter=',')
            writer.writerow(['t','Elastic energy','Dissipated energy','Total energy','Number of iterations'])
            writer.writerows(energies)
    fig1, ax1=plt.subplots(1,3)
    ax1[0].plot(energies[:,0],energies[:,4],label='AltMin iterations')
        
    fig2, ax2=plt.subplots(1,3)
    ax2[0].plot(energies[:,0],energies[:,1],label='AltMin')
    ax2[1].plot(energies[:,0],energies[:,2],label='AltMin')
    ax2[2].plot(energies[:,0],energies[:,3],label='AltMin')
    
    linesearch_list=['fp','bt','tr','ls','2step']
    # linesearch_list=['none']
    for linesearch in linesearch_list:
        
        output, energies=EX('Newton',linesearch,WriteSwitch=True)

        if WriteSwitch and rank==0:
            with open(f"output/TBL_Test_Newton_{linesearch}_energy.csv",'w') as csv.file:
                writer=csv.writer(csv.file,delimiter=',')
                writer.writerow(['t','Elastic energy','Dissipated energy','Total energy','Number of iterations'])
                writer.writerows(energies)
            with open(f"output/TBL_Test_Newton_{linesearch}_its.csv",'w') as csv.file:
                writer=csv.writer(csv.file,delimiter=',')
                writer.writerow(['Quasi-static step','Inner iteration','Outer iteration'])
                writer.writerows(output)

        ax1[0].plot(output[:,0],output[:,2],label=f"{linesearch} outer iterations")
        ax1[1].plot(output[:,0],output[:,1],label=f"{linesearch} inner iterations")
        ax1[2].plot(output[:,0],output[:,1]/output[:,2],label=linesearch)
        
        ax2[0].plot(energies[:,0],energies[:,1],label=f"{linesearch}")
        ax2[1].plot(energies[:,0],energies[:,2],label=linesearch)
        ax2[2].plot(energies[:,0],energies[:,3],label=linesearch)
    
    ax1[0].set_xlabel('t')
    ax1[0].set_ylabel('Iterations')
    ax1[0].legend()
    ax1[1].set_xlabel('t')
    ax1[1].set_ylabel('Iterations')
    ax1[1].legend()
    ax1[2].set_xlabel('t')
    ax1[2].set_ylabel('Ave. inner / outer')
    ax1[2].legend()
    fig1.savefig(f"output/FIG_Test_its.png")

    ax2[0].set_xlabel('t')
    ax2[0].set_ylabel('Elastic energy')
    ax2[0].legend()
    ax2[1].set_xlabel('t')
    ax2[1].set_ylabel('Dissipated energy')
    ax2[1].legend()
    ax2[2].set_xlabel('t')
    ax2[2].set_ylabel('Total energy')
    ax2[2].legend()
    fig2.savefig(f"output/FIG_Test_energy.png")

if __name__== "__main__":
    main()
    sys.exit()
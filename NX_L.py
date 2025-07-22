from EX_L import main as EX
import matplotlib.pyplot as plt
import csv
import sys
import numpy as np

def EX_save(method,linesearch):
    output, energies=EX(method,linesearch)
    if method!='AltMin':
        with open(f"output/TBL_L_{method}_{linesearch}_energy.csv",'w') as csv.file:
            writer=csv.writer(csv.file,delimiter=',')
            writer.writerow(['t','Elastic energy','Dissipated energy','Total energy','Number of iterations'])
            writer.writerows(energies)
        with open(f"output/TBL_L_{method}_{linesearch}_its.csv",'w') as csv.file:
            writer=csv.writer(csv.file,delimiter=',')
            writer.writerow(['Quasi-static step','Inner iteration','Outer iteration'])
            writer.writerows(output)
    else:
        with open(f"output/TBL_L_{method}_energy.csv",'w') as csv.file:
            writer=csv.writer(csv.file,delimiter=',')
            writer.writerow(['t','Elastic energy','Dissipated energy','Total energy','Number of iterations'])
            writer.writerows(energies)
    return output, energies

def EX_plot(linesearch_list):
    energies=np.loadtxt(f"output/TBL_L_AltMin_energy.csv",delimiter=',',skiprows=1)
    fig1, ax1=plt.subplots(1,3)
    ax1[0].plot(energies[:,0],energies[:,4],label='AltMin iterations')
    fig2, ax2=plt.subplots(1,3)
    ax2[0].plot(energies[:,0],energies[:,1],label='AltMin')
    ax2[1].plot(energies[:,0],energies[:,2],label='AltMin')
    ax2[2].plot(energies[:,0],energies[:,3],label='AltMin')
    
    for linesearch in linesearch_list:
        energies = np.loadtxt(f"output/TBL_L_Newton_{linesearch}_energy.csv",delimiter=',',skiprows=1)
        output = np.loadtxt(f"output/TBL_L_Newton_{linesearch}_its.csv",delimiter=',',skiprows=1)
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
    plt.savefig(f"output/FIG_L_its.png")

    ax2[0].set_xlabel('t')
    ax2[0].set_ylabel('Elastic energy')
    ax2[0].legend()
    ax2[1].set_xlabel('t')
    ax2[1].set_ylabel('Dissipated energy')
    ax2[1].legend()
    ax2[2].set_xlabel('t')
    ax2[2].set_ylabel('Total energy')
    ax2[2].legend()
    plt.savefig(f"output/FIG_L_energy.png")

def main(WriteSwitch=False):
    output, energies=EX('AltMin')
    if WriteSwitch and rank==0:
        with open(f"output/TBL_L_AltMin_energy.csv",'w') as csv.file:
            writer=csv.writer(csv.file,delimiter=',')
            writer.writerow(['t','Elastic energy','Dissipated energy','Total energy','Number of iterations'])
            writer.writerows(energies)
    fig1, ax1=plt.subplots(1,3)
    ax1[0].plot(energies[:,0],energies[:,4],label='AltMin iterations')
        
    fig2, ax2=plt.subplots(1,3)
    ax2[0].plot(energies[:,0],energies[:,1],label='AltMin')
    ax2[1].plot(energies[:,0],energies[:,2],label='AltMin')
    ax2[2].plot(energies[:,0],energies[:,3],label='AltMin')
    
    linesearch_list=['tr','ls','2step']
    # linesearch_list=['fp']
    for i, linesearch in enumerate(linesearch_list):
        
        output, energies=EX('Newton',linesearch)

        if WriteSwitch and rank==0:
            with open(f"output/TBL_L_Newton_{linesearch}_energy.csv",'w') as csv.file:
                writer=csv.writer(csv.file,delimiter=',')
                writer.writerow(['t','Elastic energy','Dissipated energy','Total energy','Number of iterations'])
                writer.writerows(energies)
            with open(f"output/TBL_L_Newton_{linesearch}_its.csv",'w') as csv.file:
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
    plt.savefig(f"output/FIG_L_its.png")

    ax2[0].set_xlabel('t')
    ax2[0].set_ylabel('Elastic energy')
    ax2[0].legend()
    ax2[1].set_xlabel('t')
    ax2[1].set_ylabel('Dissipated energy')
    ax2[1].legend()
    ax2[2].set_xlabel('t')
    ax2[2].set_ylabel('Total energy')
    ax2[2].legend()
    plt.savefig(f"output/FIG_L_energy.png")

if __name__== "__main__":
    main()
    sys.exit()
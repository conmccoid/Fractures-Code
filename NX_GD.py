from EX_GradientDamage import main as EX
import matplotlib.pyplot as plt
import csv
import sys

def main(WriteSwitch=False):
    output, energies=EX('AltMin')
    if WriteSwitch:
        with open(f"output/TBL_GD_AltMin_energy.csv",'w') as csv.file:
            writer=csv.writer(csv.file,delimiter=',')
            writer.writerow(['t','Elastic energy','Dissipated energy','Total energy','Number of iterations'])
        with open(f"output/TBL_GD_AltMin_its.csv",'w') as csv.file:
            writer=csv.writer(csv.file,delimiter=',')
            writer.writerow(['Quasi-static step','Outer iteration'])
        if rank==0:
            with open(f"output/TBL_GD_Newton_{linesearch}_energy.csv",'a') as csv.file:
                writer=csv.writer(csv.file,delimiter=',')
                writer.writerow(energies[i_t,:])
        with open(f"output/TBL_GD_Newton_{linesearch}_its.csv",'w') as csv.file:
            writer=csv.writer(csv.file,delimiter=',')
            writer.writerows(output)
    fig1, ax1=plt.subplots(1,2)
    ax1[0].plot(energies[:,0],energies[:,4],label='AltMin iterations')
        
    fig2, ax2=plt.subplots(1,3)
    ax2[0].plot(energies[:,0],energies[:,1],label='AltMin')
    ax2[1].plot(energies[:,0],energies[:,2],label='AltMin')
    ax2[2].plot(energies[:,0],energies[:,3],label='AltMin')
    
    # linesearch_list=['bt','tr','ls','2step']
    linesearch_list=['fp','2step']
    # linesearch_list=['fp']
    for i, linesearch in enumerate(linesearch_list):
        
        output, energies=EX('NewtonLS',linesearch)

        if WriteSwitch:
            with open(f"output/TBL_GD_Newton_{linesearch}_energy.csv",'w') as csv.file:
                writer=csv.writer(csv.file,delimiter=',')
                writer.writerow(['t','Elastic energy','Dissipated energy','Total energy','Number of iterations'])
            with open(f"output/TBL_GD_Newton_{linesearch}_its.csv",'w') as csv.file:
                writer=csv.writer(csv.file,delimiter=',')
                writer.writerow(['Quasi-static step','Inner iteration','Outer iteration'])
            if rank==0:
                with open(f"output/TBL_GD_Newton_{linesearch}_energy.csv",'a') as csv.file:
                    writer=csv.writer(csv.file,delimiter=',')
                    writer.writerow(energies[i_t,:])
            with open(f"output/TBL_GD_Newton_{linesearch}_its.csv",'w') as csv.file:
                writer=csv.writer(csv.file,delimiter=',')
                writer.writerows(output)

        ax1[0].plot(output[:,0],output[:,2],label=f"{linesearch} outer iterations")
        ax1[1].plot(output[:,0],output[:,1],label=f"{linesearch} inner iterations")
        
        ax2[0].plot(energies[:,0],energies[:,1],label=f"{linesearch}")
        ax2[1].plot(energies[:,0],energies[:,2],label=linesearch)
        ax2[2].plot(energies[:,0],energies[:,3],label=linesearch)
    
    ax1[0].set_xlabel('t')
    ax1[0].set_ylabel('Iterations')
    ax1[0].legend()
    ax1[1].set_xlabel('t')
    ax1[1].set_ylabel('Iterations')
    ax1[1].legend()

    ax2[0].set_xlabel('t')
    ax2[0].set_ylabel('Elastic energy')
    ax2[0].legend()
    ax2[1].set_xlabel('t')
    ax2[1].set_ylabel('Dissipated energy')
    ax2[1].legend()
    ax2[2].set_xlabel('t')
    ax2[2].set_ylabel('Total energy')
    ax2[2].legend()

if __name__== "__main__":
    main()
    sys.exit()
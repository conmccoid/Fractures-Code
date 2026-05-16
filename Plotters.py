from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt

def plotEnergyLandscape(fp, x, p):
    """
    Plot the energy landscape along the search direction.

    Parameters:
    - fp: fixed point function for AltMin
    - x: current solution vector
    - p: Newton search direction vector
    """
    # plot energy landscape
    E0=fp.updateEnergies(x) # initial energies
    alpha_list=np.linspace(-1,1,101)
    energies_list=alpha_list.copy()
    for j in range(0,101):
        xcopy=x.copy()
        xcopy.axpy(alpha_list[j],p)
        energies_list[j]=fp.updateEnergies(xcopy)[2] # total energy
        xcopy.destroy()
    plt.plot(alpha_list*p.norm(),energies_list-E0[2],'b',label='Total energy')
    plt.xlabel('Step length alpha')
    plt.ylabel('Total Energy')
    plt.show()

def plotEnergyLandscape2D(fp,x,res,p,target=None,filename=None):
    """
    Plot the energy landscape in a parallelogram bounded by the AltMin and Newton steps (both + and -)

    Parameters:
    - fp: function implementing the example
    - x: current solution
    - res: AltMin step
    - p: Newton step
    """
    E0=fp.updateEnergies(x)[2]
    angle = np.arccos(res.dot(p)/(res.norm()*p.norm())) # angle between AltMin and Newton steps
    nn=11
    alpha=np.linspace(0,2,nn)
    beta=np.linspace(-1,1,2*nn-1)
    energies=np.zeros([nn,2*nn-1])
    for i in range(0,nn):
        for j in range(0,2*nn-1):
            xcopy=x.copy()
            xcopy.axpy(alpha[i],res)
            xcopy.axpy(beta[j],p)
            energies[i][j]=fp.updateEnergies(xcopy)[2]
            xcopy.destroy()
    fig, ax = plt.subplots(1,2)
    cf=ax[0].contourf(beta, alpha, energies - E0)
    if target is not None:
        ax[0].plot(target[0], target[1], 'rx', markersize=10, label='Chosen step')
    ax[0].set_ylabel('AltMin')
    ax[0].set_xlabel('MSPIN')
    ax[0].set_title('Energy Landscape')
    fig.colorbar(cf, label='Total Energy')
    ax[1].quiver(0,0,res.norm()*np.cos(angle),res.norm()*np.sin(angle), angles='xy', scale_units='xy', scale=1, color='b', label='AltMin step')
    ax[1].quiver(0,0,p.norm(),0, angles='xy', scale_units='xy', scale=1, color='r', label='Newton step')
    ax[1].set_ylabel('AltMin')
    ax[1].set_xlabel('MSPIN')
    ax[1].set_title('Search Directions')
    ax[1].set_xlim(-1.5*p.norm(), 1.5*p.norm())
    ax[1].set_ylim(-1.5*res.norm(), 1.5*res.norm())
    plt.show()
    if filename is not None:
        fig.savefig(filename)

def plotNX(example,id_list,en_list):
    marker= ['o', 's', 'D', '^', 'v']
    color= ['g', 'r', 'c', 'm', 'y']
    ms=5
    fig1, ax1=plt.subplots(1,3)
    fig2, ax2=plt.subplots(1,3)
    fig3, ax3=plt.subplots(1,1)

    for i, identifier in enumerate(id_list):
        if en_list==None:
            energies = np.loadtxt(f"output/TBL_{identifier}.csv",
                              delimiter=',',skiprows=1)
        else:
            energies = en_list[i]
        ax1[0].plot(energies[:,0],energies[:,5],label=identifier,ls='--',marker=marker[i],color=color[i],ms=ms)
        if energies.shape[1]>5:
            ax1[1].plot(energies[:,0],energies[:,6],label=identifier,ls='--',marker=marker[i],color=color[i],ms=ms)
            ax1[2].plot(energies[:,0],energies[:,6]/np.maximum(1,energies[:,5]),label=identifier,ls='--',marker=marker[i],color=color[i],ms=ms)

        ax2[0].plot(energies[:,0],energies[:,1],label=identifier,ls='--',marker=marker[i],color=color[i],ms=ms)
        ax2[1].plot(energies[:,0],energies[:,2],label=identifier,ls='--',marker=marker[i],color=color[i],ms=ms)
        ax2[2].plot(energies[:,0],energies[:,3],label=identifier,ls='--',marker=marker[i],color=color[i],ms=ms)
        ax3.plot(energies[:,0],energies[:,4],label=identifier,ls='--',marker=marker[i],color=color[i],ms=ms)

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
    fig1.savefig(f"output/FIG_{example}_its.pdf")

    ax2[0].set_xlabel('t')
    ax2[0].set_ylabel('Elastic energy')
    # ax2[0].legend()
    ax2[1].set_xlabel('t')
    ax2[1].set_ylabel('Dissipated energy')
    # ax2[1].legend()
    ax2[2].set_xlabel('t')
    ax2[2].set_ylabel('Total energy')
    ax2[2].legend()
    fig2.savefig(f"output/FIG_{example}_energy.pdf")

    ax3.set_xlabel('t')
    ax3.set_ylabel('Time elapsed (s) per load step')
    ax3.legend()
    fig3.savefig(f"output/FIG_{example}_time.pdf")

def plotConvCrit(ConvCrit):
    # check how angle of MSPIN and AltMin change from iteration to iteration
    plt.semilogy(ConvCrit[:,0], ConvCrit[:,1], '.--', label='Step size')
    plt.xlabel('Iteration')
    plt.ylabel('Step size')
    # plt.ylim([1e-4, 1e2])
    plt.show()
    plt.plot(ConvCrit[:,0], ConvCrit[:,2]*180/np.pi, '.', label='Angle')
    plt.xlabel('Iteration')
    plt.ylabel('Angle between directions')
    plt.show()
    plt.semilogy(ConvCrit[:,0], ConvCrit[:,3], '.', label='+Alpha')
    plt.semilogy(ConvCrit[:,0], -ConvCrit[:,3], '.', label='-Alpha')
    plt.semilogy(ConvCrit[:,0], ConvCrit[:,4], 'v', label='+Beta')
    plt.semilogy(ConvCrit[:,0], -ConvCrit[:,4], '^', label='-Beta')
    # plt.semilogy(ConvCrit[:,0], ConvCrit[:,5], 'o', label='+Alpha opt')
    # plt.semilogy(ConvCrit[:,0], -ConvCrit[:,5], 'o', label='-Alpha opt')
    # plt.semilogy(ConvCrit[:,0], ConvCrit[:,6], 'd', label='+Beta opt')
    # plt.semilogy(ConvCrit[:,0], -ConvCrit[:,6], 's', label='-Beta opt')
    # plt.ylim([1e-4, 1.5])
    plt.xlabel('Iteration')
    plt.ylabel('Step percentages')
    plt.legend()
    plt.show()
    plt.semilogy(ConvCrit[:,0], ConvCrit[:,7], '.', label='+Determinant')
    plt.semilogy(ConvCrit[:,0], -ConvCrit[:,7], '.', label='-Determinant')
    plt.semilogy(ConvCrit[:,0], ConvCrit[:,8], 'd', label='+Curvature in AltMin direction')
    plt.semilogy(ConvCrit[:,0], -ConvCrit[:,8], 'd', label='-Curvature in AltMin direction')
    plt.xlabel('Iteration')
    plt.ylabel('Determinant of quadratic form')
    plt.legend()
    plt.show()

def plotStepByStep(fp, x, res):
    resu, resv = res.getNestSubVecs()
    resv0 = resv.copy()
    resv0.zeroEntries()
    resu0 = resu.copy()
    resu0.zeroEntries()
    stepu = PETSc.Vec().createNest([resu,resv0], None, fp.comm)
    stepv = PETSc.Vec().createNest([resu0,resv], None, fp.comm)
    plotEnergyLandscape(fp,x,stepu)
    plotEnergyLandscape(fp,x+stepu,stepv)

def plotDirectionChange(p, p_old):
    angle = np.arccos(p.dot(p_old)/(p.norm()*p_old.norm()))
    plt.quiver(0,-1,0,1, angles='xy', scale_units='xy', scale=1, color='r', label='Previous step')
    plt.quiver(0,0,np.sin(angle),np.cos(angle), angles='xy', scale_units='xy', scale=1, color='b', label='Current step')
    plt.title(f'Angle between steps: {angle*180/np.pi:3.2f} degrees')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.show()
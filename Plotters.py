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
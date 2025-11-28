from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt

def KSPsetUp(fp, J, type="gmres", rtol=1.0e-7, max_it=50, restarts=50, monitor='off'):
    """
    Set up a KSP solver with specified parameters.
    
    Parameters:
    - J: The Jacobian matrix to be used by the KSP.
    - type: The type of KSP solver (default is "gmres").
    - rtol: Relative tolerance for convergence (default is 1.0e-7).
    - max_it: Maximum number of iterations (default is 50).
    
    Returns:
    - ksp: Configured KSP solver.
    """
    ksp = PETSc.KSP().create(fp.comm)
    fp.Jn(None, None, J, None)  # Ensure the Jacobian is set up
    ksp.setOperators(J)
    ksp.setType(type)
    ksp.setTolerances(rtol=rtol, max_it=max_it)
    ksp.getPC().setType("none")  # Disable preconditioning
    opts= PETSc.Options()
    if monitor!='off':
        ksp.setMonitor(lambda snes, its, norm: print(f"Iteration {its}, Residual norm: {norm}"))
        opts['ksp_converged_reason'] = None # Returns reason for convergence
        if monitor=='cond':
            opts['ksp_monitor_singular_value'] = None # Returns singular values
    opts['ksp_gmres_restart']=restarts # number of iterations before restart
    ksp.setFromOptions()
    opts.destroy()
    return ksp

def DBTrick(res,p):
    proj_NFP=p.dot(res)
    DBSwitch=proj_NFP<0
    return DBSwitch

def customLineSearch(fp, p, type, DBSwitch):
    """
    Modify search direction using custom strategies

    Parameters:
    - fp: The fixed-point direction vector.
    - p: The Newton search direction vector.
    - type: The type of line search strategy ('fp', 'tr', 'ls', '2step').
    - DBSwitch: Boolean indicating whether to apply the DB trick.

    Returns:
    - The input p is modified.
    """
    if DBSwitch:
        p.setArray(-p.array) # using this method for changing value of p didn't work in the cubic backtracking case, but it's not clear why
        p.assemblyBegin()
        p.assemblyEnd()
    
    if type=='fp':
        p.setArray(fp.array)
        p.assemblyBegin()
        p.assemblyEnd()
        print('*') # indicate AltMin step
    elif type=='tr':
        diff_FP2x = fp.norm()
        diff_N2x = p.norm()
        diff_N2FP = (p - fp).norm()
        if diff_N2x <= diff_N2FP:
            p.setArray(fp.array)
            p.assemblyBegin()
            p.assemblyEnd()
            print('*') # indicate AltMin step
    elif type=='ls': # this is a custom line search based on the ratio of norms
        if p.norm()!=0:
            dist_ls = np.min([1, fp.norm()/p.norm()])
        else:
            dist_ls = 1
        p.setArray(dist_ls * p.array)
        p.assemblyBegin()
        p.assemblyEnd()
    elif type=='2step':
        diff_FP2x = fp.norm()
        diff_N2FP = (p - fp).norm()
        dist_2step = 1/(1+np.exp(diff_N2FP - diff_FP2x))
        p.setArray(fp.array + dist_2step * (p - fp).array)
        p.assemblyBegin()
        p.assemblyEnd()

def CubicBacktracking(fp,x,p,res):
    """
    Perform cubic backtracking line search.

    Parameters:
    - fp: fixed point function for AltMin
    - x: current solution vector
    - p: Newton search direction vector
    - res: AltMin step
    """

    E0 = fp.updateEnergies(x)[2]  # initial energy
    print(f"Initial Energy: {E0}")
    alpha = 1.0  # initial step length
    fp.updateGradF(x)
    gp = p.dot(fp.gradF)
    print(f"Gradient dot search direction: {gp}")
    if gp>0: # DB trick
        p=-p
    elif gp==0: # local minimum, use AltMin step
        p=res
        print("*") # indicate AltMin step
    xcopy=x.copy()
    xcopy.axpy(alpha,p)
    E1 = fp.updateEnergies(xcopy)[2]  # new energy
    Efp= fp.updateEnergies(x + res)[2] # energy after AltMin step
    print(f"Energy at Newton step: {E1}, Energy at AltMin step: {Efp}")
    first_time=True
    while E1 > E0:
        # if alpha < 1e-16:
        #     p=res
        #     alpha=1.0
        #     print("*") # indicate AltMin step
        #     break
        if first_time==True:
            alpha_0 = 1.0
            E_prev = E1
            alpha = -gp/(2*(E1 - E0 - gp))
            first_time=False
        else:
            array1=np.array([[1/alpha**2, -1/alpha_0**2],[-alpha_0/alpha**2, alpha/alpha_0**2]])
            array2=np.array([E1 - E0 - gp * alpha, E_prev - E0 - gp * alpha_0])
            array_out = array1.dot(array2) / (alpha - alpha_0)
            a=array_out[0]
            b=array_out[1]
            if a == 0:
                alpha_new = -gp/(2*b)
            else:
                alpha_new = (-b + np.sqrt(b**2 - 3*a*gp)) / (3*a)
            alpha_0 = alpha
            E_prev = E1
            alpha = alpha_new
            # if alpha > 0.5 * alpha_0: # not sure if this is necessary
            #     alpha = 0.5 * alpha_0
            # print(f"Backtracking step length: {alpha}, Energy: {E1}, Target energy: {E0}")
        xcopy.waxpy(alpha,p,x) # replacing xcopy=x and then xcopy.axpy(alpha,p) to avoid creating multiple copies, may not work
        E1 = fp.updateEnergies(xcopy)[2]
        print(f"Backtracking step length: {alpha}, Energy: {E1}, Target energy: {E0}")
    print(f"Final step length: {alpha}")
    xcopy.zeroEntries()
    p.aypx(alpha,xcopy)
    xcopy.destroy()
    return p

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
    # elastic_energies=alpha_list.copy()
    # dissipated_energies=alpha_list.copy()
    for j in range(0,101):
        xcopy=x.copy()
        xcopy.axpy(alpha_list[j],p)
        temp=fp.updateEnergies(xcopy)
        # elastic_energies[j]=temp[0]
        # dissipated_energies[j]=temp[1]
        energies_list[j]=temp[2] # total energy
    # plt.plot(alpha_list,elastic_energies-E0[0],'r',label='Elastic energy')
    # plt.plot(alpha_list,dissipated_energies-E0[1],'g',label='Dissipated energy')
    plt.plot(alpha_list,energies_list-E0[2],'b',label='Total energy')
    plt.xlabel('Step length alpha')
    plt.ylabel('Total Energy')
    plt.show()

def plotNX(example,linesearch_list):
    marker= ['o', 's', 'D', '^', 'v']
    color= ['g', 'r', 'c', 'm', 'y']
    ms=5
    energies=np.loadtxt(f"output/TBL_{example}_AltMin_fp.csv",
                        delimiter=',',skiprows=1)
    
    fig1, ax1=plt.subplots(1,3)
    ax1[0].plot(energies[:,0],energies[:,4],label='AltMin',ls='--',marker='.',color='b',ms=ms)
    fig2, ax2=plt.subplots(1,3)
    ax2[0].plot(energies[:,0],energies[:,1],label='AltMin',ls='--',marker='.',color='b',ms=ms)
    ax2[1].plot(energies[:,0],energies[:,2],label='AltMin',ls='--',marker='.',color='b',ms=ms)
    ax2[2].plot(energies[:,0],energies[:,3],label='AltMin',ls='--',marker='.',color='b',ms=ms)

    for i, linesearch in enumerate(linesearch_list):
        energies = np.loadtxt(f"output/TBL_{example}_Newton_{linesearch}.csv",
                              delimiter=',',skiprows=1)
        ax1[0].plot(energies[:,0],energies[:,4],label=linesearch,ls='--',marker=marker[i],color=color[i],ms=ms)
        ax1[1].plot(energies[:,0],energies[:,5],label=linesearch,ls='--',marker=marker[i],color=color[i],ms=ms)
        ax1[2].plot(energies[:,0],energies[:,5]/np.maximum(1,energies[:,4]),label=linesearch,ls='--',marker=marker[i],color=color[i],ms=ms)

        ax2[0].plot(energies[:,0],energies[:,1],label=linesearch,ls='--',marker=marker[i],color=color[i],ms=ms)
        ax2[1].plot(energies[:,0],energies[:,2],label=linesearch,ls='--',marker=marker[i],color=color[i],ms=ms)
        ax2[2].plot(energies[:,0],energies[:,3],label=linesearch,ls='--',marker=marker[i],color=color[i],ms=ms)
    
    ax1[0].set_xlabel('t')
    ax1[0].set_ylabel('Iterations')
    ax1[0].legend()
    ax1[0].set_title('Outer iterations')
    ax1[1].set_xlabel('t')
    ax1[1].set_ylabel('Iterations')
    ax1[1].legend()
    ax1[1].set_title('Inner iterations')
    ax1[2].set_xlabel('t')
    ax1[2].set_ylabel('Ave. inner / outer')
    ax1[2].legend()
    ax1[2].set_title('Ave. inner per outer')
    fig1.savefig(f"output/FIG_{example}_its.png")

    ax2[0].set_xlabel('t')
    ax2[0].set_ylabel('Elastic energy')
    ax2[0].legend()
    ax2[1].set_xlabel('t')
    ax2[1].set_ylabel('Dissipated energy')
    ax2[1].legend()
    ax2[2].set_xlabel('t')
    ax2[2].set_ylabel('Total energy')
    ax2[2].legend()
    fig2.savefig(f"output/FIG_{example}_energy.png")
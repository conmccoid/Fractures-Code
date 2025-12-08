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
    if gp>0: # DB trick
        p.scale(-1)
        gp = p.dot(fp.gradF)
        print("DB trick")
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

def ParallelogramBacktracking(fp, x, q, p, PlotSwitch=False):
    """
    Find the minimum of a quadratic 2D polynomial that interpolates the residual in a parallelogram

    Parameters:
    - fp: function handling example
    - x: current solution
    - q: direction towards the AltMin step
    - p: direction towards the Newton step

    nb: some kinks to work out, currently minimum could be found outside parallelogram
    also not clear what to do when r==0
    need to check if minimum lies outside, and if it does, then find the minimum on the boundary
    """
    # need to do DB trick first - does gradF need to have a commensurate sign change?
    fp.updateGradF(x)
    e=p.dot(fp.gradF)
    d=q.dot(fp.gradF)
    if e>0: # DB trick
        pcopy=-p.copy()
        e=pcopy.dot(fp.gradF)
        print("DB trick")
    else:
        pcopy=p.copy()
    E0=fp.updateEnergies(x)[2]
    Eq=fp.updateEnergies(x+q)[2]
    Ep=fp.updateEnergies(x+pcopy)[2]
    Epq=fp.updateEnergies(x+pcopy+q)[2]
    f=E0
    c=Ep-e-f
    a=Eq-d-f
    b=Epq + E0 - Ep - Eq
    r=4*a*c-b**2
    alpha = (-2*c*d + b*e)/r
    beta = (-2*a*e + b*d)/r
    E_list=[Eq, Ep, Epq]
    v_list=[q, pcopy, q + pcopy]
    beta_list=[0, 1, 1]
    alpha_list=[1, 0, 1]
    step_list=["AltMin", "Newton", "Both"]
    if (alpha>0) & (alpha<1) & (beta>0) & (beta<1):
        v=q.copy()
        v.scale(alpha)
        v.axpy(beta,pcopy)
        v_list.append(v)
        alpha_list.append(alpha)
        beta_list.append(beta)
        E_list.append(fp.updateEnergies(x+v)[2])
        step_list.append("Parallelogram interior")
    else:
        print("Minimum outside parallelogram, finding minimum on boundary")
    # beta=0 minimum
    alpha= -d/(2*a)
    beta_list.append(0)
    if alpha<0:
        v_list.append(0)
        E_list.append(E0)
        alpha_list.append(0)
        step_list.append("Origin")
    elif alpha>1:
        v_list.append(q)
        E_list.append(Eq)
        alpha_list.append(1)
        step_list.append("AltMin")
    else:
        v=q.copy()
        v.scale(alpha)
        E_list.append(fp.updateEnergies(x+v)[2])
        v_list.append(v)
        alpha_list.append(alpha)
        step_list.append("AltMin linesearch")
    
    # alpha=0
    beta= -e/(2*c)
    alpha_list.append(0)
    if beta<0:
        v_list.append(0)
        E_list.append(E0)
        beta_list.append(0)
        step_list.append("Origin")
    elif beta>1:
        v_list.append(pcopy)
        E_list.append(Ep)
        beta_list.append(1)
        step_list.append("Newton")
    else:
        v=pcopy.copy()
        v.scale(beta)
        E_list.append(fp.updateEnergies(x+v)[2])
        v_list.append(v)
        beta_list.append(beta)
        step_list.append("Newton linesearch")
    
    # beta=1
    alpha= (-d - b)/(2*a)
    beta_list.append(1)
    if alpha<0:
        v_list.append(pcopy)
        E_list.append(Ep)
        alpha_list.append(0)
        step_list.append("Newton")
    elif alpha>1:
        v_list.append(q + pcopy)
        E_list.append(Epq)
        alpha_list.append(1)
        step_list.append("Both")
    else:
        v=q.copy()
        v.scale(alpha)
        v.axpy(1,pcopy)
        E_list.append(fp.updateEnergies(x+v)[2])
        v_list.append(v)
        alpha_list.append(alpha)
        step_list.append("Newton + AltMin linesearch")
    
    # alpha=1
    beta= (-e - b)/(2*c)
    alpha_list.append(1)
    if beta<0:
        v_list.append(q)
        E_list.append(Eq)
        beta_list.append(0)
        step_list.append("AltMin")
    elif beta>1:
        v_list.append(q + pcopy)
        E_list.append(Epq)
        beta_list.append(1)
        step_list.append("Both")
    else:
        v=q.copy()
        v.scale(1)
        v.axpy(beta,pcopy)
        E_list.append(fp.updateEnergies(x+v)[2])
        v_list.append(v)
        beta_list.append(beta)
        step_list.append("AltMin + Newton linesearch")

    min_index=np.argmin(E_list)
    v=v_list[min_index]
    alpha=alpha_list[min_index]
    beta=beta_list[min_index]
    print(f"Chosen step: {step_list[min_index]}, Energy: {E_list[min_index]}")
    if PlotSwitch:
        print(f"Step in AltMin: {alpha}, Step in Newton: {beta}")
        plotEnergyLandscape2D(fp,x,q,pcopy,[beta, alpha])
    return v

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
    plt.plot(alpha_list,energies_list-E0[2],'b',label='Total energy')
    plt.xlabel('Step length alpha')
    plt.ylabel('Total Energy')
    plt.show()

def plotEnergyLandscape2D(fp,x,res,p,target=None):
    """
    Plot the energy landscape in a parallelogram bounded by the AltMin and Newton steps (both + and -)

    Parameters:
    - fp: function implementing the example
    - x: current solution
    - res: AltMin step
    - p: Newton step
    """
    E0=fp.updateEnergies(x)[2]
    nn=11
    alpha=np.linspace(0,1,nn)
    beta=np.linspace(-1,1,2*nn-1)
    energies=np.zeros([nn,2*nn-1])
    for i in range(0,nn):
        for j in range(0,2*nn-1):
            xcopy=x.copy()
            xcopy.axpy(alpha[i],res)
            xcopy.axpy(beta[j],p)
            energies[i][j]=fp.updateEnergies(xcopy)[2]
            xcopy.destroy()
    plt.contourf(beta, alpha, energies - E0)
    if target is not None:
        plt.plot(target[0], target[1], 'rx', markersize=10, label='Chosen step')
    plt.ylabel('AltMin')
    plt.xlabel('MSPIN')
    plt.colorbar(label='Total Energy')
    plt.show()

def plotNX(example,id_list,en_list):
    marker= ['o', 's', 'D', '^', 'v']
    color= ['g', 'r', 'c', 'm', 'y']
    ms=5
    fig1, ax1=plt.subplots(1,3)
    fig2, ax2=plt.subplots(1,3)

    for i, identifier in enumerate(id_list):
        if en_list==None:
            energies = np.loadtxt(f"output/TBL_{example}_{identifier}.csv",
                              delimiter=',',skiprows=1)
        else:
            energies = en_list[i]
        ax1[0].plot(energies[:,0],energies[:,4],label=identifier,ls='--',marker=marker[i],color=color[i],ms=ms)
        ax1[1].plot(energies[:,0],energies[:,5],label=identifier,ls='--',marker=marker[i],color=color[i],ms=ms)
        ax1[2].plot(energies[:,0],energies[:,5]/np.maximum(1,energies[:,4]),label=identifier,ls='--',marker=marker[i],color=color[i],ms=ms)

        ax2[0].plot(energies[:,0],energies[:,1],label=identifier,ls='--',marker=marker[i],color=color[i],ms=ms)
        ax2[1].plot(energies[:,0],energies[:,2],label=identifier,ls='--',marker=marker[i],color=color[i],ms=ms)
        ax2[2].plot(energies[:,0],energies[:,3],label=identifier,ls='--',marker=marker[i],color=color[i],ms=ms)
    
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
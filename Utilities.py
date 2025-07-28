from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt

def KSPsetUp(fp, J, type="gmres", rtol=1.0e-7, max_it=50, monitor='off'):
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
    ksp.setFromOptions()
    opts.destroy()
    return ksp

def DBTrick(res,p):
    proj_NFP=p.dot(res)
    DBSwitch=proj_NFP<0
    return DBSwitch

def customLineSearch(res, p, type, DBSwitch):
    if DBSwitch:
        p.setArray(-p.array)
        p.assemblyBegin()
        p.assemblyEnd()
    
    if type=='fp':
        p.setArray(res.array)
        p.assemblyBegin()
        p.assemblyEnd()
    elif type=='tr':
        diff_FP2x = res.norm()
        diff_N2x = p.norm()
        diff_N2FP = (p - res).norm()
        if diff_N2x <= diff_N2FP:
            p.setArray(res.array)
            p.assemblyBegin()
            p.assemblyEnd()
    elif type=='ls':
        if p.norm()!=0:
            dist_ls = np.min([1, res.norm()/p.norm()])
        else:
            dist_ls = 1
        p.setArray(dist_ls * p.array)
        p.assemblyBegin()
        p.assemblyEnd()
    elif type=='2step':
        diff_FP2x = res.norm()
        diff_N2FP = (p - res).norm()
        dist_2step = 1/(1+np.exp(diff_N2FP - diff_FP2x))
        p.setArray(res.array + dist_2step * (p - res).array)
        p.assemblyBegin()
        p.assemblyEnd()

def plotNX(example,linesearch_list):
    energies=np.loadtxt(f"output/TBL_{example}_AltMin_fp.csv",
                        delimiter=',',skiprows=1)
    
    fig1, ax1=plt.subplots(1,3)
    ax1[0].plot(energies[:,0],energies[:,4],label='AltMin iterations')
    fig2, ax2=plt.subplots(1,3)
    ax2[0].plot(energies[:,0],energies[:,1],label='AltMin')
    ax2[1].plot(energies[:,0],energies[:,2],label='AltMin')
    ax2[2].plot(energies[:,0],energies[:,3],label='AltMin')

    for linesearch in linesearch_list:
        energies = np.loadtxt(f"output/TBL_{example}_Newton_{linesearch}.csv",
                              delimiter=',',skiprows=1)
        ax1[0].plot(energies[:,0],energies[:,4],label=f"{linesearch} outer iterations")
        ax1[1].plot(energies[:,0],energies[:,5],label=f"{linesearch} inner iterations")
        ax1[2].plot(energies[:,0],energies[:,5]/energies[:,4],label=f"{linesearch} ave. inner per outer")

        ax2[0].plot(energies[:,0],energies[:,1],label=linesearch)
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
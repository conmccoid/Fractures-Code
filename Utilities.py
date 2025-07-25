from petsc4py import PETSc
import numpy as np

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

def customLineSearch(res, p, type, DBSwitch):
    if DBSwitch==True:
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
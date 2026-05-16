from petsc4py import PETSc
import numpy as np

# maybe these methods should be moved to the FPAltMin class

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

def DBTrick(fp,x,p):
    fp.updateGradF(x)
    gp = p.dot(fp.gradF)
    if gp>0:
        p.scale(-1)
        # if fp.rank==0:
        #     print("DB trick")

def boxConstraints(fp,x):
    """
    Apply box constraints to the search direction.

    Parameters:
    - fp: fixed point function for AltMin
    - x: current solution vector

    Returns:
    - The input x is modified.
    """

    fp.updateUV(x)
    # E0 = fp.updateEnergies(x)[2]
    v = fp.v.x.petsc_vec
    v_lb = fp.v_lb.x.petsc_vec # retrieve upper and lower bounds as PETSc vectors
    v_ub = fp.v_ub.x.petsc_vec
    dist_low = v.array - v_lb.array # determine distance from bounds
    dist_upp = v_ub.array - v.array
    IS_low = np.where(dist_low < 0)[0] # find indices where v is below lower bound
    IS_upp = np.where(dist_upp < 0)[0] # find indices where v is above upper bound
    # dist_total = np.sum(dist_low[IS_low]) + np.sum(dist_upp[IS_upp])
    v.array[IS_low] = v_lb.array[IS_low] # set v to boundary value if it crosses boundary
    v.array[IS_upp] = v_ub.array[IS_upp] # set v to boundary value if it crosses boundary
    v.assemblyBegin()
    v.assemblyEnd()
    _, xv = x.getNestSubVecs()
    xv.setArray(v.array)
    xv.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    # E1 = fp.updateEnergies(x)[2]
    # print(f"Applied box constraints to {len(IS_low) + len(IS_upp)} entries, total distance from bounds was {dist_total}")
    # print(f"Energy before applying constraints: {E0}, Energy after applying constraints: {E1}")
from FPAltMin_Test import FPAltMin
import numpy as np
from petsc4py import PETSc

def main():
    fp = FPAltMin()
    loads = np.linspace(0, 1, 10)  # Load values
    energies = np.zeros((loads.shape[0], 5))  # Initialize energies array

    x, J = fp.createVecMat()  # Create empty vector and matrix
    res = x.duplicate()  # Create a duplicate for the residual vector
    
    for i_t, t in enumerate(loads):
        fp.updateBCs(t)
        energies[i_t, 0] = t
        if fp.rank == 0:
            print(f"-- Solving for t = {t:3.2f} --")
        
        iteration=0
        fp.Fn(None, x, res)
        x += res  # Update the vector with the residual
        error = fp.updateError()
        fp.monitor(iteration)
        while error > 1e-8:
            iteration += 1
            fp.Fn(None, x, res)
            x+=res  # Update the vector with the residual
            error = fp.updateError()
            fp.monitor(iteration)

        energies[i_t,1:4] = fp.updateEnergies(x)

        fp.plot(x=x)

def Newton():
    fp = FPAltMin()
    loads = np.linspace(0, 1, 10)  # Load values
    energies = np.zeros((loads.shape[0], 5))  # Initialize energies array

    x, J = fp.createVecMat()  # Create empty vector and matrix
    res = x.duplicate()  # Create a duplicate for the residual vector

    SNESNewton = PETSc.SNES().create(fp.comm)
    SNESNewton.setFunction(fp.Fn, res)
    # SNESNewton.setJacobian(fp.Jn, J) # THIS IS THE FUCKING PROBLEM
    SNESNewton.setType("newtonls")
    SNESNewton.getKSP().setType("gmres")
    SNESNewton.getKSP().setTolerances(rtol=1.0e-7, max_it=50)
    opts=PETSc.Options()
    opts['snes_linesearch_type']='none'
    opts['snes_linesearch_monitor']=None
    opts['snes_converged_reason']=None
    SNESNewton.setFromOptions()
    opts.destroy()
    SNESNewton.setSolution(x)

    for i_t, t in enumerate(loads):
        fp.updateBCs(t)
        energies[i_t, 0] = t
        if fp.rank == 0:
            print(f"-- Solving for t = {t:3.2f} --")
        
        iteration=0
        SNESNewton.solve(None, x)  # Solve the nonlinear system
        fp.updateUV(x)  # Update the solution vectors
        error = fp.updateError()
        fp.monitor(iteration)

        while error > 1e-8:
            iteration += 1
            SNESNewton.solve(None, x)
            fp.updateUV(x)
            error = fp.updateError()
            fp.monitor(iteration)

        energies[i_t, 1:4] = fp.updateEnergies(x)
        fp.plot(x=x)
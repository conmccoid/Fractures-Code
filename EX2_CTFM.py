from FPAltMin_CTFM import FPAltMin
import numpy as np
from petsc4py import PETSc
from Utilities import KSPsetUp, customLineSearch
from dolfinx import io
import csv

def main(method='AltMin', linesearch='fp', WriteSwitch=False, PlotSwitch=False):
    fp = FPAltMin()
    loads = np.linspace(0, 1.5 * fp.load_c * 12 / 10, 20) # (load_c/E)*L
    # first critical load is between 0.87 and 1.31 (but sometimes up to 1.7?)
    # second critical load between 4.79 and 5.22
    if method=='AltMin':
        energies = np.zeros((loads.shape[0], 5))  # Initialize energies array
    else:
        energies = np.zeros((loads.shape[0], 6))

    x, J = fp.createVecMat()  # Create empty vector and matrix
    res = x.duplicate()  # Create a duplicate for the residual vector
    p = x.duplicate()  # Create a duplicate for the search direction

    if method!='AltMin':
        SNESKSP = KSPsetUp(fp, J, type="gmres", rtol=1.0e-7, max_it=50)  # Set up the KSP solver

    if WriteSwitch:
        with io.XDMFFile(fp.comm, f"output/EX_CTFM_{method}_{linesearch}.xdmf","w") as xdmf:
            xdmf.write_mesh(fp.dom)
        with open(f"output/TBL_CTFM_{method}_{linesearch}.csv",'w') as csv.file:
            writer=csv.writer(csv.file,delimiter=',')
            if method=='AltMin':
                writer.writerow(['t','Elastic energy','Dissipated energy','Total energy','Number of iterations'])
            else:
                writer.writerow(['t','Elastic energy','Dissipated energy','Total energy','Outer iterations', 'Inner iterations'])

    for i_t, t in enumerate(loads):
        fp.updateBCs(t)
        energies[i_t, 0] = t
        if fp.rank == 0:
            print(f"-- Solving for t = {t:3.2f} --")
        
        iteration=0
        fp.Fn(None, x, res)  # Evaluate the function
        if method=='AltMin':
            x += res  # Update the vector with the residual
        else:
            SNESKSP.solve(res, p)  # Solve the linear system
            customLineSearch(res, p, type=linesearch, DBSwitch=False)
            x += p  # Update the solution vector
            energies[i_t,5]=SNESKSP.getIterationNumber()
        fp.updateUV(x)  # Update the solution vectors
        error = fp.updateError()
        fp.monitor(iteration)

        while error > 1e-4:
            iteration += 1
            fp.Fn(None, x, res)
            if method=='AltMin':
                x+=res
            else:
                SNESKSP.solve(res, p)  # Solve the linear system
                customLineSearch(res, p, type=linesearch, DBSwitch=False)
                x += p  # Update the solution vector
                energies[i_t,5]+=SNESKSP.getIterationNumber()
            fp.updateUV(x)
            error = fp.updateError()
            fp.monitor(iteration)

        energies[i_t, 1:4] = fp.updateEnergies(x)
        energies[i_t, 4] = iteration

        if PlotSwitch:
            fp.plot(x=x)

        if WriteSwitch:
            with io.XDMFFile(fp.comm, f"output/EX_CTFM_{method}_{linesearch}.xdmf","a") as xdmf:
                xdmf.write_function(fp.u, t)
                xdmf.write_function(fp.v, t)
        
    if WriteSwitch:
        with open(f"output/TBL_CTFM_{method}_{linesearch}.csv",'a') as csv.file:
            writer=csv.writer(csv.file,delimiter=',')
            writer.writerows(energies)

    return energies
from FPAltMin_GD import FPAltMin
import numpy as np
from petsc4py import PETSc
from Utilities import KSPsetUp, customLineSearch, DBTrick, CubicBacktracking, ParallelogramBacktracking, plotEnergyLandscape
from dolfinx import io
import csv

def main(method='AltMin', linesearch=None, maxit=100, WriteSwitch=False, PlotSwitch=False):
    if linesearch is None:
        identifier=f"{method}"
    else:
        identifier=f"{method}_{linesearch}"
    fp = FPAltMin()
    loads = np.linspace(0, 1, 11)  # Load values
    energies = np.zeros((loads.shape[0], 6)) # intialize energy storage

    x, J = fp.createVecMat()  # Create empty vector and matrix
    res = x.duplicate()  # Create a duplicate for the residual vector
    p = x.duplicate()  # Create a duplicate for the search direction

    if method!='AltMin':
        SNESKSP = KSPsetUp(fp, J, type="gmres", rtol=1.0e-7, max_it=1000, restarts=1000, monitor='off')  # Set up the KSP solver

    if WriteSwitch:
        with io.XDMFFile(fp.comm, f"output/EX_GD_{identifier}.xdmf","w") as xdmf:
            xdmf.write_mesh(fp.dom)
        with open(f"output/TBL_GD_{identifier}.csv",'w') as csv.file:
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
        fp.Fn(None, x, res)  # Evaluate the function: run one iteration of AltMin to satisfy BCs
        if PlotSwitch:
            plotEnergyLandscape(fp,x,res) # temporary
            print(f"Energy: {fp.updateEnergies(x)[2]}") # temporary
        x+=res
        fp.updateUV(x)  # Update the solution vectors
        error = fp.updateError()
        fp.monitor(iteration)

        while error > 1e-4 and iteration < maxit:
            iteration += 1
            fp.Fn(None, x, res)
            if method=='AltMin':
                if PlotSwitch:
                    plotEnergyLandscape(fp,x,res) # temporary
                    print(f"Energy: {fp.updateEnergies(x)[2]}") # temporary
                x+=res
            elif method=='CubicBacktracking': # Run cubic backtracking in situ
                SNESKSP.solve(res, p)  # Solve the linear system
                energies[i_t,5]=SNESKSP.getIterationNumber()
                DBTrick(fp,x,p) # apply DB trick to search direction
                if PlotSwitch:
                    plotEnergyLandscape(fp,x,p)
                p = CubicBacktracking(fp, x, p, res)
                x += p # update solution
            elif method=='Parallelogram':
                SNESKSP.solve(res, p)  # Solve the linear system
                energies[i_t,5]=SNESKSP.getIterationNumber()
                DBTrick(fp,x,p) # apply DB trick to search direction
                v = ParallelogramBacktracking(fp, x, res, p, PlotSwitch=PlotSwitch)
                x += v # update solution
            else:
                SNESKSP.solve(res, p)  # Solve the linear system
                customLineSearch(res, p, type=linesearch, DBSwitch=DBTrick(res, p))
                x += p  # Update the solution vector
                energies[i_t,5]+=SNESKSP.getIterationNumber()
            fp.updateUV(x)
            error = fp.updateError()
            fp.monitor(iteration)

        energies[i_t, 1:4] = fp.updateEnergies(x)[0:3]
        energies[i_t, 4] = iteration

        fp.v_lb.x.array[:] = fp.v.x.array # update lower bound for damage to ensure irreversibility
        # fp.damage_solver.setVariableBounds(v_lb.x.petsc_vec, v_ub.x.petsc_vec) # unnecessary?

        if PlotSwitch:
            fp.plot(x=x)

        if WriteSwitch:
            with io.XDMFFile(fp.comm, f"output/EX_GD_{identifier}.xdmf","a") as xdmf:
                xdmf.write_function(fp.u, t)
                xdmf.write_function(fp.v, t)
    
    if WriteSwitch:
        with open(f"output/TBL_GD_{identifier}.csv",'a') as csv.file:
            writer=csv.writer(csv.file,delimiter=',')
            writer.writerows(energies)

    return energies, identifier
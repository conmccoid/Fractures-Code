import numpy as np
from petsc4py import PETSc
from Utilities import KSPsetUp, DBTrick, boxConstraints, CubicBacktracking, ParallelogramBacktracking, plotEnergyLandscape
from dolfinx import io
import csv

class OuterSolver:
    def __init__(self, fp, example, method, loads):
        self.fp = fp
        self.example = example
        self.method = method
        self.loads = loads
        self.energies = np.zeros((loads.shape[0], 6)) # intialize energy storage
        self.setIdentifier()
        self.setUp()
        if self.method!='AltMin':
            self.SNESKSP = KSPsetUp(self.fp, self.J, type="gmres", rtol=1.0e-7, max_it=1000, restarts=1000, monitor='off')  # Set up the KSP solver

    def setIdentifier(self):
        self.identifier=f"{self.example}_{self.method}"

    def setUp(self):
        self.x, self.J = self.fp.createVecMat()  # Create empty vector and matrix
        self.res = self.x.duplicate()  # Create a duplicate for the residual vector
        self.p = self.x.duplicate()  # Create a duplicate for the search direction
    
    def solve(self, WriteSwitch=False, PlotSwitch=False, maxit=100, tol=1e-4):

        # write headers to saved files
        if WriteSwitch:
            with io.XDMFFile(self.fp.comm, f"output/EX_{self.identifier}.xdmf","w") as xdmf:
                xdmf.write_mesh(self.fp.dom)
            with open(f"output/TBL_{self.identifier}.csv",'w') as csv.file:
                writer=csv.writer(csv.file,delimiter=',')
                if self.method=='AltMin':
                    writer.writerow(['t','Elastic energy','Dissipated energy','Total energy','Number of iterations'])
                else:
                    writer.writerow(['t','Elastic energy','Dissipated energy','Total energy','Outer iterations', 'Inner iterations'])

        # main iteration
        for i_t, t in enumerate(self.loads):
            self.fp.updateBCs(t)
            self.energies[i_t, 0] = t
            if self.fp.rank == 0:
                print(f"-- Solving for t = {t:3.2f} --")
            
            iteration=0
            self.fp.Fn(None, self.x, self.res)  # Evaluate the function: run one iteration of AltMin to satisfy BCs
            if PlotSwitch:
                plotEnergyLandscape(self.fp,self.x,self.res) # temporary
                print(f"Energy: {self.fp.updateEnergies(self.x)[2]}") # temporary
            self.x+=self.res
            self.fp.updateUV(self.x)  # Update the solution vectors
            error = self.fp.updateError() # calculate error
            self.fp.monitor(iteration) # monitor convergence

            while error > tol and iteration < maxit:
                iteration += 1
                self.fp.Fn(None, self.x, self.res)
                if self.method=='AltMin':
                    if PlotSwitch:
                        plotEnergyLandscape(self.fp,self.x,self.res) # temporary
                        print(f"Energy: {self.fp.updateEnergies(self.x)[2]}") # temporary
                    self.x+=self.res
                else:
                    self.SNESKSP.solve(self.res, self.p)  # Solve the linear system
                    self.energies[i_t,5]=self.SNESKSP.getIterationNumber()
                    DBTrick(self.fp,self.x,self.p) # apply DB trick to search direction
                    if self.method=='CubicBacktracking': # Run cubic backtracking in situ
                        if PlotSwitch:
                            plotEnergyLandscape(self.fp,self.x,self.p)
                        self.p = CubicBacktracking(self.fp, self.x, self.p, self.res)
                        self.x += self.p # update solution
                    elif self.method=='Parallelogram':
                        v = ParallelogramBacktracking(self.fp, self.x, self.res, self.p, PlotSwitch=PlotSwitch)
                        self.x += v # update solution
                self.fp.updateUV(self.x)
                error = self.fp.updateError()
                self.fp.monitor(iteration)
            
            if self.method=='CubicBacktracking' or self.method=='Parallelogram': # apply box constraints to final solution for backtracking methods
                boxConstraints(self.fp,self.x) # apply box constraints to final solution
                self.fp.updateUV(self.x) # update solution vectors after applying constraints
                error = self.fp.updateError() # update error after applying constraints
                self.fp.monitor(iteration)
            
            self.energies[i_t, 1:4] = self.fp.updateEnergies(self.x)[0:3]
            self.energies[i_t, 4] = iteration

            self.fp.v_lb.x.array[:] = self.fp.v.x.array # update lower bound for damage to ensure irreversibility

            if PlotSwitch:
                self.fp.plot(x=self.x)

            # write solution to file
            if WriteSwitch:
                with io.XDMFFile(self.fp.comm, f"output/EX_{self.identifier}.xdmf","a") as xdmf:
                    xdmf.write_function(self.fp.u, t)
                    xdmf.write_function(self.fp.v, t)

        # write energies to table
        if WriteSwitch:
            with open(f"output/TBL_{self.identifier}.csv",'a') as csv.file:
                writer=csv.writer(csv.file,delimiter=',')
                writer.writerows(self.energies)
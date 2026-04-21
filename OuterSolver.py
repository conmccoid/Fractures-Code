import numpy as np
from petsc4py import PETSc
from Utilities import KSPsetUp, DBTrick, boxConstraints, CubicBacktracking, ParallelogramBacktracking, plotDirectionChange, plotEnergyLandscape, plotConvCrit
from dolfinx import io
import csv

class OuterSolver:
    def __init__(self, fp, example, method, loads):
        self.fp = fp
        self.example = example
        self.method = method
        self.loads = loads
        self.energies = np.zeros((loads.shape[0], 7)) # intialize energy storage
        self.setIdentifier()
        self.setUp()
        if self.method!='AltMin':
            self.SNESKSP = KSPsetUp(self.fp, self.J, type="gmres", rtol=1.0e-7, max_it=1000, restarts=1000, monitor='off')  # Set up the KSP solver

    def setIdentifier(self):
        self.identifier=f"{self.example}_{self.method}"

    def setUp(self):
        self.x, self.J = self.fp.createVecMat()  # Create empty vector and preconditioned Jacobian
        self.res = self.x.duplicate()  # Create a duplicate for the residual vector
        self.p = self.x.duplicate()  # Create a duplicate for the search direction
    
    def solve(self, WriteSwitch=False, PlotSwitch=False, maxit=100, tol=1e-4):

        # write headers to saved files
        if WriteSwitch:
            with io.XDMFFile(self.fp.comm, f"output/EX_{self.identifier}.xdmf","w") as xdmf:
                xdmf.write_mesh(self.fp.dom)
            if self.fp.rank==0:
                with open(f"output/TBL_{self.identifier}.csv",'w') as csv.file:
                    writer=csv.writer(csv.file,delimiter=',')
                    if self.method=='AltMin':
                        writer.writerow(['t','Elastic energy','Dissipated energy','Total energy','Time elapsed','Number of iterations'])
                    else:
                        writer.writerow(['t','Elastic energy','Dissipated energy','Total energy','Time elapsed','Outer iterations','Inner iterations'])
                if self.method=='Parallelogram':
                    with open(f"output/ConvCrit_{self.identifier}.csv",'w') as csv.file:
                        writer=csv.writer(csv.file,delimiter=',')
                        writer.writerow(['Iteration','Step size','Volume','Flatness','Angle','Alpha','Beta'])

        # main iteration
        for i_t, t in enumerate(self.loads):
            self.fp.updateBCs(t)
            self.energies[i_t, 0] = t
            if self.fp.rank == 0:
                print(f"-- Solving for t = {t:3.2f} --")
            
            start_time = PETSc.Log.getTime() # or time.perf_counter()?
            iteration=0
            self.fp.Fn(None, self.x, self.res)  # Evaluate the function: run one iteration of AltMin to satisfy BCs
            if PlotSwitch:
                plotEnergyLandscape(self.fp,self.x,self.res) # temporary
                print(f"Energy: {self.fp.updateEnergies(self.x)[2]}") # temporary
            self.x.axpy(1.0,self.res) # Add the residual to the solution vector
            self.p_old = self.res.copy() # store search direction

            self.fp.updateUV(self.x)  # Update the solution vectors
            error = self.fp.updateError() # calculate error
            self.fp.monitor(iteration) # monitor convergence

            if self.method=='Parallelogram':
                ConvCrit = [] # initialize array to store convergence criteria for parallelogram backtracking

            while error > tol and iteration < maxit:
                iteration += 1
                self.fp.Fn(None, self.x, self.res)
                if self.method=='AltMin':
                    if PlotSwitch:
                        plotEnergyLandscape(self.fp,self.x,self.res) # temporary
                        print(f"Energy: {self.fp.updateEnergies(self.x)[2]}") # temporary
                    self.x.axpy(1.0,self.res) # Add the residual to the solution vector
                    plotDirectionChange(self.res, self.p_old)
                    self.p_old = self.res
                elif self.method=='qbt':
                    if PlotSwitch:
                        plotEnergyLandscape(self.fp,self.x,self.res) # temporary
                        print(f"Energy: {self.fp.updateEnergies(self.x)[2]}") # temporary
                    CubicBacktracking(self.fp, self.x, self.res, None)
                    self.x += self.res # update solution
                else:
                    # Solve the linear system
                    self.fp.PJ.updateMat() # update Jacobian matrix in Python context
                    self.fp.PJ.getKSPs() # update KSPs
                    self.SNESKSP.solve(self.res, self.p)  # Solve the linear system
                    self.fp.PJ.resetKSPs() # reset KSPs
                    self.fp.PJ.destroyMat() # destroy Jacobian matrix components to free memory before solve
                    self.energies[i_t,6]+=self.SNESKSP.getIterationNumber()
                    DBTrick(self.fp,self.x,self.p) # apply DB trick to search direction
                    if self.method=='CubicBacktracking': # Run cubic backtracking in situ
                        if PlotSwitch:
                            plotEnergyLandscape(self.fp,self.x,self.p)
                        CubicBacktracking(self.fp, self.x, self.p, self.res)
                        self.x.axpy(1.0, self.p) # update solution
                        plotDirectionChange(self.p, self.p_old)
                        self.p_old = self.p.copy() # store search direction
                    elif self.method=='Parallelogram':
                        v, vol, flat, angle, alpha, beta = ParallelogramBacktracking(self.fp, self.x, self.res, self.p, PlotSwitch=PlotSwitch)
                        temp = [v.norm(),vol,flat,angle,alpha,beta]
                        ConvCrit.append(temp)
                        if self.fp.rank==0:
                            with open(f"output/ConvCrit_{self.identifier}.csv",'a') as csv.file:
                                writer=csv.writer(csv.file,delimiter=',')
                                writer.writerow([iteration,v.norm(),vol,flat,angle,alpha,beta])
                        self.x.axpy(1.0, v) # update solution
                        plotDirectionChange(v, self.p_old)
                        self.p_old = v.copy() # store search direction
                        v.destroy() # clean up parallelogram step vector
                    boxConstraints(self.fp,self.x) # apply box constraints to solution for backtracking methods
                self.fp.updateUV(self.x)
                error = self.fp.updateError()
                self.fp.monitor(iteration)

            if self.method=='Parallelogram' and iteration>1:
                plotConvCrit(np.array(ConvCrit))

            self.energies[i_t, 1:4] = self.fp.updateEnergies(self.x)[0:3]
            end_time = PETSc.Log.getTime()
            self.energies[i_t, 4] = end_time - start_time
            self.energies[i_t, 5] = iteration

            self.fp.v_lb.x.array[:] = self.fp.v.x.array # update lower bound for damage to ensure irreversibility

            if PlotSwitch:
                self.fp.plot(x=self.x)

            # write solution to file
            if WriteSwitch:
                with io.XDMFFile(self.fp.comm, f"output/EX_{self.identifier}.xdmf","a") as xdmf:
                    xdmf.write_function(self.fp.u, t)
                    xdmf.write_function(self.fp.v, t)

        # write energies to table
        if WriteSwitch and self.fp.rank==0:
            with open(f"output/TBL_{self.identifier}.csv",'a') as csv.file:
                writer=csv.writer(csv.file,delimiter=',')
                writer.writerows(self.energies)

    def destroy(self):
        self.x.destroy()
        self.res.destroy()
        self.p.destroy()
        self.J.destroy()
        if self.method!= 'AltMin':
            self.SNESKSP.destroy()
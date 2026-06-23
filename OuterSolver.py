import numpy as np
from petsc4py import PETSc
from Utilities import KSPsetUp, DBTrick, boxConstraints
from GlobalizationTechniques import cbt, pm1, pm2, pm23, pm3, tet
from Plotters import plotEnergyLandscape, plotEnergyLandscape2D
from dolfinx import io
import csv

class OuterSolver:
    def __init__(self, fp, example, method, loads, ksp_type="gmres", rtol=1.0e-3, max_it=100, restarts=100, monitor='off'):
        
        self.fp = fp
        self.example = example
        self.method = method
        self.loads = loads
        self.energies = np.zeros((loads.shape[0], 7)) # intialize energy storage
        self.setIdentifier()

        PETSc.Log.begin()
        PETSc.Options().insertString(f"-log_view :LOG_{self.identifier}.txt")
        stage_ossetup = PETSc.Log.Stage("OS setup")
        stage_ossetup.push()

        self.setUp()
        if self.method!='AltMin':
            self.SNESKSP = KSPsetUp(self.fp, self.J, type=ksp_type, rtol=rtol, max_it=max_it, restarts=restarts, monitor=monitor)  # Set up the KSP solver

        stage_ossetup.pop()

    def setIdentifier(self):
        self.identifier=f"{self.example}_{self.method}"

    def setUp(self):
        self.x, self.J = self.fp.createVecMat()  # Create empty vector and preconditioned Jacobian
        self.res = self.x.duplicate()  # Create a duplicate for the residual vector
        self.p = self.x.duplicate()  # Create a duplicate for the search direction
    
    def solve(self, WriteSwitch=False, PlotSwitch=False, maxit=100, tol=1e-4):

        stage_altmin = PETSc.Log.Stage("AltMin step")
        stage_mspin = PETSc.Log.Stage("MSPIN solve")
        stage_global = PETSc.Log.Stage("Glob. tech.")
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
                        writer.writerow(['Iteration','Step size','Angle','Alpha','Beta','Alpha_opt','Beta_opt','Determinant','Curvature in AltMin direction'])

        # main iteration
        for i_t, t in enumerate(self.loads):
            self.fp.updateBCs(t)
            self.energies[i_t, 0] = t
            if self.fp.rank == 0:
                print(f"-- Solving for t = {t:3.2f} --")
            
            start_time = PETSc.Log.getTime() # or time.perf_counter()?
            iteration=0
            stage_altmin.push()
            self.fp.Fn(None, self.x, self.res)  # Evaluate the function: run one iteration of AltMin to satisfy BCs
            stage_altmin.pop()
            if PlotSwitch:
                plotEnergyLandscape(self.fp,self.x,self.res) # temporary
                print(f"Energy: {self.fp.updateEnergies(self.x)[2]}") # temporary
            self.x.axpy(1.0,self.res) # Add the residual to the solution vector

            self.fp.updateUV(self.x)  # Update the solution vectors
            error = self.fp.updateError() # calculate error
            self.fp.monitor(iteration) # monitor convergence

            while error > tol and iteration < maxit:
                iteration += 1
                stage_altmin.push()
                self.fp.Fn(None, self.x, self.res)
                stage_altmin.pop()
                if self.method=='AltMin':
                    if PlotSwitch:
                        plotEnergyLandscape(self.fp,self.x,self.res) # temporary
                        print(f"Energy: {self.fp.updateEnergies(self.x)[2]}") # temporary
                    self.x.axpy(1.0,self.res) # Add the residual to the solution vector
                elif self.method=='qbt':
                    if PlotSwitch:
                        plotEnergyLandscape(self.fp,self.x,self.res) # temporary
                        print(f"Energy: {self.fp.updateEnergies(self.x)[2]}") # temporary
                    cbt(self.fp, self.x, self.res, None)
                    self.x += self.res # update solution
                else:
                    # Solve the linear system
                    stage_mspin.push()
                    self.fp.PJ.updateMat() # update Jacobian matrix in Python context
                    self.fp.PJ.getKSPs() # update KSPs
                    self.SNESKSP.solve(self.res, self.p)  # Solve the linear system
                    self.fp.PJ.resetKSPs() # reset KSPs
                    self.fp.PJ.destroyMat() # destroy Jacobian matrix components to free memory before solve
                    stage_mspin.pop()
                    self.energies[i_t,6]+=self.SNESKSP.getIterationNumber()
                    DBTrick(self.fp,self.x,self.p) # apply DB trick to search direction
                    stage_global.push()
                    if self.method=='CubicBacktracking': # Run cubic backtracking in situ
                        if PlotSwitch:
                            plotEnergyLandscape(self.fp,self.x,self.p)
                        cbt(self.fp, self.x, self.p, self.res)
                        self.x.axpy(1.0, self.p) # update solution
                    elif self.method=='Parallelogram':
                        v, angle, alpha, beta, alpha_opt, beta_opt, det, curv_AltMin = pm3(self.fp, self.x, self.res, self.p)
                        if PlotSwitch:
                            print(f"Step in AltMin: {alpha}, Step in Newton: {beta}")
                            plotEnergyLandscape2D(self.fp,self.x,self.res,self.p,[beta, alpha])

                        vnorm = v.norm()
                        if self.fp.rank==0:
                            with open(f"output/ConvCrit_{self.identifier}.csv",'a') as csv.file:
                                writer=csv.writer(csv.file,delimiter=',')
                                writer.writerow([iteration,vnorm,angle,alpha,beta,alpha_opt,beta_opt,det,curv_AltMin])

                        self.x.axpy(1.0, v) # update solution
                        v.destroy() # clean up parallelogram step vector
                    elif self.method=='Triangle':
                        v, angle, alpha, beta, alpha_opt, beta_opt, det, curv_AltMin = pm23(self.fp, self.x, self.res, self.p)
                        if PlotSwitch:
                            print(f"Step in AltMin: {alpha}, Step in Newton: {beta}")
                            plotEnergyLandscape2D(self.fp,self.x,self.res,self.p,[beta, alpha])

                        vnorm = v.norm()
                        self.x.axpy(1.0, v) # update solution
                        v.destroy() # clean up parallelogram step vector
                    elif self.method=='Tetrahedron':
                        resu, resv = self.res.getNestSubVecs() # get sub-vectors for u and v residuals
                        resv0 = resv.copy()
                        resv0.zeroEntries()
                        stepu = PETSc.Vec().createNest([resu,resv0], None, self.fp.comm)
                        v, _, _, [p_opt, q_opt, r_opt] = tet(self.fp, self.x, self.res, self.p, stepu) # compute tetrahedron step
                        if PlotSwitch:
                            xr=self.x.copy()
                            xr.axpy(r_opt, stepu)
                            plotEnergyLandscape2D(self.fp,xr,self.res, self.p, target=[q_opt, p_opt])
                        self.x.axpy(1.0, v) # update solution
                        v.destroy() # clean up tetrahedron step vector
                        stepu.destroy()
                        resv0.destroy()
                    boxConstraints(self.fp,self.x) # apply box constraints to solution for backtracking methods
                    stage_global.pop()
                self.fp.updateUV(self.x)
                error = self.fp.updateError()
                self.fp.monitor(iteration)

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
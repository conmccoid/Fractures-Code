import ufl
import numpy as np
from dolfinx import fem
from petsc4py import PETSc
from mpi4py import MPI
from NewtonSolverContext import NewtonSolverContext

class Identity:
    def mult(self, mat, X, Y):
        X.copy(Y)

class NewtonSolver:
    def __init__(self, solver1, solver2, problem1, problem2, B, C):
        self.u=problem1.u
        self.v=problem2.u
        V_u = self.u.function_space
        V_v = self.v.function_space
        self.u_old = fem.Function(V_u)
        self.u_old.x.petsc_vec.setArray(self.u.x.petsc_vec.duplicate())
        self.v_old = fem.Function(V_v)
        self.v_old.x.petsc_vec.setArray(self.v.x.petsc_vec.duplicate())
        self.solver1=solver1
        self.solver2=solver2
        self.PJ=NewtonSolverContext(B, C, solver1, solver2) # preconditioned Jacobian
        # self.PJ=Identity()
        self.output=[]

    def Fn(self, snes, x, F):
        # store old u and v values
        u_store = self.u.x.petsc_vec.duplicate()
        v_store = self.v.x.petsc_vec.duplicate()
        self.u.x.petsc_vec.copy(u_store)
        self.v.x.petsc_vec.copy(v_store)
        
        xu, xv = x.getNestSubVecs()
        # None of the ghost updates seem to change the results, nor the scatterings
        xu.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        xv.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        xu.copy(self.u.x.petsc_vec)
        xv.copy(self.v.x.petsc_vec)
        # self.u.x.array[:] = xu.array # alternative that seems to work just as well
        # self.v.x.array[:] = xv.array
        self.u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.v.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        self.u_old.x.array[:] = self.u.x.array
        self.v_old.x.array[:] = self.v.x.array
        self.u_old.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.v_old.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        # res = PETSc.Vec().createNest([self.u.x.petsc_vec - self.u_old.x.petsc_vec,self.v.x.petsc_vec - self.v_old.x.petsc_vec])
        # print(f"Norm of res after updating u_old: {res.norm():3.4e}")
        
        self.solver1.solve(None, self.u.x.petsc_vec)
        self.u.x.scatter_forward() # neither of these lines appears to change anything
        self.solver2.solve(None, self.v.x.petsc_vec)
        self.v.x.scatter_forward() # should be unnecessary, depending on KSP in solver2

        # self.solver1.solve(None, xu)
        # self.u.x.scatter_forward() # neither of these lines appears to change anything
        # self.solver2.solve(None, xv)
        # self.v.x.scatter_forward() # should be unnecessary, depending on KSP in solver2

        resu, resv=F.getNestSubVecs()
        resu.setArray(self.u_old.x.petsc_vec.array - self.u.x.petsc_vec.array)
        resv.setArray(self.v_old.x.petsc_vec.array - self.v.x.petsc_vec.array)
        resu.assemblyBegin()
        resu.assemblyEnd()
        resv.assemblyBegin()
        resv.assemblyEnd()
        self.res=F
        # print(f"Norm of res after solving for u: {res.norm():3.4e}")

        # self.v_old.x.array[:] = self.v.x.array
        # self.u_old.x.array[:] = self.u.x.array

        u_store.copy(self.u.x.petsc_vec)
        v_store.copy(self.v.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.v.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        u_store.destroy()
        v_store.destroy()

    def Jn(self, snes, x, J, P):
        J.setPythonContext(self.PJ)
        J.setUp()

    def setUp(self):
        self.solver = PETSc.SNES().create(MPI.COMM_WORLD)
        
        b_u = self.u.x.petsc_vec.duplicate()
        b_v = self.v.x.petsc_vec.duplicate()
        b = PETSc.Vec().createNest([b_u,b_v])

        local_size = b.getLocalSize()
        global_size = b.getSize()
        J = PETSc.Mat().createPython(((local_size,global_size),(local_size,global_size)),NewtonSolverContext,MPI.COMM_WORLD)
        
        self.solver.setFunction(self.Fn, b)
        self.solver.setJacobian(self.Jn, J)
        self.solver.setType('newtonls') # other types that work: nrichardson
        self.solver.setTolerances(rtol=1.0e-4, max_it=1000)
        self.solver.getKSP().setType("gmres")
        self.solver.getKSP().setTolerances(rtol=1.0e-4, max_it=100)
        self.solver.getKSP().getPC().setType("none") # try different preconditioners, i.e. bjacobi
        # each preconditioner requires information from the matrix, i.e. jacobi needs a getDiagonal method
        opts=PETSc.Options()
        opts['snes_linesearch_type']='none'
        self.solver.setFromOptions()
        self.solver.setConvergenceTest(self.customConvergenceTest)
        self.solver.setMonitor(self.customMonitor)
        # self.solver.getKSP().setMonitor(lambda snes, its, norm: print(f"Iteration:{its}, Norm:{norm:3.4e}"))
        # opts['ksp_monitor_singular_value']=None # Returns estimate of condition number of system solved by KSP
        opts['ksp_converged_reason']=None # Returns reason for convergence of the KSP
        opts['ksp_gmres_restart']=100 # Number of GMRES iterations before restart (100 doesn't do too bad)
        self.solver.getKSP().setFromOptions()
        self.solver.getKSP().setPostSolve(self.customPostSolve)
        # GMRES restarts after 30 iterations; stopping at a multiple of 30 iterations indicates breakdown and generally a singularity
        self.solver.setLineSearchPreCheck(self.customLineSearch)
        # self.solver.setForceIteration(True)

    def customMonitor(self, snes, its, norm):
        """Returns the same L2-nrom as AltMin for comparable convergence"""
        print(f"Iteration {its}: Residual Norm = {self.error_L2:3.4e}")

    def customConvergenceTest(self, snes, it, reason):
        """Calculates the same L2-norm as AltMin for comparable convergence"""
        atol, rtol, stol, max_it = snes.getTolerances()
        F = snes.getFunction()
        _, res_v = F[0].getNestSubVecs()
        bv = fem.Function(self.v.function_space)
        bv.x.petsc_vec.setArray(res_v.array)
        L2_error = ufl.inner(bv,bv) * ufl.dx
        self.error_L2 = np.sqrt(MPI.COMM_WORLD.allreduce(fem.assemble_scalar(fem.form(L2_error)), op=MPI.SUM))
        if self.error_L2 <= atol:
            snes.setConvergedReason(PETSc.SNES.ConvergedReason.CONVERGED_FNORM_ABS)
            return 1
        elif it >= max_it:
            snes.setConvergedReason(PETSc.SNES.ConvergedReason.DIVERGED_MAX_IT)
            return -1
        return 0

    def customLineSearch(self, x, y):
        """Used as a pre-check, this allows custom line search methods
        -- x: current solution
        -- y: current search direction"""

        # best strategy so far: if a/c>1 then Newton, otherwise AltMin
        # close second: if (a/c>1) OR (a/c<0) then Newton, otherwise AltMin
        diff = y - self.res
        c = self.res.norm()**2 + diff.norm()**2 - y.norm()**2
        a = self.res.norm()**2
        b = diff.norm()/self.res.norm()
        if c==0:
            trust=1
        else:
            trust=a/c
        print(f"    Fixed point iteration: {self.res.norm():3.4e}, Newton step: {y.norm():3.4e}, Relative difference: {b:3.4e}, Trust: {trust:%}")
        if self.solver.getKSP().getConvergedReason()==10:
            print(f"LINEAR SOLVE FAILURE")
            y.array[:]=self.res.array
            print(f"    AltMin step")
        elif a>c or c<0:
            print(f"    NewtonLS step")
        else:
            y.array[:]=self.res.array
            print(f"    AltMin step")
        # in practice we'll want a combination of self.res and y
        # save iteration count
        if self.solver.getIterationNumber()==0:
            self.output.append(['Elastic its','Damage its','Newton inner its','FP step','Newton step'])
        self.output.append([
            self.solver1.getKSP().getIterationNumber(),
            self.solver2.getKSP().getIterationNumber(),
            self.solver.getKSP().getIterationNumber(),
            self.res.norm(),
            y.norm()
        ])

    def customPostSolve(self,ksp,rhs,x):
        """Post solve function for the KSP in an attempt to rout unnecessary divergence breaks"""
        reason=ksp.getConvergedReason()
        if reason<0:
            print(f"    KSP diverged with reason {reason}")
            ksp.setConvergedReason(10)
            x.zeroEntries()
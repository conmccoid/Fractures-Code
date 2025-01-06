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
        self.u_old.x.array[:] = self.u.x.array
        self.v_old = fem.Function(V_v)
        self.v_old.x.array[:] = self.v.x.array
        self.solver1=solver1
        self.solver2=solver2
        self.PJ=NewtonSolverContext(B, C, solver1, solver2) # preconditioned Jacobian
        # self.PJ=Identity()

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

        res = PETSc.Vec().createNest([self.u.x.petsc_vec - self.u_old.x.petsc_vec,self.v.x.petsc_vec - self.v_old.x.petsc_vec])
        F.array[:] = -res.array
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
        
        b_u = PETSc.Vec().create()
        b_u.setType('mpi')
        b_u.setSizes(self.u.x.petsc_vec.getSize())
        b_v = PETSc.Vec().create()
        b_v.setType('mpi')
        b_v.setSizes(self.v.x.petsc_vec.getSize())
        b = PETSc.Vec().createNest([b_u,b_v])
        
        J = PETSc.Mat().createPython(self.u.x.petsc_vec.getSize()+self.v.x.petsc_vec.getSize())

        self.solver.setFunction(self.Fn, b)
        self.solver.setJacobian(self.Jn, J)
        self.solver.setType('newtonls')
        self.solver.setTolerances(rtol=1.0e-8, max_it=50)
        self.solver.getKSP().setType("gmres")
        self.solver.getKSP().setTolerances(rtol=1.0e-9, max_it=b.getSize())
        self.solver.getKSP().getPC().setType("none")
        opts=PETSc.Options()
        opts['snes_linesearch_type']='none'
        self.solver.setFromOptions()
        self.solver.setConvergenceTest(self.customConvergenceTest)
        self.solver.setMonitor(self.customMonitor)
        # self.solver.getKSP().setMonitor(lambda snes, its, norm: print(f"Iteration:{its}, Norm:{norm:3.4e}"))
        # opts=PETSc.Options()
        # opts['ksp_monitor_singular_value']=None # Returns estimate of condition number of system solved by KSP
        opts['ksp_converged_reason']=None # Returns reason for convergence of the KSP
        opts['ksp_gmres_restart']=b.getSize() # Number of GMRES iterations before restart (100 doesn't do too bad)
        self.solver.getKSP().setFromOptions()
        # GMRES restarts after 30 iterations; stopping at a multiple of 30 iterations indicates breakdown and generally a singularity
        self.solver.setLineSearchPreCheck(self.customLineSearch)

    def customMonitor(self, snes, its, norm):
        """Returns the same L2-nrom as AltMin for comparable convergence"""
        print(f"Iteration {its}: Residual Norm = {self.error_L2:3.4e}")

    def customConvergenceTest(self, snes, it, reason):
        """Calculates the same L2-norm as AltMin for comparable convergence"""
        atol, rtol, stol, max_it = snes.getTolerances()
        F = snes.getFunction()
        _, res_v = F[0].getNestSubVecs()
        bv = fem.Function(self.v.function_space)
        bv.x.array[:]=res_v.array
        L2_error = ufl.inner(bv,bv) * ufl.dx
        self.error_L2 = np.sqrt(MPI.COMM_WORLD.allreduce(fem.assemble_scalar(fem.form(L2_error)), op=MPI.SUM))
        if self.error_L2 <= atol:
            snes.setConvergedReason(PETSc.SNES.ConvergedReason.CONVERGED_FNORM_ABS)
            return 1
        elif it >= max_it:
            snes.setConvergedReason(PETSc.SNES.ConvergedReason.DIVERGED_ITS)
            return -1
        return 0

    def customLineSearch(self, x, y):
        """Used as a pre-check, this allows custom line search methods
        -- x: current solution
        -- y: current search direction"""
        # self.Fn(self.solver, x, y)
        # x_new=x.x.petsc_vec.duplicate()
        res=self.res
        res_new=self.res.duplicate()
        self.Fn(self.solver,x+y,res_new)
        self.Fn(self.solver,x+res,res)
        print(f"Fixed point iteration: {res.norm()}, Newton step: {res_new.norm()}")
        if res_new.norm() > res.norm():
            y.array[:]=res.array
            print(f"AltMin step")
        else:
            print(f"NewtonLS step")
        # in practice we'll want a combination of self.res and y
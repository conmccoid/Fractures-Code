import ufl
import numpy as np
from dolfinx import fem
from petsc4py import PETSc
from mpi4py import MPI
from NewtonSolverContext import NewtonSolverContext

# plotter
import pyvista
from pyvista.utilities.xvfb import start_xvfb
from PLOT_DamageState import plot_damage_state

class Identity:
    def mult(self, mat, X, Y):
        X.copy(Y)

class NewtonSolver:
    def __init__(self, solver1, solver2, problem1, problem2, B, C, linesearch='bt'):
        self.u=problem1.u
        self.v=problem2.u
        V_u = self.u.function_space
        V_v = self.v.function_space
        self.u_old = fem.Function(V_u)
        self.v_old = fem.Function(V_v)
        self.u.x.petsc_vec.copy(self.u_old.x.petsc_vec)
        self.v.x.petsc_vec.copy(self.v_old.x.petsc_vec)
        self.solver1=solver1
        self.solver2=solver2
        self.PJ=NewtonSolverContext(B, C, solver1, solver2, self.u, self.v) # preconditioned Jacobian
        # self.PJ=Identity()
        self.output=0
        self.comm=MPI.COMM_WORLD
        self.rank=self.comm.rank
        self.linesearch=linesearch

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
        self.u.x.scatter_forward() # update ghost values before use
        self.v.x.scatter_forward()
        self.u.x.petsc_vec.copy(self.u_old.x.petsc_vec)
        self.v.x.petsc_vec.copy(self.v_old.x.petsc_vec)
        self.u_old.x.scatter_forward()
        self.v_old.x.scatter_forward()

        # AltMin step
        self.solver1.solve(None, self.u.x.petsc_vec)
        self.u.x.scatter_forward() # further update, now that they've changed
        self.solver2.solve(None, self.v.x.petsc_vec)
        self.v.x.scatter_forward()

        resu, resv=F.getNestSubVecs()
        resu.setArray(self.u.x.petsc_vec.array - self.u_old.x.petsc_vec.array)
        resv.setArray(self.v.x.petsc_vec.array - self.v_old.x.petsc_vec.array)
        # put updated ghost values back into F
        resu.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        resv.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.res=F
        # print(f"Norm of res after solving for u: {self.res.norm():3.4e}") # norm is same across processes

        # restore original values of u and v
        u_store.copy(self.u.x.petsc_vec)
        v_store.copy(self.v.x.petsc_vec)
        self.u.x.scatter_forward()
        self.v.x.scatter_forward()
        u_store.destroy()
        v_store.destroy()

    def Jn(self, snes, x, J, P):
        J.setPythonContext(self.PJ)
        J.setUp()

    def setUp(self,rtol=1.0e-8, max_it_SNES=1000, max_it_KSP=100, ksp_restarts=30, monitor='off'):
        self.solver = PETSc.SNES().create(self.comm)
        
        b_u = self.u.x.petsc_vec.duplicate()
        b_v = self.v.x.petsc_vec.duplicate()
        b = PETSc.Vec().createNest([b_u,b_v],None,self.comm)

        local_size = b.getLocalSize()
        global_size = b.getSize()
        J = PETSc.Mat().createPython(((local_size,global_size),(local_size,global_size)),NewtonSolverContext,self.comm)
        
        self.solver.setFunction(self.Fn, b)
        self.solver.setJacobian(self.Jn, J)
        self.solver.setType('newtonls') # other types that work: nrichardson
        self.solver.setTolerances(rtol=rtol, max_it=max_it_SNES)
        self.solver.getKSP().setType("gmres")
        self.solver.getKSP().setTolerances(rtol=rtol, max_it=max_it_KSP)
        self.solver.getKSP().getPC().setType("none") # try different preconditioners, i.e. bjacobi
        # each preconditioner requires information from the matrix, i.e. jacobi needs a getDiagonal method
        opts=PETSc.Options()
        if self.linesearch!='bt':
            opts['snes_linesearch_type']='none'
        else:
            opts['snes_linesearch_type']='bt'
        opts['snes_converged_reason']=None
        opts['snes_linesearch_monitor']=None
        self.solver.setFromOptions()
        self.solver.setConvergenceTest(self.customConvergenceTest)
        if monitor!='off':
            self.solver.setMonitor(self.customMonitor)
            if monitor=='ksp':
                self.solver.getKSP().setMonitor(lambda snes, its, norm: print(f"Iteration:{its}, Norm:{norm:3.4e}"))
                opts['ksp_converged_reason']=None # Returns reason for convergence of the KSP
            elif monitor=='cond':
                opts['ksp_monitor_singular_value']=None # Returns estimate of condition number of system solved by KSP
        opts['ksp_gmres_restart']=ksp_restarts # Number of GMRES iterations before restart (default 30)
        self.solver.getKSP().setFromOptions()
        self.solver.getKSP().setPostSolve(self.customPostSolve) # in the event of a failed solve of the Newton direction, falls back to a fixed point iteration
        self.solver.setLineSearchPreCheck(self.customLineSearch) # forces a custom line search
        # self.solver.setForceIteration(True)
        opts.destroy() # destroy options database so it isn't used elsewhere by accident

    def customMonitor(self, snes, its, norm):
        """Returns the same L2-norm as AltMin for comparable convergence"""
        if self.rank==0:
            print(f"Iteration {its}: Residual Norm = {self.error_L2:3.4e}, KSP Iterations = {self.solver.getKSP().getIterationNumber()}")

    def customConvergenceTest(self, snes, it, reason):
        """Calculates the same L2-norm as AltMin for comparable convergence"""
        atol, rtol, stol, max_it = snes.getTolerances()
        F = snes.getFunction()
        _, res_v = F[0].getNestSubVecs()
        bv = fem.Function(self.v.function_space)
        bv.x.petsc_vec.setArray(res_v.array)
        bv.x.scatter_forward()
        L2_error = ufl.inner(bv,bv) * ufl.dx
        self.error_L2 = np.sqrt(self.comm.allreduce(fem.assemble_scalar(fem.form(L2_error)), op=MPI.SUM))
        if self.error_L2 <= atol:
            snes.setConvergedReason(PETSc.SNES.ConvergedReason.CONVERGED_FNORM_ABS)
            return 1
        elif it >= max_it:
            snes.setConvergedReason(PETSc.SNES.ConvergedReason.DIVERGED_MAX_IT)
            return -1
        else:
            return 0

    def customLineSearch(self, x, y):
        """Used as a pre-check, this allows custom line search methods
        -- x: current solution
        -- y: current search direction (nb: the iteration performs x-y, not x+y)
        -- self.res: current fixed point direction (FP=x + self.res)"""

        # set up DB trick
        proj_NFP = y.dot(-self.res)
        if proj_NFP < 0:
            y.setArray(-y.array)
            y.assemblyBegin()
            y.assemblyEnd()

        # use self.linesearch to choose augmented Newton type
        if self.linesearch!='bt':

            # initialize LinSolveStatus
            if self.solver.getIterationNumber()==0:
                self.LinSolveStatus=1
                self.output=0

            # if Newton fails to converge, do AltMin (nb: looks like some numerical error between this and AltMin)
            if self.LinSolveStatus==0 or self.solver.getKSP().getConvergedReason()<0 or self.linesearch=='fp':
                diff_FP2x= -self.res
                diff_FP2x.copy(y)
                if self.rank==0:
                    print(f"    LINEAR SOLVE FAILURE: AltMin step")
            elif self.linesearch=='tr':
                # trust region
                diff_FP2x= -self.res
                diff_N2x = y
                diff_N2FP= y + self.res
                if diff_N2x.norm() <= diff_N2FP.norm():
                    diff_FP2x.copy(y)
            elif self.linesearch=='ls':
                # line search
                dist_ls = np.min([1,self.res.norm()/y.norm()])
                # dist_ls = 1/(1+np.exp(y.norm() - self.res.norm())) # sigmoid activation function, no good for line search?
                y.setArray(dist_ls*y.array)
            elif self.linesearch=='2step':
                # 2-step line search
                diff_FP2x= -self.res
                diff_N2FP= y + self.res
                # if diff_N2FP.norm()==0:
                #     dist_2step = 1
                # else:
                #     dist_2step = np.min([1,diff_FP2x.norm()/diff_N2FP.norm()])
                dist_2step = 1/(1+np.exp(diff_N2FP.norm() - diff_FP2x.norm())) # sigmoid activation function, seems pretty successful, but what should the threshold be?
                y.setArray(diff_FP2x.array + dist_2step*diff_N2FP.array)
        
        # save iteration count
        if self.rank==0:
            self.output+=self.solver.getKSP().getIterationNumber()

    def customPostSolve(self,ksp,rhs,x):
        """Post solve function for the KSP in an attempt to rout unnecessary divergence breaks"""
        reason=ksp.getConvergedReason()
        if reason<0:
            if self.rank==0:
                print(f"    KSP diverged with reason {reason}")
                print(f"    LINEAR SOLVE FAILURE: AltMin step")
            ksp.setConvergedReason(10)
            x.setArray(x.array + self.res.array)
            x.assemblyBegin()
            x.assemblyEnd()
            if self.linesearch=='tr' or self.linesearch=='ls':
                self.LinSolveStatus=0
            else:
                self.LinSolveStatus=1 # while it appears better to switch permanently for each iteration, this can be changed to 1 to only switch temporarily
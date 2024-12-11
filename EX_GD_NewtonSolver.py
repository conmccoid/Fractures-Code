from dolfinx import fem
from petsc4py import PETSc
from mpi4py import MPI
from NewtonSolverContext import NewtonSolverContext

class Identity:
    def mult(self, mat, X, Y):
        X.copy(Y)

class NewtonSolver:
    def __init__(self, solver1, solver2, problem1, problem2, A, B, C, D):
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
        # self.PJ=NewtonSolverContext(A, B, C, D, problem1, problem2) # preconditioned Jacobian
        self.PJ=Identity()

    def Fn(self, snes, x, F):
        # store old u and v values?
        
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
        res = PETSc.Vec().createNest([self.u.x.petsc_vec - self.u_old.x.petsc_vec,self.v.x.petsc_vec - self.v_old.x.petsc_vec])
        print(f"Norm of res after updating u_old: {res.norm():3.4e}")
        
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
        print(f"Norm of res after solving for u: {res.norm():3.4e}")

        # self.v_old.x.array[:] = self.v.x.array
        # self.u_old.x.array[:] = self.u.x.array

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
        self.solver.setTolerances(rtol=1.0e-7, max_it=50)
        self.solver.getKSP().setType("gmres")
        self.solver.getKSP().setTolerances(rtol=1.0e-16)
        self.solver.getKSP().getPC().setType("none")
        opts=PETSc.Options()
        opts['snes_linesearch_type']='none'
        self.solver.setFromOptions()
        self.solver.setMonitor(lambda snes, it, norm: print(f"Iteration {it}: Residual Norm = {norm:3.4e}"))
        self.solver.setMonitor(self.customMonitor)
        # no idea what's wrong now

    def customMonitor(self, snes,its,norm):
        x = snes.getSolution()
        F = snes.getFunction()
        
        res = PETSc.Vec().createNest([self.u.x.petsc_vec - self.u_old.x.petsc_vec,self.v.x.petsc_vec - self.v_old.x.petsc_vec])

        test = res - F[0]
        print(f"Norm of Newton step: {res.norm():3.4e}, Norm of residual: {F[0].norm():3.4e}")
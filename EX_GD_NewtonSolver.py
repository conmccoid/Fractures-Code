from dolfinx import fem
from petsc4py import PETSc
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
        xu, xv = x.getNestSubVecs()
        xu.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        xv.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        xu.copy(self.u.vector)
        xv.copy(self.v.vector)
        self.v_old.x.array[:] = self.v.x.array
        self.u_old.x.array[:] = self.u.x.array
        self.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.v.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        
        self.solver1.solve(None, self.u.vector)
        self.u.x.scatter_forward()
        self.solver2.solve(None, self.v.vector)
        self.v.x.scatter_forward() # should be unnecessary, depending on KSP in solver2

        res_u = self.u.vector - self.u_old.vector
        res_v = self.v.vector - self.v_old.vector
        F = PETSc.Vec().createNest([res_u,res_v])
        F.assemble()
        print(f"Residual: {res_v.norm():3.4e}")
        
        # self.v_old.x.array[:] = self.v.x.array
        # self.u_old.x.array[:] = self.u.x.array

    def Jn(self, snes, x, J, P):
        J = PETSc.Mat().createPython(J.getSize())
        J.setPythonContext(self.PJ)
        J.setUp()

    def setUp(self):
        self.solver = PETSc.SNES().create()
        
        b_u = PETSc.Vec().create()
        b_u.setType('mpi')
        b_u.setSizes(self.u.vector.getSize())
        b_v = PETSc.Vec().create()
        b_v.setType('mpi')
        b_v.setSizes(self.v.vector.getSize())
        b = PETSc.Vec().createNest([b_u,b_v])
        
        J_uu = PETSc.Mat().create()
        J_uu.setType('seqaij')
        J_uu.setSizes(self.u.vector.getSizes())
        J_vv = PETSc.Mat().create()
        J_vv.setType('seqaij')
        J_vv.setSizes(self.v.vector.getSizes())
        J_uv = PETSc.Mat().create()
        J_uv.setType('seqaij')
        J_uv.setSizes([self.u.vector.getSize(), self.v.vector.getSize()])
        J_vu = PETSc.Mat().create()
        J_vu.setType('seqaij')
        J_vu.setSizes([self.v.vector.getSize(), self.u.vector.getSize()])
        J = PETSc.Mat().createNest([[J_uu,J_uv],[J_vu,J_vv]])

        self.solver.setFunction(self.Fn, b)
        self.solver.setJacobian(self.Jn, J)
        self.solver.setType('newtonls')
        self.solver.setTolerances(rtol=1.0e-9, max_it=50)
        self.solver.getKSP().setType("preonly")
        self.solver.getKSP().setTolerances(rtol=1.0e-9)
        self.solver.getKSP().getPC().setType("lu")
        opts=PETSc.Options()
        opts['snes_linesearch_type']='none'
        self.solver.setFromOptions()
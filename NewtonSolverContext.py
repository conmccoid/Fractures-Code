from dolfinx import fem
from dolfinx.fem import petsc
from petsc4py import PETSc
from mpi4py import MPI

class NewtonSolverContext:
    def __init__(self,Euv,Evu, elastic_solver, damage_solver):
        # self.E_uu=Euu
        self.E_uv=Euv
        self.E_vu=Evu
        # self.E_vv=Evv
        # self.elastic_problem=elastic_problem
        # self.damage_problem=damage_problem
        self.elastic_solver=elastic_solver
        self.damage_solver=damage_solver
        
    def mult(self, mat, X, Y):
        x1, x2 = X.getNestSubVecs()
        w1, v2 = Y.getNestSubVecs()
        # self.Euu = petsc.assemble_matrix(fem.form(self.E_uu))
        # self.Evv = petsc.assemble_matrix(fem.form(self.E_vv))
        # self.elastic_problem.Jn(None,None,self.Euu,None)
        # self.damage_problem.Jn(None,None,self.Evv,None)
        Euu, _, _ = self.elastic_solver.getJacobian()
        Evv, _, _ = self.damage_solver.getJacobian()
        self.Euv = petsc.assemble_matrix(fem.form(self.E_uv))
        self.Evu = petsc.assemble_matrix(fem.form(self.E_vu))
        
        # assemble matrices
        self.Euv.assemble()
        # self.Euv.zeroEntries() # testing equivalence with AltMin
        self.Evu.assemble()
        Euu.assemble()
        Evv.assemble()
        
        # initialize vectors
        y1=Euu.createVecLeft()
        y2=Evv.createVecLeft()
        z1=self.Euv.createVecLeft()
        z2=Evv.createVecLeft()
        
        # initialize KSPs
        ksp_uu=PETSc.KSP().create(MPI.COMM_WORLD) # re-use, and re-use previous results as initial guess, ksp.setInitialGuessNonzero
        ksp_uu.setOperators(Euu)
        ksp_uu.setType('preonly')
        ksp_uu.getPC().setType('lu')
        ksp_vv=PETSc.KSP().create(MPI.COMM_WORLD)
        ksp_vv.setOperators(Evv)
        ksp_vv.setType('preonly')
        ksp_vv.getPC().setType('lu')
        
        # multiply by Jacobian
        Euu.mult(x1,y1)
        Evv.mult(x2,y2)
        self.Euv.multAdd(x2,y1,z1)
        self.Evu.multAdd(x1,y2,z2)
        
        # multiply by preconditioner
        ksp_uu.solve(z1,w1)
        # self.elastic_solver.solve(z1,w1) # this performs the nonlinear solve, not the linear solve we need
        w2=Evv.createVecRight()
        self.Evu.multAdd(-w1,z2,w2)
        ksp_vv.solve(w2,v2)
        # self.damage_solver.solve(w2,v2)
        # then w1 and v2 get put into Y

        # destroy KSPs (& vectors?)
        ksp_uu.destroy()
        ksp_vv.destroy()
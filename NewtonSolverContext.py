from dolfinx import fem
from dolfinx.fem import petsc
from petsc4py import PETSc
from mpi4py import MPI

class NewtonSolverContext:
    def __init__(self,Euv,Evu, elastic_solver, damage_solver):
        self.E_uv=fem.form(Euv)
        self.E_vu=fem.form(Evu)
        self.elastic_solver=elastic_solver
        self.damage_solver=damage_solver
        # self.count = 0 # for debugging
        
    def mult(self, mat, X, Y):
        x1, x2 = X.getNestSubVecs()
        x1.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x2.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        w1, v2 = Y.getNestSubVecs()
        Euu, _, _ = self.elastic_solver.getJacobian()
        Evv, _, _ = self.damage_solver.getJacobian()
        Euv = petsc.assemble_matrix(self.E_uv)
        Evu = petsc.assemble_matrix(self.E_vu) # issue here: this way of assembling these matrices will not work in parallel
        
        # assemble matrices
        Euv.assemble() # test equivalence with AltMin by commenting out this line
        Evu.assemble()
        # Euv.assemblyBegin()
        # Euv.assemblyEnd()
        # Evu.assemblyBegin()
        # Evu.assemblyEnd()
        Euu.assemble()
        Evv.assemble()
        # self.count+=1
        # if self.count==3:
        #     viewer = PETSc.Viewer().createASCII("output/Euv_matrix_parallel.txt",mode="w",comm=MPI.COMM_WORLD)
        #     Euv.view(viewer)
        
        # initialize vectors
        # y1=Euu.createVecLeft()
        # y2=Evv.createVecLeft()
        # z1=self.Euv.createVecLeft()
        # z2=Evv.createVecLeft()
        y1=x1.duplicate()
        y2=x2.duplicate()
        z1=x1.duplicate()
        z2=x2.duplicate()
        
        # initialize KSPs
        # ksp_uu=PETSc.KSP().create(MPI.COMM_WORLD) # re-use, and re-use previous results as initial guess, ksp.setInitialGuessNonzero
        ksp_uu=self.elastic_solver.getKSP()
        ksp_vv=self.damage_solver.getKSP()
        opts=PETSc.Options()
        opts['ksp_reuse_preconditioner']=True
        ksp_uu.setFromOptions()
        ksp_vv.setFromOptions()
        opts.destroy()
        
        # multiply by Jacobian
        Euu.mult(x1,y1)
        Evv.mult(x2,y2)
        y1.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        y2.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        Euv.multAdd(x2,y1,z1)
        Evu.multAdd(x1,y2,z2)
        z1.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        z2.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        # multiply by preconditioner
        ksp_uu.solve(z1,w1)
        z1.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        # self.elastic_solver.solve(z1,w1) # this performs the nonlinear solve, not the linear solve we need
        # w2=Evv.createVecRight()
        w2=x2.duplicate()
        Evu.multAdd(-w1,z2,w2)
        w2.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        ksp_vv.solve(w2,v2)
        v2.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        # self.damage_solver.solve(w2,v2)
        # then w1 and v2 get put into Y

        # # destroy KSPs (& vectors?)
        # ksp_uu.destroy()
        # ksp_vv.destroy()
        Evu.destroy()
        Euv.destroy()
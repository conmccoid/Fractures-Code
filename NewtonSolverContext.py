from dolfinx import fem
from dolfinx.fem import petsc
from petsc4py import PETSc
from mpi4py import MPI

class NewtonSolverContext:
    def __init__(self,Euv,Evu, elastic_solver, damage_solver, u, v):
        self.E_uv=fem.form(Euv)
        self.E_vu=fem.form(Evu)
        self.elastic_solver=elastic_solver
        self.damage_solver=damage_solver
        self.u = u
        self.v = v
        
    def mult(self, mat, X, Y):
        print(f"X type: {X.getType()}, X size: {X.getSize()}")
        if X.getType() != PETSc.Vec.Type.NEST:
            is_u = PETSc.IS().createStride(self.u.x.petsc_vec.getSize())
            is_v = PETSc.IS().createStride(self.v.x.petsc_vec.getSize(),self.u.x.petsc_vec.getSize())
            x1 = X.getSubVector(is_u)
            x2 = X.getSubVector(is_v)
            x1.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            x2.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            w1 = Y.getSubVector(is_u)
            v2 = Y.getSubVector(is_v)
        else:
            x1, x2 = X.getNestSubVecs()
            x1.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            x2.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            w1, v2 = Y.getNestSubVecs()
        Euu, _, _ = self.elastic_solver.getJacobian()
        Evv, _, _ = self.damage_solver.getJacobian()
        self.u.x.scatter_forward()
        self.v.x.scatter_forward()
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

        # initialize vectors
        y1=x1.duplicate()
        y2=x2.duplicate()
        z1=x1.duplicate()
        z2=x2.duplicate()
        
        # initialize KSPs
        ksp_uu=self.elastic_solver.getKSP()
        ksp_vv=self.damage_solver.getKSP()
        opts=PETSc.Options()
        opts['ksp_reuse_preconditioner']=True
        ksp_uu.setFromOptions()
        ksp_vv.setFromOptions()
        
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

        # clean-up
        opts['ksp_reuse_preconditioner']=False
        ksp_uu.setFromOptions()
        ksp_vv.setFromOptions()
        opts.destroy()
        Evu.destroy()
        Euv.destroy()
        Euu.destroy()
        Evv.destroy()

    def getDiagonal(self, mat, diag):
        Euu, _, _ = self.elastic_solver.getJacobian()
        Evv, _, _ = self.damage_solver.getJacobian()
        Euu.assemble()
        Evv.assemble()
        diag1=Euu.getDiagonal()
        diag2=Evv.getDiagonal()
        diag_temp = PETSc.Vec().createNest([diag1, diag2])
        diag.setArray(diag_temp)
        diag_temp.destroy()
        diag1.destroy()
        diag2.destroy()
        Euu.destroy()
        Evv.destroy()
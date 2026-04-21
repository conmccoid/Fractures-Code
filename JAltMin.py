from dolfinx import fem
from dolfinx.fem import petsc
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

class JAltMin:
    def __init__(self, elastic_solver, damage_solver, E_uv, E_vu):
        self.Euv = E_uv
        self.Evu = E_vu
        self.elastic_solver = elastic_solver
        self.damage_solver = damage_solver
        self.rank = MPI.COMM_WORLD.rank

    def updateMat(self):
        self.Euu, _, _ = self.elastic_solver.getJacobian()
        self.Evv, _, _ = self.damage_solver.getJacobian()
        self.Euu.assemble()
        self.Euv.assemble()
        self.Evu.assemble()
        self.Evv.assemble()

    def destroyMat(self):
        # self.Euv.destroy()
        # self.Evu.destroy()
        self.IS_u.destroy()
        pass
    
    def getKSPs(self):
        self.ksp_uu=self.elastic_solver.getKSP()
        self.ksp_vv=self.damage_solver.getKSP()
        self.opts=PETSc.Options()
        self.opts['ksp_reuse_preconditioner'] = True
        self.ksp_uu.setFromOptions()
        self.ksp_vv.setFromOptions()
        self.IS_u=PETSc.IS().createStride(self.Euu.getSize()[0], comm=MPI.COMM_WORLD)
        self.IS_v=self.damage_solver.getVIInactiveSet() # get inactive set from damage solver

    def resetKSPs(self):
        self.opts['ksp_reuse_preconditioner'] = False
        self.ksp_uu.setFromOptions()
        self.ksp_vv.setFromOptions()
        self.opts.destroy()

    def mult(self, mat, X, Y):
        x1, x2 = X.getNestSubVecs()
        y1, y2 = Y.getNestSubVecs()

        n_temp=self.ksp_vv.getOperators()[0].getSize()[0]
        m_temp=y2.getSize()
        if m_temp == n_temp: # inactive set is entire set, no need for IS
            # multiply by J
            self.Euu.mult(x1, y1)
            self.Evv.mult(x2, y2)
            self.Euv.multAdd(x2, y1, y1)
            self.Evu.multAdd(x1, y2, y2)

            # multiply by P
            self.ksp_uu(y1, y1)
            y1.scale(-1.0)
            self.Evu.multAdd(y1,y2,y2)
            self.ksp_vv(y2,y2)
        
        else: # inactive set is smaller than entire set, need to use IS
            # extract components associated with inactive set
            x2_IS=x2.getSubVector(self.IS_v)
            y2_IS=y2.getSubVector(self.IS_v)
            Evv_IS=self.Evv.createSubMatrix(self.IS_v,self.IS_v)
            Evu_IS=self.Evu.createSubMatrix(self.IS_v,self.IS_u)
            Euv_IS=self.Euv.createSubMatrix(self.IS_u,self.IS_v)

            # multiply by J
            self.Euu.mult(x1,y1)
            Evv_IS.mult(x2_IS,y2_IS)
            Euv_IS.multAdd(x2_IS,y1,y1)
            Evu_IS.multAdd(x1,y2_IS,y2_IS)

            # multiply by P
            self.ksp_uu(y1,y1)
            y1.scale(-1.0)
            Evu_IS.multAdd(y1,y2_IS,y2_IS)
            self.ksp_vv(y2_IS,y2_IS)
            y2.restoreSubVector(self.IS_v,y2_IS)

            # clean-up
            Evv_IS.destroy()
            Evu_IS.destroy()
            Euv_IS.destroy()
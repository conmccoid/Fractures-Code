from dolfinx import fem
from dolfinx.fem import petsc
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

class JAltMin:
    def __init__(self, elastic_solver, damage_solver, E_uv, E_vu):
        self.EuvForm = fem.form(E_uv)
        self.EvuForm = fem.form(E_vu)
        self.elastic_solver = elastic_solver
        self.damage_solver = damage_solver

    def updateMat(self):
        self.Euu, _, _ = self.elastic_solver.getJacobian()
        self.Euv = petsc.assemble_matrix(self.EuvForm)
        self.Evu = petsc.assemble_matrix(self.EvuForm)
        self.Evv, _, _ = self.damage_solver.getJacobian()
        self.Euu.assemble()
        self.Euv.assemble()
        self.Evu.assemble()
        self.Evv.assemble()
    
    def getKSPs(self):
        ksp_uu=self.elastic_solver.getKSP()
        ksp_vv=self.damage_solver.getKSP()
        # v_inactive=self.damage_solver.getVIInactiveSet() # need the inactive set when using VI Newton
        opts=PETSc.Options()
        opts['ksp_reuse_preconditioner'] = True
        ksp_uu.setFromOptions()
        ksp_vv.setFromOptions()
        opts.destroy()
        return ksp_uu, ksp_vv #, v_inactive
    
    def resetKSPs(self, ksp):
        opts= PETSc.Options()
        opts['ksp_reuse_preconditioner'] = False
        ksp.setFromOptions()
        opts.destroy()

    def mult(self, mat, X, Y):
        x1, x2 = X.getNestSubVecs()
        y1, y2 = Y.getNestSubVecs()

        self.updateMat()

        # multiply by J
        self.Euu.mult(x1, y1)
        self.Evv.mult(x2, y2)
        self.Euv.multAdd(x2, y1, y1)
        self.Evu.multAdd(x1, y2, y2)

        # multiply by P
        ksp_uu, ksp_vv = self.getKSPs()
        ksp_uu.solve(y1, y1)
        self.Evu.multAdd(-y1,y2,y2)
        v_temp=ksp_vv.getSolution()
        n_temp=len(v_temp.array)

        # ksp_vv.solve(y2, y2)
        # # trying to restrict to inactive set in the ksp_vv.solve()
        if len(y2.array) != n_temp:
            IS=self.getInactiveSet(n_temp)
            y2_inactive=PETSc.Vec().create() # make sure this gets the same comm as y2
            y2.getSubVector(IS, y2_inactive)
            try:
                ksp_vv.solve(y2_inactive, y2_inactive)
            except:
                print("Failed to solve inactive system")
                print(f"size of IS: {IS.size}, size of v_temp: {len(v_temp.array)}")

            y2.restoreSubVector(IS, y2_inactive)
            y2_inactive.destroy()
            IS.destroy()
        else:
            ksp_vv.solve(y2, y2)
        self.resetKSPs(ksp_uu)
        self.resetKSPs(ksp_vv)
    
    def getInactiveSet(self,n):
        v_ext=self.damage_solver.getSolution() # get current damage solution (external)
        v_lb, v_ub = self.damage_solver.getVariableBounds() # get current bounds

        # determine distance from bounds
        dist_low=v_ext.array - v_lb.array
        dist_upp=v_ub.array - v_ext.array
        dist=np.minimum(dist_low, dist_upp)
        is_temp = np.sort(np.argsort(dist)[-n:]).astype(PETSc.IntType) # sort and take largest n indices

        # tol_bds=1e-8
        # cond_int = (v_ext.array > v_lb.array+tol_bds) & (v_ext.array < v_ub.array-tol_bds)
        # is_temp = np.where(cond_int)[0].astype(PETSc.IntType)

        # f=self.damage_solver.getFunction()[0]
        # tol_fun=1e-10
        # cond_low = (np.abs(v_ext.array-v_lb.array)<tol_bds) & (f.array < -tol_fun)
        # cond_upp = (np.abs(v_ext.array - v_ub.array) < tol_bds) & (f.array > tol_fun)
        # is_temp = np.where(cond_int | cond_low | cond_upp)[0].astype(PETSc.IntType)
        # print(f"Interior inactive:{sum(cond_int)}, lower inactive:{sum(cond_low)}, upper inactive:{sum(cond_upp)}")


        # IS_f=f.array.nonzero()[0].astype(PETSc.IntType)
        # IS_f=np.where(np.abs(f.array) > 1e-10)[0].astype(PETSc.IntType)
        # print(f"nnz of solution update: {len(IS_f)}")
        # ps=np.setdiff1d(is_temp, IS_f, assume_unique=True)
        # if len(ps)>0:
        #     print(f.array[ps])
        
        IS = PETSc.IS().createGeneral(is_temp, comm=v_ext.comm) # construct PETSc index set
        return IS
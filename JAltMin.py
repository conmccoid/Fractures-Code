from dolfinx import fem
from dolfinx.fem import petsc
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

from Utilities import monitorMem, storeMem

class JAltMin:
    def __init__(self, elastic_solver, damage_solver, E_uv, E_vu):
        self.EuvForm = fem.form(E_uv)
        self.EvuForm = fem.form(E_vu)
        self.elastic_solver = elastic_solver
        self.damage_solver = damage_solver
        self.rank = MPI.COMM_WORLD.rank
        # self.mem=[0,0,0]

    def updateMat(self):
        self.Euu, _, _ = self.elastic_solver.getJacobian()
        self.Euv = petsc.assemble_matrix(self.EuvForm)
        self.Evu = petsc.assemble_matrix(self.EvuForm)
        self.Evv, _, _ = self.damage_solver.getJacobian()
        self.Euu.assemble()
        self.Euv.assemble()
        self.Evu.assemble()
        self.Evv.assemble()

    def destroyMat(self):
        self.Euv.destroy()
        self.Evu.destroy()
    
    def getKSPs(self):
        self.ksp_uu=self.elastic_solver.getKSP()
        self.ksp_vv=self.damage_solver.getKSP()
        # ksp_elastic=self.elastic_solver.getKSP()
        # ksp_damage=self.damage_solver.getKSP()
        # self.ksp_uu=PETSc.KSP().create(ksp_elastic.comm)
        # self.ksp_vv=PETSc.KSP().create(ksp_damage.comm)
        # self.ksp_uu.setOperators(ksp_elastic.getOperators()[0])
        # self.ksp_vv.setOperators(ksp_damage.getOperators()[0])
        # self.ksp_uu.setType(ksp_elastic.getType())
        # self.ksp_vv.setType(ksp_damage.getType())
        # rtol, atol, dtol, max_it = ksp_elastic.getTolerances()
        # self.ksp_uu.setTolerances(rtol, atol, dtol, max_it)
        # rtol, atol, dtol, max_it = ksp_damage.getTolerances()
        # self.ksp_vv.setTolerances(rtol, atol, dtol, max_it)
        # self.ksp_uu.setPC(ksp_elastic.getPC())
        # self.ksp_vv.setPC(ksp_damage.getPC())

        self.opts=PETSc.Options()
        self.opts['ksp_reuse_preconditioner'] = True
        self.ksp_uu.setFromOptions()
        self.ksp_vv.setFromOptions()
        self.IS=self.damage_solver.getVIInactiveSet() # get inactive set from damage solver

    def resetKSPs(self):
        self.opts['ksp_reuse_preconditioner'] = False
        self.ksp_uu.setFromOptions()
        self.ksp_vv.setFromOptions()
        self.opts.destroy()
        # self.IS.destroy()
        # self.ksp_uu.destroy()
        # self.ksp_vv.destroy()

    def mult(self, mat, X, Y):
        stage=PETSc.Log.Stage("mult")
        stage.push()
        x1, x2 = X.getNestSubVecs()
        y1, y2 = Y.getNestSubVecs()

        # multiply by J
        self.Euu.mult(x1, y1)
        self.Evv.mult(x2, y2)
        self.Euv.multAdd(x2, y1, y1)
        self.Evu.multAdd(x1, y2, y2)

        #====Memory leak====#
        # multiply by P
        # mem_u1=monitorMem(self.rank, 'pre-solve u')
        self.ksp_uu(y1, y1)
        # mem_u2=monitorMem(self.rank, 'post-solve u')
        # self.mem[0]+=mem_u2-mem_u1
        y1.scale(-1.0)
        self.Evu.multAdd(y1,y2,y2)
        n_temp=self.ksp_vv.getOperators()[0].getSize()[0]
        m_temp=y2.getSize()

        if m_temp != n_temp:
            # IS=self.getInactiveSet(n_temp)
            y2_rhs=y2.getSubVector(self.IS)
            y2_sol=y2_rhs.duplicate()
            # mem_v1=monitorMem(self.rank, 'pre-solve inactive')
            self.ksp_vv(y2_rhs, y2_sol) # ***MEMORY LEAK***
            # mem_v2=monitorMem(self.rank, 'post-solve inactive')
            # self.mem[2]+=mem_v2-mem_v1
            y2.restoreSubVector(self.IS, y2_sol)
            y2_rhs.destroy()
        else:
            # mem_v1=monitorMem(self.rank, 'pre-solve v')
            self.ksp_vv(y2, y2)
            # mem_v2=monitorMem(self.rank, 'post-solve v')
            # self.mem[1]+=mem_v2-mem_v1
        # storeMem(self.rank, self.mem)
        stage.pop()
        #===#
    
    def getInactiveSet(self,n):
        v_ext=self.damage_solver.getSolution() # get current damage solution (external)
        v_lb, v_ub = self.damage_solver.getVariableBounds() # get current bounds

        # determine distance from bounds
        dist_low=v_ext.array - v_lb.array
        dist_upp=v_ub.array - v_ext.array
        dist=np.minimum(dist_low, dist_upp)
        is_temp = np.sort(np.argsort(dist)[-n:]).astype(PETSc.IntType) # sort and take largest n indices

        # # conditions used in PETSc VI, but non-functional here
        # f,F=self.damage_solver.getFunction() # get damage function and storage vector
        # F[0](F[1],v_ext,f)  # evaluate Jacobian at current damage solution
        # tol_bds=1e-8
        # cond_low = v_ext.array > v_lb.array + tol_bds
        # cond_upp = v_ext.array < v_ub.array - tol_bds
        # grad_low = f.array <= 0.0
        # grad_upp = f.array >= 0.0
        # is_temp = np.where((cond_low | grad_low) & (cond_upp | grad_upp))[0].astype(PETSc.IntType)

        IS = PETSc.IS().createGeneral(is_temp, comm=v_ext.comm) # construct PETSc index set
        return IS
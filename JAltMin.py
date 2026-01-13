from dolfinx import fem
from dolfinx.fem import petsc
from mpi4py import MPI
from petsc4py import PETSc

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
        # debug
        # v_temp=ksp_vv.getSolution()
        # v_lb, v_ub = self.damage_solver.getVariableBounds()
        # inactive_count = ((v_temp > v_lb) & (v_temp < v_ub)).sum()
        # if inactive_count < len(y2):
        #     print(f"Warning: inactive count {inactive_count} less than vector length {len(y2)}")

        ksp_vv.solve(y2, y2)
        # # trying to restrict to inactive set in the ksp_vv.solve()
        # y2_inactive=PETSc.Vec().create() # make sure this gets the same comm as y2
        # y2.getSubVector(v_inactive, y2_inactive)
        # ksp_vv.solve(y2_inactive, y2_inactive)
        # y2.restoreSubVector(v_inactive, y2_inactive)
        # y2_inactive.destroy()
        # v_inactive.destroy()
        self.resetKSPs(ksp_uu)
        self.resetKSPs(ksp_vv)
from dolfinx import fem, la, io
from dolfinx.fem import petsc
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl

import pyvista
from pyvista.plotting.utilities import start_xvfb

from EX_GD_Domain import domain, BCs, VariationalFormulation
from Solvers import Elastic, Damage
from PLOT_DamageState import plot_damage_state
from JAltMin import JAltMin

class FPAltMin:
    def __init__(self):
        self.comm=MPI.COMM_WORLD
        self.rank=self.comm.rank

        L=1.
        H=0.3
        cell_size=0.1/6
        self.u, self.v, self.dom=domain(L,H,cell_size)
        V_u=self.u.function_space
        V_v=self.v.function_space
        bcs_u, bcs_v, self.u_D = BCs(self.u,self.v,self.dom,L,H)
        E_u, E_v, E_uu, E_vv, E_uv, E_vu, self.elastic_energy, self.dissipated_energy, load_C, E, self.total_energy = VariationalFormulation(self.u,self.v,self.dom)

        elastic_problem, self.elastic_solver = Elastic(E_u, self.u, bcs_u, E_uu)
        damage_problem, self.damage_solver = Damage(E_v, self.v, bcs_v, E_vv)

        self.u_old = fem.Function(V_u)
        self.v_old = fem.Function(V_v)

        # initialize damage bounds
        self.v_lb =  fem.Function(V_v, name="Lower bound")
        self.v_ub =  fem.Function(V_v, name="Upper bound")
        self.v_lb.x.array[:] = 0.0
        self.v_ub.x.array[:] = 1.0
        self.damage_solver.setVariableBounds(self.v_lb.x.petsc_vec,self.v_ub.x.petsc_vec)

        self.PJ = JAltMin(self.elastic_solver, self.damage_solver, E_uv, E_vu, self.v_lb, self.v_ub)

        self.Eu = fem.form(E_u)
        self.Ev = fem.form(E_v)
        self.gradF = PETSc.Vec().createNest([self.u.x.petsc_vec.duplicate(), self.v.x.petsc_vec.duplicate()], None, self.comm) # I think there's a better way to initialize this using Eu and Ev

        start_xvfb(wait=0.5)
    
    def createVecMat(self):
        b_u = self.u.x.petsc_vec.duplicate()
        b_v = self.v.x.petsc_vec.duplicate()
        b = PETSc.Vec().createNest([b_u, b_v],None,self.comm)
        b.zeroEntries()

        local_size = b.getLocalSize()
        global_size = b.getSize()
        J = PETSc.Mat().createPython(
            ((local_size, local_size), (global_size, global_size)),
            JAltMin,
            comm=self.comm
        )
        return b, J

    def updateBCs(self, t):
        self.u_D.value = t

    def updateUV(self,x):
        xu, xv = x.getNestSubVecs()
        self.u.x.petsc_vec.setArray(xu.array)
        self.v.x.petsc_vec.setArray(xv.array)
        self.u.x.scatter_forward()
        self.v.x.scatter_forward()

    def updateUV_old(self):
        self.u_old.x.petsc_vec.setArray(self.u.x.array)
        self.v_old.x.petsc_vec.setArray(self.v.x.array)

    def updateEnergies(self, x):
        self.updateUV(x)
        energies=np.zeros((3,))
        energies[0] = self.comm.allreduce(
            fem.assemble_scalar(fem.form(self.elastic_energy)),
            op=MPI.SUM
        )
        energies[1] = self.comm.allreduce(
            fem.assemble_scalar(fem.form(self.dissipated_energy)),
            op=MPI.SUM
        )
        energies[2] = self.comm.allreduce(
            fem.assemble_scalar(fem.form(self.total_energy)),
            op=MPI.SUM
        )
        return energies
    
    def updateGradF(self,x):
        self.updateUV(x)
        Eu, Ev = self.gradF.getNestSubVecs()
        with Eu.localForm() as f_local:
            f_local.set(0.0)
        with Ev.localForm() as f_local:
            f_local.set(0.0)
        petsc.assemble_vector(Eu, self.Eu)
        petsc.assemble_vector(Ev, self.Ev)

    def Fn(self, snes, x, F):
        self.updateUV(x)
        self.updateUV_old()

        self.elastic_solver.solve(None, self.u.x.petsc_vec)
        self.u.x.scatter_forward()
        self.damage_solver.solve(None, self.v.x.petsc_vec)

        resu, resv = F.getNestSubVecs()
        resu.setArray(self.u.x.array - self.u_old.x.array)
        resv.setArray(self.v.x.array - self.v_old.x.array)
        resu.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        resv.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        F.assemblyBegin()
        F.assemblyEnd()

    def Jn(self, snes, x, J, P):
        J.setPythonContext(self.PJ)
        J.setUp()

    def plot(self, x, size=(800, 300)):
        self.updateUV(x)
        plot_damage_state(self.u, self.v, None, size)

    def updateError(self):
        error = ufl.inner(self.v - self.v_old, self.v - self.v_old) * ufl.dx + ufl.inner(self.u - self.u_old, self.u - self.u_old) * ufl.dx
        self.error_L2 = np.sqrt(self.comm.allreduce(fem.assemble_scalar(fem.form(error)), op=MPI.SUM))
        return self.error_L2
    
    def monitor(self, iteration):
        if self.rank == 0:
            print(f"Iteration: {iteration}, Error: {self.error_L2: 3.4e}")
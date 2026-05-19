from dolfinx import fem, la, io
from dolfinx.fem import petsc
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import ufl

import pyvista

from Solvers import Elastic, Damage
from PLOT_DamageState import plot_damage_state
from JAltMin import JAltMin

class FPAltMin:
    def setUp(self,E_u, E_v, E_uu, E_vv, E_uv, E_vu, bcs_u, bcs_v):
        
        #===debug===#
        # PETSc.Log.begin()
        # PETSc.Options().insertString("-log_view")

        #===set up communicator===#
        # self.comm=MPI.COMM_WORLD
        self.comm=self.u.function_space.mesh.comm
        self.rank=self.comm.rank

        # function spaces
        V_u=self.u.function_space
        V_v=self.v.function_space

        # field-split solvers
        elastic_problem, self.elastic_solver = Elastic(E_u, self.u, bcs_u, E_uu)
        damage_problem, self.damage_solver = Damage(E_v, self.v, bcs_v, E_vv)

        # store old solutions
        self.u_old = fem.Function(V_u)
        self.v_old = fem.Function(V_v)

        # initialize damage bounds (irreversibility condition)
        self.v_lb =  fem.Function(V_v, name="Lower bound")
        self.v_ub =  fem.Function(V_v, name="Upper bound")
        self.v_lb.x.array[:] = 0.0
        self.v_ub.x.array[:] = 1.0
        self.damage_solver.setVariableBounds(self.v_lb.x.petsc_vec,self.v_ub.x.petsc_vec)

        # off-diagonal blocks for Jacobian
        Euv = petsc.assemble_matrix(fem.form(E_uv), bcs=bcs_u, diag=0.0)
        Evu = petsc.assemble_matrix(fem.form(E_vu), bcs=bcs_v, diag=0.0)
        self.PJ = JAltMin(self.elastic_solver, self.damage_solver, Euv, Evu)

        # gradient set-up
        self.Eu = fem.form(E_u)
        self.Ev = fem.form(E_v)
        self.bcs_u = bcs_u
        self.bcs_v = bcs_v
        ucopy=self.u.x.petsc_vec.duplicate()
        vcopy=self.v.x.petsc_vec.duplicate()
        self.gradF = PETSc.Vec().createNest([ucopy, vcopy], None, self.comm) # I think there's a better way to initialize this using Eu and Ev
        ucopy.destroy()
        vcopy.destroy()

        # energies set-up
        self.elastic_energy = fem.form(self.elastic_energy)
        self.dissipated_energy = fem.form(self.dissipated_energy)
        self.total_energy = fem.form(self.total_energy)
        self.error = fem.form(ufl.inner(self.v - self.v_old, self.v - self.v_old) * ufl.dx + ufl.inner(self.u - self.u_old, self.u - self.u_old) * ufl.dx)

        # initialize pyvista for plotting (optional)
        pyvista.OFF_SCREEN = True
        pyvista.set_jupyter_backend('trame')

    def createVecMat(self):
        """Create the nested vector and matrix for the main system."""
        b_u = self.u.x.petsc_vec.duplicate()
        b_v = self.v.x.petsc_vec.duplicate()
        b = PETSc.Vec().createNest([b_u, b_v],None,self.comm)
        b.zeroEntries()
        b_u.destroy()
        b_v.destroy()

        local_size = b.getLocalSize()
        global_size = b.getSize()
        J = PETSc.Mat().createPython(
            ((local_size, global_size), (local_size, global_size)),
            self.PJ,
            comm=self.comm
        )
        return b, J

    def updateUV(self,x):
        """Update the solution vectors u and v from the nested vector x."""
        xu, xv = x.getNestSubVecs()
        self.u.x.petsc_vec.setArray(xu.array)
        self.v.x.petsc_vec.setArray(xv.array)
        self.u.x.scatter_forward()
        self.v.x.scatter_forward()

    def updateUV_old(self):
        """Store previous solution vectors."""
        self.u_old.x.petsc_vec.setArray(self.u.x.petsc_vec.getArray())
        self.v_old.x.petsc_vec.setArray(self.v.x.petsc_vec.getArray())
        self.u_old.x.scatter_forward()
        self.v_old.x.scatter_forward()

    def updateEnergies(self, x):
        """Update and return the energies based on the current solution."""
        self.updateUV(x)
        energies=np.zeros((3,))
        energies[0] = self.comm.allreduce(
            fem.assemble_scalar(self.elastic_energy),
            op=MPI.SUM
        )
        energies[1] = self.comm.allreduce(
            fem.assemble_scalar(self.dissipated_energy),
            op=MPI.SUM
        )
        energies[2] = self.comm.allreduce(
            fem.assemble_scalar(self.total_energy),
            op=MPI.SUM
        )
        return energies
    
    def updateGradF(self,x):
        """
        Update the gradient vector based on the current solution.
        
        nb: this might be bugged, but you shouldn't need it immediately
        """
        self.updateUV(x)
        Eu, Ev = self.gradF.getNestSubVecs()
        with Eu.localForm() as f_local:
            f_local.set(0.0)
        with Ev.localForm() as f_local:
            f_local.set(0.0)
        petsc.assemble_vector(Eu, self.Eu)
        petsc.assemble_vector(Ev, self.Ev)
        petsc.set_bc(Eu, self.bcs_u)
        petsc.set_bc(Ev, self.bcs_v)

    def Fn(self, snes, x, F):
        """Evaluate AltMin step and return the residual F."""
        self.updateUV(x)
        self.updateUV_old()

        self.elastic_solver.solve(None, self.u.x.petsc_vec)
        self.u.x.scatter_forward()
        self.damage_solver.solve(None, self.v.x.petsc_vec)
        self.v.x.scatter_forward()

        resu, resv = F.getNestSubVecs()
        resu.setArray(self.u.x.petsc_vec.getArray() - self.u_old.x.petsc_vec.getArray())
        resv.setArray(self.v.x.petsc_vec.getArray() - self.v_old.x.petsc_vec.getArray())
        resu.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        resv.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        F.assemblyBegin()
        F.assemblyEnd()

    def Jn(self, snes, x, J, P):
        # J.setPythonContext(self.PJ)
        # J.setUp()
        pass

    def plot(self, x, size=(800, 300)):
        """Plot the current displacement and damage state."""
        self.updateUV(x)
        plot_damage_state(self.u, self.v, None, size)

    def updateError(self):
        """Calculate the L2 error between the current and previous solutions."""
        self.error_L2 = np.sqrt(self.comm.allreduce(fem.assemble_scalar(self.error), op=MPI.SUM))
        return self.error_L2
    
    def monitor(self, iteration):
        if self.rank == 0:
            print(f"Iteration: {iteration}, Error: {self.error_L2: 3.4e}")

    def destroy(self):
        """Clean up PETSc objects."""
        self.gradF.destroy()
        self.elastic_solver.destroy()
        self.damage_solver.destroy()
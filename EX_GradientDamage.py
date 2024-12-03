import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem

from EX_GD_Domain import domain, BCs, VariationalFormulation
from EX_GD_Solvers import Elastic, Damage, Newton, alternate_minimization, AMEN
from EX_GD_Visualization import plot_damage_state

L=1.
H=0.3
cell_size=0.1/6
u, v, dom=domain(L,H,cell_size)
V_u=u.function_space
V_v=v.function_space
bcs_u, bcs_v, u_D = BCs(u,v,dom,L,H)
E_u, E_v, E_uu, E_vv, E_uv, E_vu, elastic_energy, dissipated_energy = VariationalFormulation(u,v,dom)

# now we want to solve E_u(u,v)=0 and E_v(u,v)=0 with alternate minimization with a Newton accelerator
# first set up solvers for the individual minimizations

elastic_problem, elastic_solver = Elastic(E_u, u, bcs_u, E_uu)
damage_problem, damage_solver = Damage(E_v, v, bcs_v, E_vv)
v_lb =  fem.Function(V_v, name="Lower bound")
v_ub =  fem.Function(V_v, name="Upper bound")
v_lb.x.array[:] = 0.0
v_ub.x.array[:] = 1.0
damage_solver.setVariableBounds(v_lb.vector,v_ub.vector)
EN_solver = Newton(E_uu, E_vv, E_uv, E_vu, elastic_problem, damage_problem)

# Solving the problem and visualizing
import pyvista
from pyvista.utilities.xvfb import start_xvfb
start_xvfb(wait=0.5)

load_c = 0.19 * L  # reference value for the loading (imposed displacement)
loads = np.linspace(0, 1.5 * load_c, 20)

# Array to store results
energies = np.zeros((loads.shape[0], 3))

for i_t, t in enumerate(loads):
    u_D.value = t
    energies[i_t, 0] = t

    # Update the lower bound to ensure irreversibility of damage field.
    v_lb.x.array[:] = v.x.array

    print(f"-- Solving for t = {t:3.2f} --")
    # alternate_minimization(u, v, elastic_solver, damage_solver)
    AMEN(u,v, elastic_solver, damage_solver, EN_solver)
    plot_damage_state(u, v)

    # Calculate the energies
    energies[i_t, 1] = MPI.COMM_WORLD.allreduce(
        fem.assemble_scalar(fem.form(elastic_energy)),
        op=MPI.SUM,
    )
    energies[i_t, 2] = MPI.COMM_WORLD.allreduce(
        fem.assemble_scalar(fem.form(dissipated_energy)),
        op=MPI.SUM,
    )
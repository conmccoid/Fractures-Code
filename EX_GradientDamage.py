import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io

import pyvista
from pyvista.utilities.xvfb import start_xvfb
import matplotlib.pyplot as plt
import csv
import sys

from EX_GD_Domain import domain, BCs, VariationalFormulation
from Solvers import Elastic, Damage, alternate_minimization
from PLOT_DamageState import plot_damage_state
from NewtonSolver import NewtonSolver

def main(method='AltMin',linesearch='bt',PlotSwitch=False,WriteSwitch=False):
    comm=MPI.COMM_WORLD
    rank=comm.rank
    
    L=1.
    H=0.3
    cell_size=0.1/6
    u, v, dom=domain(L,H,cell_size)
    V_u=u.function_space
    V_v=v.function_space
    bcs_u, bcs_v, u_D = BCs(u,v,dom,L,H)
    E_u, E_v, E_uu, E_vv, E_uv, E_vu, elastic_energy, dissipated_energy, load_c, E, total_energy = VariationalFormulation(u,v,dom)
    
    # now we want to solve E_u(u,v)=0 and E_v(u,v)=0 with alternate minimization with a Newton accelerator
    # first set up solvers for the individual minimizations
    
    elastic_problem, elastic_solver = Elastic(E_u, u, bcs_u, E_uu)
    damage_problem, damage_solver = Damage(E_v, v, bcs_v, E_vv)
    v_lb =  fem.Function(V_v, name="Lower bound")
    v_ub =  fem.Function(V_v, name="Upper bound")
    v_lb.x.array[:] = 0.0
    v_ub.x.array[:] = 1.0
    # damage_solver.setVariableBounds(v_lb.x.petsc_vec,v_ub.x.petsc_vec)

    if method=='Newton':
        EN=NewtonSolver(elastic_solver, damage_solver,
                        elastic_problem, damage_problem,
                        E_uv, E_vu,
                       linesearch=linesearch)
        EN.setUp(rtol=1.0e-8,max_it_SNES=100,max_it_KSP=100,ksp_restarts=100, monitor='off')
        uv = PETSc.Vec().createNest([u.x.petsc_vec,v.x.petsc_vec])
    
    # Solving the problem and visualizing
    if PlotSwitch:
        start_xvfb(wait=0.5)
    
    # load_c = 0.19 * L  # reference value for the loading (imposed displacement)
    loads = np.linspace(0, 1.5 * load_c * L / 10, 20) # (load_c/E)*L
    
    # Array to store results
    energies = np.zeros((loads.shape[0], 5))
    output=np.zeros((loads.shape[0],3))

    if WriteSwitch:
        with io.XDMFFile(dom.comm, f"output/EX_GD_{method}_{linesearch}.xdmf","w") as xdmf:
            xdmf.write_mesh(dom)

    for i_t, t in enumerate(loads):
        u_D.value = t
        energies[i_t, 0] = t
    
        # Update the lower bound to ensure irreversibility of damage field.
        v_lb.x.array[:] = v.x.array

        if rank==0:
            print(f"-- Solving for t = {t:3.2f} --")
        if method=='Newton':
            EN.solver.solve(None,uv)
            energies[i_t,4] = EN.solver.getIterationNumber()
            output[i_t,:]=[t,EN.output,EN.solver.getIterationNumber()]
        else:
            iteration = alternate_minimization(u, v, elastic_solver, damage_solver, 1e-8, 100, monitor=False)
            output[i_t,:]=[t,0,iteration]
            energies[i_t,4] = iteration

        if PlotSwitch:
            plot_damage_state(u, v, None, [800,300])
    
        # Calculate the energies
        energies[i_t, 1] = MPI.COMM_WORLD.allreduce(
            fem.assemble_scalar(fem.form(elastic_energy)),
            op=MPI.SUM,
        )
        energies[i_t, 2] = MPI.COMM_WORLD.allreduce(
            fem.assemble_scalar(fem.form(dissipated_energy)),
            op=MPI.SUM,
        )
        energies[i_t, 3] = comm.allreduce(
            fem.assemble_scalar(fem.form(total_energy)),
            op=MPI.SUM,
        )
        if WriteSwitch:
            with io.XDMFFile(dom.comm, f"output/EX_GD_{method}_{linesearch}.xdmf","a") as xdmf:
                xdmf.write_function(u, t)
                xdmf.write_function(v, t)

    return output, energies

if __name__ == "__main__":
    pyvista.OFF_SCREEN=True
    main('Newton','2step')
    sys.exit()
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io

import pyvista
from pyvista.utilities.xvfb import start_xvfb
import matplotlib.pyplot as plt
import csv
import sys

from EX_L_Domain import domain, BCs, VariationalFormulation
from Solvers import Elastic, Damage, alternate_minimization
from PLOT_DamageState import plot_damage_state
from NewtonSolver import NewtonSolver

def main(method='AltMin',linesearch='bt',PlotSwitch=False,WriteSwitch=False):
    comm=MPI.COMM_WORLD
    rank=comm.rank
    rtol=1.0e-6
    
    u, v, dom =domain()
    V_u=u.function_space
    V_v=v.function_space
    E_u, E_v, E_uu, E_vv, E_uv, E_vu, elastic_energy, dissipated_energy, p, total_energy = VariationalFormulation(u,v,dom)
    bcs_u, bcs_v, uD = BCs(u,v,dom)
    
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
        EN.setUp(rtol=rtol,max_it_SNES=1000,max_it_KSP=100,ksp_restarts=100,monitor='off')
        uv = PETSc.Vec().createNest([u.x.petsc_vec,v.x.petsc_vec])#,None,MPI.COMM_WORLD)
    
    # Solving the problem and visualizing
    if PlotSwitch:
        start_xvfb(wait=0.5)
    
    loads = np.linspace(0,0.6,81)
    
    # Array to store results
    energies = np.zeros((loads.shape[0], 5 ))
    output=np.zeros((loads.shape[0],3))

    if WriteSwitch:
        with io.XDMFFile(dom.comm, f"output/EX_L_{method}_{linesearch}.xdmf",'w') as xdmf:
            xdmf.write_mesh(dom)

    for i_t, t in enumerate(loads):
        uD.value=t
        energies[i_t, 0] = t
    
        # Update the lower bound to ensure irreversibility of damage field.
        v_lb.x.array[:] = v.x.array

        if rank==0:
            print(f"-- Solving for t = {t:3.2e} --")
        if method=='Newton':
            EN.solver.solve(None,uv)
            energies[i_t,4] = EN.solver.getIterationNumber()
            output[i_t,:] = [t,EN.output,EN.solver.getIterationNumber()]
        else:
            iteration = alternate_minimization(u, v, elastic_solver, damage_solver, rtol, 1000, monitor=True)
            energies[i_t,4] = iteration
            output[i_t,:]=[t,0,iteration]

        if PlotSwitch:
            plot_damage_state(u, v, None, [1400, 850])
        
        # Calculate the energies
        energies[i_t, 1] = comm.allreduce(
            fem.assemble_scalar(fem.form(elastic_energy)),
            op=MPI.SUM,
        )
        energies[i_t, 2] = comm.allreduce(
            fem.assemble_scalar(fem.form(dissipated_energy)),
            op=MPI.SUM,
        )
        energies[i_t, 3] = comm.allreduce(
            fem.assemble_scalar(fem.form(total_energy)),
            op=MPI.SUM,
        )
        if WriteSwitch:
            with io.XDMFFile(dom.comm, f"output/EX_L_{method}_{linesearch}.xdmf","a") as xdmf:
                xdmf.write_function(u, t)
                xdmf.write_function(v, t)
    return output, energies

if __name__ == "__main__":
    pyvista.OFF_SCREEN=True
    main('NewtonLS')
    sys.exit()
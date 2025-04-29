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

def main(method='AltMin'):
    comm=MPI.COMM_WORLD
    rank=comm.rank
    
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
    EN=NewtonSolver(elastic_solver, damage_solver,
                    elastic_problem, damage_problem,
                    E_uv, E_vu,
                   'backtrack')
    EN.setUp(rtol=1.0e-4,max_it_SNES=1000,max_it_KSP=100,ksp_restarts=100)
    uv = PETSc.Vec().createNest([u.x.petsc_vec,v.x.petsc_vec])#,None,MPI.COMM_WORLD)
    
    # Solving the problem and visualizing
    start_xvfb(wait=0.5)
    
    loads = np.linspace(0,0.6,81)
    
    # Array to store results
    energies = np.zeros((loads.shape[0], 5 ))
    iter_count=[]
    with open(f"output/TBL_L_{method}_energy.csv",'w') as csv.file:
        writer=csv.writer(csv.file,delimiter=',')
        writer.writerow(['t','Elastic energy','Dissipated energy','Total energy','Number of iterations'])
    with io.XDMFFile(dom.comm, f"output/EX_L_{method}.xdmf","w") as xdmf:
        xdmf.write_mesh(dom)

    for i_t, t in enumerate(loads):
        uD.value=t
        energies[i_t, 0] = t
    
        # Update the lower bound to ensure irreversibility of damage field.
        v_lb.x.array[:] = v.x.array

        if rank==0:
            print(f"-- Solving for t = {t:3.2e} --")
        if method=='NewtonLS':
            EN.solver.solve(None,uv)
            energies[i_t,4] = EN.solver.getIterationNumber()
        else:
            iter_count, iteration = alternate_minimization(u, v, elastic_solver, damage_solver, 1e-6, 1000, True, iter_count)
            energies[i_t,4] = iteration
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
        if rank==0:
            with open(f"output/TBL_L_{method}_energy.csv",'a') as csv.file:
                writer=csv.writer(csv.file,delimiter=',')
                writer.writerow(energies[i_t,:])
        with io.XDMFFile(dom.comm, f"output/EX_L_{method}.xdmf","a") as xdmf:
            xdmf.write_function(u, t)
            xdmf.write_function(v, t)
    with open(f"output/TBL_L_{method}_its.csv",'w') as csv.file:
        writer=csv.writer(csv.file,delimiter=',')
        if method=='NewtonLS':
            writer.writerows(EN.output) # find some way to combine these files into one
        elif method=='AltMin':
            writer.writerows(iter_count)

    fig, ax=plt.subplots()
    ax.plot(energies[:,0],energies[:,1],label='elastic energy')
    ax.plot(energies[:,0],energies[:,2],label='dissipated energy')
    ax.plot(energies[:,0],energies[:,3],label='total energy')
    ax.set_xlabel('t')
    ax.set_ylabel('Energy')
    ax.legend()
    plt.savefig(f"output/FIG_L_{method}_energy.png")
    # plt.show()

if __name__ == "__main__":
    pyvista.OFF_SCREEN=True
    main('NewtonLS')
    sys.exit()
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem

import pyvista
from pyvista.utilities.xvfb import start_xvfb
import matplotlib.pyplot as plt
import csv
import sys

from EX_CTFM_Domain import domain, BCs, VariationalFormulation
from EX_GD_Solvers import Elastic, Damage, Newton, alternate_minimization, AMEN
from PLOT_DamageState import plot_damage_state
from EX_GD_NewtonSolver import NewtonSolver

def main(method='AltMin'):
    u, v, dom, cell_tags, facet_tags=domain()
    V_u=u.function_space
    V_v=v.function_space
    bcs_u, bcs_v, t1, t2 = BCs(u,v,dom,cell_tags, facet_tags)
    E_u, E_v, E_uu, E_vv, E_uv, E_vu, elastic_energy, dissipated_energy, load_c, total_energy = VariationalFormulation(u,v,dom,cell_tags,facet_tags)
    
    # now we want to solve E_u(u,v)=0 and E_v(u,v)=0 with alternate minimization with a Newton accelerator
    # first set up solvers for the individual minimizations
    
    elastic_problem, elastic_solver = Elastic(E_u, u, bcs_u, E_uu)
    damage_problem, damage_solver = Damage(E_v, v, bcs_v, E_vv)
    v_lb =  fem.Function(V_v, name="Lower bound")
    v_ub =  fem.Function(V_v, name="Upper bound")
    v_lb.x.array[:] = 0.0
    v_ub.x.array[:] = 1.0
    # damage_solver.setVariableBounds(v_lb.x.petsc_vec,v_ub.x.petsc_vec)
    EN_solver = Newton(E_uv, E_vu, elastic_solver, damage_solver)
    EN=NewtonSolver(elastic_solver, damage_solver,
                    elastic_problem, damage_problem,
                    E_uv, E_vu)
    EN.setUp()
    uv = PETSc.Vec().createNest([u.x.petsc_vec,v.x.petsc_vec])#,None,MPI.COMM_WORLD)
    
    # Solving the problem and visualizing
    start_xvfb(wait=0.5)
    
    # load_c = 0.19 * L  # reference value for the loading (imposed displacement)
    loads = np.linspace(0, 1.5 * load_c * 12 / 10, 20) # (load_c/E)*L
    
    # Array to store results
    energies = np.zeros((loads.shape[0], 4 ))
    with open('output/TBL_CTFM_energy.csv','w') as csv.file:
        writer=csv.writer(csv.file,delimiter=',')
        writer.writerow(['t','Elastic energy','Dissipated energy','Total energy'])
    
    for i_t, t in enumerate(loads):
        t1.value = t
        t2.value =-t
        energies[i_t, 0] = t
    
        # Update the lower bound to ensure irreversibility of damage field.
        v_lb.x.array[:] = v.x.array
    
        print(f"-- Solving for t = {t:3.2f} --")
        if method=='AMEN':
            AMEN(u,v, elastic_solver, damage_solver, EN_solver)
        elif method=='NewtonLS':
            EN.solver.solve(None,uv)
        else:
            iter_count = alternate_minimization(u, v, elastic_solver, damage_solver)
        plot_damage_state(u, v, None, [1400, 850])
    
        # Calculate the energies
        energies[i_t, 1] = MPI.COMM_WORLD.allreduce(
            fem.assemble_scalar(fem.form(elastic_energy)),
            op=MPI.SUM,
        )
        energies[i_t, 2] = MPI.COMM_WORLD.allreduce(
            fem.assemble_scalar(fem.form(dissipated_energy)),
            op=MPI.SUM,
        )
        energies[i_t, 3] = MPI.COMM_WORLD.allreduce(
            fem.assemble_scalar(fem.form(total_energy)),
            op=MPI.SUM,
        )
        with open('output/TBL_CTFM_energy.csv','w') as csv.file:
            writer=csv.writer(csv.file,delimiter=',')
            writer.writerow(energies[i_t,:])
    with open(f"output/TBL_CTFM_its.csv",'w') as csv.file:
        writer=csv.writer(csv.file,delimiter=',')
        # writer.writerow(['Elastic its','Damage its','Newton inner its','FP step','Newton step'])
        writer.writerows(EN.output) # find some way to combine these files into one

    fig, ax=plt.subplots()
    ax.plot(energies[:,0],energies[:,1],label='elastic energy')
    ax.plot(energies[:,0],energies[:,2],label='dissipated energy')
    ax.plot(energies[:,0],energies[:,3],label='total energy')
    ax.set_xlabel('t')
    ax.set_ylabel('Energy')
    ax.legend()
    plt.savefig('output/FIG_CTFM_energy.png')
    # plt.show()
    # plot_damage_state(u, v, None, [1400, 850],'output/FIG_CTFM_final.png')

if __name__ == "__main__":
    pyvista.OFF_SCREEN=True
    main('NewtonLS')
    sys.exit()
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io

import pyvista
from pyvista.utilities.xvfb import start_xvfb
import matplotlib.pyplot as plt
import csv
import sys

from EX_Surf_Domain import domain, BCs, VariationalFormulation, SurfBC
from Solvers import Elastic, Damage, Newton, alternate_minimization, AMEN
from PLOT_DamageState import plot_damage_state
from NewtonSolver import NewtonSolver

def main(method='AltMin'):
    u, v, dom, cell_tags, facet_tags=domain()
    V_u=u.function_space
    V_v=v.function_space
    E_u, E_v, E_uu, E_vv, E_uv, E_vu, elastic_energy, dissipated_energy, p, total_energy = VariationalFormulation(u,v,dom,cell_tags,facet_tags)
    bcs_u, bcs_v, U, bdry_cells = BCs(u,v,dom,cell_tags, facet_tags, p)
    
    # now we want to solve E_u(u,v)=0 and E_v(u,v)=0 with alternate minimization with a Newton accelerator
    # first set up solvers for the individual minimizations
    
    elastic_problem, elastic_solver = Elastic(E_u, u, bcs_u, E_uu)
    damage_problem, damage_solver = Damage(E_v, v, bcs_v, E_vv)
    v_lb =  fem.Function(V_v, name="Lower bound")
    v_ub =  fem.Function(V_v, name="Upper bound")
    v_lb.x.array[:] = 0.0
    v_ub.x.array[:] = 1.0
    # damage_solver.setVariableBounds(v_lb.x.petsc_vec,v_ub.x.petsc_vec)
    # EN_solver = Newton(E_uv, E_vu, elastic_solver, damage_solver)
    EN=NewtonSolver(elastic_solver, damage_solver,
                    elastic_problem, damage_problem,
                    E_uv, E_vu)
    EN.setUp(rtol=1.0e-4,max_it_SNES=1000,max_it_KSP=100,ksp_restarts=100)
    uv = PETSc.Vec().createNest([u.x.petsc_vec,v.x.petsc_vec])#,None,MPI.COMM_WORLD)
    
    # Solving the problem and visualizing
    start_xvfb(wait=0.5)
    
    loads = np.linspace(0,65,14)
    # loads = np.array([25]) # 2-cycle appears when using Newton w/o line search
    
    # Array to store results
    energies = np.zeros((loads.shape[0], 4 ))
    iter_count=[]
    with open(f"output/TBL_Surf_{method}_energy.csv",'w') as csv.file:
        writer=csv.writer(csv.file,delimiter=',')
        writer.writerow(['t','Elastic energy','Dissipated energy','Total energy'])
    with io.XDMFFile(dom.comm, "output/EX_Surf.xdmf","w") as xdmf:
        xdmf.write_mesh(dom)

    for i_t, t in enumerate(loads):
        U.interpolate(lambda x: SurfBC(x,t,p),bdry_cells)
        energies[i_t, 0] = t
    
        # Update the lower bound to ensure irreversibility of damage field.
        v_lb.x.array[:] = v.x.array
    
        print(f"-- Solving for t = {t:3.2f} --")
        if method=='AMEN':
            # AMEN(u,v, elastic_solver, damage_solver, EN_solver)
            print(f"AMEN deprecated")
        elif method=='NewtonLS':
            EN.solver.solve(None,uv)
        else:
            iter_count = alternate_minimization(u, v, elastic_solver, damage_solver, 1e-4, 1000, True, iter_count)
        if i_t!=len(loads)-1:
            plot_damage_state(u, v, None, [1400, 850])
        else:
            plot_damage_state(u, v, None, [1400, 850],f"output/FIG_Surf_{method}_final.png")
    
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
        with open(f"output/TBL_Surf_{method}_energy.csv",'a') as csv.file:
            writer=csv.writer(csv.file,delimiter=',')
            writer.writerow(energies[i_t,:])
        with io.XDMFFile(dom.comm, "output/EX_Surf.xdmf","a") as xdmf:
            xdmf.write_function(u, t)
            xdmf.write_function(v, t)
            xdmf.write_function(U, t)
    with open(f"output/TBL_Surf_{method}_its.csv",'w') as csv.file:
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
    plt.savefig(f"output/FIG_Surf_{method}_energy.png")
    # plt.show()

if __name__ == "__main__":
    pyvista.OFF_SCREEN=True
    main('NewtonLS')
    sys.exit()
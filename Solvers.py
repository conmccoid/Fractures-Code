import ufl
import numpy as np
from dolfinx import fem, la
from dolfinx.fem import petsc
from SNESProblem import SNESProblem
from NewtonSolverContext import NewtonSolverContext
from petsc4py import PETSc
from mpi4py import MPI

def Elastic(E, u, bcs, J):
    V = u.function_space

    elastic_problem=SNESProblem(E, u, bcs, J)

    b_u =  la.create_petsc_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
    J_u =  petsc.create_matrix(elastic_problem.a)

    elastic_solver=PETSc.SNES().create()
    elastic_solver.setFunction(elastic_problem.Fn,b_u)
    elastic_solver.setJacobian(elastic_problem.Jn,J_u)
    elastic_solver.setType("ksponly")
    elastic_solver.setTolerances(rtol=1.0e-7, max_it=50)
    elastic_solver.getKSP().setType("preonly") # testing
    elastic_solver.getKSP().setTolerances(rtol=1.0e-9)
    elastic_solver.getKSP().getPC().setType("lu") # testing
    return elastic_problem, elastic_solver

def Damage(E, v, bcs, J):
    V = v.function_space
    damage_problem =SNESProblem(E, v, bcs, J)
 
    b_v =  la.create_petsc_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
    J_v =  petsc.create_matrix(damage_problem.a)

    damage_solver=PETSc.SNES().create()
    damage_solver.setFunction(damage_problem.Fn, b_v)
    damage_solver.setJacobian(damage_problem.Jn, J_v)
    damage_solver.setType("vinewtonrsls")
    damage_solver.setTolerances(rtol=1.0e-7, max_it=50)
    damage_solver.getKSP().setType("preonly")
    damage_solver.getKSP().setTolerances(rtol=1.0e-9)
    damage_solver.getKSP().getPC().setType("lu")

    return damage_problem, damage_solver

# AltMin definition
def alternate_minimization(u, v, elastic_solver, damage_solver, atol=1e-4, max_iterations=1000, monitor=True):
    v_old = fem.Function(v.function_space)
    v_old.x.array[:] = v.x.array
    u_old = fem.Function(u.function_space)
    u_old.x.array[:] = u.x.array

    for iteration in range(max_iterations):
        # Solve for displacement
        elastic_solver.solve(None, u.x.petsc_vec) # replace None with a rhs function
        # This forward scatter is necessary when `solver_u_snes` is of type `ksponly`.
        u.x.scatter_forward() # why isn't it necessary for v?

        # Solve for damage
        damage_solver.solve(None, v.x.petsc_vec)

        # Check error and update
        L2_error = ufl.inner(v - v_old, v - v_old) * ufl.dx + ufl.inner(u - u_old, u - u_old) * ufl.dx
        error_L2 = np.sqrt(MPI.COMM_WORLD.allreduce(fem.assemble_scalar(fem.form(L2_error)), op=MPI.SUM))

        v_old.x.array[:] = v.x.array
        u_old.x.array[:] = u.x.array

        if monitor:
            if MPI.COMM_WORLD.rank==0:
                print(f"Iteration: {iteration}, Error: {error_L2:3.4e}")

        if error_L2 <= atol:
          return iteration #(error_L2,iteration)

    raise RuntimeError(
        f"Could not converge after {max_iterations} iterations, error {error_L2:3.4e}"
    )
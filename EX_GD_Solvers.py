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
    elastic_solver.setTolerances(rtol=1.0e-9, max_it=50)
    elastic_solver.getKSP().setType("preonly")
    elastic_solver.getKSP().setTolerances(rtol=1.0e-9)
    elastic_solver.getKSP().getPC().setType("lu")
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
    damage_solver.setTolerances(rtol=1.0e-9, max_it=50)
    damage_solver.getKSP().setType("preonly")
    damage_solver.getKSP().setTolerances(rtol=1.0e-9)
    damage_solver.getKSP().getPC().setType("lu")

    return damage_problem, damage_solver

def Newton(E_uu, E_vv, E_uv, E_vu, elastic_problem, damage_problem):
    Euu = petsc.assemble_matrix(fem.form(E_uu))
    Evv = petsc.assemble_matrix(fem.form(E_vv))
    elastic_problem.Jn(None,None,Euu,None)
    damage_problem.Jn(None,None,Evv,None)
    Euv = petsc.assemble_matrix(fem.form(E_uv))
    Evu = petsc.assemble_matrix(fem.form(E_vu))
    EN=NewtonSolverContext(E_uu, E_uv, E_vu, E_vv,elastic_problem,damage_problem) # replacing Euv with E_uv also works? ditto Evu

    A_test = PETSc.Mat().createNest([[Euu,Euv],[Evu,Evv]])
    # there has to be a better way!

    A = PETSc.Mat().createPython(A_test.getSize())
    A.setPythonContext(EN)
    A.setUp()

    EN_solver = PETSc.KSP().create()
    EN_solver.setOperators(A)
    EN_solver.setType('gmres')
    EN_solver.getPC().setType('none') # there are no PCs for Python matrices (that I've found)
    return EN_solver

# AltMin definition
def alternate_minimization(u, v, elastic_solver, damage_solver, atol=1e-8, max_iterations=100, monitor=True):
    v_old = fem.Function(v.function_space)
    v_old.x.array[:] = v.x.array

    for iteration in range(max_iterations):
        # Solve for displacement
        elastic_solver.solve(None, u.vector) # replace None with a rhs function
        # This forward scatter is necessary when `solver_u_snes` is of type `ksponly`.
        u.x.scatter_forward() # why isn't it necessary for v?

        # Solve for damage
        damage_solver.solve(None, v.vector)

        # Check error and update
        L2_error = ufl.inner(v - v_old, v - v_old) * ufl.dx
        error_L2 = np.sqrt(MPI.COMM_WORLD.allreduce(fem.assemble_scalar(fem.form(L2_error)), op=MPI.SUM))
        v_old.x.array[:] = v.x.array

        if monitor:
          print(f"Iteration: {iteration}, Error: {error_L2:3.4e}")

        if error_L2 <= atol:
          return (error_L2,iteration)

    raise RuntimeError(
        f"Could not converge after {max_iterations} iterations, error {error_L2:3.4e}"
    )

# AMEN definition
def AMEN(u, v, elastic_solver, damage_solver, EN_solver, atol=1e-8, max_iterations=100, monitor=True):
    V_u = u.function_space
    V_v = v.function_space
    v_old = fem.Function(V_v)
    v_old.x.array[:] = v.x.array
    u_old = fem.Function(V_u)
    u_old.x.array[:] = u.x.array

    # initialize Newton step direction
    p_u = fem.Function(u.function_space)
    p_v = fem.Function(v.function_space)
    p = PETSc.Vec().createNest([p_u.vector,p_v.vector])

    for iteration in range(max_iterations):
        # Solve for displacement
        elastic_solver.solve(None, u.vector) # replace None with a rhs function
        # This forward scatter is necessary when `solver_u_snes` is of type `ksponly`.
        u.x.scatter_forward() # why isn't it necessary for v?

        # Solve for damage
        damage_solver.solve(None, v.vector)
        v.x.scatter_forward() # nb: either need this to update res or don't because it only is required for ksponly solvers

        # Exact Newton step
        res_u = u.vector - u_old.vector
        res_v = v.vector - v_old.vector
        res = PETSc.Vec().createNest([res_u,res_v]) # residual vector to minimize
          # nb: testing seems to indicate res needs to be redefined at each iteration; pointers are not enough
        
        EN_solver.solve(res,p) # resulting p is the Newton direction in both u and v (needs initialization)
        p_u, p_v = p.getNestSubVecs()
        u.x.array[:] = u_old.x.array + p_u # nb: should probably be some kind of line search or backtracking step
        v.x.array[:] = v_old.x.array + p_v

        # Check error and update
        L2_error = ufl.inner(v - v_old, v - v_old) * ufl.dx
        error_L2 = np.sqrt(MPI.COMM_WORLD.allreduce(fem.assemble_scalar(fem.form(L2_error)), op=MPI.SUM))
        v_old.x.array[:] = v.x.array
        u_old.x.array[:] = u.x.array

        if monitor:
          print(f"Iteration: {iteration}, Error: {error_L2:3.4e}, Max alpha: {np.max(v.x.array):3.4e}, Min alpha: {np.min(v.x.array):3.4e}")

        if error_L2 <= atol:
          return (error_L2,iteration)

    raise RuntimeError(
        f"Could not converge after {max_iterations} iterations, error {error_L2:3.4e}"
    )
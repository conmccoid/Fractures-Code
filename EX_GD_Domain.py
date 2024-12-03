import ufl
import basix
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem

def w(v): # dissipated energy function (of dmg)
  return v

def a(v, k_ell=1.e-6): # stiffness modulation (of dmg & ?)
  return (1-v)**2 + k_ell

def eps(u): # strain tensor (of displacement)
  return ufl.sym(ufl.grad(u))

def sigma_0(u, nu,E,ndim): # stress tensor of undamaged material (of disp)
  mu = E / (2.0* (1.0+nu))
  lmbda = E * nu / (1.0 - nu**2)
  return 2.0 * mu * eps(u) + lmbda * ufl.tr(eps(u)) * ufl.Identity(ndim)

def sigma(u,v, nu,E,ndim): # stress tensor of damaged material (of disp & dmg)
  return a(v) * sigma_0(u, nu, E, ndim)

def domain(L=1.,H=0.3,cell_size=0.1/6):
    # Domain set-up
    nx=int(L/cell_size)
    ny=int(H/cell_size)

    domain=mesh.create_rectangle(
        MPI.COMM_WORLD, 
        [(0.0,0.0),(L,H)],
        [nx,ny],
        cell_type=mesh.CellType.quadrilateral
        )

    # Function space and solution initialization
    element_u = basix.ufl.element("Lagrange", domain.basix_cell(), degree=1, shape=(2,)) 
    element_v = basix.ufl.element("Lagrange", domain.basix_cell(), degree=1)
    V_u = fem.functionspace(domain, element_u)
    V_v = fem.functionspace(domain, element_v)
    u = fem.Function(V_u, name="Displacement")
    v = fem.Function(V_v, name="Damage")
    return u, v, domain

def BCs(u,v,domain,L=1.,H=0.3):
    V_u=u.function_space
    V_v=v.function_space
    fdim=domain.topology.dim-1

    # Boundary conditions (using characteristic functions)
    def bottom(x):
        return np.isclose(x[1], 0.0)

    def top(x):
        return np.isclose(x[1], H)

    def right(x):
        return np.isclose(x[0], L)

    def left(x):
        return np.isclose(x[0], 0.0)

    b_facets= mesh.locate_entities_boundary(domain, fdim, bottom)
    t_facets= mesh.locate_entities_boundary(domain, fdim, top)
    r_facets= mesh.locate_entities_boundary(domain, fdim, right)
    l_facets= mesh.locate_entities_boundary(domain, fdim, left)

    b_boundary_dofs_uy= fem.locate_dofs_topological(V_u.sub(1), fdim, b_facets)
    t_boundary_dofs_uy= fem.locate_dofs_topological(V_u.sub(1), fdim, t_facets)
    r_boundary_dofs_ux= fem.locate_dofs_topological(V_u.sub(0), fdim, r_facets)
    l_boundary_dofs_ux= fem.locate_dofs_topological(V_u.sub(0), fdim, l_facets)

    u_D= fem.Constant(domain,PETSc.ScalarType(1.))
    bc_u_l= fem.dirichletbc(0.0, l_boundary_dofs_ux, V_u.sub(0))
    bc_u_r= fem.dirichletbc(u_D, r_boundary_dofs_ux, V_u.sub(0)) # nonhomog. bc
    bc_u_t= fem.dirichletbc(0.0, t_boundary_dofs_uy, V_u.sub(1))
    bc_u_b= fem.dirichletbc(0.0, b_boundary_dofs_uy, V_u.sub(1))
    bcs_u = [bc_u_l,bc_u_r]

    r_boundary_dofs_v= fem.locate_dofs_topological(V_v, fdim, r_facets)
    l_boundary_dofs_v= fem.locate_dofs_topological(V_v, fdim, l_facets)
    bc_v_l= fem.dirichletbc(0.0, l_boundary_dofs_v, V_v)
    bc_v_r= fem.dirichletbc(0.0, r_boundary_dofs_v, V_v)
    bcs_v = [bc_v_l,bc_v_r]
    return bcs_u, bcs_v, u_D

def VariationalFormulation(u,v,domain):
    V_u=u.function_space
    V_v=v.function_space
    ndim=domain.geometry.dim

    # Variational formulation
    E= fem.Constant(domain, PETSc.ScalarType(100.0))
    nu= fem.Constant(domain, PETSc.ScalarType(0.3))

    dx = ufl.Measure("dx",domain=domain)
    ds = ufl.Measure("ds",domain=domain)

    Gc=  fem.Constant(domain, PETSc.ScalarType(1.0))
    ell= fem.Constant(domain, PETSc.ScalarType(0.1))
    cw=  fem.Constant(domain, PETSc.ScalarType(8/3))
    f =  fem.Constant(domain, PETSc.ScalarType((0.,0.)))

    elastic_energy = 0.5 * ufl.inner(sigma(u,v,nu,E,ndim), eps(u)) * dx
    dissipated_energy= Gc/cw * ( w(v) / ell + ell * ufl.inner(ufl.grad(v), ufl.grad(v))) * dx
    external_work=ufl.inner(f,u)*dx
    total_energy=elastic_energy + dissipated_energy - external_work

    E_u = ufl.derivative(total_energy, u, ufl.TestFunction(V_u))
    E_uu= ufl.derivative(E_u,u,ufl.TrialFunction(V_u))
    E_v = ufl.derivative(total_energy, v, ufl.TestFunction(V_v))
    E_vv= ufl.derivative(E_v,v,ufl.TrialFunction(V_v))
    E_uv= ufl.derivative(E_u,v,ufl.TrialFunction(V_v))
    E_vu= ufl.derivative(E_v,u,ufl.TrialFunction(V_u))

    return E_u, E_v, E_uu, E_vv, E_uv, E_vu, elastic_energy, dissipated_energy
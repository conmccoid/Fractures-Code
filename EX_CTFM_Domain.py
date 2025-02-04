import ufl
import basix
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem
from dolfinx.io.gmshio import read_from_msh

def w(v): # dissipated energy function (of dmg)
  return v**2

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

def domain():
    # Domain set-up
    (domain, cell_tags, facet_tags) = read_from_msh('CTFM.msh',MPI.COMM_WORLD)

    # Function space and solution initialization
    element_u = basix.ufl.element("Lagrange", domain.basix_cell(), degree=1, shape=(3,)) 
    element_v = basix.ufl.element("Lagrange", domain.basix_cell(), degree=1)
    V_u = fem.functionspace(domain, element_u)
    V_v = fem.functionspace(domain, element_v)
    u = fem.Function(V_u, name="Displacement")
    v = fem.Function(V_v, name="Damage")
    return u, v, domain, cell_tags, facet_tags

def BCs(u,v,domain, cell_tags, facet_tags):
    V_u=u.function_space
    V_v=v.function_space
    fdim=domain.topology.dim-1

    circle1_facets=facet_tags.find(20)
    circle2_facets=facet_tags.find(30)
    crack_facets=facet_tags.find(40)

    circle1_dofsx=fem.locate_dofs_topological(V_u.sub(0), fdim, circle1_facets)
    circle2_dofsx=fem.locate_dofs_topological(V_u.sub(0), fdim, circle2_facets)
    circle1_dofsy=fem.locate_dofs_topological(V_u.sub(1), fdim, circle1_facets)
    circle2_dofsy=fem.locate_dofs_topological(V_u.sub(1), fdim, circle2_facets)
    crack_dofs=fem.locate_dofs_topological(V_v, fdim, crack_facets)

    t1=fem.Constant(domain, PETSc.ScalarType(1.0))
    t2=fem.Constant(domain, PETSc.ScalarType(-1.0))
    circle1_bcsx=fem.dirichletbc( t1, circle1_dofsx, V_u.sub(0))
    circle2_bcsx=fem.dirichletbc( t2, circle2_dofsx, V_u.sub(0))
    circle1_bcsy=fem.dirichletbc( t1, circle1_dofsy, V_u.sub(1))
    circle2_bcsy=fem.dirichletbc( t2, circle2_dofsy, V_u.sub(1))
    crack_bcs=fem.dirichletbc(fem.Constant(domain, PETSc.ScalarType(1.)), crack_dofs, V_v)

    bcs_u=[circle1_bcsx, circle2_bcsx, circle1_bcsy, circle2_bcsy]
    bcs_v=[crack_bcs]
    return bcs_u, bcs_v, t1, t2

def VariationalFormulation(u,v,domain,cell_tags,facet_tags):
    V_u=u.function_space
    V_v=v.function_space
    ndim=domain.geometry.dim

    # Variational formulation
    E= fem.Constant(domain, PETSc.ScalarType(100.0))
    nu= fem.Constant(domain, PETSc.ScalarType(0.3))

    dx = ufl.Measure("dx",domain=domain, subdomain_data=cell_tags)
    ds = ufl.Measure("ds",domain=domain, subdomain_data=facet_tags)

    Gc=  fem.Constant(domain, PETSc.ScalarType(1.0))
    ell= fem.Constant(domain, PETSc.ScalarType(0.1))
    cw=  fem.Constant(domain, PETSc.ScalarType(1/2))
    f =  fem.Constant(domain, PETSc.ScalarType((0.,0.,0.)))
    load_c = np.sqrt(27 * Gc.value * E.value / (256 * ell.value) ) # AT2

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

    return E_u, E_v, E_uu, E_vv, E_uv, E_vu, elastic_energy, dissipated_energy, load_c
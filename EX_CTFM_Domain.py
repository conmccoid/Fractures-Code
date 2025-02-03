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
    (mesh, cell_tags, facet_tags) = read_from_msh('CTFM.msh',MPI.COMM_WORLD)
    ndim = mesh.geometry.dim
    domain=mesh.ufl_domain()

    # Function space and solution initialization
    element_u = basix.ufl.element("Lagrange", domain.ufl_cell(), degree=1, shape=(2,)) 
    element_v = basix.ufl.element("Lagrange", domain.ufl_cell(), degree=1)
    V_u = fem.functionspace(domain, element_u)
    V_v = fem.functionspace(domain, element_v)
    u = fem.Function(V_u, name="Displacement")
    v = fem.Function(V_v, name="Damage")
    return u, v, domain
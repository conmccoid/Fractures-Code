import ufl
import basix
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem
from dolfinx.io.gmshio import read_from_msh

class Parameters:
    def __init__(self,domain):
        self.Gc =fem.Constant(domain, PETSc.ScalarType(1.0))
        self.ell=fem.Constant(domain, PETSc.ScalarType(0.5))
        self.cw =fem.Constant(domain, PETSc.ScalarType(1/2))
        self.E  =fem.Constant(domain, PETSc.ScalarType(1.0))
        self.nu =fem.Constant(domain, PETSc.ScalarType(0.3))
        self.load_c = np.sqrt(27 * self.Gc.value * self.E.value / (256 * self.ell.value) ) # AT2
        self.kappa = (3.0-self.nu)/(1.0+self.nu)
        self.mu = self.E / (2.0* (1.0+self.nu))
        self.lmbda = self.E * self.nu / (1.0 - self.nu**2)
        self.K = fem.Constant(domain, PETSc.ScalarType(1.0))

def w(v): # dissipated energy function (of dmg)
  return v**2

def a(v, k_ell=1.e-6): # stiffness modulation (of dmg & ?)
  return (1-v)**2 + k_ell

def eps(u): # strain tensor (of displacement)
  return ufl.sym(ufl.grad(u))

def sigma_0(u, p,ndim): # stress tensor of undamaged material (of disp)
  return 2.0 * p.mu * eps(u) + p.lmbda * ufl.tr(eps(u)) * ufl.Identity(ndim)

def sigma(u,v, p,ndim): # stress tensor of damaged material (of disp & dmg)
  return a(v) * sigma_0(u, p, ndim)

def domain():
    # Domain set-up
    (domain, cell_tags, facet_tags) = read_from_msh('Surf.msh',MPI.COMM_WORLD,gdim=2)

    # Function space and solution initialization
    element_u = basix.ufl.element("Lagrange", domain.basix_cell(), degree=1, shape=(2,)) 
    element_v = basix.ufl.element("Lagrange", domain.basix_cell(), degree=1)
    V_u = fem.functionspace(domain, element_u)
    V_v = fem.functionspace(domain, element_v)
    u = fem.Function(V_u, name="Displacement")
    v = fem.Function(V_v, name="Damage")
    return u, v, domain, cell_tags, facet_tags

def SurfBC(x,t,p):
    r=np.sqrt((x[0]-t)**2 + x[1]**2)
    theta=np.arctan2(x[1],x[0]-t)
    Ux= (p.K/(2*p.mu)) * np.sqrt(r/(2*np.pi)) * (p.kappa - np.cos(theta)) * np.cos(theta/2)
    Uy= (p.K/(2*p.mu)) * np.sqrt(r/(2*np.pi)) * (p.kappa - np.cos(theta)) * np.sin(theta/2)
    return np.vstack((Ux,Uy))

def BCs(u,v,domain, cell_tags, facet_tags, p):
    V_u=u.function_space
    V_v=v.function_space
    fdim=domain.topology.dim-1

    crack_facets=facet_tags.find(40)
    bdry_facets=facet_tags.find(20)

    crack_dofs_v=fem.locate_dofs_topological(V_v, fdim, crack_facets)
    # bdry_dofs_u =fem.locate_dofs_topological(V_u, fdim, bdry_facets)
    bdry_dofs_ux=fem.locate_dofs_topological(V_u.sub(0), fdim, bdry_facets)
    bdry_dofs_uy=fem.locate_dofs_topological(V_u.sub(1), fdim, bdry_facets)

    U=fem.Function(V_u)
    # U.interpolate(lambda x: SurfBC(x,0.0,p), bdry_facets)
    U.interpolate(lambda x: np.vstack((10,0)), bdry_facets)
    bc_ux = fem.dirichletbc(U.sub(0), bdry_dofs_ux)
    bc_uy = fem.dirichletbc(U.sub(1), bdry_dofs_uy)

    crack_bcs=fem.dirichletbc(fem.Constant(domain, PETSc.ScalarType(1.)), crack_dofs_v, V_v)
    bcs_u=[bc_ux, bc_uy]
    bcs_v=[crack_bcs]
    return bcs_u, bcs_v, U, bdry_facets

def VariationalFormulation(u,v,domain,cell_tags,facet_tags):
    V_u=u.function_space
    V_v=v.function_space
    ndim=domain.geometry.dim

    # Variational formulation
    p=Parameters(domain)
    f =  fem.Constant(domain, PETSc.ScalarType((0.,0.)))
    dx = ufl.Measure("dx",domain=domain, subdomain_data=cell_tags)
    ds = ufl.Measure("ds",domain=domain, subdomain_data=facet_tags)

    elastic_energy = 0.5 * ufl.inner(sigma(u,v,p,ndim), eps(u)) * dx
    dissipated_energy= p.Gc/p.cw * ( w(v) / p.ell + p.ell * ufl.inner(ufl.grad(v), ufl.grad(v))) * dx
    external_work=ufl.inner(f,u)*dx
    total_energy=elastic_energy + dissipated_energy - external_work

    E_u = ufl.derivative(total_energy, u, ufl.TestFunction(V_u))
    E_uu= ufl.derivative(E_u,u,ufl.TrialFunction(V_u))
    E_v = ufl.derivative(total_energy, v, ufl.TestFunction(V_v))
    E_vv= ufl.derivative(E_v,v,ufl.TrialFunction(V_v))
    E_uv= ufl.derivative(E_u,v,ufl.TrialFunction(V_v))
    E_vu= ufl.derivative(E_v,u,ufl.TrialFunction(V_u))

    return E_u, E_v, E_uu, E_vv, E_uv, E_vu, elastic_energy, dissipated_energy, p, total_energy
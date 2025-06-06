import ufl
import basix
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io
import meshio
from dolfinx.io.gmshio import read_from_msh

class Parameters:
    def __init__(self,domain):
        self.Gc =fem.Constant(domain, PETSc.ScalarType(8.9e-5)) # in kN/mm
        self.ell=fem.Constant(domain, PETSc.ScalarType(5.0)) #2x mesh parameter
        self.cw =fem.Constant(domain, PETSc.ScalarType(2.0)) #?
        self.E  =fem.Constant(domain, PETSc.ScalarType(1.0)) #?
        # self.nu =fem.Constant(domain, PETSc.ScalarType(0.2)) #?
        # self.load_c = np.sqrt(27 * self.Gc.value * self.E.value / (256 * self.ell.value) ) # AT2
        # self.k = (3.0-self.nu)/(1.0+self.nu) #? 1e-9?
        self.mu = 10.95 #in kN/mm^2
        self.lmbda = 6.16 #in kN/mm^2
        # self.K = fem.Constant(domain, PETSc.ScalarType(1.5))

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
    # cell='quadrilateral'
    # co_el=basix.ufl.element("Lagrange",cell,1,shape=(2,))
    # el=basix.ufl.element("Lagrange",cell,1)
    # msh = meshio.read('3K replication/L_shape.e')
    # points_2d = msh.points[:,:2]
    # cells_2d = msh.cells[0].data
    # domain = mesh.create_mesh(
    #     comm=MPI.COMM_WORLD,
    #     cells=cells_2d,
    #     x=points_2d,
    #     e=ufl.Mesh(co_el)
    # )
    (domain, _, _) = read_from_msh('L_shape.msh',MPI.COMM_WORLD,gdim=2)
    
    # Function space and solution initialization
    element_u = basix.ufl.element("Lagrange", domain.basix_cell(), degree=1, shape=(2,)) 
    element_v = basix.ufl.element("Lagrange", domain.basix_cell(), degree=1)
    V_u = fem.functionspace(domain, element_u)
    V_v = fem.functionspace(domain, element_v)
    u = fem.Function(V_u, name="Displacement")
    v = fem.Function(V_v, name="Damage")
    return u, v, domain

def BCs(u,v,domain):
    V_u=u.function_space
    V_v=v.function_space
    fdim=domain.topology.dim-1

    def forcepoint(x): # point at which force is applied
        return np.isclose(x[1],250,atol=2.5) & np.isclose(x[0],470,atol=20)
    # Boundaries: start at bottom and proceed clockwise
    def bdry1(x):
        return np.isclose(x[1],0)
    def bdry2(x):
        return np.isclose(x[0],0)
    def bdry3(x):
        return np.isclose(x[1],500)
    def bdry4(x):
        return np.isclose(x[0],500)
    def bdry5(x):
        return np.isclose(x[1],250) & np.greater(x[0],250)
    def bdry6(x):
        return np.isclose(x[0],250) & np.less(x[1],250)
    def bdry_u(x):
        return bdry1(x) | bdry2(x) | bdry3(x) | bdry4(x) | bdry6(x)

    fp_facets = mesh.locate_entities_boundary(domain, fdim, forcepoint)
    bdry_u_facets=mesh.locate_entities_boundary(domain, fdim, bdry1)
    fp_dofs_ux = fem.locate_dofs_topological(V_u.sub(0), fdim, fp_facets)
    fp_dofs_uy = fem.locate_dofs_topological(V_u.sub(1), fdim, fp_facets)
    bdry_dofs_ux=fem.locate_dofs_topological(V_u.sub(0),fdim, bdry_u_facets)
    bdry_dofs_uy=fem.locate_dofs_topological(V_u.sub(1),fdim, bdry_u_facets)
    uD = fem.Constant(domain,PETSc.ScalarType(1.))
    bc_ux_fp = fem.dirichletbc(0.0,fp_dofs_ux, V_u.sub(0))
    bc_uy_fp = fem.dirichletbc(uD, fp_dofs_uy, V_u.sub(1))
    bc_ux_bdry = fem.dirichletbc(0.0, bdry_dofs_ux, V_u.sub(0))
    bc_uy_bdry = fem.dirichletbc(0.0, bdry_dofs_uy, V_u.sub(1))
    
    bcs_u= [bc_ux_fp, bc_uy_fp, bc_ux_bdry, bc_uy_bdry]
    
    def outer_bdry(x):
        return np.isclose(x[0],0) | np.isclose(x[1],0) | np.isclose(x[0],500) | np.isclose(x[1],500)
    def corner(x):
        return (np.isclose(x[0],250) & np.less(x[1],200)) | (np.isclose(x[1],250) & np.greater(x[0],200))

    def bdry(x):
        return bdry1(x) | bdry2(x) | bdry3(x) | bdry4(x) | bdry5(x) | bdry6(x)
    def interior(x):
        return np.less(x[1],200) | np.greater(x[0],300)

    bdry_facets=mesh.locate_entities_boundary(domain, fdim, bdry)
    int_facets =mesh.locate_entities(domain, fdim, interior)
    # corner_facets=mesh.locate_entities_boundary(domain, fdim, corner)
    bdry_dofs=fem.locate_dofs_topological(V_v, fdim, bdry_facets)
    int_dofs =fem.locate_dofs_topological(V_v, fdim, int_facets)
    # corner_dofs=fem.locate_dofs_topological(V_v, fdim, corner_facets)
    bc_v_bdry = fem.dirichletbc(0.0, bdry_dofs, V_v)
    bc_v_int  = fem.dirichletbc(0.0, int_dofs, V_v)
    # bc_v_corner = fem.dirichletbc(0.0, corner_dofs, V_v)
    bcs_v = []#[bc_v_bdry, bc_v_int]
    return bcs_u, bcs_v, uD

def VariationalFormulation(u,v,domain):
    V_u=u.function_space
    V_v=v.function_space
    ndim=domain.topology.dim

    # Variational formulation
    p=Parameters(domain)
    f=fem.Constant(domain, PETSc.ScalarType((0.,0.)))

    elastic_energy = 0.5 * ufl.inner(sigma(u,v,p,ndim), eps(u)) * ufl.dx
    dissipated_energy= p.Gc/p.cw * ( w(v) / p.ell + p.ell * ufl.inner(ufl.grad(v), ufl.grad(v))) * ufl.dx
    external_work=ufl.inner(f,u)*ufl.dx
    total_energy=elastic_energy + dissipated_energy - external_work

    E_u = ufl.derivative(total_energy, u, ufl.TestFunction(V_u))
    E_uu= ufl.derivative(E_u,u,ufl.TrialFunction(V_u))
    E_v = ufl.derivative(total_energy, v, ufl.TestFunction(V_v))
    E_vv= ufl.derivative(E_v,v,ufl.TrialFunction(V_v))
    E_uv= ufl.derivative(E_u,v,ufl.TrialFunction(V_v))
    E_vu= ufl.derivative(E_v,u,ufl.TrialFunction(V_u))

    return E_u, E_v, E_uu, E_vv, E_uv, E_vu, elastic_energy, dissipated_energy, p, total_energy
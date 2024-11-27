import ufl
import basix
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, la
from dolfinx.fem import petsc

# custom classes
class GlobalProblem:
    def __init__(self, Fu, u, bcs_u, Fv, v, bcs_v, Ju=None, Jv=None):
      Vu = u.function_space
      Vv = v.function_space
      du = ufl.TrialFunction(Vu)
      dv = ufl.TrialFunction(Vv)
      self.Lu = fem.form(Fu)
      self.Lv = fem.form(Fv)
      if Ju is None:
         self.au = fem.form(ufl.derivative(Fu,u,du))
      else:
         self.au = fem.form(Ju)
      if Jv is None:
         self.av = fem.form(ufl.derivative(Fv,v,dv))
      else:
         self.av = fem.form(Jv)
      self.bcs_u = bcs_u
      self.bcs_v = bcs_v
      self._F, self._J = None, None
      self.u = u
      self.v = v

    def Fn(self, snes, x, F):
      """Assemble nested residual vector."""
      xu, xv = x.getNestSubVecs()
      xu.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
      xu.copy(self.u.vector)
      self.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
      xv.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
      xv.copy(self.v.vector)
      self.v.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

      with F.localForm() as f_local:
         f_local.set(0.0)
      Fu, Fv = F.getNestSubVecs()
      petsc.assemble_vector(Fu,self.Lu)
      petsc.assemble_vector(Fv,self.Lv)
      petsc.apply_lifting(Fu, [self.au], bcs=[self.bcs_u], x0=[xu], scale=-1.0)
      petsc.apply_lifting(Fv, [self.av], bcs=[self.bcs_v], x0=[xv], scale=-1.0)
      Fu.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
      Fv.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
      petsc.set_bc(Fu,self.bcs_u, xu, -1.0)
      petsc.set_bc(Fv,self.bcs_v, xv, -1.0)

    def Jn(self, snes, x, J, P):
       """Assemble nested Jacobian matrix."""
       Ju = J.getNestSubMatrix(0,0)
       Jv = J.getNestSubMatrix(1,1)
       Ju.zeroEntries()
       Jv.zeroEntries()
       petsc.assemble_matrix(Ju, self.au, bcs=self.bcs_u)
       petsc.assemble_matrix(Jv, self.av, bcs=self.bcs_v)
       Ju.assemble()
       Jv.assemble()
      
class SNESProblem:
    def __init__(self, F, u, bcs, J=None):
        V = u.function_space
        du = ufl.TrialFunction(V)
        self.L = fem.form(F)
        if J is None:
            self.a = fem.form(ufl.derivative(F, u, du))
        else:
            self.a = fem.form(J)
        self.bcs = bcs
        self._F, self._J = None, None
        self.u = u

    def Fn(self, snes, x, F):
        """Assemble residual vector."""
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        with F.localForm() as f_local:
            f_local.set(0.0)
        petsc.assemble_vector(F, self.L)
        petsc.apply_lifting(F, [self.a], bcs=[self.bcs], x0=[x], scale=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(F, self.bcs, x, -1.0)

    def Jn(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        J.zeroEntries()
        petsc.assemble_matrix(J, self.a, bcs=self.bcs)
        J.assemble()

class NewtonSolverContext:
    # nb: needs to take as input two Jacobian matrices, probably in the form of PETSc solvers
    # nb: needs also one Jacobian matrix that can multiply a vector
    def __init__(self,Euu,Euv,Evu,Evv):
        # get sizes of submatrices, Nu, Nv
        self.Euu=Euu
        
    def mult(self, mat, X, Y):
        x1, x2 = X.getNestSubVecs()
        w1, v2 = Y.getNestSubVecs()
        Euu = mat.getNestSubMatrix(0,0) # nb: if using this set-up, need to update to not use self.Euu, etc.
        Euv = mat.getNestSubMatrix(0,1)
        Evu = mat.getNestSubMatrix(1,0)
        Evv = mat.getNestSubMatrix(1,1)
        # self.Euu.assemble()
        # self.Euv.assemble()
        # self.Evu.assemble()
        # self.Evv.assemble()
        y1=self.Euu.createVectorRight()
        y2=self.Evv.createVectorRight()
        z1=self.Euv.createVectorRight()
        z2=self.Evv.createVectorRight()
        self.Euu.mult(x1,y1)
        self.Evv.mult(x2,y2)
        self.Euv.multAdd(x2,y1,z1)
        self.Evu.multAdd(x1,y2,z2)
        # w1=self.Euu.createVectorRight()
        self.Euu.matSolve(z1,w1) # nb: likely will need to change this to a proper KSP solve
        w2=self.Evv.createVectorRight()
        self.Evu.multAdd(-w1,z2,w2)
        self.Evv.matSolve(w2,v2)
        # then w1 and v2 get put into Y

# Domain set-up
L=1.; H=0.3
cell_size=0.1/6
nx=int(L/cell_size)
ny=int(H/cell_size)

domain=mesh.create_rectangle(
  MPI.COMM_WORLD, 
  [(0.0,0.0),(L,H)],
  [nx,ny],
  cell_type=mesh.CellType.quadrilateral
)
ndim=domain.geometry.dim

# Function space and solution initialization
element_u = basix.ufl.element("Lagrange", domain.basix_cell(), degree=1, shape=(2,)) 
element_v = basix.ufl.element("Lagrange", domain.basix_cell(), degree=1)
V_u = fem.functionspace(domain, element_u)
V_v = fem.functionspace(domain, element_v)
u = fem.Function(V_u, name="Displacement")
v = fem.Function(V_v, name="Damage")

# Boundary conditions (using characteristic functions)
def bottom(x):
  return np.isclose(x[1], 0.0)

def top(x):
  return np.isclose(x[1], H)

def right(x):
  return np.isclose(x[0], L)

def left(x):
  return np.isclose(x[0], 0.0)

fdim=domain.topology.dim-1

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

# Variational formulation
E= fem.Constant(domain, PETSc.ScalarType(100.0))
nu= fem.Constant(domain, PETSc.ScalarType(0.3))

def w(v): # dissipated energy function (of dmg)
  return v

def a(v, k_ell=1.e-6): # stiffness modulation (of dmg & ?)
  return (1-v)**2 + k_ell

def eps(u): # strain tensor (of displacement)
  return ufl.sym(ufl.grad(u))

def sigma_0(u): # stress tensor of undamaged material (of disp)
  mu = E / (2.0* (1.0+nu))
  lmbda = E * nu / (1.0 - nu**2)
  return 2.0 * mu * eps(u) + lmbda * ufl.tr(eps(u)) * ufl.Identity(ndim)

def sigma(u,v): # stress tensor of damaged material (of disp & dmg)
  return a(v) * sigma_0(u)

dx = ufl.Measure("dx",domain=domain)
ds = ufl.Measure("ds",domain=domain)

Gc=  fem.Constant(domain, PETSc.ScalarType(1.0))
ell= fem.Constant(domain, PETSc.ScalarType(0.1))
cw=  fem.Constant(domain, PETSc.ScalarType(8/3))
f =  fem.Constant(domain, PETSc.ScalarType((0.,0.)))

elastic_energy = 0.5 * ufl.inner(sigma(u,v), eps(u)) * dx
dissipated_energy= Gc/cw * ( w(v) / ell + ell * ufl.inner(ufl.grad(v), ufl.grad(v))) * dx
external_work=ufl.inner(f,u)*dx
total_energy=elastic_energy + dissipated_energy - external_work

E_u = ufl.derivative(total_energy, u, ufl.TestFunction(V_u))
E_uu= ufl.derivative(E_u,u,ufl.TrialFunction(V_u))
E_v = ufl.derivative(total_energy, v, ufl.TestFunction(V_v))
E_vv= ufl.derivative(E_v,v,ufl.TrialFunction(V_v))
E_uv= ufl.derivative(E_u,v,ufl.TrialFunction(V_v))
E_vu= ufl.derivative(E_v,u,ufl.TrialFunction(V_u))

# now we want to solve E_u(u,v)=0 and E_v(u,v)=0 with alternate minimization with a Newton accelerator
# first set up solvers for the individual minimizations

# Nest the two problems into one: switching to Jacobi preconditioner

# Custom class for the problems to be solved by SNES
elastic_problem=SNESProblem(E_u, u, bcs_u, E_uu)
damage_problem =SNESProblem(E_v, v, bcs_v, E_vv)

# Initializations of storage objects
b_u =  la.create_petsc_vector(V_u.dofmap.index_map, V_u.dofmap.index_map_bs)
b_v =  la.create_petsc_vector(V_v.dofmap.index_map, V_v.dofmap.index_map_bs)
J_u =  petsc.create_matrix(elastic_problem.a)
J_v =  petsc.create_matrix(damage_problem.a)

# Solvers and their options
elastic_solver=PETSc.SNES().create()
damage_solver=PETSc.SNES().create()

elastic_solver.setFunction(elastic_problem.Fn,b_u)
damage_solver.setFunction(damage_problem.Fn, b_v)
elastic_solver.setJacobian(elastic_problem.Jn,J_u)
damage_solver.setJacobian(damage_problem.Jn, J_v)

elastic_solver.setType("ksponly")
damage_solver.setType("vinewtonrsls")
elastic_solver.setTolerances(rtol=1.0e-9, max_it=50)
damage_solver.setTolerances(rtol=1.0e-9, max_it=50)
elastic_solver.getKSP().setType("preonly")
damage_solver.getKSP().setType("preonly")
elastic_solver.getKSP().setTolerances(rtol=1.0e-9)
damage_solver.getKSP().setTolerances(rtol=1.0e-9)
elastic_solver.getKSP().getPC().setType("lu")
damage_solver.getKSP().getPC().setType("lu")

v_lb =  fem.Function(V_v, name="Lower bound")
v_ub =  fem.Function(V_v, name="Upper bound")
v_lb.x.array[:] = 0.0
v_ub.x.array[:] = 1.0
damage_solver.setVariableBounds(v_lb.vector,v_ub.vector)

# A global solver that doesn't work
global_solver=PETSc.SNES().create()
global_problem=GlobalProblem(E_u, u, bcs_u, E_v, v, bcs_v)
b=PETSc.Vec().createNest([b_u,b_v])
J=PETSc.Mat().createNest([[J_u,None],[None,J_v]])
global_solver.setFunction(global_problem.Fn,b)
global_solver.setJacobian(global_problem.Jn,J)

global_solver.setType('ksponly')
global_solver.setTolerances(rtol=1.0e-9, max_it=50)
global_solver.getKSP().setType('preonly')
global_solver.getKSP().setTolerances(rtol=1.0e-9)
global_solver.getKSP().getPC().setType('lu')

# AltMin definition
def alternate_minimization(u, v, atol=1e-8, max_iterations=100, monitor=True):
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
        L2_error = ufl.inner(v - v_old, v - v_old) * dx
        error_L2 = np.sqrt(MPI.COMM_WORLD.allreduce(fem.assemble_scalar(fem.form(L2_error)), op=MPI.SUM))
        v_old.x.array[:] = v.x.array

        if monitor:
          print(f"Iteration: {iteration}, Error: {error_L2:3.4e}")

        if error_L2 <= atol:
          return (error_L2,iteration)

    raise RuntimeError(
        f"Could not converge after {max_iterations} iterations, error {error_L2:3.4e}"
    )

# visualization function
def plot_damage_state(u, alpha, load=None):
    """
    Plot the displacement and damage field with pyvista
    """
    assert u.function_space.mesh == alpha.function_space.mesh
    mesh = u.function_space.mesh

    plotter = pyvista.Plotter(
        title="damage, warped by displacement", window_size=[800, 300], shape=(1, 2)
    )

    topology, cell_types, geometry = plot.vtk_mesh(mesh)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    plotter.subplot(0, 0)
    if load is not None:
        plotter.add_text(f"displacement - load {load:3.3f}", font_size=11)
    else:
        plotter.add_text("displacement", font_size=11)

    values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
    values[:, : len(u)] = u.x.array.real.reshape((geometry.shape[0], len(u)))
    grid["u"] = values
    warped = grid.warp_by_vector("u", factor=0.1)
    _ = plotter.add_mesh(warped, show_edges=False)
    plotter.view_xy()

    plotter.subplot(0, 1)
    if load is not None:
        plotter.add_text(f"damage - load {load:3.3f}", font_size=11)
    else:
        plotter.add_text("damage", font_size=11)
    grid["alpha"] = alpha.x.array.real
    grid.set_active_scalars("alpha")
    plotter.add_mesh(grid, show_edges=False, show_scalar_bar=True, clim=[0, 1])
    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()

# Solving the problem and visualizing
from dolfinx import plot
import pyvista
from pyvista.utilities.xvfb import start_xvfb
start_xvfb(wait=0.5)

load_c = 0.19 * L  # reference value for the loading (imposed displacement)
loads = np.linspace(0, 1.5 * load_c, 20)

# Array to store results
energies = np.zeros((loads.shape[0], 3))

for i_t, t in enumerate(loads):
    u_D.value = t
    energies[i_t, 0] = t

    # Update the lower bound to ensure irreversibility of damage field.
    v_lb.x.array[:] = v.x.array

    print(f"-- Solving for t = {t:3.2f} --")
    alternate_minimization(u, v)
    plot_damage_state(u, v)

    # Calculate the energies
    energies[i_t, 1] = MPI.COMM_WORLD.allreduce(
        fem.assemble_scalar(fem.form(elastic_energy)),
        op=MPI.SUM,
    )
    energies[i_t, 2] = MPI.COMM_WORLD.allreduce(
        fem.assemble_scalar(fem.form(dissipated_energy)),
        op=MPI.SUM,
    )
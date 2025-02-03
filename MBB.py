#%%
import click
from mpi4py import MPI
import numpy as np
import ufl

from dolfinx.fem.petsc import LinearProblem
# import utils
import pyvista
import os
import basix

import dolfinx
from dolfinx.io import XDMFFile
from dolfinx.io.gmshio import read_from_msh
import dolfinx.fem as fem
from dolfinx import default_scalar_type



pyvista.OFF_SCREEN = True
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numProc = comm.Get_size()

#%% Double well potential
def W(x):
    return x**2 * (1.0-x)**2
def Wprime(x):
    return 2.0*x * (1.0-x)**2 - 2.0 * x**2 * (1.0 - x) 
def a(x):
    return x**2

@click.command()

@click.option('--output', '-o', type=click.STRING, default='rho.xdmf', help = 'result file')
@click.option('--input', '-i', type=click.STRING, default='MBB.msh', help = 'mesh')
@click.option('--step', '-s', type=click.FLOAT, help="step size ",default=1.e-3)
@click.option('--maxiter', '-m', type=click.INT, help="maximum number of iterations",default=10000)
@click.option('--tol', type=click.FLOAT, help="tolerance",default=1.e-4)
@click.option('--saveinterval', type=click.INT, help="save interval",default=100)


def main(**options):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    (msh, cell_tags, facet_tags) = read_from_msh(options['input'],comm)
    ndim = msh.geometry.dim

    myCS = set(cell_tags.values)
    allCS = tuple(msh.comm.allreduce(myCS,MPI.BOR))
    numCS = len(allCS)

    myFS = set(facet_tags.values)
    allFS = tuple(msh.comm.allreduce(myFS,MPI.BOR))
    numFS = len(allFS)

    # Domain measure.
    dx = ufl.Measure("dx", domain=msh, subdomain_data = cell_tags)
    ds = ufl.Measure("ds", domain=msh, subdomain_data = facet_tags)


    element_u = basix.ufl.element("Lagrange", msh.basix_cell(), degree=1)
    V = fem.functionspace(msh, element_u)

    # Trial and test functions
    u = ufl.TrialFunction(V)
    phi_u = ufl.TestFunction(V)

    rho = ufl.TrialFunction(V)
    phi_rho = ufl.TestFunction(V)

    bc_u = fem.Function(V, name="boundary displacement")
    bc_u.interpolate(lambda x: 0*x[1])
    bcs_u = []
    for fs in (30,):
        bc_facets = facet_tags.find(fs)
        bc_dofs   = fem.locate_dofs_topological(V, msh.topology.dim - 1, bc_facets)
        bcs_u.append(fem.dirichletbc(bc_u, bc_dofs))

    t = 1.
    f = fem.Constant(msh, dolfinx.default_scalar_type(t))
    rho_n = fem.Function(V) 
    # rho_n.interpolate(lambda x: (1+np.sin(2*np.pi * x[0]) * np.sin(3 * x[1] * np.pi))/2.)
    # rho_n.interpolate(lambda x: 1.*x[1])
    A_state = a(rho_n) * ufl.inner(ufl.grad(u), ufl.grad(phi_u)) * dx
    L_state = f * phi_u * dx
    stateEquation = LinearProblem(A_state(u,phi_u), L_state(phi_u), bcs=bcs_u)

    for t in np.linspace(0., 1., 11):
        # f.value = t
        rho_n.interpolate(lambda x: 0.*x[1]+t+1)
        u = stateEquation.solve()
        u_min  = comm.allreduce(np.min(u.x.array),op=MPI.MIN)
        u_max  = comm.allreduce(np.max(u.x.array),op=MPI.MAX)
        if rank == 0:
            print(f"t {t:.4e}: u min: {u_min:5.02e}, max: {u_max:5.02e}")
                  
if __name__ == '__main__':
    main()

# %%

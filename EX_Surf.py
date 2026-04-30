import numpy as np
from DOM_Surf import domain, BCs, VariationalFormulation, SurfBC
from FPAltMin import FPAltMin
from OuterSolver import OuterSolver

class FP(FPAltMin):
    def __init__(self, ell=0.5):
        '''Initialize the problem by defining the domain, variational formulation, and boundary conditions.
        - p: problem parameters
        '''
        self.u, self.v, self.dom, cell_tags, facet_tags=domain()
        E_u, E_v, E_uu, E_vv, E_uv, E_vu, self.elastic_energy, self.dissipated_energy, self.p, self.total_energy = VariationalFormulation(self.u, self.v, self.dom, cell_tags, facet_tags, ell)
        bcs_u, bcs_v, self.U, self.bdry_cells = BCs(self.u, self.v, self.dom, cell_tags, facet_tags, self.p)
        self.setUp(E_u, E_v, E_uu, E_vv, E_uv, E_vu, bcs_u, bcs_v)

    def updateBCs(self, t):
        self.U.interpolate(lambda x: SurfBC(x,t,self.p),self.bdry_cells)

def main(method='AltMin', maxit=1000, tol=1e-4, WriteSwitch=False, PlotSwitch=False, ell=0.5):
    fp = FP(ell=ell)
    example = 'Surf'
    loads = np.linspace(0.5, 6, 18)  # Load values
    os = OuterSolver(fp, example, method, loads)
    os.solve(WriteSwitch=WriteSwitch, PlotSwitch=PlotSwitch, maxit=maxit, tol=tol)
    energies = os.energies.copy()
    identifier = os.identifier
    os.destroy()
    fp.destroy()
    return energies, identifier

import argparse
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the FP problem with specified parameters.')
    parser.add_argument('--method', type=str, default='AltMin', help='Optimization method to use (default: AltMin)')
    parser.add_argument('--maxit', type=int, default=1000, help='Maximum number of iterations (default: 100)')
    parser.add_argument('--tol', type=float, default=1e-4, help='Tolerance for convergence (default: 1e-4)')
    parser.add_argument('--write', action='store_true', default=False, help='Write results to file')
    parser.add_argument('--plot', action='store_true', default=False, help='Plot results')
    args = parser.parse_args()
    energies, identifier = main(method=args.method, maxit=args.maxit, tol=args.tol, WriteSwitch=args.write, PlotSwitch=args.plot)
    sys.exit()
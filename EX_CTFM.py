import numpy as np
from OuterSolver import OuterSolver
from DOM_CTFM import domain, BCs, VariationalFormulation
from FPAltMin import FPAltMin

class FP(FPAltMin):
    def __init__(self):
        self.u, self.v, self.dom, cell_tags, facet_tags=domain()
        bcs_u, bcs_v, self.t1, self.t2 = BCs(self.u,self.v,self.dom,cell_tags, facet_tags)
        E_u, E_v, E_uu, E_vv, E_uv, E_vu, self.elastic_energy, self.dissipated_energy, self.load_c, self.total_energy = VariationalFormulation(self.u,self.v,self.dom,cell_tags, facet_tags)
        self.setUp(E_u, E_v, E_uu, E_vv, E_uv, E_vu, bcs_u, bcs_v)

    def updateBCs(self, t):
        self.t1.value = t
        self.t2.value =-t

def main(method='AltMin', WriteSwitch=False, PlotSwitch=False):
    fp = FP()
    example='CTFM'
    loads = np.linspace(0, 1.5 * fp.load_c * 12 / 10, 20) # (load_c/E)*L
    # first critical load is between 0.87 and 1.31 (but sometimes up to 1.7?)
    # second critical load between 4.79 and 5.22
    os = OuterSolver(fp, example, method, loads)
    os.solve(WriteSwitch=WriteSwitch, PlotSwitch=PlotSwitch, maxit=1000, tol=1e-4)
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
    parser.add_argument('--maxit', type=int, default=100, help='Maximum number of iterations (default: 100)')
    parser.add_argument('--tol', type=float, default=1e-4, help='Tolerance for convergence (default: 1e-4)')
    parser.add_argument('--write', action='store_true', default=False, help='Write results to file')
    parser.add_argument('--plot', action='store_true', default=False, help='Plot results')
    args = parser.parse_args()
    energies, identifier = main(method=args.method, maxit=args.maxit, tol=args.tol, WriteSwitch=args.write, PlotSwitch=args.plot)
    sys.exit() # is this necessary?
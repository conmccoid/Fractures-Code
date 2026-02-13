from OuterSolver import OuterSolver
from FPAltMin import FPAltMin
import numpy as np
from EX_GD_Domain import domain, BCs, VariationalFormulation

class FP(FPAltMin):
    def __init__(self):
        L=1.
        H=0.3
        cell_size=0.1/6
        self.u, self.v, self.dom=domain(L,H,cell_size)
        bcs_u, bcs_v, self.u_D = BCs(self.u,self.v,self.dom,L,H)
        E_u, E_v, E_uu, E_vv, E_uv, E_vu, self.elastic_energy, self.dissipated_energy, load_C, E, self.total_energy = VariationalFormulation(self.u,self.v,self.dom)
        self.setUp(E_u, E_v, E_uu, E_vv, E_uv, E_vu, bcs_u, bcs_v)

    def updateBCs(self, t):
        self.u_D.value = t

def main(method='AltMin', maxit=100, tol=1e-4, WriteSwitch=False, PlotSwitch=False):

    fp=FP()
    example='GD'
    loads = np.linspace(0, 1, 11)  # Load values

    os=OuterSolver(fp, example, method, loads)
    os.solve(WriteSwitch=WriteSwitch, PlotSwitch=PlotSwitch, maxit=maxit, tol=tol)
    return os.energies, os.identifier

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the FP problem with specified parameters.')
    parser.add_argument('--method', type=str, default='AltMin', help='Optimization method to use (default: AltMin)')
    parser.add_argument('--maxit', type=int, default=100, help='Maximum number of iterations (default: 100)')
    parser.add_argument('--tol', type=float, default=1e-4, help='Tolerance for convergence (default: 1e-4)')
    parser.add_argument('--write', action='store_true', default=False, help='Write results to file')
    parser.add_argument('--plot', action='store_true', default=False, help='Plot results')
    args = parser.parse_args()
    energies, identifier = main(method=args.method, maxit=args.maxit, tol=args.tol, WriteSwitch=args.write, PlotSwitch=args.plot)
import numpy as np
from EX_L_Domain import domain, BCs, VariationalFormulation
from FPAltMin import FPAltMin
from OuterSolver import OuterSolver

class FP(FPAltMin):
    def __init__(self):
        self.u, self.v, self.dom=domain()
        bcs_u, bcs_v, self.u_D = BCs(self.u,self.v,self.dom)
        E_u, E_v, E_uu, E_vv, E_uv, E_vu, self.elastic_energy, self.dissipated_energy, p, self.total_energy = VariationalFormulation(self.u,self.v,self.dom)
        self.setUp(E_u, E_v, E_uu, E_vv, E_uv, E_vu, bcs_u, bcs_v)

    def updateBCs(self, t):
        self.u_D.value = t

def main(method='AltMin', WriteSwitch=False, PlotSwitch=False):
    fp = FP()
    example='L'
    loads = np.linspace(0,0.6,81)

    os=OuterSolver(fp, example, method, loads)
    os.solve(WriteSwitch=WriteSwitch, PlotSwitch=PlotSwitch, maxit=1000, tol=1e-6)
    return os.energies, os.identifier
import numpy as np
from OuterSolver import OuterSolver
from EX_CTFM_Domain import domain, BCs, VariationalFormulation
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
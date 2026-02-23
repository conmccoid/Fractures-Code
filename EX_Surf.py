import numpy as np
from DOM_Surf import domain, BCs, VariationalFormulation, SurfBC
from FPAltMin import FPAltMin
from OuterSolver import OuterSolver

class FP(FPAltMin):
    def __init__(self):
        '''Initialize the problem by defining the domain, variational formulation, and boundary conditions.
        - p: problem parameters
        '''
        u, v, dom, cell_tags, facet_tags=domain()
        E_u, E_v, E_uu, E_vv, E_uv, E_vu, self.elastic_energy, self.dissipated_energy, self.p, self.total_energy = VariationalFormulation(u,v,dom,cell_tags,facet_tags)
        bcs_u, bcs_v, self.U, self.bdry_cells = BCs(u,v,dom,cell_tags, facet_tags, self.p)
        self.setUp(E_u, E_v, E_uu, E_vv, E_uv, E_vu, bcs_u, bcs_v)

    def updateBCs(self, t):
        self.U.interpolate(lambda x: SurfBC(x,t,self.p),self.bdry_cells)

def main(method='AltMin', WriteSwitch=False, PlotSwitch=False):
    fp = FP()
    example = 'Surf'
    loads = np.linspace(0.5, 6, 18)  # Load values
    os = OuterSolver(fp, example, method, loads)
    os.solve(WriteSwitch=WriteSwitch, PlotSwitch=PlotSwitch, maxit=1000, tol=1e-4)
    return os.energies, os.identifier
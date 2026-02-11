from EX_CTFM_Domain import domain, BCs, VariationalFormulation
from FPAltMin import FPAltMin

class FPAltMin_CTFM(FPAltMin):
    def __init__(self):
        self.u, self.v, self.dom, cell_tags, facet_tags=domain()
        bcs_u, bcs_v, self.t1, self.t2 = BCs(self.u,self.v,self.dom,cell_tags, facet_tags)
        E_u, E_v, E_uu, E_vv, E_uv, E_vu, self.elastic_energy, self.dissipated_energy, self.load_c, self.total_energy = VariationalFormulation(self.u,self.v,self.dom,cell_tags, facet_tags)
        self.setUp(E_u, E_v, E_uu, E_vv, E_uv, E_vu, bcs_u, bcs_v)

    def updateBCs(self, t):
        self.t1.value = t
        self.t2.value =-t
from EX_GD_Domain import domain, BCs, VariationalFormulation
from FPAltMin import FPAltMin

class FPAltMin_GD(FPAltMin):
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
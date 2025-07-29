from FPAltMin_CTFM import FPAltMin
from dolfinx import io
import numpy as np

def main(WriteSwitch=False, PlotSwitch=False):
    fp = FPAltMin()
    loads = np.linspace(0, 1.5 * fp.load_c * 12 / 10, 20) # (load_c/E)*L
    crit = 4

    x, _ = fp.createVecMat()  # Create empty vector and matrix
    res = x.duplicate()  # Create a duplicate for the residual vector

    if WriteSwitch:
        with io.XDMFFile(fp.comm, f"output/PLOT_CTFM_propagation.xdmf","w") as xdmf:
            xdmf.write_mesh(fp.dom)

    for t in loads[0:crit]:
        iteration=0
        if fp.rank == 0:
            print(f"-- Solving for load = {t:3.2f} --")
        fp.updateBCs(t)
        fp.Fn(None, x, res)  # Evaluate the function
        x += res  # Update the vector with the residual
        fp.updateUV(x)  # Update the solution vectors
        error = fp.updateError()
        fp.monitor(iteration)
        while error > 1e-4:
            iteration += 1
            fp.Fn(None, x, res)
            x += res
            fp.updateUV(x)
            error = fp.updateError()
            fp.monitor(iteration)

    iteration=0
    if fp.rank == 0:
        print(f"-- Solving for load = {loads[crit]:3.2f} --")
    fp.updateBCs(loads[crit])
    fp.Fn(None, x, res)  # Evaluate the function
    x+=res
    fp.updateUV(x)  # Update the solution vectors
    error = fp.updateError()
    fp.monitor(iteration)

    if PlotSwitch:
        fp.plot(x=x)
    
    if WriteSwitch:
        with io.XDMFFile(fp.comm, f"output/PLOT_CTFM_propagation.xdmf","a") as xdmf:
            xdmf.write_function(fp.u, iteration)
            xdmf.write_function(fp.v, iteration)
    
    while error > 1e-4:
        iteration += 1
        fp.Fn(None, x, res)
        x+=res
        fp.updateUV(x) # Update the solution vectors
        error = fp.updateError()
        fp.monitor(iteration)

        if PlotSwitch:
            fp.plot(x=x)

        if WriteSwitch:
            with io.XDMFFile(fp.comm, f"output/PLOT_CTFM_propagation.xdmf","a") as xdmf:
                xdmf.write_function(fp.u, iteration)
                xdmf.write_function(fp.v, iteration)
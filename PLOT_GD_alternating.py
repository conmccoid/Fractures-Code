from FPAltMin_GD import FPAltMin
from dolfinx import io
import numpy as np

def main(WriteSwitch=False, PlotSwitch=False):
    """
    Show how AltMin alternates minimizations with the GD example.
    """
    fp = FPAltMin()
    loads = np.linspace(0, 1, 10)  # Load values
    crit = 4

    x, _ = fp.createVecMat()  # Create empty vector and matrix
    res = x.duplicate()  # Create a duplicate for the residual vector

    if WriteSwitch:
        with io.XDMFFile(fp.comm, f"output/PLOT_GD_alternating.xdmf","w") as xdmf:
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
    resu, resv= res.getNestSubVecs()
    xu, xv= x.getNestSubVecs()
    xu+=resu
    fp.updateUV(x) # update after minimizing over displacement
    if WriteSwitch:
        with io.XDMFFile(fp.comm, f"output/PLOT_GD_alternating.xdmf","a") as xdmf:
            xdmf.write_function(fp.u, 2*iteration)
            xdmf.write_function(fp.v, 2*iteration)
    xv+=resv
    fp.updateUV(x) # update after minimizing over damage
    if WriteSwitch:
        with io.XDMFFile(fp.comm, f"output/PLOT_GD_alternating.xdmf","a") as xdmf:
            xdmf.write_function(fp.u, 2*iteration+1)
            xdmf.write_function(fp.v, 2*iteration+1)
    error = fp.updateError()
    fp.monitor(iteration)

    if PlotSwitch:
        fp.plot(x=x)
    
    while error > 1e-4:
        iteration += 1
        fp.Fn(None, x, res)
        resu, resv= res.getNestSubVecs()
        xu, xv= x.getNestSubVecs()
        xu+=resu
        fp.updateUV(x) # update after minimizing over displacement
        if WriteSwitch:
            with io.XDMFFile(fp.comm, f"output/PLOT_GD_alternating.xdmf","a") as xdmf:
                xdmf.write_function(fp.u, 2*iteration)
                xdmf.write_function(fp.v, 2*iteration)
        xv+=resv
        fp.updateUV(x) # update after minimizing over damage
        if WriteSwitch:
            with io.XDMFFile(fp.comm, f"output/PLOT_GD_alternating.xdmf","a") as xdmf:
                xdmf.write_function(fp.u, 2*iteration+1)
                xdmf.write_function(fp.v, 2*iteration+1)
        error = fp.updateError()
        fp.monitor(iteration)

        if PlotSwitch:
            fp.plot(x=x)
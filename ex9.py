#!/usr/bin/env python3
"""
2D obstacle problem WITHOUT using DMDA - pure Vec/Mat interface.

Solves the Laplace equation u_xx + u_yy = 0 with the constraint
that u(x,y) >= psi(x,y) where psi is a hemispherical obstacle.

This demonstrates active set methods using only Vec and Mat objects.

Example usage:
    python ex9.py -n 9 -snes_vi_monitor
    python ex9.py -n 17 -snes_type vinewtonssls -show_inactive_set
    python ex9.py -n 33 -snes_monitor_short
"""

import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np

# Constants for the exact solution
AFREE = 0.697965148223374
A = 0.680259411891719
B = 0.471519893402112


def psi(x, y):
    """Hemispherical obstacle with C^1 'skirt' at r=r0."""
    r = x*x + y*y
    r0 = 0.9
    psi0 = np.sqrt(1.0 - r0*r0)
    dpsi0 = -r0 / psi0
    
    if r <= r0:
        return np.sqrt(1.0 - r)
    else:
        return psi0 + dpsi0 * (r - r0)


def u_exact(x, y):
    """Exact solution to the obstacle problem."""
    r = np.sqrt(x*x + y*y)
    if r <= AFREE:
        return psi(x, y)
    else:
        return -A * np.log(r) + B


class Grid2D:
    """Simple 2D grid structure to replace DMDA functionality."""
    
    def __init__(self, nx, ny, xmin=-2.0, xmax=2.0, ymin=-2.0, ymax=2.0):
        self.nx = nx
        self.ny = ny
        self.n = nx * ny
        
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        
        self.dx = (xmax - xmin) / (nx - 1)
        self.dy = (ymax - ymin) / (ny - 1)
        
        # Create coordinate arrays
        self.x = np.linspace(xmin, xmax, nx)
        self.y = np.linspace(ymin, ymax, ny)
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing='ij')
    
    def ij_to_index(self, i, j):
        """Convert (i,j) grid indices to global vector index."""
        return i + j * self.nx
    
    def index_to_ij(self, idx):
        """Convert global vector index to (i,j) grid indices."""
        i = idx % self.nx
        j = idx // self.nx
        return i, j
    
    def is_boundary(self, i, j):
        """Check if (i,j) is on the boundary."""
        return i == 0 or j == 0 or i == self.nx - 1 or j == self.ny - 1


def formExactSolution(grid, u):
    """Fill vector u with exact solution values."""

    for j in range(grid.ny):
        for i in range(grid.nx):
            idx = grid.ij_to_index(i, j)
            x = grid.x[i]
            y = grid.y[j]
            u.array[idx] = u_exact(x, y)

def computeBounds(grid, xl, xu):
    """Compute lower bounds (obstacle) and upper bounds (infinity)."""

    for j in range(grid.ny):
        for i in range(grid.nx):
            idx = grid.ij_to_index(i, j)
            x = grid.x[i]
            y = grid.y[j]
            xl.array[idx] = psi(x, y)
    
    xu.setArray(PETSc.INFINITY)

def formFunction(snes, u, f):
    """Form the residual function for the Laplace equation."""
    grid = snes.getAppCtx()
    
    for j in range(grid.ny):
        for i in range(grid.nx):
            idx = grid.ij_to_index(i, j)
            x = grid.x[i]
            y = grid.y[j]
            
            # Boundary conditions
            if grid.is_boundary(i, j):
                f.array[idx] = 4.0 * (u.array_r[idx] - u_exact(x, y))
            else:
                # Interior: discretized Laplacian
                # Get neighbor values
                idx_w = grid.ij_to_index(i - 1, j)
                idx_e = grid.ij_to_index(i + 1, j)
                idx_s = grid.ij_to_index(i, j - 1)
                idx_n = grid.ij_to_index(i, j + 1)
                
                # Handle neighbors on boundary with exact solution
                uw = u_exact(x - grid.dx, y) if i - 1 == 0 else u.array_r[idx_w]
                ue = u_exact(x + grid.dx, y) if i + 1 == grid.nx - 1 else u.array_r[idx_e]
                us = u_exact(x, y - grid.dy) if j - 1 == 0 else u.array_r[idx_s]
                un = u_exact(x, y + grid.dy) if j + 1 == grid.ny - 1 else u.array_r[idx_n]

                f.array[idx] = (-(grid.dy/grid.dx) * (uw - 2.0*u.array_r[idx] + ue) -
                                (grid.dx/grid.dy) * (us - 2.0*u.array_r[idx] + un))

def formJacobian(snes, u, J, P):
    """Form the Jacobian matrix for the Laplace equation."""
    grid = snes.getAppCtx()
    
    oxx = grid.dy / grid.dx
    oyy = grid.dx / grid.dy
    
    P.zeroEntries()
    
    for j in range(grid.ny):
        for i in range(grid.nx):
            row = grid.ij_to_index(i, j)
            
            # Boundary points
            if grid.is_boundary(i, j):
                P.setValue(row, row, 4.0)
            else:
                # Interior points: 5-point stencil
                cols = [row]
                vals = [2.0 * (oxx + oyy)]
                
                # West neighbor
                if i - 1 > 0:
                    cols.append(grid.ij_to_index(i - 1, j))
                    vals.append(-oxx)
                
                # East neighbor
                if i + 1 < grid.nx - 1:
                    cols.append(grid.ij_to_index(i + 1, j))
                    vals.append(-oxx)
                
                # South neighbor
                if j - 1 > 0:
                    cols.append(grid.ij_to_index(i, j - 1))
                    vals.append(-oyy)
                
                # North neighbor
                if j + 1 < grid.ny - 1:
                    cols.append(grid.ij_to_index(i, j + 1))
                    vals.append(-oyy)
                
                P.setValues(row, cols, vals)
    
    P.assemblyBegin()
    P.assemblyEnd()
    
    if J != P:
        J.assemblyBegin()
        J.assemblyEnd()


def analyzeInactiveSet(snes, u, grid):
    """
    Demonstrate how to use getVIInactiveSet() to analyze the active set.
    """
    # Get the inactive set (variables not constrained)
    inactive_is = snes.getVIInactiveSet()
    PETSc.Sys.Print(f"DEBUG: got inactive set")
    
    if inactive_is is None:
        PETSc.Sys.Print("Warning: Could not retrieve inactive set")
        return
    
    try:
        total_size = inactive_is.getSize()
        PETSc.Sys.Print(f"DEBUG: total size via getSize() = {total_size}")
    except Exception as e:
        PETSc.Sys.Print(f"ERROR calling getSize(): {e}")
        return
    
    # Now try getLocalSize
    try:
        inactive_size = inactive_is.getLocalSize()
        PETSc.Sys.Print(f"DEBUG: inactive_size via getLocalSize() = {inactive_size}")
    except Exception as e:
        PETSc.Sys.Print(f"ERROR calling getLocalSize(): {e}")
        # Use total_size as fallback
        inactive_size = total_size
        PETSc.Sys.Print(f"Using getSize() result instead: {inactive_size}")
    # # Get local size and inactive indices
    # inactive_size = inactive_is.getLocalSize()
    # PETSc.Sys.Print(f"DEBUG: inactive size = {inactive_size}")
    # inactive_indices = inactive_is.getIndices()
    
    total_dofs = grid.n
    active_size = total_dofs - inactive_size
    
    PETSc.Sys.Print("=" * 60)
    PETSc.Sys.Print("Active Set Analysis (without DMDA):")
    PETSc.Sys.Print(f"  Grid:               {grid.nx} x {grid.ny}")
    PETSc.Sys.Print(f"  Total DOFs:         {total_dofs}")
    PETSc.Sys.Print(f"  Inactive DOFs:      {inactive_size} ({100*inactive_size/total_dofs:.1f}%)")
    PETSc.Sys.Print(f"  Active DOFs:        {active_size} ({100*active_size/total_dofs:.1f}%)")
    PETSc.Sys.Print("=" * 60)
    
    # Get bounds to verify
    xl, xu = snes.getVariableBounds()
    u_array = u.getArray()
    xl_array = xl.getArray()
    
    # Check how many variables are at the lower bound
    at_bound = np.sum(np.abs(u_array - xl_array) < 1e-10)
    PETSc.Sys.Print(f"  Variables at lower bound: {at_bound}")
    
    # Show some spatial information about active set
    if grid.nx <= 20:
        PETSc.Sys.Print("\nActive set visualization (* = active, . = inactive):")
        for j in range(grid.ny - 1, -1, -1):  # Top to bottom
            line = ""
            for i in range(grid.nx):
                idx = grid.ij_to_index(i, j)
                if idx in inactive_indices:
                    line += ". "
                else:
                    line += "* "
            PETSc.Sys.Print(f"  {line}")
        PETSc.Sys.Print("")
    
    # Show physical location of active region
    active_indices = [idx for idx in range(total_dofs) if idx not in inactive_indices]
    if active_indices:
        active_radii = []
        for idx in active_indices:
            i, j = grid.index_to_ij(idx)
            x, y = grid.x[i], grid.y[j]
            r = np.sqrt(x*x + y*y)
            active_radii.append(r)
        
        PETSc.Sys.Print(f"  Active set radius range: [{min(active_radii):.3f}, {max(active_radii):.3f}]")
        PETSc.Sys.Print(f"  Theoretical free boundary: r = {AFREE:.3f}")
    
    PETSc.Sys.Print("=" * 60)
    
    u.setArray(u_array)
    xl.setArray(xl_array)


def main(n=9,show_inactive=True):
    # Parse options
    opts = PETSc.Options()
    
    PETSc.Sys.Print(f"\nSolving 2D obstacle problem on {n}x{n} grid (without DMDA)")
    
    # Create grid structure
    grid = Grid2D(n, n)
    
    # Create vectors
    u = PETSc.Vec().createSeq(grid.n)
    u.zeroEntries()
    
    xl = PETSc.Vec().createSeq(grid.n)
    xu = PETSc.Vec().createSeq(grid.n)
    computeBounds(grid, xl, xu)
    
    # Create matrix for Jacobian (5-point stencil = max 5 nonzeros per row)
    J = PETSc.Mat().createAIJ([grid.n, grid.n], nnz=5*grid.n)
    J.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    J.setUp()
    
    # Create SNES solver
    snes = PETSc.SNES().create(PETSc.COMM_SELF)
    snes.setType(PETSc.SNES.Type.VINEWTONRSLS)
    
    # Store grid in application context
    snes.setAppCtx(grid)
    
    # Set up the variational inequality
    u.assemble()
    xl.assemble()
    xu.assemble()
    J.assemble()
    snes.setFunction(formFunction, u.duplicate())
    snes.setJacobian(formJacobian, J, J)
    snes.setVariableBounds(xl, xu)
    snes.setFromOptions()
    
    # Solve
    PETSc.Sys.Print("Solving...")
    snes.solve(None, u)
    
    # Check convergence
    reason = snes.getConvergedReason()
    its = snes.getIterationNumber()
    PETSc.Sys.Print(f"Converged in {its} iterations (reason: {reason})")
    
    # Compute errors
    u_ex = PETSc.Vec().createSeq(grid.n)
    formExactSolution(grid, u_ex)
    
    u.axpy(-1.0, u_ex)  # u = u - u_exact
    error1 = u.norm(PETSc.NormType.NORM_1)
    error1 /= grid.n  # Average error
    errorinf = u.norm(PETSc.NormType.NORM_INFINITY)
    
    PETSc.Sys.Print(f"\nerrors on {grid.nx} x {grid.ny} grid:  "
                    f"av |u-uexact| = {error1:.3e},  "
                    f"|u-uexact|_inf = {errorinf:.3e}")
    
    # Demonstrate getVIInactiveSet() usage
    if show_inactive or opts.getBool('snes_vi_monitor', False):
        PETSc.Sys.Print(f"DEBUG: attempt to use analyzeInactiveSet()")
        # Restore u to actual solution (not error)
        u.axpy(1.0, u_ex)
        analyzeInactiveSet(snes, u, grid)
        PETSc.Sys.Print(f"DEBUG: success using analyzeInactiveSet()")
    
    # Optional: save solution for visualization
    if opts.getBool('save_solution', False):
        u.axpy(1.0, u_ex)  # Restore if we subtracted
        u_array = u.getArray()
        u_2d = u_array.reshape((grid.ny, grid.nx))
        
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(14, 5))
            
            # Surface plot
            ax1 = fig.add_subplot(131, projection='3d')
            ax1.plot_surface(grid.xx, grid.yy, u_2d.T, cmap='viridis', alpha=0.8)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('u')
            ax1.set_title('Solution u(x,y)')
            
            # Contour plot
            ax2 = fig.add_subplot(132)
            contour = ax2.contourf(grid.xx, grid.yy, u_2d.T, levels=20, cmap='viridis')
            plt.colorbar(contour, ax=ax2)
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.set_title('Solution contours')
            ax2.set_aspect('equal')
            
            # Active set visualization
            ax3 = fig.add_subplot(133)
            inactive_is = snes.getVIInactiveSet()
            inactive_indices = inactive_is.getIndices()
            active_mask = np.ones(grid.n)
            active_mask[inactive_indices] = 0
            active_2d = active_mask.reshape((grid.ny, grid.nx))
            
            ax3.imshow(active_2d.T, origin='lower', extent=[grid.xmin, grid.xmax, grid.ymin, grid.ymax],
                      cmap='RdYlGn_r', interpolation='nearest')
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')
            ax3.set_title('Active Set (red) vs Inactive (green)')
            ax3.set_aspect('equal')
            
            plt.tight_layout()
            plt.savefig('obstacle_solution.png', dpi=150)
            PETSc.Sys.Print("\nSolution saved to obstacle_solution.png")
            plt.show()
            
            u.setArray(u_array)
        except ImportError:
            PETSc.Sys.Print("\nInstall matplotlib for visualization: pip install matplotlib")
    
    # Cleanup
    u_ex.destroy()
    xl.destroy()
    xu.destroy()
    u.destroy()
    J.destroy()
    snes.destroy()


if __name__ == '__main__':
    main()
    PETSc.Sys.Print("\nDone!")
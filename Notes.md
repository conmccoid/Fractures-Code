# Phase-field fracture models

## Research program

We now have for comparison three methods:
- AltMin, using SNESVINewton for irreversibility
- MSPIN with cubic backtracking, using ??? for irreversibility
- Parallelogram interpolation (and various generalizations), using active set for irreversibility

- [ ] Implement irreversibility
- [ ] Parallelize code
- [ ] Run examples
  - [ ] Check efficiency, both in FLOPs and wall clock time
  - [ ] Check scaling
- [ ] Write paper
  - [ ] Section: background on MSPIN for phase-field
  - [ ] Section: parallelogram interpolation
    - [ ] Subsection: active set method in parallelogram setting
  - [ ] Section: numerical comparisons

## Running examples

### Fractures-code

```
docker run --rm --name phase-field -it -v ${pwd}:/Fractures-code -w /Fractures-code -p 8888:8888 dolfinx/lab:stable

docker exec -it phase-field bash

pip install meshio
mpirun -n 8 python EX.py
```

### firebreak

```
docker run --rm --name firebreak -it -v ${pwd}:/firebreak -w /firebreak -p 8888:8888 firedrakeproject/firedrake

docker exec -it firebreak bash

pip install meshio
python3

import meshio
mesh=meshio.read("mesh.e")
meshio.write("mesh.msh",mesh)
exit()
```

The firebreak repo has its own Dockerfile (slightly out of date, this has been mostly fixed).
```
docker build -t firebreak ./
```

### ParaView: warp and colour

XDMF outputs from the examples store data in blocks.
To warp by displacement but colour by damage:
1. Extract Block filter on main data; do this twice, once for each block of data.
2. Append Attributes filter on both extracted blocks.
3. Warp by Vector filter on appended blocks.
4. Change the Coloring variable from the dropdown to the scalar block.

## Coding up MSPEN

### Concatenating vectors

I need to concatenate two PETSc vector arrays in such a way that I can easily pull them apart.
I need to be able to put the concatenated vector into a PETSc solver.

There seems to be a deprecated method `PETSc.Vec.concatenate()` which would have done exactly as I needed.
Now I'm hoping that functionality has been moved to `PETSc.Vec().createNest()`.

I can nest vectors with:
```python
vecs = PETSc.Vec().createNest([u.x.petsc_vec,v.x.petsc_vec])
```
If I want to extract vectors from the nest, I do:
```python
u_out, v_out = vecs.getNestSubVecs()
```

### Creating a .msh from a .geo (copied from MEF++ cheatsheet)

Convert a .geo into a .msh with the line
```
gmsh fichier.geo -3 -o nomsortie.msh
```
Replace `-3` with `-2` when using 2D meshes.

### Running code in Docker container from command line

First make sure the Docker container is running.
Open bash in the container with
```
docker exec -it <container name> sh
```
Change directory to the shared folder:
```
cd shared
```
Now anything can be run in the shared folder with standard commands:
```
mpiexec -n 8 python -u EX_CTFM.py
```
(Include the `-u` to output to the terminal.)

### Parallelization

It's important that objects used in multiple places are properly gathered and scattered when they are needed.

For example, a recent hurdle, the vectors 'u' and 'v' must be updated when assembling 'Euv' and 'Evu'.
This is not done in the course of the iteration, and so must be done when 'Euv' and 'Evu' are needed when multiplying by the Jacobian.

## SSH into Graham cluster

`ssh mccoidc@graham.alliancecan.ca`, using SSH key and DUO multi-factor app.

### Examples

#### Basic shear?

#### L-shaped

## Cubic backtracking implementation

I'm now mapping out the energy landscape along the search directions, either Newton or fixed point.
However, this is showing that the initial step at each new load for AltMin actually increases the energy before optimizing.
Possible explanations:
- something isn't updating before the landscape is plotted
- there's a bug when solving the AltMin process
- something isn't assembled at the right time
- a previous load or solution is being plotted
- to satisfy the boundary conditions, energy must first increase (this HAS to be it!)

AltMin is giving bad steps when used as a fallback for cubic backtracking.
I'm concerned that either the boundary conditions are suddenly not satisfied as part of cubic backtracking, or something within the cubic backtracking algorithm messes with the AltMin step.

Important notes:
- need to enforce boundary conditions before entering into Newton steps
- often the energy for the AltMin step will be lower even after cubic backtracking, so it's probably a good idea to switch when that happens
- each step of cubic backtracking requires a handful of FLOPs and an evaluation of the objective function (energies); this means it's fairly cheap, so the only serious additional costs of MSPIN over AltMin will be the Newton KSP solve
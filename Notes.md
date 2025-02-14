# Phase-field fracture models

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

### Custom line search

I need to be able to switch between MSPIN and AltMin based on the residual.
This will require a custom line search, since this will take advantage of the implicit fixed point iteration,
which is not common for SNES problems.

Might be an idea to adapt Powell's dogleg method, which chooses a step optimized along a line between two vectors.
Those two vectors would be the step from the fixed point iteration and the Newton step.

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

## SSH into Graham cluster

`ssh mccoidc@graham.alliancecan.ca`, using SSH key and DUO multi-factor app.
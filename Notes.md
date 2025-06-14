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

New idea (03.13.2025): Davidenko-Branin trick, using analysis on 2D hyperplane containing fixed point iteration and the line connecting the Newton and anti-Newton directions.

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

## 3K replication

### Building MOOSE

1. Create a volume called `projects`:
```
docker volume create projects
```
2. Build the latest image of MOOSE into the volume:
```
docker run -it -v projects:/projects idaholab/moose:latest
```
3. Test the MOOSE build by running an electromagnetics example:
```
cd /projects
moose-opt --copy-inputs electromagnetics
cd moose/electromagnetics
moose-opt --run -j 4
```

### Building Utopia in the new volume

1. Use the same `projects` folder as MOOSE.
2. Build the image:
```
docker run -it -v projects:/projects utopiadev/utopia
```

This did not work.
I need to somehow combine the two into a single volume/container.

### Building a unified Docker image

Combining the two images has proven incredibly difficult.
I am now trying to use one of the images (MOOSE, the larger) and clone the git repo of the other (Utopia).

I am doing this through a Dockerfile.
A Dockerfile is a plain text (no extension) file called `Dockerfile`.
To run a Dockerfile, run `docker build -t build-name /path/to/Dockerfile`, or navigate to the directory containing Dockerfile and run `docker build .`.
A Dockerfile saves the method for how to build an image, which is very useful.
It should include the update and installation of all dependencies.

1. Load the MOOSE image as base:
```
FROM idaholab/moose:latest AS moose
```
- `FROM` says where to start from, in this case an image online.
- `AS` gives a name to this step that can act as a placeholder.
Having another `FROM moose AS setup` line will split up the build, with the next part starting from the previous.
Using `--target moose` in the `docker build` will cause the build to stop after the first part.


1. Update and install dependencies for Utopia:
```
RUN dnf update -y && dnf install -y gcc-c++ make cmake openmpi openmpi-devel blas-devel git && dnf clean all
ENV MPI_C=mpi-cc MPI_CXX=mpi-c++ CMAKE_PREFIX_PATH=/usr/lib64/openmpi LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH PATH=/usr/lib64/openmpi/bin:$PATH
```
- The package manager `dnf` is for the Rocky-linux used by MOOSE.
- The `-y` tells the terminal to type `Y` when a `[y/n]` option comes up.
- The `&&` indicates new commands.
- To separate out commands, use `\` to break up lines.
- `ENV` creates some environment variables to add `openmpi` to the path.

1. Set the work directory for Utopia and clone from bitbucket, then follow installation directions from that repo:
```
WORKDIR /
RUN git clone --recurse-submodules https://bitbucket.org/zulianp/utopia.git
ENV UTOPIA_DIR=/utopia
WORKDIR /utopia/utopia
RUN mkdir bin && cd bin && cmake .. -DCMAKE_INSTALL_PREFIX=$UTOPIA_DIR && make && make install
```

4. Add a new non-root user:
```
RUN useradd -m appuser
USER appuser
```
- The image of MOOSE declares the default user to be `root`, which causes security risks when running in parallel with MPI.
A new user is needed that owns everything in the directory.

5. Once the container is built and run, before anything else can be done the environment must be sourced:
```
source /environment
```

__This approach ended up being ineffective because the version of MOOSE required by pf_frac_spin is outdated.__

### Building a bespoke Docker image

`docker build -t 3k-build .`

1. Add a base image: `FROM ubuntu 22.04 AS base`
2. Install dependencies: `RUN apt-get install -y ...`
3. Install Utopia from the specific commit using `cmake`.
4. Install MOOSE from the specific commit. Use scripts to install dependencies: `RUN cd moose/scripts && ./update_and_rebuild_petsc.sh && ./update_and_rebuild_libmesh.sh`. Then `make` MOOSE.
5. Install pf_frac_spin.

`docker run --rm --name 3k-container -it -v projects:/projects 3k-build`

Test MOOSE: `cd ~/projects/moose/test && ./run_tests -j 6`
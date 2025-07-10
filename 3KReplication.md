
# 3K replication

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

1. Install the git repo for `pf_frac_spin`, creating the directory `pf_frac_spin`; create a Dockerfile in this directory
2. Load the MOOSE image as base: `FROM idaholab/moose-dev:latest AS moose`
   - `FROM` says where to start from, in this case an image online.
   - `AS` gives a name to this step that can act as a placeholder.
   - Having another `FROM moose AS setup` line will split up the build, with the next part starting from the previous.
   - Using `--target moose` in the `docker build` will cause the build to stop after the first part.
3. Copy the Utopia image: `COPY --from=utopiadev/utopia:latest /utopia /utopia`
   - `--from=utopiadev/utopia:latest` indicates the image to copy from
   - `/utopia /utopia_image` copies the directory `/utopia` into one in the image called `/utopia_image`
4. Add environment variables to indicate the directories of MOOSE and Utopia: `ENV UTOPIA_DIR=/utopia` and `ENV MOOSE_DIR=/opt/moose`
5. Source the environment (?): `RUN source /environment`
6. Add environment variables to override warnings from MPI about running as root user: `ENV OMPI_ALLOW_RUN_AS_ROOT=1` and `ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1`
   - The image of MOOSE declares the default user to be `root`, which causes security risks when running in parallel with MPI
7. Make a `pf_frac_spin/CMakeLists.txt` with the information for `cmake`
8. Build the image: `docker build -t pf_frac_spin .` (replace `.` with the path to `pf_frac_spin/`)
   1. `-t pf_frac_spin` sets the tag of the build as `pf_frac_spin`
   2. `.` is the path to the Dockerfile
9.  Run the Docker container: `docker run --rm --name pf_frac_spin -it -v .:pf_frac_spin -w /pf_frac_spin pf_frac_spin`
    -  `--rm` tells Docker to clean up the container once we're finished with it
    -  `--name pf_frac_spin` gives the container a specific name
    -  `-it` (not sure)
    -  `-v .:/pf_frac_spin` uses the current directory as a volume, sharing the files there on the host with the container, placing them in a directory called `/pf_frac_spin`
    -  `-w /pf_frac_spin` sets the working directory as the new directory shared from the host
    -  `pf_frac_spin` is the tag of the build

#### Proscribed workflows

Idaho Lab suggests using the `moose-dev` docker image when developing MOOSE-based applclications. The example thiey give follows these steps:
1. Create a new volume
2. Run Docker, moutning this volume with th e`moose-dev:latest` image
3. In the volume, clone the MOOSE git repo
4. Install MOOSE (this step doesn't work, error 127, `-std=c++17: command not found`)
5. Test the installation of MOOSE

##### Historic

Update and install dependencies for Utopia:
```
RUN dnf update -y && dnf install -y gcc-c++ make cmake openmpi openmpi-devel blas-devel git && dnf clean all
ENV MPI_C=mpi-cc MPI_CXX=mpi-c++ CMAKE_PREFIX_PATH=/usr/lib64/openmpi LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH PATH=/usr/lib64/openmpi/bin:$PATH
```
- The package manager `dnf` is for the Rocky-linux used by MOOSE.
- The `-y` tells the terminal to type `Y` when a `[y/n]` option comes up.
- The `&&` indicates new commands.
- To separate out commands, use `\` to break up lines.
- `ENV` creates some environment variables to add `openmpi` to the path.

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

##### new options 26.06.2025
- reduced to simplified image with all files (except pf_frac_spin)
- running a script in simplified image: `./build_dependencies.sh`
- create new image from the resulting container: `docker commit` (? check syntax)

Alena (Kopanicakova) has told me she was the only one working on Utopia at USI, and now that she's not, the code base is defunct and has not kept up with its dependencies.
Pablo thinks that, because it uses Cmake, it could be salvageable, but this is almost certainly not worth the effort since Alena has started migrating the code base to Firedrake.
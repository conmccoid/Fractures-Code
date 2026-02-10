FROM dolfinx/lab:stable

# install frequent dependencies
USER root
RUN apt-get update && apt-get install -y xvfb
RUN pip install meshio

# clone PETSc repo
WORKDIR /tmp
RUN git clone -b release https://gitlab.com/petsc/petsc.git petsc

# patch PETSc to fix getVIInactiveSet bug
WORKDIR /tmp/petsc
RUN sed -i '506d' src/snes/impls/vi/rs/virs.c

# configure and build PETSc (not sure about these config options)
ENV PETSC_DIR=/tmp/petsc
ENV PETSC_ARCH=linux-gnu-real64-32
RUN ./configure
RUN make all check

# install petsc4py
RUN python3 -m pip uninstall petsc4py
RUN python3 -m pip install src/binding/petsc4py
RUN cd src/binding/petsc4py && python3 test/runtests.py

# looks like this tmp folder is now necessary
# # clean up
# RUN rm -rf /tmp/petsc
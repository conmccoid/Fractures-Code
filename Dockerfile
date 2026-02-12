FROM dolfinx/lab:stable

# install frequent dependencies
USER root
RUN apt-get update && apt-get install -y xvfb
RUN pip install meshio

# clone PETSc repo
WORKDIR /usr/local/petsc
RUN git reset --hard # restore the source code which is deleted for some reason
RUN sed -i '24s/IS_inact/IS_inact_prev/' src/snes/impls/vi/rs/virs.c #Patch SNESVIGetInactiveSet
RUN make all

FROM idaholab/moose:latest AS moose
# RUN dnf update -y && \
#     dnf install -y \
#         gcc-c++ \
#         make \
#         cmake \
#         openmpi \
#         openmpi-devel \
#         blas-devel \
#         lapack-devel \
#         yaml-cpp-devel \
#         suitesparse-devel \
#         git && \
#     dnf clean all
# ENV MPI_C=mpi-cc \
#     MPI_CXX=mpi-c++ \
#     MPI_Fortran=mpi-f90 \
#     CMAKE_PREFIX_PATH=/usr/lib64/openmpi \
#     CMAKE_PREFIX_PATH=/usr/include/suitesparse \
#     LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH \
#     PATH=/usr/lib64/openmpi/bin:$PATH

FROM moose AS utopia
COPY --from=utopiadev/utopia:latest /utopia /utopia
# WORKDIR /
# RUN git clone --recurse-submodules https://bitbucket.org/zulianp/utopia.git
# ENV UTOPIA_DIR=/utopia
# CMD ["/bin/bash"]
# # WORKDIR /utopia/utopia
# # RUN mkdir build_dep
# # WORKDIR /utopia/utopia/build_dep
# # RUN cmake .. && make petsc && make trilinos
# WORKDIR /utopia/utopia
# RUN mkdir bin && \
#     cd bin && \
#     cmake .. -DCMAKE_INSTALL_PREFIX=$UTOPIA_DIR && \
#     make && \
#     make install

FROM utopia AS replication
# RUN useradd -m appuser
# USER appuser
WORKDIR /files
RUN git clone https://bitbucket.org/alena_kopanicakova/pf_frac_spin/src/master/
RUN sed -i 's|\.\./framework|/opt/moose|g' /files/master/Makefile
RUN source /environment
WORKDIR /files/master
COPY CMakeLists.txt /files/master/
ENV MOOSE_DIR=/opt/moose
ENV UTOPIA_DIR=/utopia/utopia
RUN mkdir build && cd build && \
    source /environment && \
    cmake .. && \
    make -j$(nproc)

RUN useradd -m appuser
# RUN chown -R appuser:appuser /master
# USER appuser
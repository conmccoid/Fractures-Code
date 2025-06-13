FROM ubuntu:22.04 AS base
RUN apt-get update -y && apt-get install -y \
    build-essential \
    cmake \
    git \
    m4 \
    openmpi-bin \
    libopenmpi-dev \
    bison \
    flex \
    python3-dev
    # clang-format \
    # slepc-dev \
    # libboost-all-dev \
    # rm -rf /var/lib/apt/lists/*

WORKDIR /src
# Utopia
RUN git clone --recurse-submodules https://bitbucket.org/zulianp/utopia.git && \
    cd utopia && \
    git checkout 1074078cf4c97346c354767793a8d9374d4aed21

ENV UTOPIA_DIR=/src/utopia
RUN cd utopia/utopia && \
    mkdir bin && \
    cd bin && \
    cmake .. -DCMAKE_INSTALL_PREFIX=${UTOPIA_DIR} && \
    make -j$(nproc) && \
    make install

# MOOSE
RUN git clone https://github.com/idaholab/moose.git && \
    cd moose && \
    git checkout 90123e7b6bd52f1bc36e68aac5d1fa95e76aeb91 && \
    git submodule update --init --recursive
RUN cd moose/scripts && \
    ./update_and_rebuild_petsc.sh && \
    ./update_and_rebuild_libmesh.sh
# RUN cd moose/libmesh && \
#     ./configure --prefix=/src/moose/libmesh/installed && \
#     make -j$(nproc) install
# RUN sed -i 's/^\([[:space:]]*\)\(LocalRankConfig buildComm(bool batch_mode);\)/\1virtual \2/' \
    # framework/include/multiapps/MultiApp.h
RUN cd moose/test && \
    make -j 6

# 3K files
RUN git clone https://bitbucket.org/alena_kopanicakova/pf_frac_spin.git
RUN cd pf_frac_spin && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install

WORKDIR /
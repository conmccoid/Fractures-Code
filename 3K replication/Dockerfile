FROM idaholab/moose:latest AS moose
COPY --from=utopiadev/utopia:latest /utopia /utopia
ENV UTOPIA_DIR=/utopia
ENV MOOSE_DIR=/moose
COPY /pf_frac_spin /pf_frac_spin
WORKDIR /pf_frac_spin
RUN mkdir build && \
    cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install
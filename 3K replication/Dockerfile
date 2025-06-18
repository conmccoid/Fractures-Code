FROM idaholab/moose:latest AS moose
COPY --from=utopiadev/utopia:latest /utopia /utopia
ENV UTOPIA_DIR=/utopia
ENV MOOSE_DIR=/moose
RUN git clone https://bitbucket.org/alena_kopanicakova/pf_frac_spin.git
WORKDIR /pf_frac_spin
# RUN mkdir build && \
#     cd build && \
#     cmake .. && \
#     make -j$(nproc) && \
#     make install
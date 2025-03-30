# Journal

## Project goals

- [ ] Replicate results of Kopanicakova, Kothari and Krause (3K)
- [x] Write an MSPIN solver for phase-field fracture models based on alternate minimization
  - [ ] include a variatonal inequality module
- [ ] Replace IN(K) with alternative optimizers, such as BFGS, Richardson, etc.
  - [ ] Construct a modified Newton method that leverages the underlying fixed-point iteration

## Tasks

###  July 2024 - March 2025

- Write MSPIN solver
- Compare MSPIN and AltMin
  - Gradient damage example
  - CTFM three circles example
  - Surfing BCs example
- Use AT2 model to avoid variational inequality issues
- Manual and tutorial for using FEniCSx and petsc4py in solving phase-field fracture models
- Improved interfacing with PETSc, including bespoke line searches, monitors and convergence tests

### April 2025

- [ ] Replicate 3K results
  - [ ] Working installation of MOOSE, either locally or on cluster
  - [ ] Examine models used; are these standard? Have they been modified?
  - [ ] Compare AltMin and MSPIN code vs. ours
- [ ] Parallelize MSPIN; currently getting different results compared with serial runs on Surf example
- [ ] Construct modified Newton
  - [ ] Theory on dynamical systems
  - [ ] Generalize thesis results from 1D to nD
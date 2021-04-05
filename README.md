# Parallel Molecular Dynamics

One of the most frequent applications for high performance computing is the area of molecular dynamics, a study of movement and interaction of molecules by solving the classical equations of motion for a set of molecules. The interactions of a collection of atoms can be modelled by using force functions derived from potential energy functions e.g. the Lennard-Jones potential. This project considers molecular dynamics on a 2D torus, using a velocity Verlet numerical integration scheme. There is a serial (`md_serial.cc`) implementation and two parallel (`pmd_cell_1.cc` and `pmd_cell_2.cc`) implementations, and for further details, see **Name of Report**.

## Getting Started

These instructions will get you a copy of the project up and running on your local or remote machine for development and testing purposes.

### Prerequisites

1. a C++ compiler
2. an implementation of MPI

E.g. for the above, we used:

1. GNU compiler
2. OpenMPI

## Parameters in the Code

Due to the quantity of parameters required by molecular dynamics, no parameter is fed in through the command line. Every parameter is set in `main` or at the top of the code as a global variable. 

### Physical parameters

Here, we include a description of all the physical parameters in the code. We flag derived parameters with a :triangular_flag_on_post:;

1. `sigma` : $\sigma$ in the Lennard-Jones potential
2. `L` : The length of one side of the 2D torus. :triangular_flag_on_post:
3. `R` : The cut-off distance, $R$, for atom-atom interaction through the Lennard-Jones potential
4. `eps` : $\epsilon$ is a small number, which determines where we begin the interpolation of the potential and force function i.e. at $R-\epsilon$. 
5. `lat-size` : The number of atoms along a horizontal or vertical axis in the full torus, at the Cartesian grid initialisation stage.
6. `Ntot` : The total number of atoms in the full torus. :triangular_flag_on_post:
7. `delta_init` : The initial separation, $\delta_{\text{init}}$, between atoms, at the Cartesian grid initialisation stage. Note that $L = \delta_{\text{init}} \times \text{lat_size}$.
8. `tf` : Simulation length (in seconds).

### Numerical parameters

1. `n` : Number of Verlet iterations.
2. `boundary_type` : Set to `"periodic"` for periodic boundary conditions or `"reflective"` for reflective boundary conditions (just serial code at the moment).
3. `write_pos` : Set to `"True"` to print the positions and velocities of the atoms at every iteration.

### Parallelisation parameters

1. `cell-size` : The number of atoms along a horizontal or vertical axis in a processor's sub-domain of the full torus, at the Cartesian grid initialisation stage.
2. `nproc` : Number of processors used. Note that is just set to however many processors are used to run the code, and that it has to be a square number.
3. `M` : The length of one side of a processor's sub-domain of the full torus. Note $M = \delta_{\text{init}} \times \text{cell-size} = L/\sqrt{\text{nproc}}$. Note that we require $M \ge R$.  :triangular_flag_on_post:

## Running the Serial Code

First set the above parameters as required. Depending on the compiler used, this code will change, but continuing with our example from above:

`g++ md_serial.cc -o md_serial_exe`

## Running the Parallel Code

First set the above parameters as required. Ensure that you use a square number of processors. Depending on the compiler used, this code will change, but continuing with our example from above:

`mpicxx pmd_cell_1.cc -o pmd_cell_1_exe`
or
`mpicxx pmd_cell_2.cc -o pmd_cell_2_exe`

And then

`mpirun -np <Number of Processors> pmd_cell_1_exe`
or
`mpirun -np <Number of Processors> pmd_cell_2_exe`

## Authors

Aidan Tully, Andrew Cleary, Karolina Benkova

## Acknowledgments

The authors would like to thank Prof. Ben Leimkuhler and René Lohmann for their help with this project. 


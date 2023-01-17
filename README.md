# Ising2D
An MPI-parallel heat-bath Monte Carlo code to evolve the 2D Ising model and study its phase transition

## Usage
1. Tune the parameters (macros living in Parameters.hh)
2. Compile by running `make options=<full_path_to_optionlist>` (see directory `Optionlists` for some example optionlists)
3. Run with e.g. `mpiexec -np <nprocs> ./Ising2D`

## Things to keep in mind
1. `nprocs` should be equal to `NPROCS_X` * `N_PROCS_Y`. If that's not the case, the code will abort execution
2. Saving the lattice to file during thermalization can take time if the lattice is big
3. Generating a single output file takes more time than generating chunked output

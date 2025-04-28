# Ising2D
An MPI-parallel heat-bath Monte Carlo code to evolve the 2D Ising model and study its phase transition

## Author(s)
Lorenzo Ennoggi (<le8016@rit.edu> / <lorenzo.ennoggi@gmail.com>)

## Maintainer(s)
Lorenzo Ennoggi (<le8016@rit.edu> / <lorenzo.ennoggi@gmail.com>)


## Minimal requirements
- A C++ compiler supporting the `C++17` standard
- An MPI library
- The HDF5 library with MPI support

## Usage
1. Tune the parameters -- macros living in `Parameters.hh`
2. Compile
   ```
   make -j4 options=OptionLists/<optionlist>
   ```
   where `<optionlist>` is the list of compiler options for your machine
3. Edit the job submission script for your machine as needed, or create a new one if your machine is not listed there
4. Run
   ```
   cd RunScripts
   ./<runscript>
   ```
5. To remove the executable and all object files, type
   ```
   make clean
   ```

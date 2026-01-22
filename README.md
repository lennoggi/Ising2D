[![CI](https://github.com/lennoggi/Ising2D/actions/workflows/CI.yml/badge.svg)](https://github.com//lennoggi/Ising2D/actions/workflows/CI.yml)

# Ising2D
A hybrid MPI/CUDA heat-bath Monte Carlo code to evolve the 2D Ising model and study its phase transition

## Author(s)
Lorenzo Ennoggi (<le8016@rit.edu> / <lorenzo.ennoggi@gmail.com>)

## Maintainer(s)
Lorenzo Ennoggi (<le8016@rit.edu> / <lorenzo.ennoggi@gmail.com>)


## Minimal requirements
- A C++ compiler supporting the `C++17` standard
- An MPI library
- The HDF5 library with MPI support
- The CUDA library if you want to offload calculations to the GPU

## Usage
1. Tune the parameters -- macros living in `Parameters.hh`
2. Compile
  - CPU
   ```
   make -j5 options=OptionLists/<optionlist>
   ```
  - GPU
   ```
   make -j10 options=OptionLists/<optionlist>
   ```
   where `<optionlist>` is the list of compiler options for your machine. For GPU builds, see the example optionlist `OptionLists/Vista_NVIDIA_CUDA.cfg`.

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

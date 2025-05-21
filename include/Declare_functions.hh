#ifndef DECLARE_FUNCTIONS_HH
#define DECLARE_FUNCTIONS_HH

#include <array>
#include <random>
#include <hdf5.h>
#include "Declare_variables.hh"


std::array<int, 6>
set_indices_neighbors(const int &rank,
                      const int &nprocs);

void update(const int                                    &rank,
                  std::mt19937                           &gen,
                  std::uniform_real_distribution<double> &dist,
            const std::array<int, 6>                     &indices_neighbors,
                  std::array<int, nx1locp2_nx2locp2>     &local_lattice);

void write_lattice(const int &rank,
                   const int &nprocs,
                   const int &x1index,
                   const int &x2index,
                   const int &n,
                   const std::array<int, nx1locp2_nx2locp2> &local_lattice,
                   const hid_t &file_id);


#ifdef USE_CUDA

#include <cuda_runtime.h>

int *allocate_int_device(const int    &rank,
                         const size_t &size);

void copy_device(const int    &rank,
                       void           *dest,
                 const void           *src,
                 const size_t         &size,
                 const cudaMemcpyKind &kind);

void free_device(const int  &rank,
                       void *device_ptr);
#endif


#endif  // DECLARE_FUNCTIONS_HH

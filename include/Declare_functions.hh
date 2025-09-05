#ifndef DECLARE_FUNCTIONS_HH
#define DECLARE_FUNCTIONS_HH

#include <array>
#include <random>
#include <hdf5.h>
#include "Declare_variables.hh"


std::array<int, 7>
set_indices_neighbors_parity(const int &rank,
                             const int &nprocs);

void exchange_ghosts(const int &rank,
                     const std::array<int, 7> &indices_neighbors_parity,
                           int *local_lattice);

void update(const int                                    &rank,
                  std::mt19937                           &gen,
                  std::uniform_real_distribution<double> &dist,
            const std::array<int, 7>                     &indices_neighbors_parity,
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

template <typename T>
T *allocate_device(const int    &rank,
                   const size_t &num_elements);

template <typename T>
void copy_device(const int    &rank,
                       T      *dest,
                       T      *src,
                 const size_t &num_elements,
                 const cudaMemcpyKind &copy_kind);

template <typename T>
void copy_device_2D(const int    &rank,
                          T      *dest,
                    const size_t &dest_stride,
                          T      *src,
                    const size_t &src_stride,
                    const size_t &width,
                    const size_t &height,
                    const cudaMemcpyKind &copy_kind);

void free_device(const int  &rank,
                       void *device_ptr);

template <typename T> __global__
void init_rng_device_kernel(T *rng_states_device,
                            const size_t  seed,
                            const size_t  nx,
                            const size_t  ny,
                                  int    *out_of_bounds_device_ptr);

template <typename T>
void init_rng_device(const int &rank,
                     T *rng_states_device,
                     const size_t &seed);

template <typename T> __global__
void update_device_kernel(T   *rng_states_device,
                          int *local_lattice_device,
                          const size_t nx,
                          const size_t ny,
                                int    *out_of_bounds_device_ptr);

template <typename T>
void update_device(const int &rank,
                         T   *rng_states_device,
                   const std::array<int, 7> &indices_neighbors_parity,
                         int *local_lattice_device);
#endif


#endif

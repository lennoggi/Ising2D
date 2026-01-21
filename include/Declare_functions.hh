#ifndef DECLARE_FUNCTIONS_HH
#define DECLARE_FUNCTIONS_HH

#include <array>
#include <vector>
#include <random>

#include <hdf5.h>

#include "Declare_variables.hh"


std::array<int, 7>
set_indices_neighbors_parity(const int &rank,
                             const int &nprocs);

void update(const int                                    &rank,
                  std::mt19937                           &gen,
                  std::uniform_real_distribution<double> &dist,
            const std::array<int, 7>                     &indices_neighbors_parity,
                  std::array<int, nx1locp2_nx2locp2>     &local_lattice);

void calc_obs_corr(const int &rank,
                   const std::array<int, nx1locp2_nx2locp2> &local_lattice,
                   const hsize_t &n,
                   const int &x1index,
                   const int &x2index,
                   const int &rank_x1,
                   const int &rank_x2,
                   const MPI_Comm &comm_x1,
                   const MPI_Comm &comm_x2,
                   std::vector<int> &sums_x1_loc,
                   std::vector<int> &sums_x2_loc,
                   std::vector<int> &sums_x1_loc_reduced,
                   std::vector<int> &sums_x2_loc_reduced,
                   std::vector<int> &mag_energy_vec_int,
                   std::vector<int> &sums_x1_vec,
                   std::vector<int> &sums_x2_vec);

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

void set_int_device(const int   &rank,
                          int   *device_ptr,
                    const int   &value,
                   const size_t &num_elements);

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
                     const size_t &seed,
                           int    *error_flag_device_ptr);

template <typename T> __global__
void update_device_kernel(T   *rng_states_device,
                          int *local_lattice_device,
                          const size_t imin,
                          const size_t jmin,
                          const size_t nx,
                          const size_t ny,
                          const size_t ypitch,
                          const bool   update_odd_sites,
                                int    *out_of_bounds_device_ptr);

template <typename T>
void update_device(const int &rank,
                         T   *rng_states_device,
                   const std::array<int, 7> &indices_neighbors_parity,
                         int *local_lattice_device,
                         int *error_flag_device_ptr);

__global__
void calc_obs_corr_device_kernel(const int   *local_lattice_device,
                                 const size_t nx,
                                 const size_t ny,
                                 const size_t ypitch,
                                       int   *obs_loc_device,
                                       int   *sums_x1_loc_device,
                                       int   *sums_x2_loc_device,
                                       int   *error_flag_device_ptr);

void calc_obs_corr_device(const int &rank,
                          const int *local_lattice_device,
                          const hsize_t &n,
                          const int &x1index,
                          const int &x2index,
                          const int &rank_x1,
                          const int &rank_x2,
                          const MPI_Comm &comm_x1,
                          const MPI_Comm &comm_x2,
                                int *obs_loc_device,
                                int *sums_x1_loc_device,
                                int *sums_x2_loc_device,
                                int *sums_x1_loc_reduced_device,
                                int *sums_x2_loc_reduced_device,
                                int *mag_energy_vec_int_device,
                                int *sums_x1_vec_device,
                                int *sums_x2_vec_device,
                                int *error_flag_device_ptr);

template<typename T_in, typename T_out> __global__
void cast_and_scale_two_device_kernel(const T_in  *v_in,
                                            T_out *v1_out,
                                            T_out *v2_out,
                                      const T_out  a1,
                                      const T_out  a2,
                                      const size_t vout_size);

template<typename T_in, typename T_out>
void cast_and_scale_two_device(const int    &rank,
                               const T_in   *v_in,
                                     T_out  *v1_out,
                                     T_out  *v2_out,
                               const T_out  &a1,
                               const T_out  &a2,
                               const size_t &vout_size);
#endif


#endif

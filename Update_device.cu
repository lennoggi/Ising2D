#include <cmath>  // Supports calling math functions from within CUDA kernels
#include <array>
#include <curand_kernel.h>

#include "include/Declare_variables.hh"
#include "include/Declare_functions.hh"
#include "include/Macros.hh"

#include "Parameters.hh"

using namespace std;



/* ============================================================================
 * Kernel updating the process-local lattice by first sweeping only over "red"
 * sites, whose neighbors are all "black", and then sweeping only over "black"
 * sites, whose neighbors are all "red"
 * ============================================================================ */
template <typename T> __global__
void update_device_kernel(T   *rng_states_device,
                          int *local_lattice_device,
                          const int    color,  // NOTE: MUST BE 0 or 1, but the kernel doesn't check!
                          const size_t nx,
                          const size_t ny,
                                int    *out_of_bounds_device_ptr)
{
    constexpr auto _2beta = 2.*BETA;

    const auto i = blockIdx.x*blockDim.x + threadIdx.x;
    const auto j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i >= nx or i >= ny) {
        // Capture out-of-bounds error via an error flag
        atomicExch(out_of_bounds_device_ptr, 1);
        return;
    }

    /* NOTE: i and j above (the thread indices) are calculated wrt the INTERIOR
     *   of the process-local lattice size, so we need to add 1 to both of them
     *   to index the process-local lattice, which includes the ghosts          */
    if ((i + j) & 1 == color) {
        const auto ij  = i*ny + j;
    
        const auto i_lat = i + 1;
        const auto j_lat = j + 1;
    
        const auto ny_lat = ny + 1;
    
        const auto ij_lat = i_lat*ny_lat + j_lat;
    
        const auto ipj_lat = ij_lat + ny_lat;    // (i_lat + 1)*ny_lat + j_lat
        const auto imj_lat = ij_lat - ny_lat;    // (i_lat - 1)*ny_lat + j_lat
    
        const auto ijp_lat = ij_lat + 1;  // i_lat*ny_lat + (j_lat + 1) 
        const auto ijm_lat = ij_lat - 1;  // i_lat*ny_lat + (j_lat - 1) 

        const auto   f    = -(local_lattice_device[ipj_lat] + local_lattice_device[imj_lat] + local_lattice_device[ijp_lat] + local_lattice_device[ijm_lat]);
        const double prob = 1./(1. + exp(_2beta*f));  // exp(-BETA*f)/(exp(-BETA*f) + exp(BETA*f))

        /* NOTE: rng_states_device only covers the interior, so it must be
         *       indexed using i and j                                          */
              auto   rng_state = rng_states_device[ij];
        const double trial     = curand_uniform_double(&rng_state);
        rng_states_device[ij]  = rng_state;  // Update the RNG state after curand_uniform_double() has modified it

        local_lattice_device[ij_lat] = (trial < prob) ? 1 : -1;
    }

    return;
}

template __global__ void
update_device_kernel<curandStatePhilox4_32_10_t>(curandStatePhilox4_32_10_t *rng_states_device,
                                                 int *local_lattice_device,
                                                 const int    color,
                                                 const size_t nx,
                                                 const size_t ny,
                                                       int    *out_of_bounds_device_ptr);



/* =========================================================
 * Wrapper routine around update_device_kernel() (see above)
 * ========================================================= */
template <typename T>
void update_device(const int &rank,
                         T   *rng_states_device,
                   const array<int, 6> &indices_neighbors,
                         int    *local_lattice_device,
                   const size_t &nx,
                   const size_t &ny,
                   const size_t &block_size_x,
                   const size_t &block_size_y) {
    // Exchange the ghost site between neighboring processes
    exchange_ghosts(rank, indices_neighbors, local_lattice_device);

    // Shape of the CUDA thread block
    dim3 block(block_size_x, block_size_y);

    // Shape of the CUDA block grid
    dim3 grid((nx + block.x - 1)/block.x,   // block.x == block_size_x
              (ny + block.y - 1)/block.y);  // block.y == block_size_y

    // No out-of-bounds errors to begin with
    int  out_of_bounds = 0;
    int *out_of_bounds_device_ptr = allocate_device<int>(rank, 1);
    copy_device<int>(rank, out_of_bounds_device_ptr, &out_of_bounds, 1, cudaMemcpyHostToDevice);

    // Launch the process-local lattice update kernel on "red" and "black" sites
    update_device_kernel<T><<<grid, block>>>(rng_states_device, local_lattice_device, 0, nx, ny, out_of_bounds_device_ptr);
    CHECK_ERROR_CUDA(rank, cudaGetLastError());  // Capture potential errors from the kernel
    CHECK_ERROR_CUDA(rank, cudaDeviceSynchronize());

    copy_device<int>(rank, &out_of_bounds, out_of_bounds_device_ptr, 1, cudaMemcpyDeviceToHost);

    if (out_of_bounds == 1) {
        ERROR(rank, "init_rng_device_kernel() returned out-of-bounds error (spin color = 0)");
    }

    out_of_bounds = 0;

    update_device_kernel<T><<<grid, block>>>(rng_states_device, local_lattice_device, 1, nx, ny, out_of_bounds_device_ptr);
    CHECK_ERROR_CUDA(rank, cudaGetLastError());  // Capture potential errors from the kernel
    CHECK_ERROR_CUDA(rank, cudaDeviceSynchronize());

    copy_device<int>(rank, &out_of_bounds, out_of_bounds_device_ptr, 1, cudaMemcpyDeviceToHost);
    free_device(rank, out_of_bounds_device_ptr);

    if (out_of_bounds == 1) {
        ERROR(rank, "init_rng_device_kernel() returned out-of-bounds error (spin color = 1)");
    }

    return;
}

template void
update_device<curandStatePhilox4_32_10_t>(const int &rank,
                                          curandStatePhilox4_32_10_t *rng_states_device,
                                          const array<int, 6>        &indices_neighbors,
                                                int    *local_lattice_device,
                                          const size_t &nx,
                                          const size_t &ny,
                                          const size_t &block_size_x,
                                          const size_t &block_size_y); 

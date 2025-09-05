#include <curand_kernel.h>
#include "include/Declare_functions.hh"
#include "include/Macros.hh"

#include "Parameters.hh"


/* =======================================================================
 * Actual kernel initializing the RNG on the device (i.e., wrapper routine
 * around curand_init())
 * ======================================================================= */
template <typename T> __global__
void init_rng_device_kernel(T *rng_states_device,
                            const size_t seed,
                            const size_t nx,  // nx1loc
                            const size_t ny,  // nx2loc
                                  int    *out_of_bounds_device_ptr) {
    const auto i = blockIdx.x*blockDim.x + threadIdx.x;
    const auto j = blockIdx.y*blockDim.y + threadIdx.y;

    /* Capture out-of-bounds errors via an error flag
     * NOTE: no need to check if i<0 or j<0 because nx and ny are size_t        */
    if (i >= nx or j >= ny) {
        atomicExch(out_of_bounds_device_ptr, 1);
        return;
    }

    const auto ij = i*ny + j;

    // 0 is the offset in the sequence of pseudo/quasi-random numbers generated
    curand_init(seed, ij, 0, &rng_states_device[ij]);  // curand_init() returns void

    return;
}

/* NOTE: explicit instantiation of init_rng_device_kernel() must appear BEFORE
 *   the kernel is used (used in init_rng_device() below)                       */
// TODO: add more RNG types
template __global__ void
init_rng_device_kernel<curandStatePhilox4_32_10_t>(curandStatePhilox4_32_10_t *rng_states_device,
                                                   const size_t seed,
                                                   const size_t nx,
                                                   const size_t ny,
                                                         int    *out_of_bounds_device_ptr);



/* ===========================================================
 * Wrapper routine around init_rng_device_kernel() (see above)
 * =========================================================== */
template <typename T>
void init_rng_device(const int &rank,
                     T *rng_states_device,
                     const size_t &seed) {
    // Shape of the CUDA thread block
    /* NOTE: launch the RNG kernel on the process-local lattice using a single
     *   block if the process-local lattice is small enough, or use multiple
     *   blocks of MAX_BLOCK_SIZE_X1*MAX_BLOCK_SIZE_X2 threads each otherwise   */
    constexpr int block_size_x1 = std::min(nx1loc, MAX_BLOCK_SIZE_X1);
    constexpr int block_size_x2 = std::min(nx2loc, MAX_BLOCK_SIZE_X2);
    dim3 block(block_size_x1, block_size_x2);

    // Shape of the CUDA block grid
    constexpr int grid_size_x1 = std::ceil(nx1loc/block_size_x1);
    constexpr int grid_size_x2 = std::ceil(nx2loc/block_size_x2);
    dim3 grid(grid_size_x1, grid_size_x2);
    //dim3 grid((nx1loc + block.x - 1)/block.x,   // block.x == block_size_x1
    //          (nx2loc + block.y - 1)/block.y);  // block.y == block_size_x2

    // No out-of-bounds errors to begin with
    int  out_of_bounds = 0;
    int *out_of_bounds_device_ptr = allocate_device<int>(rank, 1);
    copy_device<int>(rank, out_of_bounds_device_ptr, &out_of_bounds, 1, cudaMemcpyHostToDevice);

    init_rng_device_kernel<T><<<grid, block>>>(rng_states_device, seed, nx1loc, nx2loc, out_of_bounds_device_ptr);

    CHECK_ERROR_CUDA(rank, cudaGetLastError());  // Capture potential errors from the kernel
    CHECK_ERROR_CUDA(rank, cudaDeviceSynchronize());

    copy_device<int>(rank, &out_of_bounds, out_of_bounds_device_ptr, 1, cudaMemcpyDeviceToHost);
    free_device(rank, out_of_bounds_device_ptr);

    if (out_of_bounds == 1) {
        ERROR(rank, "init_rng_device_kernel() returned out-of-bounds error");
    }

    return;
}

// TODO: add more RNG types
template void
init_rng_device<curandStatePhilox4_32_10_t>(const int &rank,
                                            curandStatePhilox4_32_10_t *rng_states_device,
                                            const size_t &seed);

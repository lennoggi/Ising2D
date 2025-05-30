#include <curand_kernel.h>
#include "include/Declare_functions.hh"
#include "include/Macros.hh"


/* =======================================================================
 * Actual kernel initializing the RNG on the device (i.e., wrapper routine
 * around curand_init())
 * ======================================================================= */
template <typename T> __global__
void init_rng_device_kernel(T *rng_states_device,
                            const size_t seed,
                            const size_t nx,
                            const size_t ny,
                                  int    *out_of_bounds_device_ptr) {
    const auto i  = blockIdx.x*blockDim.x + threadIdx.x;
    const auto j  = blockIdx.y*blockDim.y + threadIdx.y;

    /* Safety step: make sure the threads don't access memory they shouldn't
     * NOTE: this should never happen if there aren't more threads than points
     *   in the process-local lattice along any given dimension, which we
     *   enforce in include/Check_parameters.hh
     * NOTE: we actually enforce that there are no more threads than HALF of the
     *   number of process-local lattice sites along each dimension because of
     *   the "red/black" checkerboard lattice update pattern                    */
    if (i >= nx or j >= ny) {
        // Capture out-of-bounds error via an error flag
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
                     const size_t &seed,
                     const size_t &nx,
                     const size_t &ny,
                     const size_t &block_size_x,
                     const size_t &block_size_y) {
    // Shape of the CUDA thread block
    dim3 block(block_size_x, block_size_y);

    // Shape of the CUDA block grid
    dim3 grid((nx + block.x - 1)/block.x,   // block.x == block_size_x
              (ny + block.y - 1)/block.y);  // block.y == block_size_y

    // No out-of-bounds errors to begin with
    int  out_of_bounds = 0;
    int *out_of_bounds_device_ptr = allocate_device<int>(rank, 1);
    copy_device<int>(rank, out_of_bounds_device_ptr, &out_of_bounds, 1, cudaMemcpyHostToDevice);

    init_rng_device_kernel<T><<<grid, block>>>(rng_states_device, seed, nx, ny, out_of_bounds_device_ptr);

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
                                            const size_t &seed,
                                            const size_t &nx,
                                            const size_t &ny,
                                            const size_t &block_size_x,
                                            const size_t &block_size_y);

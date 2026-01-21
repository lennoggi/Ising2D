#include <algorithm>  // For std::min
#include <stdexcept>

#include "include/Macros.hh"


/* =============================================================================
 * CUDA kernel casting an input array of type T_in and size 2*N into a couple of
 * arrays of type T_out and size N, assuming every other element in the input
 * array belongs to the same output array. Finally, the output arrays are scaled
 * by user-defined scalars.
 * ============================================================================= */
template<typename T_in, typename T_out> __global__
void cast_and_scale_two_device_kernel(const T_in  *v_in,    // Size: 2*vout_size
                                            T_out *v1_out,  // Size:   vout_size
                                            T_out *v2_out,  // Size:   vout_size
                                      const T_out  a1,
                                      const T_out  a2,
                                      const size_t vout_size) {
    /* **** Example ****
     * gridDim.x  = 3  =>  blockIdx.x  = {0,1,2}
     * blockDim.x = 4  =>  threadIdx.x = {0,1,2,3}    (Number of blocks in each grid)
     * Total number of threads = gridDim.x*blockDim.x = 4*3 = 12
     *
     * If the size of the output arrays is vout_size=20, then:
     *    Input index:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20  21  22  23  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39
     *   Thread index: T0 T0 T1 T1 T2 T2 T3 T3 T4 T4 T5 T5 T6 T6 T7 T7 T8 T8 T9 T9 T10 T10 T11 T11 T0  T0  T1  T1  T2  T2  T3  T3  T4  T4  T5  T5  T6  T6  T7  T7
     *   Output index: A0 B0 A1 B1 A2 B2 A3 B3 A4 B4 A5 B5 A6 B6 A7 B7 A8 B8 A9 B9 A10 B10 A11 B11 A12 B12 A13 B13 A14 B14 A15 B15 A16 B16 A17 B17 A18 B18 A19 B19 */
    const auto global_thread_idx = blockIdx.x*blockDim.x + threadIdx.x;  // Lies in {0, ..., 12} in the example above
    const auto nthreads_tot      = gridDim.x*blockDim.x;                 // Equals 12 in the example above

    /* NOTE: the 'i < vout_size' condition in the loop automatically guarantees no
     *   thread accesses any array element past the array's bounds              */ 
    for (auto i = decltype(vout_size){global_thread_idx}; i < vout_size; i += nthreads_tot) {
        const auto _2i = 2*i;
        v1_out[i] = a1*static_cast<T_out>(v_in[_2i]); 
        v2_out[i] = a2*static_cast<T_out>(v_in[_2i+1]); 
    }
}

template __global__ void
cast_and_scale_two_device_kernel<int, double>(const int    *v_in,    // Size: 2*vout_size
                                                    double *v1_out,  // Size:   vout_size
                                                    double *v2_out,  // Size:   vout_size
                                              const double  a1,
                                              const double  a2,
                                              const size_t  vout_size);



/* =====================================================================
 * Wrapper routine around cast_and_scale_two_device_kernel() (see above)
 * ===================================================================== */
template<typename T_in, typename T_out>
void cast_and_scale_two_device(const int    &rank,
                               const T_in   *v_in,    // Size: 2*vout_size
                                     T_out  *v1_out,  // Size:   vout_size
                                     T_out  *v2_out,  // Size:   vout_size
                               const T_out  &a1,
                               const T_out  &a2,
                               const size_t &vout_size) {
    if (vout_size == 0) {
        throw std::runtime_error("cast_and_scale_two_device(): can't have vout_size == 0");
    }

    const auto block_size = std::min(vout_size, static_cast<size_t>(256));
    const auto grid_size  = ((vout_size + block_size - 1)/block_size);

    cast_and_scale_two_device_kernel<T_in, T_out><<<static_cast<int>(grid_size), static_cast<int>(block_size)>>>(v_in, v1_out, v2_out, a1, a2, vout_size);

    CHECK_ERROR(rank, cudaGetLastError());
    CHECK_ERROR(rank, cudaDeviceSynchronize());

    return;
}

template void 
cast_and_scale_two_device<int, double>(const int    &rank,
                                       const int    *v_in,
                                             double *v1_out,
                                             double *v2_out,
                                       const double &a1,
                                       const double &a2,
                                       const size_t &vout_size);

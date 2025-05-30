/* ===========================================================================
 * This file contains wrappers around CUDA routines, which cannot be compiled
 * with regular C++ compilers. We could, in principle, compile everything with
 * nvcc, but not all of the modern C++ features are supported and the MPI
 * library must be linked manually.
 * NOTE: the nvc++ compiler understands CUDA routines and kernels, but for
 *       maximum portability all the CUDA-related routines are here, and this
 *       file must be compiled using nvcc.
 * =========================================================================== */
#include <curand_kernel.h>
#include "include/Declare_functions.hh"
#include "include/Macros.hh"

using namespace std;


/* ===================================
 * Wrapper routine around cudaMalloc()
 * =================================== */
template <typename T>
T *allocate_device(const int    &rank,
                   const size_t &num_elements) {
    T *device_ptr;
    CHECK_ERROR_CUDA(rank, cudaMalloc(reinterpret_cast<void**>(&device_ptr), num_elements*sizeof(T)));
    return device_ptr;
}

template int*
allocate_device<int>(const int    &rank,
                     const size_t &size);

template curandStatePhilox4_32_10_t*
allocate_device<curandStatePhilox4_32_10_t>(const int    &rank,
                                            const size_t &size);


/* ===================================
 * Wrapper routine around cudaMemcpy()
 * =================================== */
template <typename T>
void copy_device(const int    &rank,
                       T      *dest,
                       T      *src,
                 const size_t &num_elements,
                 const cudaMemcpyKind &copy_kind) {
    CHECK_ERROR_CUDA(rank, cudaMemcpy(reinterpret_cast<void*>(dest), reinterpret_cast<void*>(src),
                                      num_elements*sizeof(T), copy_kind));
    return;
}

template void
copy_device<int>(const int    &rank,
                       int    *dest,
                       int    *src,
                 const size_t &num_elements,
                 const cudaMemcpyKind &copy_kind);



/* =====================================
 * Wrapper routine around cudaMemcpy2D()
 * ===================================== */
template <typename T>
void copy_device_2D(const int    &rank,
                          T      *dest,
                    const size_t &dest_stride,  // Number of elements between successive elements in the destination memory chunk. Usually set to 1 if contiguous or to n_columns if not.
                          T      *src,
                    const size_t &src_stride,   // Number of elements between successive elements in the source memory chunk. Usually set to 1 if contiguous or to n_columns if not.
                    const size_t &width,        // Number of elements per row to copy
                    const size_t &height,       // Number of rows to copy
                    const cudaMemcpyKind &copy_kind) {
    CHECK_ERROR_CUDA(rank, cudaMemcpy2D(reinterpret_cast<void*>(dest), dest_stride*sizeof(T),
                                        reinterpret_cast<void*>(src),   src_stride*sizeof(T),
                                        width*sizeof(T), height, copy_kind));
    return;
}

template void
copy_device_2D<int>(const int    &rank,
                          int    *dest,
                    const size_t &dest_stride,
                          int    *src,
                    const size_t &src_stride,
                    const size_t &width,
                    const size_t &height,
                    const cudaMemcpyKind &copy_kind);



/* =================================
 * Wrapper routine around cudaFree()
 * ================================= */
void free_device(const int  &rank,
                       void *device_ptr) {
    CHECK_ERROR_CUDA(rank, cudaFree(device_ptr));
    return;
}

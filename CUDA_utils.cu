#include "include/Macros.hh"


/* ================================================
 * Wrapper routine around cudaMalloc() for integers
 * ================================================ */
int *allocate_int_device(const int    &rank,
                         const size_t &size) {
    int *device_ptr;
    CHECK_ERROR_CUDA(rank, cudaMalloc((void**) &device_ptr, size));
    return device_ptr;
}


/* ===================================
 * Wrapper routine around cudaMemcpy()
 * =================================== */
void copy_device(const int            &rank,
                       void           *dest,
                 const void           *src,
                 const size_t         &size,
                 const cudaMemcpyKind &kind) {
    CHECK_ERROR_CUDA(rank, cudaMemcpy(dest, src, size, kind));
    return;
}


/* =================================
 * Wrapper routine around cudaFree()
 * ================================= */
void free_device(const int  &rank,
                       void *device_ptr) {
    CHECK_ERROR_CUDA(rank, cudaFree(device_ptr));
    return;
}

#include <cmath>  // Supports calling math functions from within CUDA kernels
#include <array>
#include <curand_kernel.h>

#include "include/Declare_variables.hh"
#include "include/Declare_functions.hh"
#include "include/Macros.hh"

#include "Parameters.hh"

using namespace std;



/* ===========================================================================
 * Kernel updating a quarter of the process-local lattice (however, the kernel
 * only knows it has to update a grid of points)
 * =========================================================================== */
template <typename T> __global__
void update_device_kernel(T      *rng_states_device,
                          int *local_lattice_device,
                          const size_t imin,
                          const size_t jmin,
                          const size_t nx,      // nx1loc_div2
                          const size_t ny,      // nx2loc_div4
                          const size_t ypitch,  // nx2loc_p2
                          const bool   update_odd_sites,
                                int    *error_flag_device_ptr)
{
    const auto i_thread = blockIdx.x*blockDim.x + threadIdx.x;
    const auto j_thread = blockIdx.y*blockDim.y + threadIdx.y;

    /* Capture errors via an error flag
     * NOTE: no need to check if i<0 or j<0 because nx and ny are size_t        */
    if (i_thread >= nx or j_thread >= ny or
        imin < 1 or jmin < 1 or
        ypitch <= 2)
    {
        atomicExch(error_flag_device_ptr, 1);
        return;
    }

    const auto iloc =   i_thread;
    const auto jloc = 2*j_thread + ((iloc + static_cast<int>(update_odd_sites)) & 1);

    const auto i = imin + iloc;
    const auto j = jmin + jloc;

    const auto ij  = i*ypitch + j;
    const auto ipj = ij + ypitch;  // (i+1)*ypitch + j
    const auto imj = ij - ypitch;  // (i-1)*ypitch + j
    const auto ijp = ij + 1;       // i*ypitch + (j+1)
    const auto ijm = ij - 1;       // i*ypitch + (j-1)

    const auto f = -(  local_lattice_device[ipj]
                     + local_lattice_device[imj]
                     + local_lattice_device[ijp]
                     + local_lattice_device[ijm]);
    constexpr auto _2beta = 2.*BETA;
    const double    prob  = 1./(1. + exp(_2beta*f));  // exp(-BETA*f)/(exp(-BETA*f) + exp(BETA*f))

    /* NOTE: rng_states_device only covers the interior, so it must be
     *       indexed using i-1, j-1 with ypitch-2(=nx2loc) columns              */
    const auto   idx_rng   = (i-1)*(ypitch-2) + j-1;
          auto   rng_state = rng_states_device[idx_rng];
    const double trial     = curand_uniform_double(&rng_state);

    rng_states_device[idx_rng] = rng_state;  // Update the RNG state after curand_uniform_double() has modified it
    local_lattice_device[ij]   = (trial < prob) ? 1 : -1;

    return;
}

template __global__ void
update_device_kernel<curandStatePhilox4_32_10_t>(curandStatePhilox4_32_10_t *rng_states_device,
                                                       int                  *local_lattice_device,
                                                 const size_t imin,
                                                 const size_t jmin,
                                                 const size_t nx,
                                                 const size_t ny,
                                                 const size_t ypitch,
                                                 const bool   update_odd_sites,
                                                       int    *error_flag_device_ptr);



/* =========================================================
 * Wrapper routine around update_device_kernel() (see above)
 * ========================================================= */
template <typename T>
void update_device(const int &rank,
                         T   *rng_states_device,
                   const array<int, 7> &indices_neighbors_parity,
                         int *local_lattice_device,
                         int *error_flag_device_ptr) {
    const auto &[x1index, x2index, x1down, x1up, x2down, x2up, parity] = indices_neighbors_parity;

    /* Arrays encoding which processes rows (i.e. x data chunks) and columns
     * (i.e. y data chunks) should be sent to and received from. The
     * process-local lattice is assumed to be split in four parts like this:
     *   0  1
     *   2  3
     * and element i in each array corresponds to the ith part of the process.
     * NOTE: think of the full grid e.g. as:
     *        |----------------|
     *   ^  3 | 15 16 17 18 19 |  Example with NPROCS_X1=4, NPROCS_X2=5
     *   |  2 | 10 11 12 13 14 |
     *   x1 1 | 5  6  7  8  9  |
     *      0 | 0  1  2  3  4  |
     *        |----------------|
     *          0  1  2  3  4
     *                  x2 ->
     * with both x1 and x2 being periodic (torus topology)                      */
    const array<int, 4> x1send = {x1up,   x1up,   x1down, x1down};
    const array<int, 4> x1recv = {x1down, x1down, x1up,   x1up};
    const array<int, 4> x2send = {x2down, x2up,   x2down, x2up};
    const array<int, 4> x2recv = {x2up,   x2down, x2up,   x2down};

    // Communication tags
    const int tag1 = 1;
    const int tag2 = 2;
    const int tag3 = 3;
    const int tag4 = 4;


    /* Update the process-local lattice one quarter at a time to make sure the
     * boundary spins see updated ghosts. In other words, if a boundary spin is
     * updated on one process, its neighbors in the neighboring process need to
     * know, so the updated spin must be copied to the appropriate ghost site in
     * the neighboring process. On top of that, inter-process communications
     * should be limited as much as possible, so we exchange arrays instead of
     * individual spins.                                                        */
    int count = 0;

    for (int kx1 = 0; kx1 < 2; ++kx1) {
        const auto sx1  = kx1*nx1loc_div2;
        const auto imin = sx1 + 1;
        //const auto imax = imin + nx1loc_div2;
        const auto idx_imin = imin*nx2loc_p2;

        const auto isend = (kx1 == 0) ? 1 : nx1loc;
        const auto irecv = (kx1 == 0) ? nx1loc_p1 : 0;

        const auto isend_idx = isend*nx2loc_p2;
        const auto irecv_idx = irecv*nx2loc_p2;

        for (int kx2 = 0; kx2 < 2; ++kx2) {
            const auto sx2  = kx2*nx2loc_div2;
            const auto jmin = sx2 + 1;
            //const int jmax = jmin + nx2loc_div2;

            /* -------------------------------------------------------
             * Update the current quarter of the process-local lattice
             * ------------------------------------------------------- */
            /* Shape of the CUDA thread block
             * NOTE: launch the lattice update kernel on each quarter of the
             *   process-local lattice, but in a 'checkerboard' fashion' in
             *   order to preserve detailed balance; note that this means
             *   updating HALF of the spins on the process-local lattice within
             *   each kernel. Do it on a single block if the block is small
             *   enough, or use multiple blocks of
                 MAX_BLOCK_SIZE_X1*MAX_BLOCK_SIZE_X2 threads each otherwise.    */
            constexpr int block_size_x1 = std::min(nx1loc_div2, MAX_BLOCK_SIZE_X1);
            constexpr int block_size_x2 = std::min(nx2loc_div4, MAX_BLOCK_SIZE_X2);
            dim3 block(block_size_x1, block_size_x2);

            // Shape of the CUDA block grid
            dim3 grid(static_cast<int>((nx1loc_div2 + block.x - 1)/block.x),   // block.x == block_size_x1
                      static_cast<int>((nx2loc_div4 + block.y - 1)/block.y));  // block.y == block_size_x2

            set_int_device(rank, error_flag_device_ptr, 0, 1);  // Set one element to 0


            /* Update every other site---so that all four neighbors either have
             * or haven't been updated---to preserve detailed balance.
               Begin, e.g., from even sites.                                    */
            bool update_odd_sites = false;
            update_device_kernel<<<grid, block>>>(rng_states_device, local_lattice_device,
                                                  imin, jmin,                // Start indices
                                                  nx1loc_div2, nx2loc_div4,  // Row and column extents
                                                  nx2loc_p2,                 // Column pitch
                                                  update_odd_sites,
                                                  error_flag_device_ptr);

            CHECK_ERROR_CUDA(rank, cudaGetLastError());
            CHECK_ERROR_CUDA(rank, cudaDeviceSynchronize());

            int out_of_bounds = 0;
            copy_device<int>(rank, &out_of_bounds, error_flag_device_ptr, 1, cudaMemcpyDeviceToHost);

            if (out_of_bounds != 0) {
                ERROR(rank, "update_device_kernel() returned out-of-bounds error (quarter ("
                         << kx1 << ", " << kx2 << ") of the process-local lattice)");
            }

            update_odd_sites = true;
            update_device_kernel<<<grid, block>>>(rng_states_device, local_lattice_device,
                                                  imin, jmin,                // Start indices
                                                  nx1loc_div2, nx2loc_div4,  // Row and column extents
                                                  nx2loc_p2,                 // Column pitch
                                                  update_odd_sites,
                                                  error_flag_device_ptr);

            CHECK_ERROR_CUDA(rank, cudaGetLastError());
            CHECK_ERROR_CUDA(rank, cudaDeviceSynchronize());

            out_of_bounds = 0;
            copy_device<int>(rank, &out_of_bounds, error_flag_device_ptr, 1, cudaMemcpyDeviceToHost);

            if (out_of_bounds != 0) {
                ERROR(rank, "update_device_kernel() returned out-of-bounds error (quarter ("
                         << kx1 << ", " << kx2 << ") of the process-local lattice)");
            }



            /* ---------------
             * Exchange ghosts
             * --------------- */
            // Helper variables for communication
            const auto jsend = (kx2 == 0) ? 1 : nx2loc;
            const auto jrecv = (kx2 == 0) ? nx2loc_p1 : 0;

            const auto isend_idx_psx2_p1 = isend_idx + jmin;  // isend_idx + sx2 + 1
            const auto irecv_idx_psx2_p1 = irecv_idx + jmin;  // irecv_idx + sx2 + 1

            /* Set up the column chunks to be sent out
             * NOTE: no need to copy the row data to a separate buffer, since
             *       all the spins along a given row are contiguous in memory   */
            int *x2out_device = allocate_device<int>(rank, nx1loc_div2);
            int *x2in_device  = allocate_device<int>(rank, nx1loc_div2);

            copy_device_2D<int>(rank,
                // Destination: 1 element between successive elements (row vector)
                 x2out_device, 1,
                // Source: first element has [i,j] = [imin, jsend], and there are nx2loc_p2 elements between successive elements in the column
                &local_lattice_device[idx_imin + jsend], nx2loc_p2,
                // Copy 1 elements per row, nx1loc_div2 rows (i.e., copy a column of local_lattice_device)
                1, nx1loc_div2,
                // NOTE: using a GPU buffer makes the copy faster, but assumes a GPU-aware MPI implementation (see below)
                cudaMemcpyDeviceToDevice);

            /* Exchange the current quarter's ghosts
             * NOTE: this assumes the MPI implementation in use is GPU-aware,
             *       i.e., MPI_Send() and MPI_Recv() must be able to handle GPU
             *       pointers  */
            if (parity) {
                CHECK_ERROR(rank, MPI_Send(&local_lattice_device[isend_idx_psx2_p1], nx2loc_div2, MPI_INT, x1send.at(count), tag1, MPI_COMM_WORLD));
                CHECK_ERROR(rank, MPI_Recv(&local_lattice_device[irecv_idx_psx2_p1], nx2loc_div2, MPI_INT, x1recv.at(count), tag2, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                CHECK_ERROR(rank, MPI_Send(x2out_device,                             nx1loc_div2, MPI_INT, x2send.at(count), tag3, MPI_COMM_WORLD));
                CHECK_ERROR(rank, MPI_Recv(x2in_device,                              nx1loc_div2, MPI_INT, x2recv.at(count), tag4, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            } else {
                CHECK_ERROR(rank, MPI_Recv(&local_lattice_device[irecv_idx_psx2_p1], nx2loc_div2, MPI_INT, x1recv.at(count), tag1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                CHECK_ERROR(rank, MPI_Send(&local_lattice_device[isend_idx_psx2_p1], nx2loc_div2, MPI_INT, x1send.at(count), tag2, MPI_COMM_WORLD));
                CHECK_ERROR(rank, MPI_Recv(x2in_device,                              nx1loc_div2, MPI_INT, x2recv.at(count), tag3, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                CHECK_ERROR(rank, MPI_Send(x2out_device,                             nx1loc_div2, MPI_INT, x2send.at(count), tag4, MPI_COMM_WORLD));
            }

            // Store the column chunk received into the ghost column
            copy_device_2D<int>(rank,
                // Destination: first element has [i,j] = [imin, jrecv], and there are nx2loc_p2 elements between successive elements in the column
                &local_lattice_device[idx_imin + jrecv], nx2loc_p2,
                // Source: 1 element between successive elements (row vector)
                 x2in_device, 1,
                // CudaMemcpy2D() treats x2in_device as a 2D buffer (column vector), so we still copy 1 element per row and nx1loc_div2 rows
                1, nx1loc_div2,
                // NOTE: using a GPU buffer makes the copy faster, but assumes a GPU-aware MPI implementation (see above)
                cudaMemcpyDeviceToDevice);

            free_device(rank, x2out_device);
            free_device(rank, x2in_device);

            // Move to the next quarter
            ++count;
        }
    }

    return;
}



template void
update_device<curandStatePhilox4_32_10_t>(const int &rank,
                                          curandStatePhilox4_32_10_t *rng_states_device,
                                          const array<int, 7> &indices_neighbors_parity,
                                                int *local_lattice_device,
                                                int *error_flag_device_ptr);

#include <cmath>  // Supports calling math functions from within CUDA kernels
#include <array>
#include <curand_kernel.h>

#include "include/Declare_variables.hh"
#include "include/Declare_functions.hh"
#include "include/Macros.hh"

#include "Parameters.hh"

using namespace std;



/* ========================================================================
 * Kernel updating the a quarter of the process-local lattice (however, the
 * kernel only knows it has to update a grid of points)
 * ======================================================================== */
template <typename T> __global__
void update_device_kernel(T      *rng_states_device_quarter,
                          int *local_lattice_device_quarter,
                          const size_t nx,  // nx1loc_half
                          const size_t ny,  // nx1loc_half
                                int    *out_of_bounds_device_ptr) {
    const auto i = blockIdx.x*blockDim.x + threadIdx.x;
    const auto j = blockIdx.y*blockDim.y + threadIdx.y;

    /* Capture out-of-bounds errors via an error flag
     * NOTE: no need to check if i<0 or j<0 because nx and ny are size_t        */
    if (i >= nx or j >= ny) {
        atomicExch(out_of_bounds_device_ptr, 1);
        return;
    }

    const auto ij  = i*ny + j;
    const auto ipj = ij + ny;  // (i+1)*ny + j
    const auto imj = ij - ny;  // (i-1)*ny + j
    const auto ijp = ij + 1;   // i*ny + (j+1)
    const auto ijm = ij - 1;   // i*ny + (j-1)

    const auto f = -(  local_lattice_device_quarter[ipj]
                       + local_lattice_device_quarter[imj]
                       + local_lattice_device_quarter[ijp]
                       + local_lattice_device_quarter[ijm]);
    constexpr auto _2beta = 2.*BETA;
    const double    prob  = 1./(1. + exp(_2beta*f));  // exp(-BETA*f)/(exp(-BETA*f) + exp(BETA*f))

    /* NOTE: rng_states_device only covers the interior, so it must be
     *       indexed using i and j                                          */
          auto   rng_state        = rng_states_device_quarter[ij];
    const double trial            = curand_uniform_double(&rng_state);
    rng_states_device_quarter[ij] = rng_state;  // Update the RNG state after curand_uniform_double() has modified it

    local_lattice_device_quarter[ij] = (trial < prob) ? 1 : -1;

    return;
}

template __global__ void
update_device_kernel<curandStatePhilox4_32_10_t>(curandStatePhilox4_32_10_t *rng_states_device_quarter,
                                                       int                  *local_lattice_device_quarter,
                                                 const size_t nx,
                                                 const size_t ny,
                                                       int    *out_of_bounds_device_ptr);



/* =========================================================
 * Wrapper routine around update_device_kernel() (see above)
 * ========================================================= */
template <typename T>
void update_device(const int &rank,
                         T   *rng_states_device,
                   const array<int, 7> &indices_neighbors_parity,
                         int *local_lattice_device) {
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

    /* Buffers needed to store the RNG states and the lattice data for one
     * quarter of the process-local lattice                                     */
    T   *rng_states_device_quarter    = allocate_device<T>  (rank, nx1lochalf_nx2lochalf);
    int *local_lattice_device_quarter = allocate_device<int>(rank, nx1lochalf_nx2lochalf);

    for (int kx1 = 0; kx1 < 2; ++kx1) {
        const auto sx1  = kx1*nx1loc_half;
        //const auto imin = sx1 + 1;  // XXX: not needed
        //const auto imax = imin + nx1loc_half;

        const auto isend = (kx1 == 0) ? 1 : nx1loc;
        const auto irecv = (kx1 == 0) ? nx1loc_p1 : 0;

        const auto isend_idx = isend*nx2loc_p2;
        const auto irecv_idx = irecv*nx2loc_p2;

        for (int kx2 = 0; kx2 < 2; ++kx2) {
            const int sx2  = kx2*nx2loc_half;
            const int jmin = sx2 + 1;
            //const int jmax = jmin + nx2loc_half;  // XXX: not needed

            /* -------------------------------------------------------
             * Update the current quarter of the process-local lattice
             * ------------------------------------------------------- */
            /* Shape of the CUDA thread block
             * NOTE: launch the lattice update kernel on each quarter of the
                 process-local lattice using a single block if the quarter is
                 small enough, or use multiple blocks of
                 MAX_BLOCK_SIZE_X1*MAX_BLOCK_SIZE_X2 threads each otherwise     */
            constexpr int block_size_x1_quarter = std::min(nx1loc_half, MAX_BLOCK_SIZE_X1);
            constexpr int block_size_x2_quarter = std::min(nx2loc_half, MAX_BLOCK_SIZE_X2);
            dim3 block(block_size_x1_quarter, block_size_x2_quarter);

            // Shape of the CUDA block grid
            constexpr int grid_size_x1_quarter = std::ceil(nx1loc_half/block_size_x1_quarter);
            constexpr int grid_size_x2_quarter = std::ceil(nx2loc_half/block_size_x2_quarter);
            dim3 grid(grid_size_x1_quarter, grid_size_x2_quarter);
            //dim3 grid((nx1loc_half + block.x - 1)/block.x,   // block.x == block_size_x1_quarter
            //          (nx2loc_half + block.y - 1)/block.y);  // block.y == block_size_x2_quarter

            // No out-of-bounds errors to begin with
            int  out_of_bounds = 0;
            int *out_of_bounds_device_ptr = allocate_device<int>(rank, 1);
            copy_device<int>(rank, out_of_bounds_device_ptr, &out_of_bounds, 1, cudaMemcpyHostToDevice);


            /* Pack the current quarter's data for 'rng_states_device' and
             * 'local_lattice_device' into temporary buffers and launch the
             * update kernel on the current quarter                             */
            for (auto i = decltype(nx1loc_half){1}; i <= nx1loc_half; ++i) {
                const auto i_idx_quarter_m1 =     (i-1)*nx2loc_p2 - 1;
                const auto i_idx_full_p_sx2 = (i + sx1)*nx2loc_p2 + sx2;

                for (auto j = decltype(nx2loc_half){1}; j <= nx2loc_half; ++j) {
                    const auto idx_quarter = i_idx_quarter_m1 + j;  //     (i-1)*nx2loc_p2 + (j-1) 
                    const auto idx_full    = i_idx_full_p_sx2 + j;  // (i + sx1)*nx2loc_p2 + (j + sx2)

                       rng_states_device_quarter[idx_quarter] =    rng_states_device[idx_full];
                    local_lattice_device_quarter[idx_quarter] = local_lattice_device[idx_full];
                }
            }

            update_device_kernel<<<grid, block>>>(rng_states_device_quarter, local_lattice_device_quarter, nx1loc_half, nx2loc_half, out_of_bounds_device_ptr);


            // Capture potential errors from the kernel
            CHECK_ERROR_CUDA(rank, cudaGetLastError());
            CHECK_ERROR_CUDA(rank, cudaDeviceSynchronize());

            copy_device<int>(rank, &out_of_bounds, out_of_bounds_device_ptr, 1, cudaMemcpyDeviceToHost);
            free_device(rank, out_of_bounds_device_ptr);

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
            array<int, nx1loc_half> x2out, x2in;

            for (auto i = decltype(nx1loc_half){1}; i <= nx1loc_half; ++i) {
                x2out.at(i-1) = local_lattice_device[(i + sx1)*nx2loc_p2 + jsend];  // local_lattice_device[i + sx1][jsend]
            }

            // Exchange the current quarter's ghosts
            if (parity) {
                CHECK_ERROR(rank, MPI_Send(&local_lattice_device[isend_idx_psx2_p1], nx2loc_half, MPI_INT, x1send.at(count), tag1, MPI_COMM_WORLD));
                CHECK_ERROR(rank, MPI_Recv(&local_lattice_device[irecv_idx_psx2_p1], nx2loc_half, MPI_INT, x1recv.at(count), tag2, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                CHECK_ERROR(rank, MPI_Send(x2out.data(),                             nx1loc_half, MPI_INT, x2send.at(count), tag3, MPI_COMM_WORLD));
                CHECK_ERROR(rank, MPI_Recv(x2in.data(),                              nx1loc_half, MPI_INT, x2recv.at(count), tag4, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            } else {
                CHECK_ERROR(rank, MPI_Recv(&local_lattice_device[irecv_idx_psx2_p1], nx2loc_half, MPI_INT, x1recv.at(count), tag1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                CHECK_ERROR(rank, MPI_Send(&local_lattice_device[isend_idx_psx2_p1], nx2loc_half, MPI_INT, x1send.at(count), tag2, MPI_COMM_WORLD));
                CHECK_ERROR(rank, MPI_Recv(x2in.data(),                              nx1loc_half, MPI_INT, x2recv.at(count), tag3, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                CHECK_ERROR(rank, MPI_Send(x2out.data(),                             nx1loc_half, MPI_INT, x2send.at(count), tag4, MPI_COMM_WORLD));
            }

            // Store the column chunk received into the ghost column
            for (auto i = decltype(nx1loc_half){1}; i <= nx1loc_half; ++i) {
                local_lattice_device[(i + sx1)*nx2loc_p2 + jrecv] = x2in.at(i-1);  // local_lattice[i + sx1][jrecv]
            }

            // Move to the next quarter
            ++count;
        }
    }

    free_device(rank, rng_states_device_quarter);
    free_device(rank, local_lattice_device_quarter);

    return;
}



template void
update_device<curandStatePhilox4_32_10_t>(const int &rank,
                                          curandStatePhilox4_32_10_t *rng_states_device,
                                          const array<int, 7> &indices_neighbors_parity,
                                                int    *local_lattice_device);

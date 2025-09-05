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
void update_device_kernel(T      *rng_states_device_quarter,
                          int *local_lattice_device_quarter,
                          const size_t nx,
                          const size_t ny,
                                int    *out_of_bounds_device_ptr)
{
    constexpr auto _2beta = 2.*BETA;

    const auto i = blockIdx.x*blockDim.x + threadIdx.x;
    const auto j = blockIdx.y*blockDim.y + threadIdx.y;

    // Capture out-of-bounds error via an error flag
    if (i < 0 or i >= nx or
        j < 0 or j >= ny)
    {
        atomicExch(out_of_bounds_device_ptr, 1);
        return;
    }

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
                   const array<int, 7> &indices_neighbors_parity,
                         int    *local_lattice_device,
                   const size_t &nx,
                   const size_t &ny,
                   const size_t &block_size_x,
                   const size_t &block_size_y) {
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
        const auto imin = sx1 + 1;
        const auto imax = imin + nx1loc_half;

        const auto isend = (kx1 == 0) ? 1 : nx1loc;
        const auto irecv = (kx1 == 0) ? nx1loc_p1 : 0;

        const auto isend_idx = isend*nx2loc_p2;
        const auto irecv_idx = irecv*nx2loc_p2;

        for (int kx2 = 0; kx2 < 2; ++kx2) {
            const int sx2  = kx2*nx2loc_half;
            const int jmin = sx2 + 1;
            const int jmax = jmin + nx2loc_half;

            /* -------------------------------------------------------
             * Update the current quarter of the process-local lattice
             * ------------------------------------------------------- */
            /* Shape of the CUDA thread block
             * NOTE: launch the lattice update kernel on each quarter of the
                 process-local lattice using a single block if the quarter is
                 small enough, or use multiple blocks of
                 MAX_BLOCK_SIZE_X1*MAX_BLOCK_SIZE_X2 threads each otherwise     */
            constexpr inline int block_size_x1 = std::min(nx1loc_half, MAX_BLOCK_SIZE_X1);
            constexpr inline int block_size_x2 = std::min(nx2loc_half, MAX_BLOCK_SIZE_X2);
            dim3 block(block_size_x1, block_size_x2);

            // Shape of the CUDA block grid
            constexpr inline int grid_size_x1 = std::ceil(nx1loc_half/block_size_x1);
            constexpr inline int grid_size_x2 = std::ceil(nx2loc_half/block_size_x2);
            dim3 grid(grid_size_x1, grid_size_x2);
            //dim3 grid((nx1loc_half + block.x - 1)/block.x,   // block.x == block_size_x1
            //          (nx2loc_half + block.y - 1)/block.y);  // block.y == block_size_x2

            // No out-of-bounds errors to begin with
            int  out_of_bounds = 0;
            int *out_of_bounds_device_ptr = allocate_device<int>(rank, 1);
            copy_device<int>(rank, out_of_bounds_device_ptr, &out_of_bounds, 1, cudaMemcpyHostToDevice);






            // TODO TODO TODO TODO TODO TODO
            // TODO TODO TODO TODO TODO TODO
            // TODO TODO TODO TODO TODO TODO
            // TODO: FROM HERE
            /* Pack the current quarter's data for 'rng_states_device' and
             * 'local_lattice_device' into temporary buffers and launch the
             * update kernel on the current quarter                             */
            // TODO TODO TODO TODO TODO TODO
            // TODO TODO TODO TODO TODO TODO
            // TODO TODO TODO TODO TODO TODO
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


















            update_device_kernel<T><<<grid, block>>>(rng_states_device, local_lattice_device, nx, ny, out_of_bounds_device_ptr);
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
                x2out.at(i-1) = local_lattice.at((i + sx1)*nx2loc_p2 + jsend);  // local_lattice[i + sx1][jsend]
            }

            // Exchange the current quarter's ghosts
            if (parity) {
                CHECK_ERROR(rank, MPI_Send(&local_lattice.at(isend_idx_psx2_p1), nx2loc_half, MPI_INT, x1send.at(count), tag1, MPI_COMM_WORLD));
                CHECK_ERROR(rank, MPI_Recv(&local_lattice.at(irecv_idx_psx2_p1), nx2loc_half, MPI_INT, x1recv.at(count), tag2, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                CHECK_ERROR(rank, MPI_Send(x2out.data(),                         nx1loc_half, MPI_INT, x2send.at(count), tag3, MPI_COMM_WORLD));
                CHECK_ERROR(rank, MPI_Recv(x2in.data(),                          nx1loc_half, MPI_INT, x2recv.at(count), tag4, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            } else {
                CHECK_ERROR(rank, MPI_Recv(&local_lattice.at(irecv_idx_psx2_p1), nx2loc_half, MPI_INT, x1recv.at(count), tag1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                CHECK_ERROR(rank, MPI_Send(&local_lattice.at(isend_idx_psx2_p1), nx2loc_half, MPI_INT, x1send.at(count), tag2, MPI_COMM_WORLD));
                CHECK_ERROR(rank, MPI_Recv(x2in.data(),                          nx1loc_half, MPI_INT, x2recv.at(count), tag3, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                CHECK_ERROR(rank, MPI_Send(x2out.data(),                         nx1loc_half, MPI_INT, x2send.at(count), tag4, MPI_COMM_WORLD));
            }

            // Store the column chunk received into the ghost column
            for (auto i = decltype(nx1loc_half){1}; i <= nx1loc_half; ++i) {
                local_lattice.at((i + sx1)*nx2loc_p2 + jrecv) = x2in.at(i-1);  // local_lattice[i + sx1][jrecv]
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
                                          const array<int, 6>        &indices_neighbors,
                                                int    *local_lattice_device,
                                          const size_t &nx,
                                          const size_t &ny,
                                          const size_t &block_size_x,
                                          const size_t &block_size_y); 

#include <cassert>
#include <cstring>
#include <array>

#include <mpi.h>

#include "include/Declare_variables.hh"
#include "include/Declare_functions.hh"
#include "include/Macros.hh"
#include "Parameters.hh"

using namespace std;


/* ==========================================================================
 * Kernel calculating the magnetization, energy, and spin sums over rows and
 * columns (which are needed to compute the correlation function(s))
 * ========================================================================== */
__global__
void calc_obs_corr_device_kernel(const int   *local_lattice_device,
                                 const size_t nx,      // nx1loc
                                 const size_t ny,      // nx2loc
                                 const size_t ypitch,  // nx2loc_p2
                                       int   *obs_loc_device,
                                       int   *sums_x1_loc_device,
                                       int   *sums_x2_loc_device,
                                       int   *error_flag_device_ptr) {
    const auto i_thread = blockIdx.x*blockDim.x + threadIdx.x;
    const auto j_thread = blockIdx.y*blockDim.y + threadIdx.y;

    /* Capture errors via an error flag
     * NOTE: no need to check if i<0 or j<0 because nx and ny are size_t        */
    if (i_thread >= nx or j_thread >= ny or
        ypitch <= 2)
    {
        atomicExch(error_flag_device_ptr, 1);
        return;
    }

    // Skip the ghost layer
    const auto i = i_thread + 1;
    const auto j = j_thread + 1;

    const auto ij  = i*ypitch + j;
    const auto ipj = ij + ypitch;  // (i+1)*ypitch + j
    const auto imj = ij - ypitch;  // (i-1)*ypitch + j
    const auto ijp = ij + 1;       // i*ypitch + (j+1)
    const auto ijm = ij - 1;       // i*ypitch + (j-1)

    const auto f = -(  local_lattice_device[ipj]
                     + local_lattice_device[imj]
                     + local_lattice_device[ijp]
                     + local_lattice_device[ijm]);

    const auto local_lattice_device_ij = local_lattice_device[ij];

    atomicAdd(obs_loc_device,      local_lattice_device_ij);    // Magnetization
    atomicAdd(obs_loc_device + 1, -local_lattice_device_ij*f);  // Energy

    atomicAdd(sums_x1_loc_device + i_thread, local_lattice_device_ij);
    atomicAdd(sums_x2_loc_device + j_thread, local_lattice_device_ij);

    return;
}



/* ============================================================================
 * Wrapper routine around calc_obs_corr_device_kernel() (see above)
 * NOTE: this routine assumes the MPI implementation in use is GPU-aware, i.e.,
 *       MPI_Reduce() and MPI_Gather() must be able to handle GPU pointers
 * ============================================================================ */
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
                                int *error_flag_device_ptr) {
    // Shape of the CUDA thread block
    constexpr int block_size_x1 = std::min(nx1loc, MAX_BLOCK_SIZE_X1);
    constexpr int block_size_x2 = std::min(nx2loc, MAX_BLOCK_SIZE_X2);
    dim3 block(block_size_x1, block_size_x2);

    // Shape of the CUDA block grid
    dim3 grid((nx1loc + block.x - 1)/block.x,   // block.x == block_size_x1
              (nx2loc + block.y - 1)/block.y);  // block.y == block_size_x2


    // Calculate the magnetization, energy, and spin sums over x1 and x2
    set_int_device(rank, error_flag_device_ptr, 0, 1);    // Set one element  to 0
    set_int_device(rank, obs_loc_device, 0, 2);           // Set two elements to 0

    set_int_device(rank, sums_x1_loc_device, 0, nx1loc);  // Set nx1loc elements to 0
    set_int_device(rank, sums_x2_loc_device, 0, nx2loc);  // Set nx2loc elements to 0

    calc_obs_corr_device_kernel<<<grid, block>>>(local_lattice_device, nx1loc, nx2loc, nx2loc_p2,
                                                 obs_loc_device, sums_x1_loc_device, sums_x2_loc_device,
                                                 error_flag_device_ptr);
    CHECK_ERROR_CUDA(rank, cudaGetLastError());
    CHECK_ERROR_CUDA(rank, cudaDeviceSynchronize());

    int out_of_bounds = 0;
    copy_device<int>(rank, &out_of_bounds, error_flag_device_ptr, 1, cudaMemcpyDeviceToHost);

    if (out_of_bounds != 0) {
        ERROR(rank, "calc_obs_device_kernel() returned out-of-bounds error");
    }


    // Sum up the magnetization and energy over all MPI processes
    CHECK_ERROR(rank,
        MPI_Reduce(obs_loc_device, mag_energy_vec_int_device + n*2,
                   2, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));


    /* Reduce spins along each row (column), then gather the sums into a global
     * array on the master MPI process                                          */
    // ***** x1 (rows) *****
    CHECK_ERROR(rank,
        MPI_Reduce(sums_x1_loc_device, sums_x1_loc_reduced_device,
                   nx1loc, MPI_INT, MPI_SUM, 0, comm_x1));  // NOTE: 0 is the master MPI process of comm_x1

    if (x2index == 0) {
        CHECK_ERROR(rank,
            MPI_Gather(sums_x1_loc_reduced_device, nx1loc, MPI_INT,
                       sums_x1_vec_device + n*NX1, nx1loc, MPI_INT,
                       0, comm_x2));  // NOTE: 0 is the master MPI process of comm_x2, i.e., the process with x1index == 0 within comm_x2;
    }                                 //       but that's just the master process within the world communicator (i.e., process 0 overall)

    // ***** x2 (columns) *****
    CHECK_ERROR(rank,
        MPI_Reduce(sums_x2_loc_device, sums_x2_loc_reduced_device,
                   nx2loc, MPI_INT, MPI_SUM, 0, comm_x2));  // NOTE: 0 is the master MPI process of comm_x2

    if (x1index == 0) {
        CHECK_ERROR(rank,
            MPI_Gather(sums_x2_loc_reduced_device, nx2loc, MPI_INT,
                       sums_x2_vec_device + n*NX2, nx2loc, MPI_INT,
                       0, comm_x1));  // NOTE: 0 is the master MPI process of comm_x1, i.e., the process with x2index == 0 within comm_x1;
    }                                 //       but that's just the master process within the world communicator (i.e., process 0 overall)

    return;
}

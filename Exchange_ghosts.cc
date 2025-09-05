#include <array>

#include <mpi.h>

#include "include/Declare_variables.hh"
#include "include/Declare_functions.hh"
#include "include/Macros.hh"

using namespace std;


/* ====================================================================
 * Routine exchanging ghost lattice sites between neighboring processes
 * NOTE: think of the full grid e.g. as:
 *        |----------------|
 *   ^  3 | 15 16 17 18 19 |  Example with NPROCS_X1=4, NPROCS_X2=5
 *   |  2 | 10 11 12 13 14 |
 *   x1 1 | 5  6  7  8  9  |
 *      0 | 0  1  2  3  4  |
 *        |----------------|
 *          0  1  2  3  4
 *                  x2 ->
 * with both x1 and x2 being periodic (torus topology)
 * ==================================================================== */
void exchange_ghosts(const int &rank,
                     const array<int, 6> &indices_neighbors_parity,
                           int *local_lattice)
{
    // TODO TODO TODO TODO TODO TODO
    // TODO TODO TODO TODO TODO TODO
    // TODO TODO TODO TODO TODO TODO
    // TODO: local_lattice can either be the host or device process-local
    //       lattice, but in the latter case the MPI implementation must be
    //       CUDA-AWARE!
    // TODO: support non-CUDA-aware MPI implementations (e.g. via a #define
    //       flag) too
    // TODO TODO TODO TODO TODO TODO
    // TODO TODO TODO TODO TODO TODO
    // TODO TODO TODO TODO TODO TODO
    const auto &[x1index, x2index, x1down, x1up, x2down, x2up] = indices_neighbors_parity;

    const int tag1 = 1;
    const int tag2 = 2;
    const int tag3 = 3;
    const int tag4 = 4;


    /* ------------
     * Row exchange
     * ------------
     * NOTE: index [i][j] is flattened out as i*nx2loc_p2 + j                   */
    CHECK_ERROR(rank, MPI_Sendrecv(  // Send the top row
        &local_lattice[nx1locp2_p1],          nx2loc, MPI_INT, x1up,   tag1,  // i=1,        j=1
        &local_lattice[nx1locp1_nx2locp2_p1], nx2loc, MPI_INT, x1down, tag1,  // i=nx1loc+1, j=1
        MPI_COMM_WORLD, MPI_STATUS_IGNORE));

    CHECK_ERROR(rank, MPI_Sendrecv(  // Send the bottom row
        &local_lattice[nx1loc_nx2locp2_p1], nx2loc, MPI_INT, x1down, tag2,  // i=nx1loc, j=1
        &local_lattice[1],                  nx2loc, MPI_INT, x1up,   tag2,  // i=0,      j=1
        MPI_COMM_WORLD, MPI_STATUS_IGNORE));


    /* ---------------
     * Column exchange
     * ---------------
     * NOTE: if local_lattice is a device pointer, then x2out and x2in must be
     *       device pointers as well                                            */
    #ifdef USE_CUDA
    int *x2out = allocate_device<int>(rank, nx1loc);
    int *x2in  = allocate_device<int>(rank, nx1loc);

    // Copy the right column to a device buffer
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX: cudaMemcpyDeviceToDevice can only be understood by nvc++.
    //      Modify copy_device() to pass a keyword or enum parameter which
    //      internally sets the CUDA memcpy kind.
    copy_device_2D<int>(rank, x2out, 1,                                // Destination: 1 element between successive elements (row vector)
                        &local_lattice[nx2locp2_p_nx2loc], nx2loc_p2,  // Source: first element has i=1, j=nx2loc, nx2loc_p2 elements between successive elements in the column
                        1, nx1loc, cudaMemcpyDeviceToDevice);          // Copy 1 element per row, nx1loc rows
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX

    // Exchange the buffers between neighboring processes
    CHECK_ERROR(rank, MPI_Sendrecv(
        x2out, nx1loc, MPI_INT, x2up,   tag3,
        x2in,  nx1loc, MPI_INT, x2down, tag3,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE));

    // Store the device buffer in the left ghost column
    copy_device_2D<int>(rank, &local_lattice[nx2loc_p2], nx2loc_p2,  // Destination: first element has i=1, j=0
                        x2in, 1,                                     // Source: 1 element between successive elements (row vector)
                        1, nx1loc, cudaMemcpyDeviceToDevice);        // Copy 1 element per row, nx1loc rows: x2in is treated as a column vector


    // Copy the left column to a device buffer
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX: cudaMemcpyDeviceToDevice can only be understood by nvc++.
    //      Modify copy_device() to pass a keyword or enum parameter which
    //      internally sets the CUDA memcpy kind.
    copy_device_2D(rank, x2out, 1,                          // Destination: 1 element between successive elements (row vector)
                   &local_lattice[nx2locp2_p1], nx2loc_p2,  // Source: first element has i=1, j=1, nx2loc_p2 elements between successive elements in the column
                   1, nx1loc, cudaMemcpyDeviceToDevice);    // Copy 1 element per row, nx1loc rows
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX

    // Exchange the buffers between neighboring processes
    CHECK_ERROR(rank, MPI_Sendrecv(
        x2out, nx1loc, MPI_INT, x2down, tag4,
        x2in,  nx1loc, MPI_INT, x2up,   tag4,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE));

    // Store the device buffer in the right ghost column
    copy_device_2D(rank, &local_lattice[nx2locp2_p_nx2locp1], nx2loc_p2,  // Destination: first element has i=1, j=nx2loc+1
                   x2in, 1,                                               // Source: 1 element between successive elements (row vector)
                   1, nx1loc, cudaMemcpyDeviceToDevice);                  // Copy 1 element per row, nx1loc rows: x2in is treated as a column vector

    free_device(rank, x2out);
    free_device(rank, x2in);


    #else  // i.e. if not USE_CUDA
    array<int, nx1loc> x2out, x2in;

    for (auto i = decltype(nx1loc){1}; i <= nx1loc; ++i) {
        x2out[i-1] = local_lattice[i*nx2loc_p2 + nx2loc];  // Varying i, j=nx2loc
    }

    // Exchange the right column
    CHECK_ERROR(rank, MPI_Sendrecv(
        x2out.data(), nx1loc, MPI_INT, x2up,   tag3,
        x2in.data(),  nx1loc, MPI_INT, x2down, tag3,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE));

    for (auto i = decltype(nx1loc){1}; i <= nx1loc; ++i) {
        local_lattice[i*nx2loc_p2] = x2in[i-1];  // Varying i, j=0
    }


    // Exchange the left column
    for (auto i = decltype(nx1loc){1}; i <= nx1loc; ++i) {
        x2out[i-1] = local_lattice[i*nx2loc_p2 + 1];  // Varying i, j=1
    }

    CHECK_ERROR(rank, MPI_Sendrecv(
        x2out.data(), nx1loc, MPI_INT, x2down, tag4,
        x2in.data(),  nx1loc, MPI_INT, x2up,   tag4,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE));

    for (auto i = decltype(nx1loc){1}; i <= nx1loc; ++i) {
        local_lattice[i*nx2loc_p2 + nx2loc_p1] = x2in[i-1];  // Varying i, j=nx2loc+1
    }

    #endif

    return;
}

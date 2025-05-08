#include <cmath>
#include <array>
#include <random>

#include "include/Declare_variables.hh"
#include "include/Declare_functions.hh"
#include "include/Macros.hh"

#include "Parameters.hh"

using namespace std;


/* =============================================================================
 * Routine updating the process-local lattice by dividing the latter in four
 * equal parts and performing communications on each of them one after the other
 * NOTE: the above technique ensures the ghost points used to update points at
 *       the boundary of the process-local lattice are up to date
 * =============================================================================*/
void update(const int                               &rank,
                  mt19937                           &gen,   // NOTE: can't be constant because dist changes the internal status of gen
                  uniform_real_distribution<double> &dist,  // NOTE: also non-const
            const array<int, 7>                     &indices_neighbors_parity,
                  array<int, nx1locp2_nx2locp2>     &local_lattice)
    {
    const auto &[x1index, x2index, x1down, x1up, x2down, x2up, parity] = indices_neighbors_parity;

    /* Arrays encoding which processes rows (i.e. x data chunks) and columns
     * (i.e. y data chunks) should be sent to and received from. The
     * process-local lattice is assumed to be splitted in four parts like this:
     *   0  1
     *   2  3
     * and element i in each array correspond to the ith part of the process.
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
     * NOTE: think of the process as being split up like
     *       0  1
     *       2  3                                                               */
    const array<int, 4> x1send = {x1up,   x1up,   x1down, x1down};
    const array<int, 4> x1recv = {x1down, x1down, x1up,   x1up};
    const array<int, 4> x2send = {x2down, x2up,   x2down, x2up};
    const array<int, 4> x2recv = {x2up,   x2down, x2up,   x2down};

    // Communication tags
    const int tag1 = 1;
    const int tag2 = 2;
    const int tag3 = 3;
    const int tag4 = 4;


    // Loop over the four parts of the process-local lattice
    constexpr auto _2beta = 2.*BETA;
    int count = 0;

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

            // Helper variables for communication
            const auto jsend = (kx2 == 0) ? 1 : nx2loc;
            const auto jrecv = (kx2 == 0) ? nx2loc_p1 : 0;

            const auto isend_idx_sx2 = isend_idx + sx2;
            const auto irecv_idx_sx2 = irecv_idx + sx2;

            array<int, nx2loc_half> x1out, x1in;
            array<int, nx1loc_half> x2out, x2in;

            // Set up the chunks of data to be sent out
            for (auto j = decltype(nx2loc_half){1}; j <= nx2loc_half; ++j) {
                x1out.at(j-1) = local_lattice.at(isend_idx_sx2 + j);            // local_lattice[isend][j + sx2]
            }

            for (auto i = decltype(nx1loc_half){1}; i <= nx1loc_half; ++i) {
                x2out.at(i-1) = local_lattice.at((i + sx1)*nx2loc_p2 + jsend);  // local_lattice[i + sx1][jsend]
            }

            // Communicate
            if (parity) {
                CHECK_ERROR(rank, MPI_Send(x1out.data(), nx2loc_half, MPI_INT, x1send.at(count), tag1, MPI_COMM_WORLD));
                CHECK_ERROR(rank, MPI_Recv(x1in.data(),  nx2loc_half, MPI_INT, x1recv.at(count), tag2, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                CHECK_ERROR(rank, MPI_Send(x2out.data(), nx1loc_half, MPI_INT, x2send.at(count), tag3, MPI_COMM_WORLD));
                CHECK_ERROR(rank, MPI_Recv(x2in.data(),  nx1loc_half, MPI_INT, x2recv.at(count), tag4, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            } else {
                CHECK_ERROR(rank, MPI_Recv(x1in.data(),  nx2loc_half, MPI_INT, x1recv.at(count), tag1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                CHECK_ERROR(rank, MPI_Send(x1out.data(), nx2loc_half, MPI_INT, x1send.at(count), tag2, MPI_COMM_WORLD));
                CHECK_ERROR(rank, MPI_Recv(x2in.data(),  nx1loc_half, MPI_INT, x2recv.at(count), tag3, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                CHECK_ERROR(rank, MPI_Send(x2out.data(), nx1loc_half, MPI_INT, x2send.at(count), tag4, MPI_COMM_WORLD));
            }

            // Store the chunks of data received into the ghost rows and columns
            for (auto j = decltype(nx2loc_half){1}; j <= nx2loc_half; ++j) {
                local_lattice.at(irecv_idx_sx2 + j) = x1in.at(j-1);            // local_lattice[irecv][j + sx2]
            }

            for (auto i = decltype(nx1loc_half){1}; i <= nx1loc_half; ++i) {
                local_lattice.at((i + sx1)*nx2loc_p2 + jrecv) = x2in.at(i-1);  // local_lattice[i + sx1][jrecv]
            }

            // Update the current part of the process-local lattice
            for (auto i = decltype(imax){imin}; i < imax; ++i) {
                const auto i_idx  = i*nx2loc_p2;
                const auto ip_idx = i_idx + nx2loc_p2;  // (i+1)*nx2loc_p2
                const auto im_idx = i_idx - nx2loc_p2;  // (i-1)*nx2loc_p2

                for (auto j = decltype(jmax){jmin}; j < jmax; ++j) {
                    const auto ij  = i_idx + j;
                    const auto ipj = ip_idx + j;
                    const auto imj = im_idx + j;
                    const auto ijp = ij + 1;
                    const auto ijm = ij - 1;

                    const auto   f     = -(local_lattice.at(ipj) + local_lattice.at(imj) + local_lattice.at(ijp) + local_lattice.at(ijm));
                    const double trial = dist(gen);  // NOTE: this assumes dist is a uniformly-chosen random number between 0 and 1
                    const double prob  = 1./(1. + exp(_2beta*f));  // exp(-BETA*f)/(exp(-BETA*f) + exp(BETA*f))

                    local_lattice.at(ij) = (trial < prob) ? 1 : -1;
                }
            }

            // Move to the next quadrant
            ++count;
        }
    }

    return;
} 

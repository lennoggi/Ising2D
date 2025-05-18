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
            const array<int, 6>                     &indices_neighbors,
                  array<int, nx1locp2_nx2locp2>     &local_lattice)
    {
    const auto &[x1index, x2index, x1down, x1up, x2down, x2up] = indices_neighbors;

    /* NOTE: think of the full grid e.g. as:
     *        |----------------|
     *   ^  3 | 15 16 17 18 19 |  Example with NPROCS_X1=4, NPROCS_X2=5
     *   |  2 | 10 11 12 13 14 |
     *   x1 1 | 5  6  7  8  9  |
     *      0 | 0  1  2  3  4  |
     *        |----------------|
     *          0  1  2  3  4
     *                  x2 ->
     * with both x1 and x2 being periodic (torus topology)                      */

    // Communication tags
    const int tag1 = 1;
    const int tag2 = 2;
    const int tag3 = 3;
    const int tag4 = 4;

    constexpr auto _2beta = 2.*BETA;

    /* Exchange the rows
     *  NOTE: index [i][j] is flattened out as i*nx2loc_p2 + j                  */
    CHECK_ERROR(rank, MPI_Sendrecv(  // Send the top row
        &local_lattice.at(nx1locp2_p1),          nx2loc, MPI_INT, x1up,   tag1,  // i=1,        j=1
        &local_lattice.at(nx1locp1_nx2locp2_p1), nx2loc, MPI_INT, x1down, tag1,  // i=nx1loc+1, j=1
        MPI_COMM_WORLD, MPI_STATUS_IGNORE));

    CHECK_ERROR(rank, MPI_Sendrecv(  // Send the bottom row
        &local_lattice.at(nx1loc_nx2locp2_p1), nx2loc, MPI_INT, x1down, tag2,  // i=nx1loc, j=1
        &local_lattice.at(1),                  nx2loc, MPI_INT, x1up,   tag2,  // i=0,      j=1
        MPI_COMM_WORLD, MPI_STATUS_IGNORE));


    array<int, nx1loc> x2out, x2in;

    // Send the right column
    for (auto i = decltype(nx1loc){1}; i <= nx1loc; ++i) {
        x2out.at(i-1) = local_lattice.at(i*nx2loc_p2 + nx2loc);  // Varying i, j=nx2loc
    }

    CHECK_ERROR(rank, MPI_Sendrecv(
        x2out.data(), nx1loc, MPI_INT, x2up,   tag3,
        x2in.data(),  nx1loc, MPI_INT, x2down, tag3,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE));

    for (auto i = decltype(nx1loc){1}; i <= nx1loc; ++i) {
        local_lattice.at(i*nx2loc_p2) = x2in.at(i-1);  // Varying i, j=0
    }


    // Send the left column
    for (auto i = decltype(nx1loc){1}; i <= nx1loc; ++i) {
        x2out.at(i-1) = local_lattice.at(i*nx2loc_p2 + 1);  // Varying i, j=1
    }

    CHECK_ERROR(rank, MPI_Sendrecv(
        x2out.data(), nx1loc, MPI_INT, x2down, tag4,
        x2in.data(),  nx1loc, MPI_INT, x2up,   tag4,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE));

    for (auto i = decltype(nx1loc){1}; i <= nx1loc; ++i) {
        local_lattice.at(i*nx2loc_p2 + nx2loc_p1) = x2in.at(i-1);  // Varying i, j=nx2loc+1
    }


    // Update the process-local lattice
    for (int color = 0; color <= 1; ++color) {  // 'Black/white' update
        for (auto i = decltype(nx1loc){1}; i <= nx1loc; ++i) {
            const auto i_idx  = i*nx2loc_p2;
            const auto ip_idx = i_idx + nx2loc_p2;  // (i+1)*nx2loc_p2
            const auto im_idx = i_idx - nx2loc_p2;  // (i-1)*nx2loc_p2

            for (auto j = decltype(nx2loc){1}; j <= nx2loc; ++j) {
                const auto ij  = i_idx  + j;
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
    }

    return;
} 

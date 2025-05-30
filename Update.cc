#include <cmath>
#include <array>
#include <random>

#include "include/Declare_variables.hh"
#include "include/Declare_functions.hh"
#include "include/Macros.hh"

#include "Parameters.hh"

using namespace std;


/* ============================================================================
 * Routine updating the process-local lattice by first sweeping only over "red"
 * sites, whose neighbors are all "black", and then sweeping only over "black"
 * sites, whose neighbors are all "red"
 * ============================================================================ */
void update(const int                               &rank,
                  mt19937                           &gen,   // NOTE: can't be constant because dist changes the internal status of gen
                  uniform_real_distribution<double> &dist,  // NOTE: also non-const
            const array<int, 6>                     &indices_neighbors,
                  array<int, nx1locp2_nx2locp2>     &local_lattice)
{
    // Exchange the ghost site between neighboring processes
    exchange_ghosts(rank, indices_neighbors, local_lattice.data());

    // Update the process-local lattice ("red"/"black" checkerboard pattern)
    constexpr auto _2beta = 2.*BETA;

    for (int color = 0; color <= 1; ++color) {
        for (auto i = decltype(nx1loc){1}; i <= nx1loc; ++i) {
            const auto i_idx  = i*nx2loc_p2;
            const auto ip_idx = i_idx + nx2loc_p2;  // (i+1)*nx2loc_p2
            const auto im_idx = i_idx - nx2loc_p2;  // (i-1)*nx2loc_p2

            for (auto j = decltype(nx2loc){1}; j <= nx2loc; ++j) {
                if ((i + j) & 1 == color) {
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
    }

    return;
} 

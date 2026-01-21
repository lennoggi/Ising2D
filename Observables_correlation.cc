#include <cassert>
#include <cstring>
#include <array>
#include <algorithm>

#include <mpi.h>

#include "include/Declare_functions.hh"
#include "include/Macros.hh"
#include "Parameters.hh"

using namespace std;


/* ==========================================================================
 * Routine calculating the magnetization, energy, and spin sums over rows and
 * columns (which are needed to compute the correlation function(s))
 * ========================================================================== */
void calc_obs_corr(const int &rank,
                   const array<int, nx1locp2_nx2locp2> &local_lattice,
                   const hsize_t &n,
                   const int &x1index,
                   const int &x2index,
                   const int &rank_x1,
                   const int &rank_x2,
                   const MPI_Comm &comm_x1,
                   const MPI_Comm &comm_x2,
                   vector<int> &sums_x1_loc,
                   vector<int> &sums_x2_loc,
                   vector<int> &sums_x1_loc_reduced,
                   vector<int> &sums_x2_loc_reduced,
                   vector<int> &mag_energy_vec_int,
                   vector<int> &sums_x1_vec,
                   vector<int> &sums_x2_vec) {
    // Calculate the observables and correlation
    int mag_loc    = 0;
    int energy_loc = 0;

    fill(sums_x1_loc.begin(), sums_x1_loc.end(), 0.);
    fill(sums_x2_loc.begin(), sums_x2_loc.end(), 0.);

    for (auto i = decltype(nx1loc){1}; i <= nx1loc; ++i) {
        const auto i_idx  = i*nx2loc_p2;
        const auto ip_idx = i_idx + nx2loc_p2;  // (i+1)*nx2loc_p2
        const auto im_idx = i_idx - nx2loc_p2;  // (i-1)*nx2loc_p2

        int sum_row_i = 0;

        for (auto j = decltype(nx2loc){1}; j <= nx2loc; ++j) {
            const auto ij  = i_idx + j;
            const auto ipj = ip_idx + j;
            const auto imj = im_idx + j;
            const auto ijp = ij + 1;
            const auto ijm = ij - 1;

            const auto f = -(  local_lattice.at(ipj)
                             + local_lattice.at(imj)
                             + local_lattice.at(ijp)
                             + local_lattice.at(ijm));

            const auto local_lattice_ij = local_lattice.at(ij);

            mag_loc    += local_lattice_ij;
            energy_loc -= local_lattice_ij*f;
            sum_row_i  += local_lattice_ij;
        }

        sums_x1_loc.at(i-1) = sum_row_i;
    }

    for (auto j = decltype(nx2loc){1}; j <= nx2loc; ++j) {
        int sum_col_j = 0;

        for (auto i = decltype(nx1loc){1}; i <= nx1loc; ++i) {
            sum_col_j += local_lattice.at(i*nx2loc_p2 + j);
        }

        sums_x2_loc.at(j-1) = sum_col_j;
    }


    // Sum up the magnetization and energy over all MPI processes
    const array<int, 2> obs_loc{mag_loc, energy_loc};  // Pack data to save one MPI reduction

    CHECK_ERROR(rank,
        MPI_Reduce(obs_loc.data(), mag_energy_vec_int.data() + n*2,
                   2, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));


    /* Reduce spins along each row (column), then gather the sums into a global
     * array on the master MPI process                                          */
    // ***** x1 (rows) *****
    CHECK_ERROR(rank,
        MPI_Reduce(sums_x1_loc.data(), sums_x1_loc_reduced.data(),
                   nx1loc, MPI_INT, MPI_SUM, 0, comm_x1));  // NOTE: 0 is the master MPI process of comm_x1

    if (x2index == 0) {
        CHECK_ERROR(rank,
            MPI_Gather(sums_x1_loc_reduced.data(), nx1loc, MPI_INT,
                       sums_x1_vec.data() + n*NX1, nx1loc, MPI_INT,
                       0, comm_x2));  // NOTE: 0 is the master MPI process of comm_x2, i.e., the process with x1index == 0 within comm_x2;
    }                                 //       but that's just the master process within the world communicator (i.e., process 0 overall)

    // ***** x2 (columns) *****
    CHECK_ERROR(rank,
        MPI_Reduce(sums_x2_loc.data(), sums_x2_loc_reduced.data(),
                   nx2loc, MPI_INT, MPI_SUM, 0, comm_x2));  // NOTE: 0 is the master MPI process of comm_x2

    if (x1index == 0) {
        CHECK_ERROR(rank,
            MPI_Gather(sums_x2_loc_reduced.data(), nx2loc, MPI_INT,
                       sums_x2_vec.data() + n*NX2, nx2loc, MPI_INT,
                       0, comm_x1));  // NOTE: 0 is the master MPI process of comm_x1, i.e., the process with x2index == 0 within comm_x1;
    }                                 //       but that's just the master process within the world communicator (i.e., process 0 overall)

    return;
}

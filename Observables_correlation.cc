#include <cassert>
#include <cstring>
#include <array>

#include <mpi.h>

#include "include/Declare_functions.hh"
#include "include/Macros.hh"
#include "Parameters.hh"

using namespace std;


/* =========================================================================
 * Routine calculating the magnetization, energy, and inter-spin correlation
 * ========================================================================= */
void calc_obs_corr(const int &rank,
                   const array<int, nx1locp2_nx2locp2> &local_lattice,
                   const hsize_t  &n,
                   vector<double> &mag_vec,
                   vector<double> &energy_vec,
                   vector<int>    &sums_rows_vec,
                   vector<int>    &sums_cols_vec) {
    // Calculate the observables and correlation
    int mag_loc    = 0;
    int energy_loc = 0;

    array<int, nx1loc> sums_rows_loc;
    array<int, nx2loc> sums_cols_loc;

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

            const auto f = -(local_lattice.at(ipj) + local_lattice.at(imj) + local_lattice.at(ijp) + local_lattice.at(ijm));
            const auto local_lattice_ij = local_lattice.at(ij);

            mag_loc    += local_lattice_ij;
            energy_loc -= local_lattice_ij*f;
            sum_row_i  += local_lattice_ij;
        }

        sums_rows_loc.at(i-1) = sum_row_i;
    }

    for (auto j = decltype(nx2loc){1}; j <= nx2loc; ++j) {
        int sum_col_j = 0;

        for (auto i = decltype(nx1loc){1}; i <= nx1loc; ++i) {
            sum_col_j += local_lattice.at(i*nx2loc_p2 + j);
        }

        sums_cols_loc.at(j-1) = sum_col_j;
    }


    // Gather and distribute the results
    const array<int, 2> obs_loc{mag_loc, energy_loc};  // Pack data to save one MPI reduction
          array<int, 2> obs_glob{0};

    CHECK_ERROR(rank,
        MPI_Reduce(obs_loc.data(), obs_glob.data(),
                   2, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));

    array<int, NX1> sums_rows_glob{0};
    array<int, NX2> sums_cols_glob{0};

    CHECK_ERROR(rank,
        MPI_Gather(sums_rows_loc.data(),  nx1loc, MPI_INT,
                   sums_rows_glob.data(), nx1loc, MPI_INT,
                   0, MPI_COMM_WORLD));
    CHECK_ERROR(rank,
        MPI_Gather(sums_cols_loc.data(),  nx2loc, MPI_INT,
                   sums_cols_glob.data(), nx2loc, MPI_INT,
                   0, MPI_COMM_WORLD));


    // Update the output vectors
    if (rank == 0) {
           mag_vec[n] = obs_glob[0];  // NOTE: division by   NX1*NX2 deferred to main()
        energy_vec[n] = obs_glob[1];  // NOTE: division by 2*NX1*NX2 deferred to main()

        constexpr auto size_of_int = sizeof(int);
        constexpr auto nx1_ints    = NX1*size_of_int;
        constexpr auto nx2_ints    = NX2*size_of_int;

        memcpy(sums_rows_vec.data() + n*NX1, sums_rows_glob.data(), nx1_ints);
        memcpy(sums_cols_vec.data() + n*NX2, sums_cols_glob.data(), nx2_ints);
    }

    return;
}

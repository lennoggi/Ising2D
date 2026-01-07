#include <cassert>
#include <array>

#include <mpi.h>

#include "include/Declare_functions.hh"
#include "include/Macros.hh"

using namespace std;


/* =========================================================================
 * Routine calculating the magnetization, energy, and inter-spin correlation
 * ========================================================================= */
void calc_obs_corr(const int &rank,
                   const array<int, nx1locp2_nx2locp2> &local_lattice,
                   const hsize_t &n,
                   const hid_t   &dset_mag_id,
                   const hid_t   &dset_energy_id,
                   const hid_t   &memspace_obs_id,
                   const hid_t   &dset_sums_rows_id,
                   const hid_t   &dset_sums_cols_id,
                   const hid_t   &memspace_sums_rows_id,
                   const hid_t   &memspace_sums_cols_id) {
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


    // Print the results to file from rank 0
    if (rank == 0) {
        /* Write data at the appropriate place for iteration n
         * NOTE: getting a fresh new file space from the dataset at every
         *   iteration is safer than re-using the original filespace through
         *   multiple hyperslab selections                                      */
        constexpr auto      ntot = NX1*NX2;
        const double    mag_glob = static_cast<double>(obs_glob.at(0))/static_cast<double>(   ntot);
        const double energy_glob = static_cast<double>(obs_glob.at(1))/static_cast<double>(2.*ntot);

        const auto fspace_mag_id    = H5Dget_space(dset_mag_id);
        const auto fspace_energy_id = H5Dget_space(dset_energy_id);
        assert(fspace_mag_id    > 0);
        assert(fspace_energy_id > 0);

        constexpr hsize_t one = 1;

        CHECK_ERROR(rank,
            H5Sselect_hyperslab(fspace_mag_id,    H5S_SELECT_SET,
                                &n, nullptr, &one, nullptr));
        CHECK_ERROR(rank,
            H5Sselect_hyperslab(fspace_energy_id, H5S_SELECT_SET,
                                &n, nullptr, &one, nullptr));
        CHECK_ERROR(rank,
            H5Dwrite(dset_mag_id, H5T_NATIVE_DOUBLE,
                     memspace_obs_id, fspace_mag_id,    H5P_DEFAULT, &mag_glob));
        CHECK_ERROR(rank,
            H5Dwrite(dset_energy_id, H5T_NATIVE_DOUBLE,
                     memspace_obs_id, fspace_energy_id, H5P_DEFAULT, &energy_glob));

        CHECK_ERROR(rank, H5Sclose(fspace_mag_id));
        CHECK_ERROR(rank, H5Sclose(fspace_energy_id));


        const auto fspace_sums_rows_id = H5Dget_space(dset_sums_rows_id);
        const auto fspace_sums_cols_id = H5Dget_space(dset_sums_cols_id);
        assert(fspace_sums_rows_id > 0);
        assert(fspace_sums_cols_id > 0);

        const     array<hsize_t, 2> start_sums_rows_cols{n, 0};
        constexpr array<hsize_t, 2> count_sums_rows{1, NX1};
        constexpr array<hsize_t, 2> count_sums_cols{1, NX2};

        CHECK_ERROR(rank,
            H5Sselect_hyperslab(fspace_sums_rows_id, H5S_SELECT_SET,
                                start_sums_rows_cols.data(), nullptr, count_sums_rows.data(), nullptr));
        CHECK_ERROR(rank,
            H5Sselect_hyperslab(fspace_sums_cols_id, H5S_SELECT_SET,
                                start_sums_rows_cols.data(), nullptr, count_sums_cols.data(), nullptr));
        CHECK_ERROR(rank,
            H5Dwrite(dset_sums_rows_id, H5T_NATIVE_INT,
                     memspace_sums_rows_id, fspace_sums_rows_id, H5P_DEFAULT, sums_rows_glob.data()));
        CHECK_ERROR(rank,
            H5Dwrite(dset_sums_cols_id, H5T_NATIVE_INT,
                     memspace_sums_cols_id, fspace_sums_cols_id, H5P_DEFAULT, sums_cols_glob.data()));

        CHECK_ERROR(rank, H5Sclose(fspace_sums_rows_id));
        CHECK_ERROR(rank, H5Sclose(fspace_sums_cols_id));
    }

    return;
}

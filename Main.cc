#include <cassert>
#include <iostream>
#include <array>
#include <chrono>

#ifdef USE_CUDA
#include <curand_kernel.h>
#else
#include <random>
#endif

#include <mpi.h>
#include <hdf5.h>

#include "include/Check_parameters.hh"
#include "include/Declare_variables.hh"
#include "include/Declare_functions.hh"
#include "include/Macros.hh"

#include "Parameters.hh"

using namespace std;



int main(int argc, char **argv) {
    // Can't call CHECK_ERROR() before knowing the rank ID
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    assert(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS);
    CHECK_ERROR(rank, MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
    assert(rank < nprocs);


    // Redirect stdout and stderr to process-specific files if desired
    #if (OUTERR_ALL_RANKS)
    ostringstream new_stdout_ss;
    new_stdout_ss << "rank" << rank << ".out";
    FILE* new_stdout;
    new_stdout = freopen(new_stdout_ss.str().c_str(), "w", stdout);
    assert(new_stdout != nullptr);

    ostringstream new_stderr_ss;
    new_stderr_ss << "rank" << rank << ".err";
    FILE* new_stderr;
    new_stderr = freopen(new_stderr_ss.str().c_str(), "w", stderr);
    assert(new_stderr != nullptr);
    #endif

    INFO(rank, "There are " << nprocs << " MPI processes, this is process " << rank);


    // Sanity check on the domain decomposition
    constexpr auto nprocs_expected = NPROCS_X1*NPROCS_X2;
    if (nprocs_expected != nprocs) {
        ERROR(rank, "The total number of MPI processes (" << nprocs <<
              ") doesn't match the desired decomposition of the integration surface (" <<
              NPROCS_X1 << "*" << NPROCS_X2 << "=" << nprocs_expected << ")");
        return 1;  // Not reached
    }

    /* NOTE: both nx1loc=NX1/NPROCS_X1 and nx2loc=NX2/NPROCS_X2 must be EVEN in
     *   order for the 'white/black' update technique to work.
     *   This check is performed in include/Declare_variables.hh .              */
    INFO(rank, "Process-local grid size: " << nx1loc << "*" << nx2loc);


    /* Get the neighbors and parity of the current process
     * NOTE: this process' neighbors and parity are needed by update(), so the
     *   return value of set_indices_neighbors_parity() is unpacked separately  */
    const auto &indices_neighbors_parity = set_indices_neighbors_parity(rank, nprocs);
    const auto &[x1index, x2index, x1down, x1up, x2down, x2up, parity] = indices_neighbors_parity;

    #if (VERBOSE)
    INFO(rank, "Indices of process " << rank << " along x1 and x2: " << x1index << ", " << x2index);
    INFO(rank, "Neighbors of process " << rank << ": " <<
           "x1down=" << x1down << ", x1up=" << x1up <<
         ", x2down=" << x2down << ", x2up=" << x2up);
    INFO(rank, "Parity of process " << rank << ": " << (int) parity);
    #endif


    /* Initialize all the elements of the process-local lattice to 1 (but -1
     * would work as well) to minimize the entropy, so that thermalization time
     * is minimized
     * NOTE: even if the lattice updates are offloaded to the device (i.e.,
     *   USE_CUDA is defined, and so the process-local lattice is copied to the
     *   device) the process-local lattice still needs to exist on the host to
     *   perform I/O and measurements                                           */
    array<int, nx1locp2_nx2locp2> local_lattice;

    for (auto &site : local_lattice) {
        site = 1;
    }

    #ifdef USE_CUDA
    int *local_lattice_device = allocate_device<int>(rank, nx1locp2_nx2locp2);
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX: cudaMemcpyHostToDevice can only be understood by nvc++.
    //      Modify copy_device() to pass a keyword or enum parameter which
    //      internally sets the CUDA memcpy kind.
    copy_device(rank, local_lattice_device, local_lattice.data(), nx1locp2_nx2locp2, cudaMemcpyHostToDevice);
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    #endif


    // Write the initial lattice to file
    auto fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    assert(fapl_id >= 0);
    CHECK_ERROR(rank, H5Pset_fapl_mpio(fapl_id, MPI_COMM_WORLD, MPI_INFO_NULL));
    CHECK_ERROR(rank, H5Pset_all_coll_metadata_ops(fapl_id, true));

    const auto file_lattice_id = H5Fcreate("Lattice.h5", H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    assert(file_lattice_id >= 0);

    write_lattice(rank, nprocs, x1index, x2index, 0, local_lattice, file_lattice_id);


    /* Initialize the seed of the random number generator, either on the host
     * (using the Mersenne twister 19937 algorithm from the <random> header) or
     * on the device (using cuRAND and a Philox 4*32 10 generator).
     * Also, initialize the process-local lattice on the device if needed.      */
    random_device rd;               // Use machine entropy as the random seed
    const auto seed = rd() + rank;  // Each rank must have a different seed

    #ifdef USE_CUDA
    /* One RNG per INTERIOR lattice site, since on the GPU the update step
     * happens ~simultaneously for all points within the process-local lattice  */
    curandStatePhilox4_32_10_t *rng_states_device = allocate_device<curandStatePhilox4_32_10_t>(rank, nx1loc_nx2loc);
    init_rng_device<curandStatePhilox4_32_10_t>(rank, rng_states_device, seed);
    #else
    mt19937 gen(seed);
    uniform_real_distribution<double> dist(0., 1.);
    #endif


    // Let the lattice thermalize
    #if (VERBOSE)
    INFO(rank, "beta = " << BETA);
    INFO(rank, "Begin lattice thermalization");
    #endif

    const auto thermalize_start = chrono::high_resolution_clock::now();

    for (auto n = decltype(NTHERM){1}; n <= NTHERM; ++n) {
        #ifdef USE_CUDA
        update_device<curandStatePhilox4_32_10_t>(rank, rng_states_device, indices_neighbors_parity, local_lattice_device);
        #else
        update(rank, gen, dist, indices_neighbors_parity, local_lattice);
        #endif

        #if (SAVE_LATTICE_THERM)
        if (n % LATTICE_OUT_EVERY == 0) {
            #ifdef USE_CUDA
            copy_device(rank, local_lattice.data(), local_lattice_device, nx1locp2_nx2locp2, cudaMemcpyDeviceToHost);
            #endif
            write_lattice(rank, nprocs, x1index, x2index, n, local_lattice, file_lattice_id);
            #if (VERYVERBOSE)
            INFO(rank, "Iteration " << n << " written to file");
            #endif
        }
        #endif
    }

    const auto thermalize_end  = chrono::high_resolution_clock::now();
    const auto thermalize_time = chrono::duration_cast<chrono::seconds>(thermalize_end - thermalize_start);

    #if (VERBOSE)
    INFO(rank, "Lattice has reached thermal equilibrium in " << thermalize_time.count() << " s");
    #endif

    CHECK_ERROR(rank, H5Fclose(file_lattice_id));



    /* --------------------------------------------------------------
     * Calculate observables and correlations across rows and columns
     * -------------------------------------------------------------- */
    const auto file_obscorr_id = H5Fcreate("Observables_correlation.h5", H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    assert(file_obscorr_id >= 0);

    // Write beta and lattice size to file
    constexpr hsize_t one = 1;
    constexpr hsize_t two = 2;

    const auto space_one_id = H5Screate_simple(1, &one, nullptr);
    const auto space_two_id = H5Screate_simple(1, &two, nullptr);
    assert(space_one_id > 0);
    assert(space_two_id > 0);

    const auto dset_beta_id    = H5Dcreate(file_obscorr_id, "/Beta", H5T_NATIVE_DOUBLE,
                                           space_one_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    const auto dset_latsize_id = H5Dcreate(file_obscorr_id, "/Lattice size", H5T_NATIVE_INT,
                                           space_two_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(dset_beta_id    > 0);
    assert(dset_latsize_id > 0);

    if (rank == 0) {
        constexpr auto beta = BETA;
        constexpr array<int, 2> latsize{NX1, NX2};

        CHECK_ERROR(rank,
            H5Dwrite(dset_beta_id,    H5T_NATIVE_DOUBLE,
                     space_one_id, space_one_id, H5P_DEFAULT, &beta));
        CHECK_ERROR(rank,
            H5Dwrite(dset_latsize_id, H5T_NATIVE_INT,
                     space_two_id, space_two_id, H5P_DEFAULT, latsize.data()));
    }

    CHECK_ERROR(rank, H5Dclose(dset_beta_id));
    CHECK_ERROR(rank, H5Dclose(dset_latsize_id));
    CHECK_ERROR(rank, H5Sclose(space_two_id));


    // Set up writing magnetization and energy to file
    constexpr  hsize_t ncalc = NCALC;
    const auto fspace_obs_id = H5Screate_simple(1, &ncalc, nullptr);
    assert(fspace_obs_id > 0);

    auto dset_mag_id    = H5Dcreate(file_obscorr_id, "/Magnetization", H5T_NATIVE_DOUBLE,
                                    fspace_obs_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    auto dset_energy_id = H5Dcreate(file_obscorr_id, "/Energy",       H5T_NATIVE_DOUBLE,
                                    fspace_obs_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(dset_mag_id    > 0);
    assert(dset_energy_id > 0);


    // Set up writing spin sums over rows and columns and energy to file
    constexpr array<hsize_t, 2> dims_fspace_sums_rows{NCALC, NX1};
    constexpr array<hsize_t, 2> dims_fspace_sums_cols{NCALC, NX2};

    const auto fspace_sums_rows_id = H5Screate_simple(2, dims_fspace_sums_rows.data(), nullptr);
    const auto fspace_sums_cols_id = H5Screate_simple(2, dims_fspace_sums_cols.data(), nullptr);
    assert(fspace_sums_rows_id > 0);
    assert(fspace_sums_cols_id > 0);

    auto dset_sums_rows_id = H5Dcreate(file_obscorr_id, "/x1 spin sums", H5T_NATIVE_INT,
                                       fspace_sums_rows_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    auto dset_sums_cols_id = H5Dcreate(file_obscorr_id, "/x2 spin sums", H5T_NATIVE_INT,
                                       fspace_sums_cols_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(dset_sums_rows_id > 0);
    assert(dset_sums_cols_id > 0);

    constexpr array<hsize_t, 2> dims_memspace_sums_rows{1, NX1};
    constexpr array<hsize_t, 2> dims_memspace_sums_cols{1, NX2};

    const auto memspace_sums_rows_id = H5Screate_simple(2, dims_memspace_sums_rows.data(), nullptr);
    const auto memspace_sums_cols_id = H5Screate_simple(2, dims_memspace_sums_cols.data(), nullptr);
    assert(memspace_sums_rows_id > 0);
    assert(memspace_sums_cols_id > 0);


    #if (VERBOSE)
    INFO(rank, "Begin calculating observables and correlations across rows and columns");
    #endif
    const auto calc_obs_corr_start = chrono::high_resolution_clock::now();

    for (hsize_t n = 0; n < ncalc; ++n) {
        #ifdef USE_CUDA
        // TODO: implement the calculation of observables and correlations on the GPU
        ERROR(rank, "The calculation of observables and correlations across rows and columns is not yet implemented on GPUs, aborting");
        //update_device<curandStatePhilox4_32_10_t>(rank, rng_states_device, indices_neighbors_parity, local_lattice_device);
        #else
        calc_obs_corr(rank, local_lattice, n,
                      dset_mag_id, dset_energy_id, space_one_id,
                      dset_sums_rows_id, dset_sums_cols_id, memspace_sums_rows_id, memspace_sums_cols_id);
        update(rank, gen, dist, indices_neighbors_parity, local_lattice);
        #endif

    }

    const auto calc_obs_corr_end  = chrono::high_resolution_clock::now();
    const auto calc_obs_corr_time = chrono::duration_cast<chrono::seconds>(calc_obs_corr_end - calc_obs_corr_start);

    #if (VERBOSE)
    INFO(rank, "Done calculating observables and correlations across rows and columns in " << calc_obs_corr_time.count() << " s");
    #endif

    CHECK_ERROR(rank, H5Dclose(dset_mag_id));
    CHECK_ERROR(rank, H5Dclose(dset_energy_id));
    CHECK_ERROR(rank, H5Dclose(dset_sums_rows_id));
    CHECK_ERROR(rank, H5Dclose(dset_sums_cols_id));

    CHECK_ERROR(rank, H5Sclose(space_one_id));

    CHECK_ERROR(rank, H5Sclose(fspace_obs_id));
    CHECK_ERROR(rank, H5Sclose(fspace_sums_rows_id));
    CHECK_ERROR(rank, H5Sclose(fspace_sums_cols_id));

    CHECK_ERROR(rank, H5Sclose(memspace_sums_rows_id));
    CHECK_ERROR(rank, H5Sclose(memspace_sums_cols_id));

    CHECK_ERROR(rank, H5Fclose(file_obscorr_id));
    CHECK_ERROR(rank, H5Pclose(fapl_id));

    #ifdef USE_CUDA
    free_device(rank, local_lattice_device);
    free_device(rank, rng_states_device);
    #endif

    CHECK_ERROR(rank, MPI_Finalize());

    return 0;
}

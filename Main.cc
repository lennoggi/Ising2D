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
    copy_device<int>(rank, local_lattice_device, local_lattice.data(), nx1locp2_nx2locp2, cudaMemcpyHostToDevice);
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
    int *error_flag_device_ptr = allocate_device<int>(rank, 1);
    curandStatePhilox4_32_10_t *rng_states_device = allocate_device<curandStatePhilox4_32_10_t>(rank, nx1loc_nx2loc);
    init_rng_device<curandStatePhilox4_32_10_t>(rank, rng_states_device, seed, error_flag_device_ptr);
    #else
    mt19937 gen(seed);
    uniform_real_distribution<double> dist(0., 1.);
    #endif


    // Let the lattice thermalize
    INFO(rank, "beta = " << BETA);
    INFO(rank, "Begin lattice thermalization");

    const auto thermalize_start = chrono::high_resolution_clock::now();

    for (auto n = decltype(NTHERM){1}; n <= NTHERM; ++n) {
        #ifdef USE_CUDA
        update_device<curandStatePhilox4_32_10_t>(rank, rng_states_device, indices_neighbors_parity,
                                                  local_lattice_device, error_flag_device_ptr);
        #else
        update(rank, gen, dist, indices_neighbors_parity, local_lattice);
        #endif

        #if (SAVE_LATTICE_THERM)
        if (n % LATTICE_OUT_EVERY == 0) {
            #ifdef USE_CUDA
            copy_device<int>(rank, local_lattice.data(), local_lattice_device, nx1locp2_nx2locp2, cudaMemcpyDeviceToHost);
            #endif
            write_lattice(rank, nprocs, x1index, x2index, n, local_lattice, file_lattice_id);
            #if (VERYVERBOSE)
            INFO(rank, "Thermalization step " << n << ": lattice written to file");
            #endif
        }
        #endif

        #if (VERYVERBOSE)
        INFO(rank, "Thermalization step " << n << " complete");
        #endif
    }

    const auto thermalize_end  = chrono::high_resolution_clock::now();
    const auto thermalize_time = chrono::duration_cast<chrono::seconds>(thermalize_end - thermalize_start);

    INFO(rank, "Lattice has reached thermal equilibrium in " << thermalize_time.count() << " s");

    CHECK_ERROR(rank, H5Fclose(file_lattice_id));



    /* --------------------------------------------------------------
     * Calculate observables and correlations across rows and columns
     * -------------------------------------------------------------- */
    INFO(rank, "Begin calculating observables and correlations across rows and columns");

    /* Potentially very large arrays => Allocate on the heap
     * Only allocate buffers on the master process, which is the only one
     * writing to file                                                          */
    #ifdef USE_CUDA
    int    *mag_energy_vec_int_device;
    double *mag_vec_device, *energy_vec_device;
    int    *sums_x1_vec_device, *sums_x2_vec_device;

    if (rank == 0) {
        mag_energy_vec_int_device = allocate_device<int>   (rank, static_cast<size_t>(NCALC*2));
                   mag_vec_device = allocate_device<double>(rank, static_cast<size_t>(NCALC));
                energy_vec_device = allocate_device<double>(rank, static_cast<size_t>(NCALC));
               sums_x1_vec_device = allocate_device<int>   (rank, static_cast<size_t>(NCALC*NX1));
               sums_x2_vec_device = allocate_device<int>   (rank, static_cast<size_t>(NCALC*NX2));
    }
    #else
    vector<int> sums_x1_loc(nx1loc);
    vector<int> sums_x2_loc(nx2loc);

    vector<int> mag_energy_vec_int;

    if (rank == 0) {
        mag_energy_vec_int.resize(NCALC*2);
    }
    #endif

    vector<double> mag_vec, energy_vec;
    vector<int>    sums_x1_vec, sums_x2_vec;

    if (rank == 0) {
            mag_vec.resize(NCALC);
         energy_vec.resize(NCALC);
        sums_x1_vec.resize(NCALC*NX1);
        sums_x2_vec.resize(NCALC*NX2);
    }


    /* Set up MPI communicators to reduce spin sums over rows and columns
     * NOTE: comm_x1 (comm_x2) contains all the MPI processes with the same
     *       x1index (x2index)                                                */
    MPI_Comm comm_x1, comm_x2;

    CHECK_ERROR(rank,
        MPI_Comm_split(MPI_COMM_WORLD, x1index, x2index, &comm_x1));
    CHECK_ERROR(rank,
        MPI_Comm_split(MPI_COMM_WORLD, x2index, x1index, &comm_x2));

    int rank_x1, rank_x2;  // Ranks in comm_x1 and comm_x2
    MPI_Comm_rank(comm_x1, &rank_x1);
    MPI_Comm_rank(comm_x2, &rank_x2);

    #ifdef USE_CUDA
    int *obs_loc_device     = allocate_device<int>(rank, 2);
    int *sums_x1_loc_device = allocate_device<int>(rank, nx1loc);
    int *sums_x2_loc_device = allocate_device<int>(rank, nx2loc);

    int *sums_x1_loc_reduced_device, *sums_x2_loc_reduced_device;

    if (x2index == 0) {
        sums_x1_loc_reduced_device = allocate_device<int>(rank, nx1loc);
    }

    if (x1index == 0) {
        sums_x2_loc_reduced_device = allocate_device<int>(rank, nx2loc);
    }

    #else
    vector<int> sums_x1_loc_reduced, sums_x2_loc_reduced;
    if (x2index == 0) {
        sums_x1_loc_reduced.resize(nx1loc);
    }
    if (x1index == 0) {
        sums_x2_loc_reduced.resize(nx2loc);
    }
    #endif


    const auto calc_obs_corr_start = chrono::high_resolution_clock::now();

    for (hsize_t n = 0; n < NCALC; ++n) {
        #ifdef USE_CUDA
        calc_obs_corr_device(rank, local_lattice_device, n,
                             x1index, x2index, rank_x1, rank_x2, comm_x1, comm_x2,
                             obs_loc_device, sums_x1_loc_device, sums_x2_loc_device,
                             sums_x1_loc_reduced_device, sums_x2_loc_reduced_device,
                             mag_energy_vec_int_device, sums_x1_vec_device, sums_x2_vec_device,
                             error_flag_device_ptr);
        update_device<curandStatePhilox4_32_10_t>(rank, rng_states_device, indices_neighbors_parity,
                                                  local_lattice_device, error_flag_device_ptr);
        #else
        calc_obs_corr(rank, local_lattice, n,
                      x1index, x2index, rank_x1, rank_x2, comm_x1, comm_x2,
                      sums_x1_loc, sums_x2_loc,
                      sums_x1_loc_reduced, sums_x2_loc_reduced,
                      mag_energy_vec_int, sums_x1_vec, sums_x2_vec);
        update(rank, gen, dist, indices_neighbors_parity, local_lattice);
        #endif

        #if (SAVE_LATTICE_CALC)
        if (n % LATTICE_OUT_EVERY == 0) {
            #ifdef USE_CUDA
            copy_device<int>(rank, local_lattice.data(), local_lattice_device, nx1locp2_nx2locp2, cudaMemcpyDeviceToHost);
            #endif
            write_lattice(rank, nprocs, x1index, x2index, n, local_lattice, file_lattice_id);
            #if (VERYVERBOSE)
            INFO(rank, "Thermalization step " << n << ": lattice written to file");
            #endif
        }
        #endif

        #if (VERYVERBOSE)
        INFO(rank, "Calculation step " << n << " complete");
        #endif
    }

    CHECK_ERROR(rank, MPI_Comm_free(&comm_x1));
    CHECK_ERROR(rank, MPI_Comm_free(&comm_x2));

    #ifdef USE_CUDA
    free_device(rank,     obs_loc_device);
    free_device(rank, sums_x1_loc_device);
    free_device(rank, sums_x2_loc_device);

    if (x2index == 0) {
        free_device(rank, sums_x1_loc_reduced_device);
    }

    if (x1index == 0) {
        free_device(rank, sums_x2_loc_reduced_device);
    }

    free_device(rank, error_flag_device_ptr);
    #endif


    if (rank == 0) {
        constexpr double   ntot_inv = 1./static_cast<double>(NX1*NX2);
        constexpr double _2ntot_inv = 0.5*ntot_inv;

        #ifdef USE_CUDA
        /* Copy the integer buffer 'mag_energy_vec_int_device' into a couple of
         * double precision buffers ('mag_vec_device' and 'energy_vec_device')
         * and scale these quantities by the lattice volume                     */
        cast_and_scale_two_device<int, double>(rank,
            mag_energy_vec_int_device,         // Input buffer   (size: 2*NCALC)
            mag_vec_device, energy_vec_device, // Output buffers (size:   NCALC)
            ntot_inv, _2ntot_inv,              // Scaling values for magnetization and energy (in this order)
            NCALC);

        copy_device<double>(rank,    mag_vec.data(),    mag_vec_device, static_cast<size_t>(NCALC), cudaMemcpyDeviceToHost);
        copy_device<double>(rank, energy_vec.data(), energy_vec_device, static_cast<size_t>(NCALC), cudaMemcpyDeviceToHost);

        copy_device<int>(rank, sums_x1_vec.data(), sums_x1_vec_device, static_cast<size_t>(NCALC*NX1), cudaMemcpyDeviceToHost);
        copy_device<int>(rank, sums_x2_vec.data(), sums_x2_vec_device, static_cast<size_t>(NCALC*NX2), cudaMemcpyDeviceToHost);

        free_device(rank, mag_energy_vec_int_device);
        free_device(rank, mag_vec_device);
        free_device(rank, energy_vec_device);
        free_device(rank, sums_x1_vec_device);
        free_device(rank, sums_x2_vec_device);
        #else
        for (auto n = decltype(NCALC){0}; n < NCALC; ++n) {
            const auto _2n = 2*n;
               mag_vec[n] = static_cast<double>(mag_energy_vec_int[_2n])   * ntot_inv;
            energy_vec[n] = static_cast<double>(mag_energy_vec_int[_2n+1]) *  _2ntot_inv;
        }
        #endif
    }

    const auto calc_obs_corr_end  = chrono::high_resolution_clock::now();
    const auto calc_obs_corr_time = chrono::duration_cast<chrono::seconds>(calc_obs_corr_end - calc_obs_corr_start);

    INFO(rank, "Done calculating observables and correlations across rows and columns in " << calc_obs_corr_time.count() << " s");



    /* ------------------
     * Write data to file
     * ------------------ */
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

    CHECK_ERROR(rank, H5Sclose(space_one_id));
    CHECK_ERROR(rank, H5Sclose(space_two_id));


    // Write the observables to file
    constexpr  hsize_t ncalc = NCALC;
    const auto space_obs_id  = H5Screate_simple(1, &ncalc, nullptr);
    assert(space_obs_id > 0);

    auto dset_mag_id    = H5Dcreate(file_obscorr_id, "/Magnetization", H5T_NATIVE_DOUBLE,
                                    space_obs_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    auto dset_energy_id = H5Dcreate(file_obscorr_id, "/Energy",       H5T_NATIVE_DOUBLE,
                                    space_obs_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(dset_mag_id    > 0);
    assert(dset_energy_id > 0);

    if (rank == 0) {
        CHECK_ERROR(rank,
            H5Dwrite(dset_mag_id, H5T_NATIVE_DOUBLE,
                     space_obs_id, space_obs_id, H5P_DEFAULT, mag_vec.data()));
        CHECK_ERROR(rank,
            H5Dwrite(dset_energy_id, H5T_NATIVE_DOUBLE,
                     space_obs_id, space_obs_id, H5P_DEFAULT, energy_vec.data()));
    }

    CHECK_ERROR(rank, H5Dclose(dset_mag_id));
    CHECK_ERROR(rank, H5Dclose(dset_energy_id));
    CHECK_ERROR(rank, H5Sclose(space_obs_id));


    // Write the spin sums over rows and columns to file
    constexpr array<hsize_t, 2> dims_space_sums_x1{NCALC, NX1};
    constexpr array<hsize_t, 2> dims_space_sums_x2{NCALC, NX2};

    const auto space_sums_x1_id = H5Screate_simple(2, dims_space_sums_x1.data(), nullptr);
    const auto space_sums_x2_id = H5Screate_simple(2, dims_space_sums_x2.data(), nullptr);
    assert(space_sums_x1_id > 0);
    assert(space_sums_x2_id > 0);

    auto dset_sums_x1_id = H5Dcreate(file_obscorr_id, "/x1 spin sums", H5T_NATIVE_INT,
                                     space_sums_x1_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    auto dset_sums_x2_id = H5Dcreate(file_obscorr_id, "/x2 spin sums", H5T_NATIVE_INT,
                                     space_sums_x2_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(dset_sums_x1_id > 0);
    assert(dset_sums_x2_id > 0);

    if (rank == 0) {
        CHECK_ERROR(rank,
            H5Dwrite(dset_sums_x1_id, H5T_NATIVE_INT,
                     space_sums_x1_id, space_sums_x1_id, H5P_DEFAULT, sums_x1_vec.data()));
        CHECK_ERROR(rank,
            H5Dwrite(dset_sums_x2_id, H5T_NATIVE_INT,
                     space_sums_x2_id, space_sums_x2_id, H5P_DEFAULT, sums_x2_vec.data()));
    }

    CHECK_ERROR(rank, H5Dclose(dset_sums_x1_id));
    CHECK_ERROR(rank, H5Dclose(dset_sums_x2_id));

    CHECK_ERROR(rank, H5Sclose(space_sums_x1_id));
    CHECK_ERROR(rank, H5Sclose(space_sums_x2_id));


    // Finalize
    CHECK_ERROR(rank, H5Fclose(file_obscorr_id));
    CHECK_ERROR(rank, H5Pclose(fapl_id));

    #ifdef USE_CUDA
    free_device(rank, local_lattice_device);
    free_device(rank, rng_states_device);
    #endif

    INFO(rank, "All done");
    CHECK_ERROR(rank, MPI_Finalize());

    return 0;
}

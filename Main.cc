#include <cassert>
#include <iostream>
#include <array>
#include <random>
#include <chrono>

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
     *   order for communication among MPI processes to happen smoothly using
     *   the 'parity' technique.
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
     * is minimized                                                             */
    array<int, nx1locp2_nx2locp2> local_lattice;

    for (auto &site : local_lattice) {
        site = 1;
    }


    // Write the initial lattice to file
    auto fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    assert(fapl_id >= 0);
    CHECK_ERROR(rank, H5Pset_fapl_mpio(fapl_id, MPI_COMM_WORLD, MPI_INFO_NULL));
    CHECK_ERROR(rank, H5Pset_all_coll_metadata_ops(fapl_id, true));

    const auto file_id = H5Fcreate("Lattice_global.h5", H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    assert(file_id >= 0);

    write_lattice(rank, nprocs, x1index, x2index, 0, local_lattice, file_id);


    // Initialize the seed of the random number generator
    random_device rd;          // Use machine entropy as the random seed 
    mt19937 gen(rd() + rank);  // Each rank must have a different seed


    // Let the lattice thermalize
    #if (VERBOSE)
    INFO(rank, "beta = " << BETA);
    INFO(rank, "Begin lattice thermalization");
    #endif

    const auto thermalize_start = chrono::high_resolution_clock::now();

    //for (auto n = decltype(NTHERM){0}; n < NTHERM; ++n) {
    //    update(neighbors_and_parity, local_lattice);
    //}




    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    //thermalize(nprocs, rank, left, right, up, down, parity, local_lattice);
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    // XXX XXX XXX XXX XXX XXX
    const auto thermalize_end   = chrono::high_resolution_clock::now();
    const auto thermalize_time  = chrono::duration_cast<chrono::seconds>(thermalize_end - thermalize_start);

    #if (VERBOSE)
    INFO(rank, "Lattice has reached thermal equilibrium");
    #endif


    // TODO TODO TODO TODO TODO TODO
    // TODO TODO TODO TODO TODO TODO
    // TODO TODO TODO TODO TODO TODO
    // Compute observables
    // TODO TODO TODO TODO TODO TODO
    // TODO TODO TODO TODO TODO TODO
    // TODO TODO TODO TODO TODO TODO



    CHECK_ERROR(rank, H5Fclose(file_id));
    CHECK_ERROR(rank, H5Pclose(fapl_id));
    CHECK_ERROR(rank, MPI_Finalize());

    return 0;
}

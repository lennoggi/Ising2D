#include <cassert>
#include <iostream>
#include <array>
//#include <random>
#include <chrono>

#include <mpi.h>

#include "include/Check_parameters.hh"
#include "include/Declare_variables.hh"
#include "include/Declare_functions.hh"
#include "include/Macros.hh"
#include "include/Types.hh"

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


    // Get the neighbors and parity of the current process
    //const auto &neighbors_and_parity = set_neighbors_and_parity(rank, nprocs);
    const auto &[x1down, x1up, x2down, x2up, parity] = set_neighbors_and_parity(rank, nprocs);

    #if (VERBOSE)
    INFO(rank, "Neighbors of process " << rank << ": " <<
           "x1down=" << x1down << ", x1up=" << x1up <<
         ", x2down=" << x2down << ", x2up=" << x2up);
    INFO(rank, "Parity of process " << rank << ": " << (int) parity);
    #endif


    /* Initialize all the elements of the process-local lattice to 1 (but -1
     * would work as well) to minimize the entropy, so that thermalization time
     * is minimized                                                             */
    array<array<int, nx1loc_p2>, nx2loc_p2> local_lattice;

    for (auto &row : local_lattice) {
        for (auto &site : row) {
            site = 1;
        }
    }


    // Initialize the seed of the random number generator
    //srand(time(NULL) + rank);
    // XXX: change this with
    //random_device rd;  // Use machine entropy as the random seed 
    //mt19937 gen(rd() + rank);  // Each rank has to have a different seed
    // XXX: or some other random number generator


    // Let the lattice thermalize
    #if (VERBOSE)
    INFO(rank, "beta = " << BETA);
    INFO(rank, "Begin lattice thermalization");
    #endif

    const auto thermalize_start = chrono::high_resolution_clock::now();
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


    MPI_Finalize();

    return 0;
}

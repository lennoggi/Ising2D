#include <cassert>
#include <cstdlib>
#include <ctime>
#include <array>
#include <mpi.h>

#include "include/Declarations.hh"
#include "include/Macros.hh"
#include "Parameters.hh"


#include <iostream>
using namespace std;


int main(int argc, char **argv) {
    // Initialize
    MPI_Init(&argc, &argv);

    // Get the total number of processes
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    assert(nprocs > 0);

    // Get the ID of the current process
    int proc_ID;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_ID);
    assert(proc_ID >= 0 and proc_ID < nprocs);

    // Perform sanity checks on the parameters
    check_parameters(nprocs);

    // Print some info
    #if (VERBOSE)
        INFO("beta = " << BETA);
        INFO("There are " << nprocs << " processes");
        INFO("The process-local grid size (interior only) is " <<
             nxloc << "*" << nyloc);
        INFO("The process-local grid size (including ghosts) is " <<
             nxloc_p2 << "*" << nyloc_p2);
    #endif


    /* Find out the neighbors of the current process
     * **CONVENTIONS**
     * 1. Process IDs increase by 1 along x, starting from the top-left
     * 2. Periodic BCs on both x and y                                          */
    // TODO: put this in a routine and select the BCs through a parameter
    const int left  = (proc_ID % NPROCS_X == 0) ?             // Left edge  (x = 0)
            proc_ID + NPROCS_X - 1 : proc_ID - 1;
    const int right = (proc_ID % NPROCS_X == NPROCS_X - 1) ?  // Right edge (x = nxloc)
            proc_ID - NPROCS_X + 1 : proc_ID + 1;
    const int up    = (proc_ID < NPROCS_X) ?                  // Upper edge (y = 0)
            proc_ID + nprocs - NPROCS_X : proc_ID - NPROCS_X;
    const int down  = (proc_ID >= nprocs - NPROCS_X) ?        // Lower edge (y = nyloc)
            proc_ID - nprocs + NPROCS_X : proc_ID + NPROCS_X;

    // Sanity check
    assert(left  >= 0 and left  < nprocs);
    assert(right >= 0 and right < nprocs);
    assert(up    >= 0 and up    < nprocs);
    assert(down  >= 0 and down  < nprocs);


    // Assign a 'parity' to the current process
    const int  y_index = proc_ID/NPROCS_X;
    const bool parity  = ((proc_ID + y_index) % 2 == 0) ? true : false;

    // Sanity check: the full lattice should look like a 'chessboard'
    const array<int, 4> neighbors = {left, right, up, down};
    for (const auto &n : neighbors) {
        const int y_index_n = n/NPROCS_X;
        const int parity_n  = ((n + y_index_n) % 2 == 0) ? true : false;
        assert(parity_n != parity);
    }


    // Process-local lattice
    array<array<int, nxloc_p2>, nyloc_p2> local_lattice;

    /* Initialize all the elements of the process-local lattice to 1 (but -1
     * would work as well) to minimize the entropy, so that thermalization time
     * is minimized                                                             */
    for (auto &row : local_lattice) {
        for (auto &site : row) {
            site = 1;
        }
    }


    // Initialize the seed of the random number generator
    srand(time(NULL) + proc_ID);


    // Let the lattice thermalize
    #if (VERBOSE)
        INFO("Lattice is reaching thermal equilibrium...");
    #endif

    thermalize(nprocs, proc_ID, left, right, up, down, parity, local_lattice);

    #if (VERBOSE)
        INFO("Lattice has reached thermal equilibrium");
    #endif


    // TODO: compute observables


    // Finalize
    MPI_Finalize();

    return 0;
}

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
     * 1. Process IDs increase by 1 along y
     * 2. Periodic BCs on both x and y                                          */
    // TODO: put this in a routine and select the BCs through a parameter
    const int right = (proc_ID >= nprocs - NPROCS_Y) ?        // Right edge
            proc_ID - nprocs + NPROCS_Y : proc_ID + NPROCS_Y;
    const int left  = (proc_ID < NPROCS_Y) ?                  // Left edge
            proc_ID + nprocs - NPROCS_Y : proc_ID - NPROCS_Y;
    const int up    = (proc_ID % NPROCS_Y == NPROCS_Y - 1) ?  // Upper edge
            proc_ID - NPROCS_Y + 1 : proc_ID + 1;
    const int down  = (proc_ID % NPROCS_Y == 0) ?             // Lower edge
            proc_ID + NPROCS_Y - 1 : proc_ID - 1;

    // Sanity check
    assert(right >= 0 and right < nprocs);
    assert(left  >= 0 and left  < nprocs);
    assert(up    >= 0 and up    < nprocs);
    assert(down  >= 0 and down  < nprocs);


    // Assign a 'parity' to the current process
    const int  x_index = proc_ID/NPROCS_Y;
    const bool parity  = ((proc_ID + x_index) % 2 == 0) ? true : false;

    // Sanity check: the full lattice should look like a 'chessboard'
    const array<int, 4> neighbors = {right, left, up, down};
    for (const auto &n : neighbors) {
        const int x_index_n = n/NPROCS_Y;
        const int parity_n  = ((n + x_index_n) % 2 == 0) ? true : false;
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

    // TODO: thermalize the full lattice and compute observables
    INFO("BEFORE UPDATE");
    MPI_Barrier(MPI_COMM_WORLD);
    if (proc_ID == 0) {
        for (auto &row : local_lattice) {
            for (auto &site : row) {
                cout << site << "\t";
            }

            cout << endl;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    update(right, left, up, down, parity, local_lattice);


    INFO("AFTER UPDATE");
    MPI_Barrier(MPI_COMM_WORLD);
    if (proc_ID == 0) {
        for (auto &row : local_lattice) {
            for (auto &site : row) {
                cout << site << "\t";
            }

            cout << endl;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);


    // Finalize
    MPI_Finalize();

    return 0;
}

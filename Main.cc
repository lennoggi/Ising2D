#include <cassert>
#include <cstdlib>
#include <ctime>
#include <array>
#include <mpi.h>

#include "include/Declarations.hh"
#include "include/Macros.hh"
#include "Parameters.hh"

using namespace std;


int main(int argc, char **argv) {
    // Initialize
    MPI_Init(&argc, &argv);

    // Get the total number of processes
    int N_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &N_procs);
    assert(N_procs > 0);

    // Get the ID of the current process
    int proc_ID;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_ID);
    assert(proc_ID >= 0 and proc_ID < N_procs);

    // Perform sanity checks on the parameters
    check_parameters(N_procs);

    // Compute the size of the process-local grid
    const int nx = NX/N_PROCS_X;
    const int ny = NY/N_PROCS_Y;

    // Print some info
    #if (VERBOSE)
        INFO("beta = " << BETA);
        INFO("There are " << N_procs << " processes");
        INFO("The process-local grid size is " << nx << "*" << ny);
    #endif


    /* Find out the neighbors of the current process
     * **CONVENTIONS**
     * 1. Process IDs increase by 1 along y
     * 2. Periodic BCs on both x and y                                          */
    // TODO: put this in a routine and select the BCs through a parameter
    const int right = (proc_ID >= N_procs - N_PROCS_Y) ?        // Right edge
            proc_ID - N_procs + N_PROCS_Y : proc_ID + N_PROCS_Y;
    const int left  = (proc_ID < N_PROCS_Y) ?                   // Left edge
            proc_ID + N_procs - N_PROCS_Y : proc_ID - N_PROCS_Y;
    const int up    = (proc_ID % N_PROCS_Y == N_PROCS_Y - 1) ?  // Upper edge
            proc_ID - N_PROCS_Y + 1 : proc_ID + 1;
    const int down  = (proc_ID % N_PROCS_Y == 0) ?              // Lower edge
            proc_ID + N_PROCS_Y - 1 : proc_ID - 1;

    // Sanity check
    assert(right >= 0 and right < N_procs);
    assert(left  >= 0 and left  < N_procs);
    assert(up    >= 0 and up    < N_procs);
    assert(down  >= 0 and down  < N_procs);


    // Assing a 'parity' to the current process
    const int  x_index = proc_ID/N_PROCS_Y;
    const bool parity  = ((proc_ID + x_index) % 2 == 0) ? true : false;

    // Sanity check
    const array<int, 4> neighbors = {right, left, up, down};
    for (const auto &n : neighbors) {
        const int x_index_n = n/N_PROCS_Y;
        const int parity_n  = ((n + x_index_n) % 2 == 0) ? true : false;
        assert(parity_n != parity);
    }


    // Initialize the seed of the random number generator
    srand(time(NULL) + proc_ID);

    // TODO: thermalize the lattice and compute observables

    // Finalize
    MPI_Finalize();

    return 0;
}

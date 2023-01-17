#include "include/Declarations.hh"
#include "include/Macros.hh"
#include "Parameters.hh"


void check_parameters(const int &nprocs) {
    static_assert(BETA > 0.);

    static_assert(NPROCS_X > 0);
    static_assert(NPROCS_Y > 0);

    /* Make sure processes can communicate with each other through the 'parity'
     * technique                                                                */
    static_assert(NPROCS_X % 2 == 0);
    static_assert(NPROCS_Y % 2 == 0);

    if (NPROCS_X*NPROCS_Y != nprocs) {
        ERROR("The total number of processes (" << nprocs <<
              ") doesn't match the desired domain decomposition (" <<
              NPROCS_X << "*" << NPROCS_Y << " = " <<
              NPROCS_X*NPROCS_Y << ")");
    }

    static_assert(NX >= NPROCS_X);
    static_assert(NY >= NPROCS_Y);

    static_assert(NX % NPROCS_X == 0);
    static_assert(NY % NPROCS_Y == 0);

    static_assert(NTHERM > 0);
    static_assert(NCALC  > 0);

    static_assert(SAVE_LATTICE_DURING_THERMALIZATION or not SAVE_LATTICE_DURING_THERMALIZATION);
    static_assert(OUT_EVERY < NTHERM);

    return;
}

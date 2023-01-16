#include "include/Declarations.hh"
#include "include/Macros.hh"
#include "Parameters.hh"


void check_parameters(const int &N_procs) {
    static_assert(BETA > 0.);

    static_assert(N_PROCS_X > 0);
    static_assert(N_PROCS_Y > 0);

    /* Make sure processes can communicate with each other through the 'parity'
     * technique                                                                */
    static_assert(N_PROCS_X % 2 == 0);
    static_assert(N_PROCS_Y % 2 == 0);

    if (N_PROCS_X*N_PROCS_Y != N_procs) {
        ERROR("The total number of processes (" << N_procs <<
              ") doesn't match the desired domain decomposition (" <<
              N_PROCS_X << "*" << N_PROCS_Y << " = " <<
              N_PROCS_X*N_PROCS_Y << ")");
    }

    static_assert(NX >= N_PROCS_X);
    static_assert(NY >= N_PROCS_Y);

    static_assert(NX % N_PROCS_X == 0);
    static_assert(NY % N_PROCS_Y == 0);

    static_assert(N_THERM > 0);
    static_assert(N_CALC  > 0);

    return;
}

#ifndef CHECK_PARAMETERS_HH
#define CHECK_PARAMETERS_HH

#include "../Parameters.hh"


static_assert(BETA > 0.);

/* Need at least two processes per dimension for the parity-based ghost point
 * exchange to work                                                             */
static_assert(NPROCS_X1 > 1);
static_assert(NPROCS_X2 > 1);

/* Make sure processes can be arranged like a chessboard, so that
 * communications can happen via the 'parity' technique                         */ 
static_assert(NPROCS_X1 % 2 == 0);
static_assert(NPROCS_X2 % 2 == 0);

static_assert(NX1 >= NPROCS_X1);
static_assert(NX2 >= NPROCS_X2);

static_assert(NX1 % NPROCS_X1 == 0);
static_assert(NX2 % NPROCS_X2 == 0);


static_assert(NTHERM > 0);
static_assert(NCALC  > 0);


static_assert(SAVE_LATTICE_THERM or not SAVE_LATTICE_THERM);
static_assert(SAVE_LATTICE_CALC  or not SAVE_LATTICE_CALC);

static_assert(OUT_EVERY > 0);


static_assert(OUTERR_ALL_RANKS or not OUTERR_ALL_RANKS);

static_assert(VERBOSE     or not VERBOSE);
//static_assert(VERYVERBOSE or not VERYVERBOSE);
//#if (VERYVERBOSE)
//static_assert(VERBOSE);
//#endif


#endif  // PARAMETERS_HH

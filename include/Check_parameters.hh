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

/* The number of points along each dimension in the process-local lattice must
 * be even for the checkerboard lattice update pattern (see Update.cc and
 * Update_device.cu) to work                                                    */
static_assert(NX1 % NPROCS_X1 == 0);
static_assert(NX2 % NPROCS_X2 == 0);


#ifdef USE_CUDA
static_assert(BLOCK_SIZE_X1 > 0);
static_assert(BLOCK_SIZE_X2 > 0);

static_assert(BLOCK_SIZE_X1 <= (NX1/NPROCS_X1)/2,
              "The number of CUDA threads per block along X1 can't exceed twice the number of points in the process-local lattice along X1 (checkerboard lattice update pattern => Only half of the spins are updated within a single kernel call)");
static_assert(BLOCK_SIZE_X2 <= (NX2/NPROCS_X2)/2,
              "The number of CUDA threads per block along X2 can't exceed twice the number of points in the process-local lattice along X2 (checkerboard lattice update pattern => Only half of the spins are updated within a single kernel call)");

static_assert(BLOCK_SIZE_X1*BLOCK_SIZE_X2 >= 32,
              "Less than 32 CUDA threads per block lead to poor slot usage within each CUDA's thread warp, whose size is 32");
static_assert(BLOCK_SIZE_X1*BLOCK_SIZE_X2 <= 1024,
              "More than 1024 CUDA threads per block are unsupported on most (especially older) GPU architectures, and even then you might kill efficiency");
#endif


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

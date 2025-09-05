#ifndef DECLARATIONS_HH
#define DECLARATIONS_HH

#include "../Parameters.hh"


// Process-local interior lattice size
constexpr inline int nx1loc = NX1/NPROCS_X1;
constexpr inline int nx2loc = NX2/NPROCS_X2;

/* Both nx1loc=NX1/NPROCS_X1 and nx2loc=NX2LOC/NPROCS_X2 must be EVEN for the
 * parity update method to work (see Update.cc and Update_device.cu)            */
static_assert(nx1loc % 2 == 0);
static_assert(nx2loc % 2 == 0);

// Size of the chunks of data to send to and receive from each process
constexpr inline int nx1loc_half = nx1loc/2;
constexpr inline int nx2loc_half = nx2loc/2;

// Indices of the last ghost row and column on the process-local lattice
constexpr inline int nx1loc_p1 = nx1loc + 1;
constexpr inline int nx2loc_p1 = nx2loc + 1;

/* Full extent (including ghost points) of the process-local lattice along x2
 * (needed to build the index along x1 in the flattened process-local lattice)  */
constexpr inline int nx1loc_p2 = nx1loc + 2;
constexpr inline int nx2loc_p2 = nx2loc + 2;

// Interior size of a quarter of the process-local lattice
constexpr inline int nx1lochalf_nx2lochalf = nx1loc_half*nx2loc_half;

// Total size (including ghost points) of the flattened process-local lattice
constexpr inline int nx1locp2_nx2locp2 = nx1loc_p2*nx2loc_p2;

#ifdef USE_CUDA
/* Launch the lattice update kernel on each quarter of the process-local lattice
 * using a single block if the quarter is small enough, or use multiple blocks
 * of MAX_BLOCK_SIZE_X1*MAX_BLOCK_SIZE_X2 threads each otherwise                */
constexpr inline int block_size_x1 = std::min(nx1loc_half, MAX_BLOCK_SIZE_X1);
constexpr inline int block_size_x2 = std::min(nx2loc_half, MAX_BLOCK_SIZE_X2);
#endif

// Helper variables for communication
//constexpr inline int nx1locp2_p1          = nx1loc + 3;
constexpr inline int nx1loc_nx2loc        = nx1loc*nx2loc;
//constexpr inline int nx1loc_nx2locp2_p1   = nx1loc_nx2loc + 3;
//constexpr inline int nx1locp1_nx2locp2_p1 = nx1loc_nx2loc + 4;
//constexpr inline int nx2locp2_p1          = nx2loc + 3;
//constexpr inline int nx2locp2_p_nx2loc    = nx2loc_p2 + nx2loc;
//constexpr inline int nx2locp2_p_nx2locp1  = nx2loc_p2 + nx2loc_p1;


#endif  // DECLARATIONS_HH
